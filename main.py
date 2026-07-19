"""
FPGA 频谱分析器 主窗口
======================
上半部分: 实时频谱
下半部分: 瀑布图 (历史记录)
"""

from __future__ import annotations
from dma_source import DMASpectrumSource


import sys
import time
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from data_source import DataSource, create_source
from protocol import SpectrumMessage, SignalMessage
from waterfall import WaterfallView


pg.setConfigOption("background", "#0e1116")
pg.setConfigOption("foreground", "#d0d8e0")
pg.setConfigOption("antialias", False)


# ============================================================
# 频谱视图
# ============================================================

class SpectrumView(pg.PlotWidget):
    """实时频谱曲线"""

    cursorMoved = QtCore.pyqtSignal(float, float)

    MOUSE_THROTTLE_S = 0.033

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Power", units="dB")
        self.showGrid(x=True, y=True, alpha=0.25)

        self.setDownsampling(auto=True, mode="peak")
        self.setClipToView(True)

        self._curve = self.plot(pen=pg.mkPen(color="#00d4ff", width=1.2))
        self._maxhold = self.plot(
            pen=pg.mkPen(color="#ff8c42", width=1.0,
                         style=QtCore.Qt.PenStyle.DashLine)
        )
        self._maxhold.setVisible(False)
        self._maxhold_data: Optional[np.ndarray] = None

        self._cur_freq: np.ndarray = np.array([])
        self._cur_spec: np.ndarray = np.array([])

        # 鼠标十字线 + 节流
        self._vline = pg.InfiniteLine(angle=90, pen=pg.mkPen("#555"))
        self._hline = pg.InfiniteLine(angle=0, pen=pg.mkPen("#555"))
        self.addItem(self._vline, ignoreBounds=True)
        self.addItem(self._hline, ignoreBounds=True)
        self._last_mouse_t = 0.0
        self.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def update_spectrum(self, freq_hz: np.ndarray, spec_db: np.ndarray) -> None:
        if spec_db.dtype != np.float32 or not spec_db.flags["C_CONTIGUOUS"]:
            spec_db = np.ascontiguousarray(spec_db, dtype=np.float32)
        if not freq_hz.flags["C_CONTIGUOUS"]:
            freq_hz = np.ascontiguousarray(freq_hz)
        self._cur_freq = freq_hz
        self._cur_spec = spec_db
        self._curve.setData(freq_hz, spec_db)

        if self._maxhold.isVisible():
            if self._maxhold_data is None or self._maxhold_data.shape != spec_db.shape:
                self._maxhold_data = spec_db.copy()
            else:
                np.maximum(self._maxhold_data, spec_db, out=self._maxhold_data)
            self._maxhold.setData(freq_hz, self._maxhold_data)

    def set_maxhold(self, enabled: bool) -> None:
        self._maxhold.setVisible(enabled)
        if enabled and self._cur_spec.size > 0:
            self._maxhold_data = self._cur_spec.copy()
        elif not enabled:
            self._maxhold_data = None

    def reset_maxhold(self) -> None:
        self._maxhold_data = None
        self._maxhold.setData([], [])

    def _on_mouse_moved(self, pos) -> None:
        now = time.monotonic()
        if now - self._last_mouse_t < self.MOUSE_THROTTLE_S:
            return
        self._last_mouse_t = now
        if not self.sceneBoundingRect().contains(pos):
            return
        view_pt = self.plotItem.vb.mapSceneToView(pos)
        f = view_pt.x()
        a = view_pt.y()
        self._vline.setPos(f)
        self._hline.setPos(a)
        if self._cur_freq.size > 0:
            idx = int(np.argmin(np.abs(self._cur_freq - f)))
            f = float(self._cur_freq[idx])
            a = float(self._cur_spec[idx])
        self.cursorMoved.emit(f, a)


# ============================================================
# 主窗口
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, source: DataSource):
        super().__init__()
        self.setWindowTitle("FPGA Spectrum Analyzer")
        self.resize(1280, 800)

        self.source = source
        self._fps_count = 0
        self._fps_last_time = time.monotonic()
        self._last_spectrum: Optional[SpectrumMessage] = None

        self._build_ui()
        self._build_menu()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_messages)
        self._poll_timer.start(33)

        self.source.set_error_callback(self._on_source_error)
        self.source.start()

        from receiver.integrate import attach_receiver_pages
        attach_receiver_pages(self)   # self 是 QMainWindow

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ====== 左侧: 上下分割的频谱+瀑布 ======
        left_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.spectrum_view = SpectrumView()
        self.spectrum_view.cursorMoved.connect(self._on_cursor)
        left_split.addWidget(self.spectrum_view)

        self.waterfall_view = WaterfallView(history_rows=200, display_bins=2048)
        left_split.addWidget(self.waterfall_view)

        # X 轴联动: 瀑布图缩放/平移时频谱图同步
        self.spectrum_view.setXLink(self.waterfall_view)

        left_split.setStretchFactor(0, 3)
        left_split.setStretchFactor(1, 4)

        main_layout.addWidget(left_split, stretch=4)

        # ====== 右侧: 控制面板 ======
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        panel.setFixedWidth(260)
        plyt = QtWidgets.QVBoxLayout(panel)
        plyt.setSpacing(6)

        # 显示控制
        grp_disp = QtWidgets.QGroupBox("Display")
        glyt = QtWidgets.QVBoxLayout(grp_disp)
        self.chk_maxhold = QtWidgets.QCheckBox("Max Hold")
        self.chk_maxhold.toggled.connect(self.spectrum_view.set_maxhold)
        glyt.addWidget(self.chk_maxhold)
        btn_reset_max = QtWidgets.QPushButton("Reset Max Hold")
        btn_reset_max.clicked.connect(self.spectrum_view.reset_maxhold)
        glyt.addWidget(btn_reset_max)
        btn_autoscale = QtWidgets.QPushButton("Auto Scale Y")
        btn_autoscale.clicked.connect(
            lambda: self.spectrum_view.enableAutoRange(axis="y"))
        glyt.addWidget(btn_autoscale)
        plyt.addWidget(grp_disp)

        # 瀑布图控制
        grp_wf = QtWidgets.QGroupBox("Waterfall")
        wlyt = QtWidgets.QFormLayout(grp_wf)
        self.spin_low = QtWidgets.QSpinBox()
        self.spin_low.setRange(-200, 0)
        self.spin_low.setValue(-100)
        self.spin_low.setSuffix(" dB")
        self.spin_high = QtWidgets.QSpinBox()
        self.spin_high.setRange(-100, 50)
        self.spin_high.setValue(-20)
        self.spin_high.setSuffix(" dB")
        self.spin_low.valueChanged.connect(self._on_wf_levels_changed)
        self.spin_high.valueChanged.connect(self._on_wf_levels_changed)
        wlyt.addRow("Color Min:", self.spin_low)
        wlyt.addRow("Color Max:", self.spin_high)
        btn_clear_wf = QtWidgets.QPushButton("Clear Waterfall")
        btn_clear_wf.clicked.connect(self.waterfall_view.clear)
        wlyt.addRow(btn_clear_wf)
        plyt.addWidget(grp_wf)

        # 帧信息
        grp_info = QtWidgets.QGroupBox("Frame Info")
        ilyt = QtWidgets.QFormLayout(grp_info)
        self.lbl_center = QtWidgets.QLabel("-")
        self.lbl_bw = QtWidgets.QLabel("-")
        self.lbl_points = QtWidgets.QLabel("-")
        self.lbl_res = QtWidgets.QLabel("-")
        ilyt.addRow("Center:", self.lbl_center)
        ilyt.addRow("Bandwidth:", self.lbl_bw)
        ilyt.addRow("FFT Points:", self.lbl_points)
        ilyt.addRow("Resolution:", self.lbl_res)
        plyt.addWidget(grp_info)

        # 统计
        grp_stats = QtWidgets.QGroupBox("Statistics")
        slyt = QtWidgets.QFormLayout(grp_stats)
        self.lbl_spectrum_ok = QtWidgets.QLabel("0")
        self.lbl_signal_ok = QtWidgets.QLabel("0")
        self.lbl_bad_crc = QtWidgets.QLabel("0")
        self.lbl_dropped = QtWidgets.QLabel("0")
        slyt.addRow("Spectrum frames:", self.lbl_spectrum_ok)
        slyt.addRow("Signal frames:", self.lbl_signal_ok)
        slyt.addRow("CRC errors:", self.lbl_bad_crc)
        slyt.addRow("Bytes dropped:", self.lbl_dropped)
        plyt.addWidget(grp_stats)

        plyt.addStretch(1)
        main_layout.addWidget(panel)

        # 状态栏
        self.lbl_cursor = QtWidgets.QLabel("Cursor: -")
        self.lbl_fps = QtWidgets.QLabel("FPS: 0.0")
        self.statusBar().addWidget(self.lbl_cursor, 2)
        self.statusBar().addPermanentWidget(self.lbl_fps)

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0e1116; color: #d0d8e0; }
            QGroupBox {
                border: 1px solid #2a3140;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 6px;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton {
                background: #1c2330;
                border: 1px solid #2e3a4f;
                padding: 6px 10px;
                border-radius: 3px;
            }
            QPushButton:hover { background: #2a3447; }
            QPushButton:pressed { background: #00d4ff; color: #000; }
            QSpinBox {
                background: #141923;
                border: 1px solid #2a3140;
                padding: 2px 4px;
            }
            QStatusBar { color: #aaa; }
        """)

    def _build_menu(self) -> None:
        mbar = self.menuBar()
        m_file = mbar.addMenu("&File")
        act_export = m_file.addAction("Export Current Spectrum CSV...")
        act_export.triggered.connect(self._export_csv)
        m_file.addSeparator()
        act_quit = m_file.addAction("Quit")
        act_quit.triggered.connect(self.close)

    # ---- 数据轮询 ----

    def _poll_messages(self) -> None:
        """从数据源拉取所有可用消息"""
        # 一次循环处理多条消息, 避免堆积
        processed = 0
        while processed < 10:
            msg = self.source.get_message(timeout=0)
            if msg is None:
                break
            if isinstance(msg, SpectrumMessage):
                self._handle_spectrum(msg)
            elif isinstance(msg, SignalMessage):
                self._handle_signal(msg)
            processed += 1

        # 每秒更新一次 FPS 和统计
        now = time.monotonic()
        if now - self._fps_last_time >= 1.0:
            fps = self._fps_count / (now - self._fps_last_time)
            self._fps_count = 0
            self._fps_last_time = now
            self.lbl_fps.setText(f"FPS: {fps:.1f}")
            self._refresh_stats()

    def _handle_spectrum(self, msg: SpectrumMessage) -> None:
        self._last_spectrum = msg
        self._fps_count += 1

        freq = msg.freq_axis_hz
        spec = msg.spectrum_db

        lo = float(spec.min()) - 5
        hi = float(spec.max()) + 10
        self.spectrum_view.setYRange(lo, hi)

        # 频谱图
        self.spectrum_view.update_spectrum(freq, spec)

        # 瀑布图
        self.waterfall_view.set_levels(lo, hi)
        self.waterfall_view.set_freq_range(
            float(freq[0]), float(freq[-1])
        )
        self.waterfall_view.push_row(spec)

        # 信息标签
        self.lbl_center.setText(f"{msg.center_freq_hz/1e6:.1f} MHz")
        self.lbl_bw.setText(f"{msg.bandwidth_hz/1e6:.1f} MHz")
        self.lbl_points.setText(f"{msg.fft_points}")
        res_khz = msg.bandwidth_hz / msg.fft_points / 1e3
        self.lbl_res.setText(f"{res_khz:.2f} kHz")

    def _handle_signal(self, msg: SignalMessage) -> None:
        """信号消息暂时只统计计数, 后续可加显示"""
        pass

    def _refresh_stats(self) -> None:
        stats = self.source.get_stats()
        self.lbl_spectrum_ok.setText(str(stats.get("spectrum_ok", 0)))
        self.lbl_signal_ok.setText(str(stats.get("signal_ok", 0)))
        self.lbl_bad_crc.setText(str(stats.get("bad_crc", 0)))
        self.lbl_dropped.setText(str(stats.get("bytes_dropped", 0)))

    # ---- UI 回调 ----

    def _on_cursor(self, f: float, a: float) -> None:
        self.lbl_cursor.setText(f"Cursor: {f/1e6:.4f} MHz, {a:.1f} dB")

    def _on_wf_levels_changed(self) -> None:
        low = self.spin_low.value()
        high = self.spin_high.value()
        if low >= high:
            return
        self.waterfall_view.set_levels(low, high)

    def _export_csv(self) -> None:
        if self._last_spectrum is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export CSV", "spectrum.csv", "CSV (*.csv)"
        )
        if not path:
            return
        m = self._last_spectrum
        np.savetxt(path,
                   np.column_stack([m.freq_axis_hz, m.spectrum_db]),
                   header="freq_hz,power_db", delimiter=",", comments="")

    def _on_source_error(self, exc: Exception) -> None:
        QtCore.QTimer.singleShot(0, lambda:
            QtWidgets.QMessageBox.critical(self, "Data Source Error", str(exc))
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.source.stop()
        super().closeEvent(event)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FPGA Spectrum Analyzer")
    parser.add_argument("--mode", choices=["sim", "pcie", "dma"], default="dma")
    parser.add_argument("--device", default="/dev/xdma0_c2h_0",
                        help="PCIe 字符设备路径")
    parser.add_argument("--fft", type=int, default=65535,
                        help="FFT 点数")
    parser.add_argument("--center", type=int, default=1500,
                        help="中心频率 (MHz, 仿真用)")
    parser.add_argument("--bw", type=int, default=300,
                        help="带宽 (MHz, 仿真用)")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="帧率 Hz (仿真用)")
    args = parser.parse_args()

    if args.mode == "sim":
        source = create_source(
            "sim",
            fft_points=args.fft,
            center_mhz=args.center,
            bandwidth_mhz=args.bw,
            frame_rate_hz=args.rate,
        )
    elif args.mode == "dma":
        # 真实硬件: 从 C2H 连续读取 8192 点频谱包。
        source = DMASpectrumSource(
            c2h_dev=args.device,
        )
    else:
        source = create_source(
            "pcie",
            device_path=args.device,
            fft_points=args.fft,
        )

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(source)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
