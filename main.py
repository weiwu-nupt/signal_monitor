"""
FPGA 频谱分析器 主窗口
======================
上半部分: 实时频谱
下半部分: 瀑布图 (历史记录)
"""

from __future__ import annotations
from fpga_closed_loop import (
    EnergyDetector,
    SpectrumDDCClosedLoop,
    FPGASpectrumMessage,
    TargetFilter,
)


import sys
import time
from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from data_source import DataSource
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
        self.addLegend(offset=(10, 10))

        self.setDownsampling(auto=True, mode="peak")
        self.setClipToView(True)

        self._curve = self.plot(
            pen=pg.mkPen(color="#00d4ff", width=1.2),
            name="C2H1 fine assembled",
        )
        self._wide_reference = self.plot(
            pen=pg.mkPen(color="#ffd166", width=1.4,
                         style=QtCore.Qt.PenStyle.DashLine),
            name="C2H0 coarse reference",
        )
        self._wide_reference.setVisible(True)
        self._maxhold = self.plot(
            pen=pg.mkPen(color="#ff8c42", width=1.0,
                         style=QtCore.Qt.PenStyle.DashLine)
        )
        self._maxhold.setVisible(False)
        self._maxhold_data: Optional[np.ndarray] = None

        self._cur_freq: np.ndarray = np.array([])
        self._cur_spec: np.ndarray = np.array([])
        self._signal_box_items = []

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

    def update_wide_reference(self, freq_hz: np.ndarray, spec_db: np.ndarray) -> None:
        """Overlay C2H0 on the C2H1 view without affecting waterfall data."""
        self._wide_reference.setData(freq_hz, spec_db)

    def set_wide_reference_visible(self, visible: bool) -> None:
        self._wide_reference.setVisible(visible)

    def set_signal_boxes(self, detected_targets, selected_targets) -> None:
        """Draw numbered occupied-band boxes; green boxes are IQ-selected."""
        for item in self._signal_box_items:
            self.removeItem(item)
        self._signal_box_items.clear()
        if self._cur_freq.size == 0:
            return
        selected = set(selected_targets)
        counters = {"wide": 0, "narrow": 0}
        for target in detected_targets:
            counters[target.target_type] = counters.get(target.target_type, 0) + 1
            source_tag = "W" if target.target_type == "wide" else "N"
            signal_id = f"{source_tag}{counters[target.target_type]}"
            left = target.center_hz - target.bandwidth_hz / 2.0
            right = target.center_hz + target.bandwidth_hz / 2.0
            mask = (self._cur_freq >= left) & (self._cur_freq <= right)
            if not np.any(mask):
                continue
            segment = self._cur_spec[mask]
            bottom = float(segment.min()) - 2.0
            top = float(segment.max()) + 2.0
            is_selected = target in selected
            color = "#35d07f" if is_selected else "#8a919d"
            style = QtCore.Qt.PenStyle.SolidLine if is_selected else QtCore.Qt.PenStyle.DashLine
            rect = QtWidgets.QGraphicsRectItem(left, bottom, right - left, top - bottom)
            rect.setPen(pg.mkPen(color=color, width=1.5, style=style))
            self.addItem(rect)
            label = pg.TextItem(
                html=(
                    f"<span style='color:{color}; font-weight:600'>"
                    f"#{signal_id}: {target.center_hz / 1e6:.2f} MHz, "
                    f"{target.bandwidth_hz / 1e6:.2f} MHz"
                    f"{'' if is_selected else ' (filtered)'}</span>"
                ),
                anchor=(0.5, 1),
            )
            # Putting the text in the horizontal centre keeps a full-band
            # detection legible instead of hiding its label under the legend.
            label.setPos((left + right) / 2.0, top)
            self.addItem(label)
            self._signal_box_items.extend((rect, label))

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

class FPGAMainWindow(QtWidgets.QMainWindow):
    def __init__(self, source: DataSource):
        super().__init__()
        self.setWindowTitle("FPGA Spectrum Analyzer")
        self.resize(1280, 800)

        self.source = source
        self._fps_count = 0
        self._fps_last_time = time.monotonic()
        self._last_spectrum: Optional[SpectrumMessage] = None
        self._last_wide_reference: Optional[SpectrumMessage] = None

        self._build_ui()
        self._apply_target_bandwidth_filter()
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
        self.chk_wide_reference = QtWidgets.QCheckBox("Show C2H0 Coarse Reference")
        self.chk_wide_reference.setChecked(True)
        self.chk_wide_reference.toggled.connect(self.spectrum_view.set_wide_reference_visible)
        glyt.addWidget(self.chk_wide_reference)
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

        # 实时能量检测门限；调整后从下一帧开始生效。
        grp_detect = QtWidgets.QGroupBox("Energy Detection")
        dlyt = QtWidgets.QFormLayout(grp_detect)
        self.spin_detect_snr = QtWidgets.QDoubleSpinBox()
        self.spin_detect_snr.setRange(0.5, 30.0)
        self.spin_detect_snr.setSingleStep(0.5)
        self.spin_detect_snr.setDecimals(1)
        self.spin_detect_snr.setValue(4.0)
        self.spin_detect_snr.setSuffix(" dB above noise floor")
        self.spin_detect_snr.valueChanged.connect(self._on_detect_snr_changed)
        dlyt.addRow("Threshold:", self.spin_detect_snr)
        plyt.addWidget(grp_detect)

        # DDC IQ请求筛选。频谱仍显示全部检测结果，只有范围内的才下发FPGA。
        grp_target = QtWidgets.QGroupBox("IQ Target Bandwidth")
        tlyt = QtWidgets.QFormLayout(grp_target)
        self.spin_target_bw_min = QtWidgets.QDoubleSpinBox()
        self.spin_target_bw_min.setRange(0.0, 4096.0)
        self.spin_target_bw_min.setValue(30.0)
        self.spin_target_bw_min.setDecimals(3)
        self.spin_target_bw_min.setSuffix(" MHz")
        self.spin_target_bw_max = QtWidgets.QDoubleSpinBox()
        self.spin_target_bw_max.setRange(0.0, 4096.0)
        self.spin_target_bw_max.setValue(40.0)
        self.spin_target_bw_max.setDecimals(3)
        self.spin_target_bw_max.setSuffix(" MHz")
        self.spin_target_bw_min.valueChanged.connect(self._on_target_bandwidth_changed)
        self.spin_target_bw_max.valueChanged.connect(self._on_target_bandwidth_changed)
        tlyt.addRow("Minimum:", self.spin_target_bw_min)
        tlyt.addRow("Maximum:", self.spin_target_bw_max)
        self.lbl_target_filter = QtWidgets.QLabel("Request IQ: 30.000–40.000 MHz")
        self.lbl_target_filter.setWordWrap(True)
        tlyt.addRow(self.lbl_target_filter)
        plyt.addWidget(grp_target)

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
        # C2H0 is intentionally not sent to the waterfall or the target-box
        # display.  It is an amber dashed comparison curve for checking the
        # C2H1 Main/Shift frequency stitching.
        if isinstance(msg, FPGASpectrumMessage) and msg.target_type == "wide":
            self._last_wide_reference = msg
            self.spectrum_view.update_wide_reference(msg.freq_axis_hz, msg.spectrum_db)
            return

        self._last_spectrum = msg
        self._fps_count += 1
        if self._fps_count <= 3:
            print(
                f"[GUI] spectrum rendered: center={msg.center_freq_hz:.0f} Hz, "
                f"bw={msg.bandwidth_hz:.0f} Hz, points={msg.fft_points}",
                flush=True,
            )

        freq = msg.freq_axis_hz
        spec = msg.spectrum_db

        lo = float(spec.min()) - 5
        hi = float(spec.max()) + 10
        self.spectrum_view.setYRange(lo, hi)

        # 频谱图
        self.spectrum_view.update_spectrum(freq, spec)
        if isinstance(msg, FPGASpectrumMessage):
            # Broad signals are detected from C2H0 but are intentionally drawn
            # over C2H1, the only spectrum shown as the primary trace.
            wide_detected = ()
            wide_selected = ()
            if isinstance(self._last_wide_reference, FPGASpectrumMessage):
                wide_detected = self._last_wide_reference.detected_targets
                wide_selected = self._last_wide_reference.selected_targets
            self.spectrum_view.set_signal_boxes(
                tuple(wide_detected) + tuple(msg.detected_targets),
                tuple(wide_selected) + tuple(msg.selected_targets),
            )

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

    def _on_target_bandwidth_changed(self) -> None:
        self._apply_target_bandwidth_filter()

    def _on_detect_snr_changed(self) -> None:
        setter = getattr(self.source, "set_detector_snr_db", None)
        if setter is not None:
            setter(self.spin_detect_snr.value())

    def _apply_target_bandwidth_filter(self) -> None:
        minimum = self.spin_target_bw_min.value()
        maximum = self.spin_target_bw_max.value()
        if minimum > maximum:
            self.lbl_target_filter.setText("Invalid range: minimum exceeds maximum")
            return
        self.lbl_target_filter.setText(
            f"Request IQ: {minimum:.3f}–{maximum:.3f} MHz"
        )
        setter = getattr(self.source, "set_bandwidth_filter", None)
        if setter is not None:
            setter(minimum * 1e6, maximum * 1e6)

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


# 默认入口直接使用接收机 FFM 页面，不启动旧 FPGA 采集流程。
from receiver.integrate import _StandaloneWindow


class MainWindow(_StandaloneWindow):
    pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
