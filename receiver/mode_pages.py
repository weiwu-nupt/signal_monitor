"""
模式专用页面
=============
每个模式一个完整页面: 左侧 pyqtgraph 显示区 + 右侧参数面板.
菜单切换模式时, 主窗口中央 QStackedWidget 切到对应页.

每页:
  - 自带该模式的显示部件 (来自 mode_views)
  - 复用 mode_panels 的参数面板 (右侧)
  - 顶部有 应用配置 / 开始 / 停止
  - 订阅 controller 的数据信号, 只处理本模式相关数据
  - get_config() / mode 属性给上层下发用
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .controller import ReceiverController, RxMode
from .mode_panels import (
    FFMPanel, PScanPanel, FScanPanel, MScanPanel, DDCPanel,
)
from .mode_views import (
    SpectrumPlot, WaterfallPlot, ScanPlot, FastScanPlot, ChannelBars,
)
from .rx_protocol import (
    SpectrumData, ScanData, FastScanData, LevelData, DDCData,
)


PAGE_STYLE = """
QWidget { background: #0e1116; color: #d0d8e0; }
QGroupBox {
    border: 1px solid #2a3140; border-radius: 4px;
    margin-top: 10px; padding-top: 6px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton {
    background: #1c2330; border: 1px solid #2e3a4f;
    padding: 6px 10px; border-radius: 3px;
}
QPushButton:hover { background: #2a3447; }
QPushButton:pressed { background: #00d4ff; color: #000; }
QPushButton:disabled { color: #555; border-color: #222; }
QPushButton#primary { background: #00d4ff; color: #001018; font-weight: bold; }
QPushButton#danger { background: #3a1c22; border-color: #5a2a33; }
QLineEdit, QComboBox {
    background: #141923; border: 1px solid #334155; padding: 4px 30px 4px 7px;
    border-radius: 2px;
}
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: top right;
    width: 24px; border-left: 1px solid #334155; background: #202b3b;
}
QComboBox::down-arrow {
    width: 0; height: 0; border-left: 5px solid transparent;
    border-right: 5px solid transparent; border-top: 6px solid #9fb0c0;
}
QComboBox QAbstractItemView {
    background: #141923; border: 1px solid #3b4b63;
    selection-background-color: #00a8cc; selection-color: #001018;
}
QSpinBox, QDoubleSpinBox {
    background: #141923; border: 1px solid #334155;
    border-radius: 2px;
    padding: 4px 26px 4px 7px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 22px; border-left: 1px solid #334155;
    border-top-right-radius: 2px; background: #202b3b;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 22px; border-left: 1px solid #334155;
    border-bottom-right-radius: 2px; background: #202b3b;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #31445f;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #00a8cc;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none; width: 0; height: 0;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-bottom: 6px solid #c8d5e3;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid #c8d5e3;
}
QTableWidget { background: #141923; gridline-color: #2a3140; }
QHeaderView::section { background: #1c2330; padding: 4px; border: none; }
QLabel#pageTitle { font-size: 15px; font-weight: bold; color: #00d4ff; }
"""


class _BasePage(QtWidgets.QWidget):
    """页面基类: 左显示区 + 右参数 + 顶部动作栏"""

    mode: RxMode = None
    title: str = ""

    def __init__(self, controller: ReceiverController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setStyleSheet(PAGE_STYLE)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # 顶部: 标题 + 动作
        top = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel("▶ " + self.title)
        lbl.setObjectName("pageTitle")
        top.addWidget(lbl)
        top.addStretch(1)
        self.btn_apply = QtWidgets.QPushButton("应用配置")
        self.btn_start = QtWidgets.QPushButton("开始")
        self.btn_start.setObjectName("primary")
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_stop.setObjectName("danger")
        top.addWidget(self.btn_apply)
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        root.addLayout(top)

        # 主体: 左显示 | 右参数
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.display_area = self._build_display()
        split.addWidget(self.display_area)

        right = QtWidgets.QScrollArea()
        right.setWidgetResizable(True)
        right.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        right.setMinimumWidth(410)
        right.setMaximumWidth(480)
        self.panel = self._build_panel()
        right.setWidget(self.panel)
        split.addWidget(right)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 430])
        root.addWidget(split, 1)

        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_start.clicked.connect(self.controller.start)
        self.btn_stop.clicked.connect(self.controller.stop)

        self._connect_signals()
        self.set_enabled_actions(False)

    # 子类实现
    def _build_display(self) -> QtWidgets.QWidget: ...
    def _build_panel(self) -> QtWidgets.QWidget: ...
    def _connect_signals(self): ...

    def get_config(self):
        return self.panel.get_config()

    def apply_mode(self) -> RxMode:
        """返回实际下发的 mode (PSCAN/FPSCAN 可能由复选覆盖)"""
        return self.mode

    def _on_apply(self):
        self.controller.apply_mode(self.apply_mode(), self.get_config())

    def set_enabled_actions(self, connected: bool):
        self.btn_apply.setEnabled(connected)
        self.btn_start.setEnabled(connected)
        self.btn_stop.setEnabled(connected)


# ============================================================
# FFM: 频谱 + 瀑布
# ============================================================

class FFMPage(_BasePage):
    mode = RxMode.FFM
    title = "FFM 固定频点监测"

    def _build_display(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.spectrum = SpectrumPlot()
        self.waterfall = WaterfallPlot(history_rows=200, display_bins=2048)
        self.spectrum.setXLink(self.waterfall)
        split.addWidget(self.spectrum)
        split.addWidget(self.waterfall)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        v.addWidget(split)
        return w

    def _build_panel(self):
        return FFMPanel()

    def _connect_signals(self):
        self.controller.spectrumReady.connect(self._on_spectrum)
        self._last_ffm_update_ms = 0
        # 首包频谱到达前，也使用右侧面板的默认配置显示坐标。
        self._apply_display_axes(self.get_config())

    def _apply_display_axes(self, config):
        """按 FFM 配置同步频谱和瀑布图的横纵坐标。"""
        center_hz = (config.center_mhz * 1_000_000
                     + config.offset_khz * 1_000)
        half_span_hz = config.span_khz * 1_000 / 2
        start_hz = center_hz - half_span_hz
        stop_hz = center_hz + half_span_hz
        low_dbuv = config.ref_level_dbuv - config.level_span_db * 10

        self.spectrum.enableAutoRange(axis="x", enable=False)
        self.spectrum.enableAutoRange(axis="y", enable=False)
        self.spectrum.setXRange(start_hz, stop_hz, padding=0)
        self.spectrum.set_reference_scale(config.ref_level_dbuv,
                                          config.level_span_db)
        self.waterfall.set_freq_range(start_hz, stop_hz)
        self.waterfall.set_levels(low_dbuv, config.ref_level_dbuv)

    def _on_apply(self):
        config = self.get_config()
        self._apply_display_axes(config)
        self.controller.apply_mode(self.apply_mode(), config)

    @QtCore.pyqtSlot(object)
    def _on_spectrum(self, d: SpectrumData):
        if self.controller.current_mode != RxMode.FFM:
            return
        # DPX 刷新时间控制本地曲线/瀑布重绘速率，不丢弃接收机数据链路。
        now_ms = QtCore.QDateTime.currentMSecsSinceEpoch()
        if now_ms - self._last_ffm_update_ms < self.panel.dpx_refresh.value():
            return
        self._last_ffm_update_ms = now_ms
        f = d.freq_axis_hz
        self.spectrum.update_spectrum(f, d.levels_db)
        self.waterfall.push_row(d.levels_db)


# ============================================================
# PSCAN / FPSCAN: 共用页 (复选切换)
# ============================================================

class PScanPage(_BasePage):
    mode = RxMode.PSCAN
    title = "PSCAN / FPSCAN 全景扫描"

    def _build_display(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.scan = ScanPlot()
        self.fast = FastScanPlot()
        self.waterfall = WaterfallPlot(history_rows=200, display_bins=2048)
        split.addWidget(self.scan)
        split.addWidget(self.fast)
        split.addWidget(self.waterfall)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 3)
        # 默认按普通全景显示 scan + 瀑布, fast 隐藏
        self.fast.setVisible(False)
        v.addWidget(split)
        return w

    def _build_panel(self):
        p = PScanPanel()
        p.fast.toggled.connect(self._on_fast_toggle)
        return p

    def _on_fast_toggle(self, fast: bool):
        self.fast.setVisible(fast)
        self.scan.setVisible(not fast)
        self.waterfall.setVisible(not fast)

    def apply_mode(self) -> RxMode:
        return RxMode.FPSCAN if self.panel.fast.isChecked() else RxMode.PSCAN

    def _connect_signals(self):
        self.controller.scanReady.connect(self._on_scan)
        self.controller.fastScanReady.connect(self._on_fast)

    @QtCore.pyqtSlot(object)
    def _on_scan(self, d: ScanData):
        if self.controller.current_mode != RxMode.PSCAN:
            return
        if d.scan_kind != "pscan":
            return
        f = d.freq_axis_hz
        self.scan.update_scan(f, d.levels_db)
        self.waterfall.set_freq_range(float(f[0]), float(f[-1]))
        self.waterfall.push_row(d.levels_db)

    @QtCore.pyqtSlot(object)
    def _on_fast(self, d: FastScanData):
        if self.controller.current_mode != RxMode.FPSCAN:
            return
        self.fast.update_fastscan(d.signal_freq_hz, d.signal_db, d.noise_db)


# ============================================================
# FSCAN: 扫频结果 + 当前频谱
# ============================================================

class FScanPage(_BasePage):
    mode = RxMode.FSCAN
    title = "FSCAN 扫频搜索"

    def _build_display(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.scan = ScanPlot()
        self.scan.setLabel("top", "扫频结果")
        self.spectrum = SpectrumPlot()
        self.spectrum.setLabel("top", "当前频谱")
        split.addWidget(self.scan)
        split.addWidget(self.spectrum)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        v.addWidget(split)
        return w

    def _build_panel(self):
        return FScanPanel()

    def _connect_signals(self):
        self.controller.scanReady.connect(self._on_scan)
        self.controller.spectrumReady.connect(self._on_spectrum)

    @QtCore.pyqtSlot(object)
    def _on_scan(self, d: ScanData):
        if self.controller.current_mode != RxMode.FSCAN:
            return
        if d.scan_kind != "fscan":
            return
        self.scan.update_scan(d.freq_axis_hz, d.levels_db)

    @QtCore.pyqtSlot(object)
    def _on_spectrum(self, d: SpectrumData):
        if self.controller.current_mode != RxMode.FSCAN:
            return
        self.spectrum.update_spectrum(d.freq_axis_hz, d.levels_db)


# ============================================================
# MSCAN: 信道柱状 + 频道表 (表在右侧参数里)
# ============================================================

class MScanPage(_BasePage):
    mode = RxMode.MSCAN
    title = "MSCAN 存储信道扫描"

    def _build_display(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.bars = ChannelBars()
        self.spectrum = SpectrumPlot()
        self.spectrum.setLabel("top", "驻留频谱")
        split.addWidget(self.bars)
        split.addWidget(self.spectrum)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        v.addWidget(split)
        return w

    def _build_panel(self):
        return MScanPanel()

    def _connect_signals(self):
        self.controller.scanReady.connect(self._on_scan)
        self.controller.levelReady.connect(self._on_level)
        self.controller.spectrumReady.connect(self._on_spectrum)
        self._levels = {}

    @QtCore.pyqtSlot(object)
    def _on_scan(self, d: ScanData):
        if self.controller.current_mode != RxMode.MSCAN:
            return
        if d.scan_kind != "mscan":
            return
        # 扫描数据每个点视为一个信道电平
        thr = self.panel.thr.value() if hasattr(self.panel, "thr") else None
        self.bars.update_bars(d.levels_db, threshold=thr)

    @QtCore.pyqtSlot(object)
    def _on_level(self, d: LevelData):
        if self.controller.current_mode != RxMode.MSCAN:
            return
        # 可按频点累积电平 (此处略, 交给扫描数据驱动柱状图)

    @QtCore.pyqtSlot(object)
    def _on_spectrum(self, d: SpectrumData):
        if self.controller.current_mode != RxMode.MSCAN:
            return
        self.spectrum.update_spectrum(d.freq_axis_hz, d.levels_db)


# ============================================================
# DDC: 主通道频谱 + 子通道频谱网格
# ============================================================

class DDCPage(_BasePage):
    mode = RxMode.DDC
    title = "DDC 数字下变频"

    MAX_SUB = 8   # 显示前 8 个子通道

    def _build_display(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.main_spectrum = SpectrumPlot()
        self.main_spectrum.setLabel("top", "主通道频谱")
        split.addWidget(self.main_spectrum)

        # 子通道小频谱网格
        sub_w = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(sub_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(3)
        self.sub_plots = []
        for i in range(self.MAX_SUB):
            p = pg_mini_plot(f"DDC{i}")
            self.sub_plots.append(p)
            grid.addWidget(p, i // 4, i % 4)
            p.setVisible(False)
        split.addWidget(sub_w)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        v.addWidget(split)
        return w

    def _build_panel(self):
        return DDCPanel()

    def _connect_signals(self):
        self.controller.spectrumReady.connect(self._on_spectrum)
        self.controller.ddcReady.connect(self._on_ddc)

    @QtCore.pyqtSlot(object)
    def _on_spectrum(self, d: SpectrumData):
        if self.controller.current_mode != RxMode.DDC:
            return
        self.main_spectrum.update_spectrum(d.freq_axis_hz, d.levels_db)

    @QtCore.pyqtSlot(object)
    def _on_ddc(self, d: DDCData):
        if self.controller.current_mode != RxMode.DDC:
            return
        for i, ch in enumerate(d.channels[:self.MAX_SUB]):
            p = self.sub_plots[i]
            p.setVisible(True)
            import numpy as np
            x = np.arange(len(ch.spectrum_db))
            p.curve.setData(x, ch.spectrum_db)
        # 隐藏多余的
        for i in range(len(d.channels), self.MAX_SUB):
            self.sub_plots[i].setVisible(False)


def pg_mini_plot(title: str):
    import pyqtgraph as pg
    p = pg.PlotWidget()
    p.setTitle(title, color="#8a96a5", size="8pt")
    p.showGrid(x=False, y=True, alpha=0.2)
    p.getAxis("bottom").setStyle(showValues=False)
    p.curve = p.plot(pen=pg.mkPen("#00d4ff", width=1.0))
    p.setMinimumHeight(90)
    return p


# 模式 -> 页面类
PAGE_CLASSES = {
    RxMode.FFM: FFMPage,
    RxMode.PSCAN: PScanPage,     # 同页含 FPSCAN
    RxMode.FSCAN: FScanPage,
    RxMode.MSCAN: MScanPage,
    RxMode.DDC: DDCPage,
}
