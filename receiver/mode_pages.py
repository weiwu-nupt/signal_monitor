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

import math
import numpy as np

from .controller import ReceiverController, RxMode
from .ffm_spectrum import FFMSpectrumPlot, TraceProcessor
from .inputs import NumberInput, NoWheelComboBox
from .theme import APP_STYLE
from .mode_panels import (
    FFMPanel, PScanPanel, FScanPanel, MScanPanel, DDCPanel,
)
from .mode_views import (
    SpectrumPlot, WaterfallPlot, ScanPlot, FastScanPlot, ChannelBars,
)
from .rx_protocol import (
    SpectrumData, SpectrumStitcher, ScanData, FastScanData, LevelData, DDCData,
)


PAGE_STYLE = APP_STYLE


class _BasePage(QtWidgets.QWidget):
    """页面基类: 左显示区 + 右参数 + 顶部动作栏"""

    mode: RxMode = None
    title: str = ""

    def __init__(self, controller: ReceiverController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setObjectName("receiverPage")
        self.setStyleSheet(PAGE_STYLE)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # 顶部: 标题 + 动作
        top = QtWidgets.QHBoxLayout()
        if self.title:
            lbl = QtWidgets.QLabel(self.title)
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
        right.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        right.setMinimumWidth(290)
        right.setMaximumWidth(440)
        self.panel = self._build_panel()
        right.setWidget(self.panel)
        split.addWidget(right)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([1060, 330])
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        root.addWidget(split, 1)

        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_start.clicked.connect(self._on_start)
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

    def _on_start(self):
        applied = self._on_apply()
        if applied is not False and self.controller.is_connected():
            self.controller.start()

    def set_enabled_actions(self, connected: bool):
        self.btn_apply.setEnabled(connected)
        self.btn_start.setEnabled(connected)
        self.btn_stop.setEnabled(False)


# ============================================================
# FFM: 频谱 + 瀑布
# ============================================================

class FFMPage(_BasePage):
    mode = RxMode.FFM
    title = ""

    def _build_display(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.addWidget(QtWidgets.QLabel("迹线"))
        self.trace_mode = NoWheelComboBox()
        self.trace_mode.addItems(TraceProcessor.MODES)
        self.trace_mode.setFixedWidth(128)
        toolbar.addWidget(self.trace_mode)
        self.average_label = QtWidgets.QLabel("平均次数")
        self.average_count = NumberInput(2, 100, 10, name="平均次数", decimals=0)
        self.average_count.setMaximumWidth(75)
        self.average_count.setToolTip("最近 N 帧在线性功率域平均")
        toolbar.addWidget(self.average_label)
        toolbar.addWidget(self.average_count)
        self.average_label.hide()
        self.average_count.hide()
        toolbar.addStretch()
        self.auto_adjust = QtWidgets.QPushButton("自动调整")
        self.auto_adjust.setToolTip("恢复完整频率范围，并按当前频谱自动设置电平刻度")
        toolbar.addWidget(self.auto_adjust)
        for text, step in (("↑", 1), ("↓", -1)):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setToolTip("参考电平上调一格" if step > 0 else "参考电平下调一格")
            button.clicked.connect(lambda _checked=False, direction=step: self._shift_reference(direction))
            toolbar.addWidget(button)
        layout.addLayout(toolbar)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.spectrum = FFMSpectrumPlot()
        self.waterfall = WaterfallPlot(history_rows=200, display_bins=2048)
        self.waterfall.setBackground("#050a11")
        self.waterfall.getAxis("left").setWidth(72)
        self.waterfall.setXLink(self.spectrum)
        split.addWidget(self.spectrum)
        split.addWidget(self.waterfall)
        split.setCollapsible(0, False)
        split.setCollapsible(1, False)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([450, 300])
        layout.addWidget(split, 1)
        self.trace_mode.currentTextChanged.connect(self._trace_mode_changed)
        self.average_count.editingFinished.connect(self._average_changed)
        self.auto_adjust.clicked.connect(self._auto_adjust)
        return w

    def _build_panel(self):
        return FFMPanel()

    def _connect_signals(self):
        self.controller.spectrumReady.connect(self._on_spectrum)
        self.controller.connectedChanged.connect(self._connection_changed)
        self._last_ffm_update_ms = 0
        self._refresh_ms = 100
        self._spectrum_stitcher = SpectrumStitcher()
        self._apply_display_axes(self.get_config())
        self.waterfall.clear_history()
        for editor in (self.panel.ref, self.panel.lvspan):
            editor.editingFinished.connect(self._update_level_scale)
        self.panel.dpx_refresh.editingFinished.connect(self._update_refresh)

    def _connection_changed(self, _connected):
        self.spectrum.clear_trace()
        self.waterfall.clear_history()
        self._spectrum_stitcher.configure(*self._configured_band)
        self._last_ffm_update_ms = 0

    def _trace_mode_changed(self, mode):
        self.spectrum.set_trace_mode(mode)
        self.average_label.setVisible(mode == "平均")
        self.average_count.setVisible(mode == "平均")

    def _average_changed(self):
        try:
            self.spectrum.set_average_count(self.average_count.value())
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(error))

    def _update_refresh(self):
        try:
            self._refresh_ms = self.panel.dpx_refresh.value()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(error))

    def _update_level_scale(self):
        try:
            reference, spacing, divisions = self.panel.display_scale()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(error))
            return False
        self.spectrum.set_reference_scale(reference, spacing, divisions)
        self.waterfall.set_levels(reference - spacing * divisions, reference)
        return True

    def _shift_reference(self, direction):
        try:
            reference, spacing, _ = self.panel.display_scale()
            reference = min(self.panel.ref.maximum, max(self.panel.ref.minimum, reference + direction * spacing))
            self.panel.ref.setValue(reference)
            self._update_level_scale()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(error))

    def _auto_adjust(self):
        self.spectrum.reset_view()
        values = self.spectrum.values
        values = values[np.isfinite(values)]
        if values.size < 3:
            self._update_level_scale()
            return
        lower10, lower30 = np.percentile(values, (10, 30))
        noise = float(lower30 + .7 * (lower30 - lower10))
        peak = float(np.max(values))
        required = (peak - noise + max(5, (peak - noise) * .1)) / 6
        spacing = next((x for x in (1, 2, 5, 10, 20, 50) if x >= required), 50)
        reference = min(self.panel.ref.maximum, max(self.panel.ref.minimum, math.ceil(noise + 6 * spacing)))
        self.panel.ref.setValue(reference)
        self.panel.lvspan.setValue(spacing)
        self._update_level_scale()

    def _apply_display_axes(self, config):
        center = config.center_mhz * 1e6
        span = config.span_khz * 1e3
        # 零跨距仍按 0 下发；坐标保留 1 Hz 的最小显示范围，避免除零。
        display_span = max(1.0, span)
        start, stop = center - display_span / 2, center + display_span / 2
        self.waterfall.set_freq_range(start, stop)
        self.waterfall.setLimits(xMin=start, xMax=stop, minXRange=max(1, display_span / 16384), maxXRange=display_span)
        self.spectrum.set_frequency_range(start, stop)
        self._update_level_scale()
        self._configured_band = (center, span)
        self._spectrum_stitcher.configure(center, span)

    def _on_apply(self):
        try:
            config = self.get_config()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(error))
            return False
        self._apply_display_axes(config)
        self.spectrum.clear_trace()
        self.waterfall.clear_history()
        self._spectrum_stitcher.configure(*self._configured_band)
        self._last_ffm_update_ms = 0
        self._refresh_ms = config.dpx_refresh_ms
        self.controller.apply_mode(self.mode, config)
        return True

    @QtCore.pyqtSlot(object)
    def _on_spectrum(self, data: SpectrumData):
        if self.controller.current_mode != self.mode:
            return
        complete = self._spectrum_stitcher.add(data)
        if complete is None:
            return
        # Process every completed frame so peaks/averages survive display throttling.
        self.spectrum.update_spectrum(complete.freq_axis_hz, complete.levels_db, render=False)
        now = QtCore.QDateTime.currentMSecsSinceEpoch()
        if now - self._last_ffm_update_ms < self._refresh_ms:
            return
        self._last_ffm_update_ms = now
        self.spectrum.render_trace()
        self.waterfall.push_row(complete.levels_db)


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
