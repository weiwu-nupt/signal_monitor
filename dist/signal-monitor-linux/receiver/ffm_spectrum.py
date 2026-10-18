"""FFM spectrum interaction adapted from signal_analyse; levels stay in dBµV."""
from collections import deque
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QRectF, Qt


class FrequencyAxis(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._view_factor = 1.0
        self._view_unit = "Hz"
        self._view_spacing = 1.0

    @staticmethod
    def engineering_unit(value):
        absolute = abs(value)
        if absolute >= 1_000_000_000:
            return 1_000_000_000.0, "GHz"
        if absolute >= 1_000_000:
            return 1_000_000.0, "MHz"
        if absolute >= 1_000:
            return 1_000.0, "kHz"
        return 1.0, "Hz"

    def set_view_format(self, start_hz: float, end_hz: float, spacing_hz: float) -> None:
        reference = max(abs(start_hz), abs(end_hz))
        self._view_factor, self._view_unit = self.engineering_unit(reference)
        self._view_spacing = max(1e-12, abs(spacing_hz))

    @staticmethod
    def format_frequency(value, spacing, include_unit=True, factor=None, unit=None):
        """Format a frequency, optionally omitting the engineering unit."""
        if factor is None or unit is None:
            factor, unit = FrequencyAxis.engineering_unit(value)
        unit_spacing = abs(spacing) / factor
        digits = min(9, max(0, int(np.ceil(-np.log10(max(unit_spacing, 1e-12)))) + 1))
        text = f"{value / factor:.{digits}f}"
        return f"{text} {unit}" if include_unit else text

    def tickStrings(self, values, scale, spacing):
        if len(values) == 0:
            return []
        factor, unit = self.engineering_unit(max(abs(value) for value in values))
        return [
            self.format_frequency(value, spacing, include_unit=index == 0, factor=factor, unit=unit)
            for index, value in enumerate(values)
        ]

    def generateDrawSpecs(self, painter):
        """Draw endpoint labels inside the axis instead of letting Qt clip them."""
        specs = super().generateDrawSpecs(painter)
        if specs is None or self.orientation != "bottom":
            return specs
        axis_spec, tick_specs, text_specs = specs
        bounds = self.mapRectFromParent(self.geometry())
        spacing = self._view_spacing
        for value, edge, align in ((self.range[0], "left", Qt.AlignmentFlag.AlignLeft), (self.range[1], "right", Qt.AlignmentFlag.AlignRight)):
            text = self.format_frequency(
                value,
                spacing,
                include_unit=edge == "left",
                factor=self._view_factor,
                unit=self._view_unit,
            )
            rect = painter.boundingRect(QRectF(0, 0, 200, 100), Qt.AlignmentFlag.AlignCenter, text)
            rect.setHeight(rect.height() * 0.8)
            width, height = rect.width(), rect.height()
            x = bounds.left() if edge == "left" else bounds.right() - width
            label_rect = QRectF(x, bounds.top() + self.style["tickTextOffset"][1], width, height)
            text_specs.append((label_rect, align | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextDontClip, text))
        return axis_spec, tick_specs, text_specs


class TraceProcessor:
    """Professional spectrum trace modes, including finite power averaging."""
    MODES = ("实时", "最大保持", "最小保持", "平均")

    def __init__(self) -> None:
        self.mode = "实时"; self.average_count = 10; self.reset()

    def reset(self) -> None:
        self._held = np.empty(0, dtype=np.float32)
        self._power_sum = np.empty(0, dtype=np.float64)
        self._valid_count = np.empty(0, dtype=np.int32)
        self._history = deque()

    def configure(self, mode: str, average_count: int | None = None) -> None:
        if mode not in self.MODES: raise ValueError("频谱迹线模式无效")
        count = self.average_count if average_count is None else max(2, min(100, int(average_count)))
        if mode != self.mode or count != self.average_count:
            self.mode, self.average_count = mode, count; self.reset()

    def apply(self, values) -> np.ndarray:
        current = np.asarray(values, dtype=np.float32)
        if self.mode == "实时" or current.size == 0:
            return current
        if self.mode in {"最大保持", "最小保持"}:
            if self._held.size != current.size:
                self._held = current.copy()
            elif self.mode == "最大保持":
                np.fmax(self._held, current, out=self._held)
            else:
                np.fmin(self._held, current, out=self._held)
            return self._held.copy()

        if self._power_sum.size != current.size:
            self.reset(); self._power_sum = np.zeros(current.size, dtype=np.float64); self._valid_count = np.zeros(current.size, dtype=np.int32)
        valid = np.isfinite(current)
        power = np.zeros(current.size, dtype=np.float64)
        power[valid] = np.power(10.0, current[valid].astype(np.float64) / 10.0)
        self._history.append((power, valid))
        self._power_sum += power; self._valid_count += valid
        while len(self._history) > self.average_count:
            old_power, old_valid = self._history.popleft(); self._power_sum -= old_power; self._valid_count -= old_valid
        result = np.full(current.size, np.nan, dtype=np.float32)
        present = self._valid_count > 0
        result[present] = (10.0 * np.log10(self._power_sum[present] / self._valid_count[present])).astype(np.float32)
        return result


class FFMSpectrumPlot(pg.PlotWidget):
    referenceStepRequested = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        self.frequency_axis = FrequencyAxis(orientation="bottom")
        super().__init__(parent, axisItems={"bottom": self.frequency_axis}, background="#050a11")
        self.processor = TraceProcessor()
        self.frequencies = np.empty(0)
        self.values = np.empty(0, dtype=np.float32)
        self.display_values = np.empty(0, dtype=np.float32)
        self.frequency_limits = None
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setLabel("left", "电平", units="dBµV")
        self.showGrid(x=True, y=True, alpha=.28)
        self.setDownsampling(auto=True, mode="peak")
        self.setClipToView(True)
        self.getViewBox().setMouseEnabled(x=True, y=False)
        self.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.enableAutoRange(x=False, y=False)
        self.curve = self.plot(pen=pg.mkPen("#31d855", width=1.35), connect="finite")
        self.getAxis("left").setWidth(72)
        for name in ("left", "bottom"):
            axis = self.getAxis(name)
            axis.setPen(pg.mkPen("#7894af"))
            axis.setTextPen(pg.mkPen("#a9c1dc"))
            axis.setStyle(tickTextOffset=8)
        self.getViewBox().sigRangeChanged.connect(self._sync_frequency_ticks)

        self.hover_line = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen("#35d955", width=1.2))
        self.hover_dot = pg.ScatterPlotItem(
            size=12, pen=pg.mkPen("#35d955", width=1), brush=pg.mkBrush("#35d955"))
        label_font = QtGui.QFont("Microsoft YaHei UI", 10)
        self.level_label = pg.TextItem(
            color="#061008", fill=pg.mkBrush("#35d955"), anchor=(0, .5))
        self.frequency_label = pg.TextItem(
            color="#061008", fill=pg.mkBrush("#35d955"), anchor=(.5, 1))
        self.level_label.setFont(label_font)
        self.frequency_label.setFont(label_font)
        for item in (self.hover_line, self.hover_dot,
                     self.level_label, self.frequency_label):
            self.addItem(item, ignoreBounds=True)
            item.hide()
        self._mouse_proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=30, slot=self._mouse_moved)
        self.setToolTip("滚轮横向缩放，左键拖动平移；双击恢复频率范围。纵轴使用参考电平和电平跨度调整。")

    def _sync_frequency_ticks(self, _view, ranges):
        start, end = ranges[0]
        spacing = max(1e-6, (end - start) / 8)
        axis = self.frequency_axis
        axis.set_view_format(start, end, spacing)
        axis.setTicks([[(start + spacing * index,
                        "" if index in (0, 8) else axis.format_frequency(
                            start + spacing * index, spacing, include_unit=False,
                            factor=axis._view_factor, unit=axis._view_unit))
                       for index in range(9)]])

    def set_frequency_range(self, start, stop):
        if stop <= start:
            raise ValueError("频率范围无效")
        self.frequency_limits = (start, stop)
        span = stop - start
        self.setLimits(xMin=start, xMax=stop, minXRange=max(1, span / 16384), maxXRange=span)
        self.reset_view()

    def reset_view(self):
        if self.frequency_limits:
            self.setXRange(*self.frequency_limits, padding=0)

    def set_reference_scale(self, reference, spacing, divisions=10):
        bottom = reference - spacing * divisions
        self.setLimits(yMin=bottom, yMax=reference, minYRange=reference-bottom, maxYRange=reference-bottom)
        self.setYRange(bottom, reference, padding=0)
        self.getAxis("left").setTicks([[(reference - index * spacing, f"{reference - index * spacing:g}")
                                      for index in range(divisions + 1)]])

    def update_spectrum(self, frequencies, values, *, render=True):
        frequencies = np.ascontiguousarray(frequencies, dtype=np.float64)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if frequencies.shape != values.shape:
            raise ValueError("频率与电平数据长度不一致")
        if not np.array_equal(frequencies, self.frequencies):
            self.processor.reset()
        self.frequencies = frequencies
        self.values = values.copy()
        self.display_values = self.processor.apply(values)
        if render:
            self.render_trace()

    def render_trace(self):
        self.curve.setData(self.frequencies, self.display_values)

    def set_trace_mode(self, mode):
        if mode == self.processor.mode:
            return
        self.processor.configure(mode)
        self._reseed_trace()

    def set_average_count(self, count):
        self.processor.configure(self.processor.mode, count)
        self._reseed_trace()

    def _reseed_trace(self):
        self.processor.reset()
        self.display_values = self.processor.apply(self.values)
        self.render_trace()

    def clear_trace(self):
        self.processor.reset()
        self.frequencies = np.empty(0)
        self.values = np.empty(0, dtype=np.float32)
        self.display_values = np.empty(0, dtype=np.float32)
        self.curve.setData([], [])
        self._hide_hover()

    def _hide_hover(self):
        for item in (self.hover_line, self.hover_dot,
                     self.level_label, self.frequency_label):
            item.hide()

    def _mouse_moved(self, args):
        position = args[0]
        view = self.getViewBox()
        if not view.sceneBoundingRect().contains(position) or self.frequencies.size == 0:
            self._hide_hover()
            return
        point = view.mapSceneToView(position)
        index = int(np.clip(np.searchsorted(self.frequencies, point.x()), 0, self.frequencies.size - 1))
        if index and abs(self.frequencies[index-1] - point.x()) < abs(self.frequencies[index] - point.x()):
            index -= 1
        frequency, level = self.frequencies[index], self.display_values[index]
        if not np.isfinite(level):
            self._hide_hover()
            return
        self.hover_line.setPos(frequency)
        self.hover_dot.setData([frequency], [level])
        (left, right), (bottom, top) = view.viewRange()
        self.level_label.setText(f"{level:.2f} dBµV")
        self.level_label.setPos(frequency, level)
        spacing = max(1e-6, (right - left) / 8)
        self.frequency_label.setText(
            self.frequency_axis.format_frequency(frequency, spacing))
        self.frequency_label.setPos(frequency, bottom + (top - bottom) * .015)
        for item in (self.hover_line, self.hover_dot,
                     self.level_label, self.frequency_label):
            item.show()

    def leaveEvent(self, event):
        self._hide_hover()
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


