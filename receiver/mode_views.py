"""
接收机模式显示部件 (pyqtgraph)
================================
各模式专用界面左侧"显示区"用到的可复用绘图部件:
    SpectrumPlot   - 频谱曲线 (可选 max-hold)
    WaterfallPlot  - 瀑布图 (滚动)
    ScanPlot       - 扫描曲线 (宽频段)
    FastScanPlot   - 信号点 + 噪声底
    ChannelBars    - 信道柱状图 (MSCAN)

颜色与宿主 FPGA 程序一致 (深色 + 青色 + Turbo 瀑布).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from .ffm_spectrum import FrequencyAxis


CYAN = "#00d4ff"
ORANGE = "#ff8c42"
GREEN = "#5bff8c"
GRID_A = 0.25


# ---- Turbo colormap (与 waterfall.py 一致) ----
_TURBO_STOPS = [
    (0.00, (48, 18, 59)), (0.13, (66, 75, 219)), (0.25, (35, 159, 248)),
    (0.38, (27, 226, 158)), (0.50, (138, 248, 49)), (0.63, (235, 220, 47)),
    (0.75, (252, 154, 47)), (0.88, (220, 65, 22)), (1.00, (122, 4, 3)),
]


def _turbo_lut(n: int = 256) -> np.ndarray:
    pos = np.array([s[0] for s in _TURBO_STOPS])
    cols = np.array([s[1] for s in _TURBO_STOPS], dtype=np.float32)
    xs = np.linspace(0, 1, n)
    lut = np.zeros((n, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, cols[:, c]).astype(np.uint8)
    return lut


# ============================================================
# 频谱曲线
# ============================================================

class SpectrumPlot(pg.PlotWidget):
    def __init__(self, parent=None, ylabel="电平", yunit="dBµV"):
        super().__init__(parent)
        self.setLabel("bottom", "频率", units="Hz")
        self.setLabel("left", ylabel, units=yunit)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.showGrid(x=True, y=True, alpha=GRID_A)
        self.setDownsampling(auto=True, mode="peak")
        self.setClipToView(True)
        self.curve = self.plot(pen=pg.mkPen(CYAN, width=1.2))
        self.maxhold = self.plot(
            pen=pg.mkPen(ORANGE, width=1.0, style=QtCore.Qt.PenStyle.DashLine))
        self.maxhold.setVisible(False)
        self._mh: Optional[np.ndarray] = None

    def update_spectrum(self, freq_hz, spec_db):
        freq_hz = np.ascontiguousarray(freq_hz)
        spec_db = np.ascontiguousarray(spec_db, dtype=np.float32)
        self.curve.setData(freq_hz, spec_db)
        if self.maxhold.isVisible():
            if self._mh is None or self._mh.shape != spec_db.shape:
                self._mh = spec_db.copy()
            else:
                np.maximum(self._mh, spec_db, out=self._mh)
            self.maxhold.setData(freq_hz, self._mh)

    def set_maxhold(self, on: bool):
        self.maxhold.setVisible(on)
        if not on:
            self._mh = None
            self.maxhold.setData([], [])

    def reset_maxhold(self):
        self._mh = None
        self.maxhold.setData([], [])

    def set_reference_scale(self, reference_dbuv: float, division_db: float,
                            divisions: int = 10):
        """设置频谱纵轴：参考电平为上边界，每格为 division_db。"""
        self.setYRange(reference_dbuv - division_db * divisions,
                       reference_dbuv, padding=0)


# ============================================================
# 瀑布图
# ============================================================

class WaterfallPlot(pg.PlotWidget):
    DEFAULT_LEVELS = (-100.0, -20.0)

    def __init__(self, history_rows=200, display_bins=2048, parent=None,
                 levels=None):
        self.frequency_axis = FrequencyAxis(orientation="bottom")
        super().__init__(parent, axisItems={"bottom": self.frequency_axis},
                         background="#050a11")
        self.setLabel("left", "历史", units="帧")
        self.setMenuEnabled(False)
        self.hideButtons()
        self.invertY(True)
        self.history_rows = history_rows
        self.display_bins = display_bins
        lv = levels or self.DEFAULT_LEVELS
        self._levels = lv
        self._img = np.full((history_rows, display_bins), lv[0], dtype=np.float32)
        self._item = pg.ImageItem(image=self._img, axisOrder="row-major",
                                  autoLevels=False)
        self._item.setLevels(lv)
        self._item.setLookupTable(_turbo_lut())
        self.addItem(self._item)
        self._f0 = 0.0
        self._f1 = 1.0
        self.getViewBox().setMouseEnabled(x=True, y=False)
        self.showGrid(x=True, y=False, alpha=GRID_A)
        self.setYRange(0, history_rows)
        self.getAxis("left").setWidth(72)
        for name in ("left", "bottom"):
            axis = self.getAxis(name)
            axis.setPen(pg.mkPen("#7894af"))
            axis.setTextPen(pg.mkPen("#a9c1dc"))
            axis.setStyle(tickTextOffset=8)
        self.getViewBox().sigRangeChanged.connect(self._sync_frequency_ticks)

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

    def set_freq_range(self, start_hz, stop_hz):
        bw = stop_hz - start_hz
        if bw <= 0:
            return
        self.setLimits(xMin=start_hz, xMax=stop_hz,
                       minXRange=max(1, bw / 16384), maxXRange=bw)
        if abs(start_hz - self._f0) < 1 and abs(stop_hz - self._f1) < 1:
            return
        self._f0, self._f1 = start_hz, stop_hz
        from PyQt6.QtGui import QTransform
        bw = stop_hz - start_hz
        if bw <= 0:
            return
        tr = QTransform()
        tr.translate(start_hz, 0)
        tr.scale(bw / self.display_bins, 1.0)
        self._item.setTransform(tr)
        self.setXRange(start_hz, stop_hz, padding=0)

    def set_levels(self, low, high):
        self._levels = (low, high)
        self._item.setLevels(self._levels)

    def push_row(self, spec_db):
        row = self._downsample(spec_db, self.display_bins)
        self._img[1:] = self._img[:-1]
        self._img[0] = row
        self._item.setImage(self._img, autoLevels=False)

    def clear_history(self):
        self._img.fill(self._levels[0])
        self._item.setImage(self._img, autoLevels=False)

    @staticmethod
    def _downsample(data, target):
        source = np.asarray(data, dtype=np.float32)
        n = len(source)
        if n == 0:
            return np.full(target, -np.inf, dtype=np.float32)
        if n == target:
            return source
        if n < target:
            xs = np.linspace(0, n - 1, target)
            return np.interp(xs, np.arange(n), source).astype(np.float32)
        starts = np.linspace(0, n, target, endpoint=False, dtype=np.intp)
        return np.maximum.reduceat(source, starts)


# ============================================================
# 扫描曲线 (宽频段)
# ============================================================

class ScanPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLabel("bottom", "频率", units="Hz")
        self.setLabel("left", "Level", units="dBµV")
        self.setMenuEnabled(False)
        self.hideButtons()
        self.showGrid(x=True, y=True, alpha=GRID_A)
        self.setDownsampling(auto=True, mode="peak")
        self.setClipToView(True)
        self.curve = self.plot(pen=pg.mkPen(CYAN, width=1.0))

    def update_scan(self, freq_hz, level_db):
        self.curve.setData(np.ascontiguousarray(freq_hz),
                           np.ascontiguousarray(level_db, dtype=np.float32))


# ============================================================
# 快速扫描: 信号点 + 噪声底
# ============================================================

class FastScanPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLabel("bottom", "频率", units="Hz")
        self.setLabel("left", "Level", units="dBµV")
        self.setMenuEnabled(False)
        self.hideButtons()
        self.showGrid(x=True, y=True, alpha=GRID_A)
        self.noise = self.plot(pen=pg.mkPen(ORANGE, width=1.0,
                                            style=QtCore.Qt.PenStyle.DashLine))
        self.signal = pg.ScatterPlotItem(size=4, brush=pg.mkBrush(CYAN),
                                         pen=None)
        self.addItem(self.signal)

    def update_fastscan(self, freq_hz, signal_db, noise_db):
        self.signal.setData(freq_hz, signal_db)
        self.noise.setData(freq_hz, noise_db)


# ============================================================
# 信道柱状图 (MSCAN)
# ============================================================

class ChannelBars(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLabel("bottom", "Channel")
        self.setLabel("left", "Level", units="dBµV")
        self.showGrid(x=False, y=True, alpha=GRID_A)
        self._bar: Optional[pg.BarGraphItem] = None
        self._thr_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(GREEN, width=1,
                                          style=QtCore.Qt.PenStyle.DashLine))
        self.addItem(self._thr_line)
        self._thr_line.setVisible(False)

    def update_bars(self, levels_db, threshold=None):
        x = np.arange(len(levels_db))
        if self._bar is not None:
            self.removeItem(self._bar)
        # 超过门限的柱用青色, 否则灰色
        if threshold is not None:
            brushes = [pg.mkBrush(CYAN) if v >= threshold else pg.mkBrush("#3a4452")
                       for v in levels_db]
        else:
            brushes = pg.mkBrush(CYAN)
        self._bar = pg.BarGraphItem(x=x, height=levels_db, width=0.7, brushes=brushes
                                    if isinstance(brushes, list) else None,
                                    brush=None if isinstance(brushes, list) else brushes)
        self.addItem(self._bar)
        if threshold is not None:
            self._thr_line.setPos(threshold)
            self._thr_line.setVisible(True)
