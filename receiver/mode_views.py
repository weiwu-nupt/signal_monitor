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
    def __init__(self, parent=None, ylabel="Level", yunit="dBµV"):
        super().__init__(parent)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", ylabel, units=yunit)
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


# ============================================================
# 瀑布图
# ============================================================

class WaterfallPlot(pg.PlotWidget):
    DEFAULT_LEVELS = (-100.0, -20.0)

    def __init__(self, history_rows=200, display_bins=2048, parent=None,
                 levels=None):
        super().__init__(parent)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Time", units="frames")
        self.invertY(True)
        self.history_rows = history_rows
        self.display_bins = display_bins
        lv = levels or self.DEFAULT_LEVELS
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

    def set_freq_range(self, start_hz, stop_hz):
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
        self._item.setLevels((low, high))

    def push_row(self, spec_db):
        row = self._downsample(spec_db, self.display_bins)
        self._img[1:] = self._img[:-1]
        self._img[0] = row
        self._item.setImage(self._img, autoLevels=False)

    def clear(self):
        self._img.fill(self.DEFAULT_LEVELS[0])
        self._item.setImage(self._img, autoLevels=False)

    @staticmethod
    def _downsample(data, target):
        n = len(data)
        if n == target:
            return data.astype(np.float32, copy=False)
        if n < target:
            xs = np.linspace(0, n - 1, target)
            return np.interp(xs, np.arange(n), data).astype(np.float32)
        if n % target == 0:
            return data.reshape(target, n // target).max(axis=1).astype(np.float32)
        block = (n + target - 1) // target
        padded = np.full(block * target, -np.inf, dtype=np.float32)
        padded[:n] = data
        return padded.reshape(target, block).max(axis=1)


# ============================================================
# 扫描曲线 (宽频段)
# ============================================================

class ScanPlot(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Level", units="dBµV")
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
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Level", units="dBµV")
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