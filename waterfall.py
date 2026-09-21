"""
瀑布图组件
===========
横轴 = 频率, 纵轴 = 时间 (新数据在上, 向下滚动),
颜色 = 信号强度 (dB).

实现要点:
- 用 pg.ImageItem + np.roll 滚动一个 2D 缓冲
- 频率轴用 transform 映射, 不重建图像
- 65535 点 × 200 行 ≈ 13M 像素, 太大. 显示时降采样到屏幕宽度 (e.g. 2048)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg


# Turbo colormap (matplotlib 的高对比度色图; 比默认 viridis 更适合频谱瀑布)
# 用 ColorMap 自己构造, 避免依赖 matplotlib
_TURBO_STOPS = [
    (0.00, (48, 18, 59)),
    (0.13, (66, 75, 219)),
    (0.25, (35, 159, 248)),
    (0.38, (27, 226, 158)),
    (0.50, (138, 248, 49)),
    (0.63, (235, 220, 47)),
    (0.75, (252, 154, 47)),
    (0.88, (220, 65, 22)),
    (1.00, (122, 4, 3)),
]


def _make_turbo_lut(n: int = 256) -> np.ndarray:
    pos = np.array([s[0] for s in _TURBO_STOPS])
    cols = np.array([s[1] for s in _TURBO_STOPS], dtype=np.float32)
    xs = np.linspace(0, 1, n)
    lut = np.zeros((n, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(xs, pos, cols[:, c]).astype(np.uint8)
    return lut


class WaterfallView(pg.PlotWidget):
    """
    一个滚动瀑布图.
    用法:
        wf = WaterfallView(history_rows=200, display_bins=2048)
        wf.set_freq_range(start_hz, stop_hz)
        wf.push_row(spectrum_db)       # 每来一帧调一次
    """

    DEFAULT_LEVELS = (-100.0, -20.0)   # dB 显示范围

    def __init__(self,
                 history_rows: int = 200,
                 display_bins: int = 2048,
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setLabel("bottom", "Frequency", units="Hz")
        self.setLabel("left", "Time", units="frames")
        self.invertY(True)   # 新数据在顶部, 向下滚

        self.history_rows = history_rows
        self.display_bins = display_bins

        # 图像缓冲: rows x bins, 存 float32 (dB)
        # 用 NaN 表示"未填充", 但 ImageItem 不接受 NaN; 用一个固定低值
        self._img = np.full((history_rows, display_bins),
                            self.DEFAULT_LEVELS[0], dtype=np.float32)
        # 当前已填充的行数 (用于初期未填满时的视觉)
        self._filled = 0

        # ImageItem
        self._image_item = pg.ImageItem(
            image=self._img,
            axisOrder="row-major",
            autoLevels=False,
        )
        self._image_item.setLevels(self.DEFAULT_LEVELS)
        self._image_item.setLookupTable(_make_turbo_lut())
        self.addItem(self._image_item)

        # 频率范围 (用于把图像 x 坐标映射到真实频率)
        self._freq_start_hz = 0.0
        self._freq_stop_hz = 1.0
        self._apply_transform()

        # 默认范围
        self.setYRange(0, history_rows)
        # 关闭鼠标缩放垂直方向 (瀑布图垂直缩放意义不大且容易误操作)
        self.getViewBox().setMouseEnabled(x=True, y=False)
        self.showGrid(x=True, y=False, alpha=0.25)

    # ---- 公开接口 ----

    def set_freq_range(self, start_hz: float, stop_hz: float) -> None:
        """设置 x 轴对应的频率范围. 来自频谱消息的 center ± bw/2."""
        if (abs(start_hz - self._freq_start_hz) < 1
                and abs(stop_hz - self._freq_stop_hz) < 1):
            return   # 没变化, 跳过
        self._freq_start_hz = start_hz
        self._freq_stop_hz = stop_hz
        self._apply_transform()
        self.setXRange(start_hz, stop_hz, padding=0)

    def set_levels(self, low_db: float, high_db: float) -> None:
        """调整颜色映射范围"""
        self._image_item.setLevels((low_db, high_db))

    def push_row(self, spectrum_db: np.ndarray) -> None:
        """
        推入一帧新数据 (顶部插入, 向下滚动).
        spectrum_db: 任意长度的 float 数组 (dB).
        自动降采样到 display_bins.
        """
        row = self._downsample(spectrum_db, self.display_bins)

        # 向下滚一行: 用 np.roll 比 memmove 慢, 但代码清晰
        self._img[1:] = self._img[:-1]
        self._img[0] = row

        self._filled = min(self._filled + 1, self.history_rows)

        # 通知 ImageItem 更新 (autoLevels=False, 直接传图像)
        self._image_item.setImage(self._img, autoLevels=False)

    def clear(self) -> None:
        self._img.fill(self.DEFAULT_LEVELS[0])
        self._filled = 0
        self._image_item.setImage(self._img, autoLevels=False)

    # ---- 内部 ----

    def _apply_transform(self) -> None:
        """把图像的像素坐标变换到真实的频率坐标"""
        from PyQt6.QtGui import QTransform
        bw = self._freq_stop_hz - self._freq_start_hz
        if bw <= 0 or self.display_bins <= 0:
            return
        x_scale = bw / self.display_bins
        tr = QTransform()
        tr.translate(self._freq_start_hz, 0)
        tr.scale(x_scale, 1.0)
        self._image_item.setTransform(tr)

    @staticmethod
    def _downsample(data: np.ndarray, target_bins: int) -> np.ndarray:
        """
        降采样到 target_bins, 用 max-pooling 保留峰值.
        使用均匀区间起点做 reduceat，确保每个输入点都落入某个显示列。
        """
        source = np.asarray(data, dtype=np.float32)
        n = len(source)
        if n == 0:
            return np.full(target_bins, -np.inf, dtype=np.float32)
        if n == target_bins:
            return source
        if n < target_bins:
            xs = np.linspace(0, n - 1, target_bins)
            return np.interp(xs, np.arange(n), source).astype(np.float32)

        # 每个显示列对应原始数据中的一个连续区间。不补齐到块长整数倍，
        # 否则 6401 -> 2048 这类非整除数据会在右侧产生空白列。
        starts = np.linspace(0, n, target_bins, endpoint=False, dtype=np.intp)
        return np.maximum.reduceat(source, starts)
