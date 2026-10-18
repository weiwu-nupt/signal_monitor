"""
各模式参数面板
===============
每个面板暴露 get_config() 返回对应 dataclass.
风格对齐宿主 FPGA 程序 (深色 + 青色), 不自带 stylesheet,
继承父级样式即可.
"""

from __future__ import annotations

from typing import List

from PyQt6 import QtCore, QtWidgets

from .inputs import FrequencyInput, NumberInput, NoWheelComboBox

from .controller import (
    FFMConfig, FFM_SPANS_MHZ, PScanConfig, FScanConfig,
    MScanConfig, MScanChannel, DDCConfig, DDCChannel,
)


DEMOD_MODES = ["IQ", "AM", "FM", "CW", "LSB", "USB", "PULSE", "PM", "DMR", "DPMR"]
RF_MODES = ["NORMal", "LOWNoise", "LOWDistort"]
DETECTORS = ["FAST", "POSitive", "AVERage", "RMS"]
FFM_STEPS_KHZ = ["12.5", "25", "50", "100", "200"]
FFM_BANDWIDTHS_KHZ = ["1.5", "2.4", "6", "9", "12", "15", "30", "50", "120", "150", "250", "300", "500"]


def _dspin(lo, hi, val, step=1.0, suffix="", decimals=3):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    if suffix:
        s.setSuffix(suffix)
    s.setMinimumWidth(176)
    s.setMinimumHeight(28)
    return s


def _ispin(lo, hi, val, step=1, suffix=""):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    s.setMinimumWidth(176)
    s.setMinimumHeight(28)
    return s


def _on_off_combo(default: bool = False) -> QtWidgets.QComboBox:
    """创建统一的 ON/OFF 开关下拉框。"""
    combo = QtWidgets.QComboBox()
    combo.addItem("开启", True)
    combo.addItem("关闭", False)
    combo.setCurrentIndex(0 if default else 1)
    combo.setMinimumWidth(176)
    combo.setMinimumHeight(28)
    return combo


def _is_on(combo: QtWidgets.QComboBox) -> bool:
    return bool(combo.currentData())


# ============================================================
# FFM
# ============================================================

class FFMPanel(QtWidgets.QWidget):
    """参数取自《接收机配置》图 1 / 6.5 和 SCPI 2.2；仅输入样式参考工作台。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self.parameter_tabs = QtWidgets.QTabWidget()
        root.addWidget(self.parameter_tabs)
        pages = {}
        for title in ("频率参数", "电平参数", "高级参数"):
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            page = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(page)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)
            scroll.setWidget(page)
            self.parameter_tabs.addTab(scroll, title)
            pages[title] = layout

        def field(tab, label, editor):
            container = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(5)
            caption = QtWidgets.QLabel(label)
            caption.setObjectName("muted")
            layout.addWidget(caption)
            layout.addWidget(editor)
            pages[tab].addWidget(container)
            return editor

        def number(tab, label, low, high, value, decimals=3):
            return field(tab, label, NumberInput(low, high, value, name=label, decimals=decimals))

        def combo(tab, label, choices, default=None):
            editor = NoWheelComboBox()
            for text, value in choices:
                editor.addItem(text, value)
            if default is not None:
                editor.setCurrentIndex(editor.findData(default))
            return field(tab, label, editor)

        frequency, level, advanced = pages
        self.center = field(frequency, "中心频率", FrequencyInput(
            2_000_000_000, minimum_hz=0, maximum_hz=8_000_000_000, field_name="中心频率"))
        self.span = combo(frequency, "频谱带宽",
                          [(f"{mhz} MHz", float(mhz * 1000)) for mhz in FFM_SPANS_MHZ],
                          320000.0)
        self.step = combo(frequency, "频谱步进", [(f"{x} kHz", float(x)) for x in FFM_STEPS_KHZ], 12.5)

        self.ref = number(level, "参考电平（dBµV）", -256, 256, 120, 1)
        self.lvspan = number(level, "电平跨度（dB/格）", 1, 50, 20, 0)
        self.range_hint = QtWidgets.QLabel()
        self.range_hint.setObjectName("muted")
        self.range_hint.setWordWrap(True)
        pages[level].addWidget(self.range_hint)
        self.xdb_band = number(level, "XdB 带宽（dB）", 0, 100, 26, 1)
        self.beta_band = number(level, "β% 带宽（%）", 0, 99.9, 0, 2)

        demods = ["同相正交（IQ）", "调幅（AM）", "调频（FM）", "连续波（CW）", "下边带", "上边带",
                  "脉冲", "调相（PM）", "数字移动无线电（DMR）", "数字专用移动无线电（DPMR）"]
        self.demod = combo(advanced, "解调模式", list(zip(demods, DEMOD_MODES)), "FM")
        self.bw = combo(advanced, "解调带宽", [(f"{x} kHz", float(x)) for x in FFM_BANDWIDTHS_KHZ], 500.0)
        switches = [("开启", True), ("关闭", False)]
        self.sql = combo(advanced, "静噪开关", switches, False)
        self.attauto = combo(advanced, "自动增益（AGC）", switches, False)
        self.rfatt = number(advanced, "射频衰减（dB）", 0, 50, 0, 0)
        self.ifatt = number(advanced, "中频衰减（dB）", 0, 50, 0, 0)
        rf_modes = list(zip(["正常", "低噪声", "低失真"], RF_MODES))
        self.rfmode = combo(advanced, "射频模式 1", rf_modes)
        self.meas_time = number(advanced, "测量时间（ms）", 0, 10000, 0.5, 3)
        self.detector = combo(advanced, "检波方式", list(zip(["快速", "正峰值", "平均值", "均方根"], DETECTORS)), "POSitive")
        self.dpx_refresh = number(advanced, "DPX 刷新时间（ms）", 20, 5000, 100, 0)
        for layout in pages.values():
            layout.addStretch()
        self.attauto.currentIndexChanged.connect(lambda: self._toggle_att(bool(self.attauto.currentData())))
        self.center.setToolTip("中心频率范围：0～8000 MHz；上下方向键按频率步进调谐")
        self.center.step_hz = int(self.step.currentData() * 1000)
        self.step.currentIndexChanged.connect(lambda: setattr(self.center, "step_hz", int(self.step.currentData() * 1000)))
        self.step.setToolTip("中心频率输入框上下方向键的调谐步进")
        self.dpx_refresh.setToolTip("当前用于频谱和瀑布图的显示刷新间隔；荧光谱尚未接入")
        self.beta_band.setToolTip("β% 带宽，默认 0")
        for editor in (self.ref, self.lvspan):
            editor.textChanged.connect(self._update_range_hint)
        self._update_range_hint()

    def _update_range_hint(self):
        try:
            top, spacing, count = self.display_scale()
            self.range_hint.setText(f"显示范围：{top - spacing * count:g} ～ {top:g} dBµV")
        except ValueError:
            self.range_hint.setText("请输入有效的电平参数")

    def display_scale(self):
        return self.ref.value(), self.lvspan.value(), 10

    def _toggle_att(self, auto):
        self.ifatt.setEnabled(not auto)
        self.rfatt.setEnabled(not auto)

    def get_config(self) -> FFMConfig:
        config = FFMConfig(
            center_mhz=self.center.value_hz() / 1e6,
            span_khz=self.span.currentData(),
            step_khz=self.step.currentData(),
            bandwidth_khz=self.bw.currentData(),
            ref_level_dbuv=self.ref.value(),
            level_span_db=self.lvspan.value(),
            xdb_band_db=self.xdb_band.value(),
            beta_band_pct=self.beta_band.value(),
            demod=self.demod.currentData(),
            squelch_on=bool(self.sql.currentData()),
            att_auto=bool(self.attauto.currentData()),
            if_att_db=self.ifatt.value(),
            rf_att_db=self.rfatt.value(),
            rf_mode=self.rfmode.currentData(),
            measure_time_us=int(round(self.meas_time.value() * 1000)),
            detector=self.detector.currentData(),
            dpx_refresh_ms=self.dpx_refresh.value(),
        )
        config.validate()
        return config


# ============================================================
# PSCAN / FPSCAN (共用面板, fast 复选切换)
# ============================================================

class PScanPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        f = QtWidgets.QFormLayout(self)
        f.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.start = _dspin(0, 8000, 88, 1, " MHz", 3)
        self.stop = _dspin(0, 8000, 108, 1, " MHz", 3)
        self.step = QtWidgets.QComboBox()
        self.step.addItems(["3.125", "6.25", "12.5", "25", "50", "100", "200", "500"])
        self.step.setCurrentText("25")
        self.ref = _dspin(-256, 256, 80, 5, " dBµV", 0)
        self.lvspan = _dspin(1, 50, 10, 5, " dB/格", 0)
        self.attauto = QtWidgets.QCheckBox("自动衰减 (AGC)")
        self.attauto.setChecked(True)
        self.ifatt = _dspin(0, 20, 0, 2, " dB", 0)
        self.rfatt = _dspin(0, 30, 0, 2, " dB", 0)
        self.rfmode = QtWidgets.QComboBox(); self.rfmode.addItems(RF_MODES)
        self.fast = QtWidgets.QCheckBox("快速全景 (FPSCAN, 端口 8012)")

        f.addRow("起始频率", self.start)
        f.addRow("终止频率", self.stop)
        f.addRow("扫描步进 (kHz)", self.step)
        f.addRow("参考电平", self.ref)
        f.addRow("电平跨度", self.lvspan)
        f.addRow("", self.attauto)
        f.addRow("中频衰减", self.ifatt)
        f.addRow("射频衰减", self.rfatt)
        f.addRow("射频模式", self.rfmode)
        f.addRow("", self.fast)

        self.attauto.toggled.connect(self._toggle_att)
        self._toggle_att(True)

    def _toggle_att(self, auto):
        self.ifatt.setEnabled(not auto)
        self.rfatt.setEnabled(not auto)

    def set_fast(self, fast: bool):
        self.fast.setChecked(fast)

    def get_config(self) -> PScanConfig:
        return PScanConfig(
            start_mhz=self.start.value(),
            stop_mhz=self.stop.value(),
            step_khz=float(self.step.currentText()),
            ref_level_dbuv=self.ref.value(),
            level_span_db=self.lvspan.value(),
            if_att_db=self.ifatt.value(),
            rf_att_db=self.rfatt.value(),
            rf_mode=self.rfmode.currentText(),
            att_auto=self.attauto.isChecked(),
            fast=self.fast.isChecked(),
        )


# ============================================================
# FSCAN
# ============================================================

class FScanPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        f = QtWidgets.QFormLayout(self)
        f.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.start = _dspin(0, 8000, 88, 1, " MHz", 3)
        self.stop = _dspin(0, 8000, 108, 1, " MHz", 3)
        self.step = _dspin(0.001, 1, 0.025, 0.001, " MHz", 3)
        self.bw = _dspin(1, 500, 120, 1, " kHz", 1)
        self.ref = _dspin(-256, 256, 80, 5, " dBµV", 0)
        self.lvspan = _dspin(1, 50, 10, 5, " dB/格", 0)
        self.thr = _dspin(-256, 256, 10, 5, " dBµV", 0)
        self.dwell = _dspin(0.01, 10, 0.1, 0.01, " s", 2)
        self.ifatt = _dspin(0, 20, 0, 2, " dB", 0)
        self.rfatt = _dspin(0, 30, 0, 2, " dB", 0)
        self.demod = QtWidgets.QComboBox(); self.demod.addItems(DEMOD_MODES)
        self.demod.setCurrentText("FM")

        f.addRow("起始频率", self.start)
        f.addRow("终止频率", self.stop)
        f.addRow("步进频率", self.step)
        f.addRow("中频带宽", self.bw)
        f.addRow("参考电平", self.ref)
        f.addRow("电平跨度", self.lvspan)
        f.addRow("静噪门限", self.thr)
        f.addRow("驻留时间", self.dwell)
        f.addRow("中频衰减", self.ifatt)
        f.addRow("射频衰减", self.rfatt)
        f.addRow("解调模式", self.demod)

    def get_config(self) -> FScanConfig:
        return FScanConfig(
            start_mhz=self.start.value(),
            stop_mhz=self.stop.value(),
            step_khz=self.step.value() * 1000.0,
            bandwidth_khz=self.bw.value(),
            ref_level_dbuv=self.ref.value(),
            level_span_db=self.lvspan.value(),
            threshold_dbuv=self.thr.value(),
            dwell_s=self.dwell.value(),
            if_att_db=self.ifatt.value(),
            rf_att_db=self.rfatt.value(),
            demod=self.demod.currentText(),
        )


# ============================================================
# MSCAN (频点表)
# ============================================================

class MScanPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.ref = _dspin(-256, 256, 80, 5, " dBµV", 0)
        self.lvspan = _dspin(1, 50, 10, 5, " dB/格", 0)
        self.thr = _dspin(-256, 256, 10, 5, " dBµV", 0)
        self.dwell = _dspin(0.01, 10, 0.1, 0.01, " s", 2)
        self.ifatt = _dspin(0, 20, 0, 2, " dB", 0)
        self.rfatt = _dspin(0, 30, 0, 2, " dB", 0)
        form.addRow("参考电平", self.ref)
        form.addRow("电平跨度", self.lvspan)
        form.addRow("门限电平", self.thr)
        form.addRow("驻留时间", self.dwell)
        form.addRow("中频衰减", self.ifatt)
        form.addRow("射频衰减", self.rfatt)
        v.addLayout(form)

        v.addWidget(QtWidgets.QLabel("频点表 (idx / 频率MHz / 带宽kHz / 解调):"))
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["ID", "频率(MHz)", "带宽(kHz)", "解调"])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table, 1)

        btns = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("添加频点")
        self.btn_del = QtWidgets.QPushButton("删除选中")
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        v.addLayout(btns)

        self.btn_add.clicked.connect(self._add_row)
        self.btn_del.clicked.connect(self._del_row)

        # 预置两行示例
        self._add_row(101.7, 120, "FM")
        self._add_row(102.5, 120, "FM")

    def _add_row(self, freq=100.0, bw=120.0, dem="FM"):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r)))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(freq)))
        self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(bw)))
        combo = QtWidgets.QComboBox(); combo.addItems(DEMOD_MODES)
        combo.setCurrentText(dem)
        self.table.setCellWidget(r, 3, combo)

    def _del_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._renumber()

    def _renumber(self):
        for r in range(self.table.rowCount()):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r)))

    def get_config(self) -> MScanConfig:
        chans: List[MScanChannel] = []
        for r in range(self.table.rowCount()):
            try:
                freq = float(self.table.item(r, 1).text())
                bw = float(self.table.item(r, 2).text())
            except (ValueError, AttributeError):
                continue
            combo = self.table.cellWidget(r, 3)
            dem = combo.currentText() if combo else "FM"
            chans.append(MScanChannel(idx=r, freq_mhz=freq, bw_khz=bw, demod=dem))
        return MScanConfig(
            channels=chans,
            ref_level_dbuv=self.ref.value(),
            level_span_db=self.lvspan.value(),
            threshold_dbuv=self.thr.value(),
            dwell_s=self.dwell.value(),
            if_att_db=self.ifatt.value(),
            rf_att_db=self.rfatt.value(),
        )


# ============================================================
# DDC (通道表)
# ============================================================

class DDCPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.center = _dspin(20, 8000, 101.7, 0.1, " MHz")
        self.span = _dspin(1, 40000, 1000, 100, " kHz", 1)
        self.ref = _dspin(-256, 256, 80, 5, " dBµV", 0)
        self.lvspan = _dspin(1, 50, 10, 5, " dB/格", 0)
        form.addRow("主通道中心频率", self.center)
        form.addRow("主通道跨距", self.span)
        form.addRow("参考电平", self.ref)
        form.addRow("电平跨度", self.lvspan)
        v.addLayout(form)

        v.addWidget(QtWidgets.QLabel("DDC 子通道 (idx / 频率 / 带宽 / 解调 / 数据):"))
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["ID", "频率(MHz)", "带宽(kHz)", "解调", "数据(LEV|SPEC|AUD)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table, 1)

        btns = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("添加通道")
        self.btn_del = QtWidgets.QPushButton("删除选中")
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_del)
        v.addLayout(btns)

        self.btn_add.clicked.connect(self._add_row)
        self.btn_del.clicked.connect(self._del_row)
        self._add_row(101.7, 120, "FM")

    def _add_row(self, freq=100.0, bw=120.0, dem="FM"):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r)))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(freq)))
        self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(bw)))
        combo = QtWidgets.QComboBox(); combo.addItems(DEMOD_MODES)
        combo.setCurrentText(dem)
        self.table.setCellWidget(r, 3, combo)
        self.table.setItem(r, 4, QtWidgets.QTableWidgetItem("LEV|SPECtrum|AUDio"))

    def _del_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._renumber()

    def _renumber(self):
        for r in range(self.table.rowCount()):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(r)))

    def get_config(self) -> DDCConfig:
        chans: List[DDCChannel] = []
        for r in range(self.table.rowCount()):
            try:
                freq = float(self.table.item(r, 1).text())
                bw = float(self.table.item(r, 2).text())
            except (ValueError, AttributeError):
                continue
            combo = self.table.cellWidget(r, 3)
            dem = combo.currentText() if combo else "FM"
            media_item = self.table.item(r, 4)
            media = (media_item.text().split("|") if media_item and media_item.text()
                     else ["LEV", "SPECtrum", "AUDio"])
            chans.append(DDCChannel(idx=r, freq_mhz=freq, bw_khz=bw,
                                    demod=dem, media=media))
        return DDCConfig(
            center_mhz=self.center.value(),
            span_khz=self.span.value(),
            channels=chans,
            ref_level_dbuv=self.ref.value(),
            level_span_db=self.lvspan.value(),
        )
