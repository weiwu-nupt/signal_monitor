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

from .controller import (
    FFMConfig, PScanConfig, FScanConfig,
    MScanConfig, MScanChannel, DDCConfig, DDCChannel,
)


DEMOD_MODES = ["IQ", "AM", "FM", "CW", "LSB", "USB", "PULSE", "PM", "DMR", "DPMR"]
RF_MODES = ["NORMal", "LOWNoise", "LOWDistort"]
DETECTORS = ["FAST", "POSitive", "AVERage", "RMS"]


def _dspin(lo, hi, val, step=1.0, suffix="", decimals=3):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    if suffix:
        s.setSuffix(suffix)
    return s


def _ispin(lo, hi, val, step=1, suffix=""):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    if suffix:
        s.setSuffix(suffix)
    return s


# ============================================================
# FFM
# ============================================================

class FFMPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        f = QtWidgets.QFormLayout(self)
        f.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.center = _dspin(20, 8000, 101.7, 0.1, " MHz")
        self.span = _dspin(1, 40000, 1000, 100, " kHz", 1)
        self.bw = _dspin(1, 500, 120, 1, " kHz", 1)
        self.ref = _dspin(-256, 256, 80, 5, " dBµV", 0)
        self.lvspan = _dspin(1, 50, 10, 5, " dB/格", 0)
        self.demod = QtWidgets.QComboBox(); self.demod.addItems(DEMOD_MODES)
        self.demod.setCurrentText("FM")
        self.detector = QtWidgets.QComboBox(); self.detector.addItems(DETECTORS)
        self.sql = QtWidgets.QCheckBox("静噪 SQL")
        self.thr = _dspin(-30, 130, 10, 5, " dBµV", 0)
        self.attauto = QtWidgets.QCheckBox("自动衰减 (AGC)")
        self.attauto.setChecked(True)
        self.ifatt = _dspin(0, 50, 0, 2, " dB", 0)
        self.rfatt = _dspin(0, 50, 0, 2, " dB", 0)
        self.rfmode = QtWidgets.QComboBox(); self.rfmode.addItems(RF_MODES)

        f.addRow("中心频率", self.center)
        f.addRow("频谱跨距", self.span)
        f.addRow("中频带宽", self.bw)
        f.addRow("参考电平", self.ref)
        f.addRow("电平跨度", self.lvspan)
        f.addRow("解调模式", self.demod)
        f.addRow("检波方式", self.detector)
        f.addRow("", self.sql)
        f.addRow("静噪门限", self.thr)
        f.addRow("", self.attauto)
        f.addRow("中频衰减", self.ifatt)
        f.addRow("射频衰减", self.rfatt)
        f.addRow("射频模式", self.rfmode)

        self.attauto.toggled.connect(self._toggle_att)
        self._toggle_att(True)

    def _toggle_att(self, auto):
        self.ifatt.setEnabled(not auto)
        self.rfatt.setEnabled(not auto)

    def get_config(self) -> FFMConfig:
        return FFMConfig(
            center_mhz=self.center.value(),
            span_khz=self.span.value(),
            bandwidth_khz=self.bw.value(),
            ref_level_dbuv=self.ref.value(),
            level_span_db=self.lvspan.value(),
            demod=self.demod.currentText(),
            squelch_on=self.sql.isChecked(),
            threshold_dbuv=self.thr.value(),
            att_auto=self.attauto.isChecked(),
            if_att_db=self.ifatt.value(),
            rf_att_db=self.rfatt.value(),
            rf_mode=self.rfmode.currentText(),
            detector=self.detector.currentText(),
        )


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