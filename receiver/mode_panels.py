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
FFM_SPANS_KHZ = {
    "1 MHz": 1000.0, "2 MHz": 2000.0, "5 MHz": 5000.0,
    "10 MHz": 10000.0, "20 MHz": 20000.0, "40 MHz": 40000.0,
}
FFM_STEPS_KHZ = ["6.25", "12.5", "25", "50", "100", "200"]
FFM_BANDWIDTHS_KHZ = ["6.25", "12.5", "25", "50", "100", "200", "500"]


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
    combo.addItems(["ON", "OFF"])
    combo.setCurrentText("ON" if default else "OFF")
    combo.setMinimumWidth(176)
    combo.setMinimumHeight(28)
    return combo


def _is_on(combo: QtWidgets.QComboBox) -> bool:
    return combo.currentText() == "ON"


# ============================================================
# FFM
# ============================================================

class FFMPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        f = QtWidgets.QFormLayout(self)
        f.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # 控件原始上限为 300000 MHz；按本项目接收机能力限制为 8000 MHz。
        self.center = _dspin(0, 8000, 2000, 0.1, " MHz", 6)
        self.span = QtWidgets.QComboBox(); self.span.addItems(FFM_SPANS_KHZ)
        self.span.setCurrentText("20 MHz")
        self.step = QtWidgets.QComboBox(); self.step.addItems(FFM_STEPS_KHZ)
        self.step.setCurrentText("6.25")
        self.offset = _dspin(-40000, 40000, 0, 12.5, " kHz", 3)
        self.ref_clock = QtWidgets.QComboBox(); self.ref_clock.addItems(["0", "10 MHz", "外部"])
        self.bw = QtWidgets.QComboBox(); self.bw.addItems(FFM_BANDWIDTHS_KHZ)
        self.bw.setCurrentText("500")
        self.ref = _dspin(-256, 256, 120, 5, " dBµV", 0)
        self.lvspan = _dspin(1, 50, 20, 1, " dB/格", 0)
        self.xdb_band = _dspin(0, 200, 26, 1, " dB", 0)
        self.beta_band = _dspin(0, 100, 0, 0.01, " %", 2)
        self.demod = QtWidgets.QComboBox(); self.demod.addItems(DEMOD_MODES)
        self.demod.setCurrentText("FM")
        self.detector = QtWidgets.QComboBox(); self.detector.addItems(DETECTORS)
        self.detector.setCurrentText("POSitive")
        self.sql = _on_off_combo()
        self.thr = _dspin(-256, 256, 50, 5, " dBµV", 0)
        self.level_trigger = _on_off_combo()
        self.trigger_level = _dspin(-256, 256, 0, 1, " dBµV", 0)
        self.attauto = _on_off_combo()
        self.ifatt = _dspin(0, 50, 0, 2, " dB", 0)
        self.rfatt = _dspin(0, 50, 0, 2, " dB", 0)
        self.rfmode = QtWidgets.QComboBox(); self.rfmode.addItems(RF_MODES)
        self.rfmode2 = QtWidgets.QComboBox(); self.rfmode2.addItems(RF_MODES)
        self.rf_output = QtWidgets.QComboBox(); self.rf_output.addItems(["低速", "高速"])
        self.high_performance = _on_off_combo()
        self.meas_time = QtWidgets.QComboBox(); self.meas_time.addItems(["100 µs", "500 µs", "1 ms", "5 ms", "10 ms"])
        self.meas_time.setCurrentText("500 µs")
        self.dpx_refresh = _ispin(20, 5000, 100, 10, " ms")
        self.echo_speed = _ispin(1, 100, 1, 1)
        self.red_occupancy = _dspin(0, 100, 1, 0.01, " %", 2)

        for combo in (self.span, self.step, self.ref_clock, self.bw,
                      self.demod, self.detector, self.rfmode, self.rfmode2,
                      self.rf_output, self.meas_time):
            combo.setMinimumWidth(176)
            combo.setMinimumHeight(28)

        f.addRow("中心频率", self.center)
        f.addRow("频谱跨距", self.span)
        f.addRow("频谱步进", self.step)
        f.addRow("频率偏移", self.offset)
        f.addRow("参考时钟", self.ref_clock)
        f.addRow("参考电平", self.ref)
        f.addRow("电平跨度", self.lvspan)
        f.addRow("Xdb 带宽", self.xdb_band)
        f.addRow("β% 带宽", self.beta_band)
        f.addRow("解调模式", self.demod)
        f.addRow("静噪 SQL", self.sql)
        f.addRow("静噪门限", self.thr)
        f.addRow("电平触发", self.level_trigger)
        f.addRow("触发电平", self.trigger_level)
        f.addRow("解调带宽", self.bw)
        f.addRow("射频衰减", self.rfatt)
        f.addRow("中频衰减", self.ifatt)
        f.addRow("射频模式 1", self.rfmode)
        f.addRow("射频模式 2", self.rfmode2)
        f.addRow("射频输出模式", self.rf_output)
        f.addRow("AGC", self.attauto)
        f.addRow("高性能模式", self.high_performance)
        f.addRow("测量时间", self.meas_time)
        f.addRow("检波方式", self.detector)
        f.addRow("DPX 刷新时间", self.dpx_refresh)
        f.addRow("回波速度", self.echo_speed)
        f.addRow("红色占有率", self.red_occupancy)

        self.attauto.currentTextChanged.connect(self._on_att_auto_changed)
        self._toggle_att(False)
        self.level_trigger.currentTextChanged.connect(self._on_level_trigger_changed)
        self.trigger_level.setEnabled(False)

    def _on_att_auto_changed(self, value: str):
        self._toggle_att(value == "ON")

    def _on_level_trigger_changed(self, value: str):
        self.trigger_level.setEnabled(value == "ON")

    def _toggle_att(self, auto):
        self.ifatt.setEnabled(not auto)
        self.rfatt.setEnabled(not auto)

    def get_config(self) -> FFMConfig:
        return FFMConfig(
            center_mhz=self.center.value(),
            span_khz=FFM_SPANS_KHZ[self.span.currentText()],
            step_khz=float(self.step.currentText()),
            offset_khz=self.offset.value(),
            bandwidth_khz=float(self.bw.currentText()),
            ref_clock=self.ref_clock.currentText(),
            ref_level_dbuv=self.ref.value(),
            level_span_db=self.lvspan.value(),
            xdb_band_db=self.xdb_band.value(),
            beta_band_pct=self.beta_band.value(),
            demod=self.demod.currentText(),
            squelch_on=_is_on(self.sql),
            threshold_dbuv=self.thr.value(),
            level_trigger_on=_is_on(self.level_trigger),
            level_trigger_dbuv=self.trigger_level.value(),
            att_auto=_is_on(self.attauto),
            if_att_db=self.ifatt.value(),
            rf_att_db=self.rfatt.value(),
            rf_mode=self.rfmode.currentText(),
            rf_mode_2=self.rfmode2.currentText(),
            rf_output_mode=self.rf_output.currentText(),
            high_performance=_is_on(self.high_performance),
            measure_time_us={"100 µs": 100, "500 µs": 500, "1 ms": 1000,
                             "5 ms": 5000, "10 ms": 10000}[self.meas_time.currentText()],
            detector=self.detector.currentText(),
            dpx_refresh_ms=self.dpx_refresh.value(),
            echo_speed=self.echo_speed.value(),
            red_occupancy_pct=self.red_occupancy.value(),
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
