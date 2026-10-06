"""Optional recognition bandwidth range, in MHz; blank means unbounded."""
import math
from PyQt6 import QtCore, QtWidgets
from fpga_closed_loop import TargetFilter
from .inputs import NoWheelComboBox


class RecognitionBandwidth(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal(object)
    rateMultiplierChanged = QtCore.pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__('识别信号带宽范围', parent)
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QGridLayout()
        self.minimum = QtWidgets.QLineEdit()
        self.maximum = QtWidgets.QLineEdit()
        for index, (label, editor) in enumerate((('最小 MHz', self.minimum), ('最大 MHz', self.maximum))):
            editor.setPlaceholderText('不限')
            editor.setMaximumWidth(110)
            row.addWidget(QtWidgets.QLabel(label), index, 0)
            row.addWidget(editor, index, 1)
        self.apply_button = QtWidgets.QPushButton('应用范围')
        row.addWidget(self.apply_button, 0, 2, 2, 1)
        layout.addLayout(row)
        rate_row = QtWidgets.QHBoxLayout()
        rate_row.addWidget(QtWidgets.QLabel('IQ采样率 / 带宽'))
        self.rate_multiplier = NoWheelComboBox()
        for value in (3.0, 3.5, 4.0):
            self.rate_multiplier.addItem(f'{value:g} 倍', value)
        self.rate_multiplier.setCurrentIndex(2)
        self.rate_multiplier.setToolTip('默认4倍；按原始采样率限制在3～4倍之间，下轮生效')
        rate_row.addWidget(self.rate_multiplier)
        layout.addLayout(rate_row)
        self.rate_multiplier.currentIndexChanged.connect(
            lambda _index: self.rateMultiplierChanged.emit(float(self.rate_multiplier.currentData())))
        self.summary = QtWidgets.QLabel('不限：识别全部；≤20 MHz 窄带 128 路，>20 MHz 宽带 2 路，超出轮询')
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.current = TargetFilter()
        self.apply_button.clicked.connect(self.apply)

    def value(self):
        values = []
        for editor in (self.minimum, self.maximum):
            text = editor.text().strip()
            value = float(text)*1e6 if text else None
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError('识别带宽必须是非负有限数值，留空表示不限')
            values.append(value)
        if all(v is not None for v in values) and values[0] > values[1]:
            raise ValueError('最小识别带宽不能大于最大识别带宽')
        return TargetFilter(*values)

    def apply(self):
        try:
            self.current = self.value()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, '识别带宽无效', str(exc))
            return
        lo, hi = self.current.bandwidth_min_hz, self.current.bandwidth_max_hz
        self.summary.setText(f'已应用：{lo/1e6 if lo is not None else "不限"} ～ '
                             f'{hi/1e6 if hi is not None else "不限"} MHz；下一轮生效。窄带128路 / 宽带2路')
        self.changed.emit(self.current)
