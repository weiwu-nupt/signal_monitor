"""Reference-project style editors adapted to this application's PyQt6 runtime."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QComboBox, QWidget, QHBoxLayout, QLineEdit, QSizePolicy


class _FrequencyUnitComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class FrequencyInput(QWidget):
    """Compact value + unit editor whose public value is always integer Hz."""

    editingFinished = QtCore.pyqtSignal()
    textChanged = QtCore.pyqtSignal(str)
    UNITS = (("Hz", 1), ("kHz", 1_000), ("MHz", 1_000_000), ("GHz", 1_000_000_000))
    UNIT_FACTORS = dict(UNITS)

    def __init__(
        self,
        value_hz: int | float | str | None = 0,
        parent=None,
        *,
        minimum_hz: int | None = None,
        maximum_hz: int | None = None,
        allow_empty: bool = False,
        field_name: str = "频率",
    ) -> None:
        super().__init__(parent)
        self.minimum_hz = minimum_hz
        self.maximum_hz = maximum_hz
        self.allow_empty = allow_empty
        self.field_name = field_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.editor = QLineEdit()
        self.step_hz = 0
        self.editor.installEventFilter(self)
        validator = QtGui.QDoubleValidator(-1e15, 1e15, 9, self.editor)
        validator.setLocale(QtCore.QLocale.c())
        validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        self.editor.setValidator(validator)
        self.editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.unit = _FrequencyUnitComboBox()
        self.unit.addItems([name for name, _factor in self.UNITS])
        self.unit.setFixedWidth(max(88, self.unit.fontMetrics().horizontalAdvance("GHz") + 48))
        layout.addWidget(self.editor, 1)
        layout.addWidget(self.unit)

        self.editor.editingFinished.connect(self.editingFinished)
        self.editor.textChanged.connect(lambda text: self.textChanged.emit(text))
        self.unit.currentTextChanged.connect(lambda _unit: self.textChanged.emit(self.editor.text()))
        self.setText("" if value_hz is None else str(value_hz))

    @staticmethod
    def _plain_number(value: Decimal) -> str:
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    def eventFilter(self, watched, event):
        if watched is self.editor and event.type() == QtCore.QEvent.Type.KeyPress and self.step_hz:
            if event.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):
                try:
                    value = self.value_hz() + (self.step_hz if event.key() == QtCore.Qt.Key.Key_Up else -self.step_hz)
                    if self.minimum_hz is not None:
                        value = max(self.minimum_hz, value)
                    if self.maximum_hz is not None:
                        value = min(self.maximum_hz, value)
                    self.setText(value)
                    self.editingFinished.emit()
                except ValueError:
                    pass
                return True
        return super().eventFilter(watched, event)

    @classmethod
    def _best_unit(cls, value_hz: Decimal) -> tuple[str, int]:
        magnitude = abs(value_hz)
        for name, factor in reversed(cls.UNITS):
            if magnitude >= factor:
                return name, factor
        return cls.UNITS[0]

    def value_hz(self) -> int | None:
        value_text = self.editor.text().strip()
        if not value_text:
            if self.allow_empty:
                return None
            raise ValueError(f"{self.field_name}不能为空")
        try:
            value = Decimal(value_text) * self.UNIT_FACTORS[self.unit.currentText()]
        except (InvalidOperation, KeyError) as error:
            raise ValueError(f"{self.field_name}请输入有效数值") from error
        if not value.is_finite():
            raise ValueError(f"{self.field_name}请输入有限数值")
        rounded = int(value.to_integral_value(rounding=ROUND_HALF_UP))
        if self.minimum_hz is not None and rounded < self.minimum_hz:
            raise ValueError(f"{self.field_name}不能小于 {self.format_hz(self.minimum_hz)}")
        if self.maximum_hz is not None and rounded > self.maximum_hz:
            raise ValueError(f"{self.field_name}不能大于 {self.format_hz(self.maximum_hz)}")
        return rounded

    def text(self) -> str:
        value = self.value_hz()
        return "" if value is None else str(value)

    def setText(self, value_hz: str | int | float) -> None:
        if value_hz is None or str(value_hz).strip() == "":
            self.editor.clear()
            return
        try:
            value = Decimal(str(value_hz).strip())
        except InvalidOperation:
            self.editor.setText(str(value_hz))
            return
        unit_name, factor = self._best_unit(value)
        self.editor.setText(self._plain_number(value / factor))
        self.unit.setCurrentText(unit_name)

    def clear(self) -> None:
        self.editor.clear()

    def setPlaceholderText(self, text: str) -> None:
        self.editor.setPlaceholderText(text)

    def display_text(self) -> str:
        text = self.editor.text().strip()
        return f"{text} {self.unit.currentText()}" if text else ""

    @classmethod
    def format_hz(cls, value_hz: int | float | str) -> str:
        value = Decimal(str(value_hz))
        name, factor = cls._best_unit(value)
        return f"{cls._plain_number(value / factor)} {name}"


class NoWheelComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(6)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def wheelEvent(self, event):
        event.ignore()


class NumberInput(QLineEdit):
    """Plain numeric editor, with explicit validation before applying settings."""
    def __init__(self, minimum, maximum, value, *, name, decimals=3, parent=None):
        super().__init__(parent)
        self.minimum, self.maximum = minimum, maximum
        self.field_name, self.decimals = name, decimals
        validator = QtGui.QDoubleValidator(minimum, maximum, decimals, self)
        validator.setLocale(QtCore.QLocale.c())
        validator.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
        self.setValidator(validator)
        self.setValue(value)

    def value(self):
        try:
            number = float(self.text())
        except ValueError as error:
            raise ValueError(f"{self.field_name}请输入有效数值") from error
        if not math.isfinite(number) or not self.minimum <= number <= self.maximum:
            raise ValueError(f"{self.field_name}范围为 {self.minimum:g}～{self.maximum:g}")
        if self.decimals == 0 and not number.is_integer():
            raise ValueError(f"{self.field_name}请输入整数")
        return int(number) if self.decimals == 0 else number

    def setValue(self, value):
        self.setText(f"{value:.{self.decimals}f}".rstrip("0").rstrip(".") if self.decimals else str(int(value)))


