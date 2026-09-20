"""Shared dark controls, matching the reference signal workbench."""
from pathlib import Path

APP_STYLE = """
/* Set a foreground colour on every widget.  Some Windows themes otherwise
   give unstyled child controls a black foreground, even in a dark window. */
* { font-size: 10pt; color: #dce9fa; }
QMainWindow, QWidget#contentRoot { background: #0a1220; color: #dce9fa; }
QFrame#sidebar { background: #0d1a2b; border-right: 1px solid #23364d; }
QLabel#brand { color: #29d4d8; font-size: 19px; font-weight: 700; letter-spacing: 1px; }
QLabel#pageTitle { color: #f4f8ff; font-size: 25px; font-weight: 700; }
QLabel#subtitle, QLabel#muted { color: #8095b2; }
/* Do not inherit the operating system's light-theme text colour.  Widgets
   such as QCheckBox otherwise fall back to black on some computers. */
QLabel, QAbstractButton, QToolButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
QDateTimeEdit, QPlainTextEdit, QTextEdit, QListView, QListWidget,
QTableView, QTableWidget, QGroupBox, QCheckBox, QRadioButton, QMenu {
    color: #dce9fa;
}
QCheckBox, QRadioButton, QGroupBox { color: #dce9fa; }
QCheckBox:disabled, QRadioButton:disabled, QGroupBox:disabled { color: #6f829a; }
QCheckBox::indicator, QRadioButton::indicator { width: 17px; height: 17px; }
QCheckBox::indicator:unchecked, QRadioButton::indicator:unchecked { background: #0b1625; border: 2px solid #52657e; border-radius: 4px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #2e80e6; border: 2px solid #63b1ff; border-radius: 4px; }
QPushButton { background: #1b2b40; color: #dce9fa; border: 1px solid #2a405b; border-radius: 6px; padding: 8px 14px; }
QPushButton:hover { background: #274667; border-color: #3e78aa; }
QPushButton:checked { background: #1c5ab8; border-color: #48a4ff; color: white; }
QPushButton#primary { background: #286de2; border-color: #4c91ff; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #3d82ee; }
QPushButton#danger { background: #8d2b42; border-color: #c84765; color: white; }
QToolButton#sidebarToggle { background: #1b2b40; color: #dce9fa; border: 1px solid #2a405b; border-radius: 5px; font-size: 20px; min-width: 28px; min-height: 28px; }
QToolButton#sidebarToggle:hover { background: #274667; border-color: #3e78aa; }
QFrame#card { background: #111e30; border: 1px solid #233a55; border-radius: 9px; }
/* Keep splitters easy to grab on high-DPI screens without stealing useful space. */
QSplitter::handle { background: #1b3048; }
QSplitter::handle:hover { background: #3c78a8; }
QSplitter::handle:horizontal { width: 6px; }
QSplitter::handle:vertical { height: 6px; }
QTableView, QTableWidget { background: #0f1a2a; alternate-background-color: #132238; border: 1px solid #263e5b; border-radius: 6px; gridline-color: #263e5b; color: #dce9fa; selection-background-color: #235cae; }
QTreeView { background: #0f1a2a; alternate-background-color: #132238; border: 1px solid #263e5b; color: #dce9fa; selection-background-color: #235cae; selection-color: #ffffff; }
QHeaderView::section { background: #172a43; color: #a9c0dd; border: 0; border-bottom: 1px solid #304966; padding: 9px; font-weight: 600; }
QHeaderView { background: #172a43; } QTableCornerButton::section { background: #172a43; border: 0; border-bottom: 1px solid #304966; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit, QPlainTextEdit { background: #0b1625; color: #e8f1ff; border: 1px solid #304966; border-radius: 5px; padding: 7px; min-height: 18px; selection-background-color: #286de2; selection-color: #ffffff; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QDateTimeEdit:disabled, QPlainTextEdit:disabled { background: #142236; color: #91a4bd; border-color: #263b53; }
QMenu { background: #111e30; color: #dce9fa; border: 1px solid #304966; padding: 4px; }
QMenu::item { background: transparent; color: #dce9fa; padding: 7px 24px 7px 10px; }
QMenu::item:selected { background: #286de2; color: #ffffff; }
QMenu::item:disabled { color: #63768e; }
QMenu::separator { height: 1px; background: #304966; margin: 4px 7px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 22px; background: #1a2c44; border-left: 1px solid #304966; border-bottom: 1px solid #304966; border-top-right-radius: 4px; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 22px; background: #1a2c44; border-left: 1px solid #304966; border-bottom-right-radius: 4px; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover { background: #274667; }
QComboBox::drop-down { border: 0; width: 22px; } QComboBox QAbstractItemView { background: #132238; color: #e8f1ff; selection-background-color: #235cae; }
QListWidget { background: #0f1a2a; color: #dce9fa; border: 1px solid #263e5b; border-radius: 6px; selection-background-color: #235cae; selection-color: #ffffff; }
QProgressBar { background: #0b1625; color: #e8f1ff; border: 1px solid #304966; border-radius: 5px; text-align: center; }
QProgressBar::chunk { background: #286de2; }
QTabWidget::pane { border: 1px solid #263e5b; background: #0f1a2a; } QTabBar::tab { background: #16253a; color: #91a7c4; padding: 9px 16px; } QTabBar::tab:selected { background: #286de2; color: white; }
QScrollArea, QScrollArea > QWidget > QWidget { background: #0a1220; border: 0; }
QStatusBar { background: #0d1a2b; color: #8196b5; } QDialog { background: #111e30; color: #dce9fa; }
QFileDialog, QFileDialog QWidget { background: #111e30; color: #dce9fa; }
QFileDialog QListView, QFileDialog QTreeView { background: #0f1a2a; color: #dce9fa; alternate-background-color: #132238; selection-background-color: #235cae; selection-color: #ffffff; }
QFileDialog QToolButton { background: #1b2b40; color: #dce9fa; border: 1px solid #2a405b; border-radius: 4px; }

QWidget#receiverPage { background: #0a1220; }
QLabel#muted { color: #8095b2; }
QPushButton:disabled { background: #142236; color: #6f829a; border-color: #263b53; }
QPushButton#primary:disabled, QPushButton#danger:disabled { background: #142236; color: #6f829a; border-color: #263b53; }
QTabBar::tab { padding: 8px 12px; }
QMenuBar { background: #0d1a2b; color: #dce9fa; }
QMenuBar::item:selected { background: #1b3048; }
QToolButton { background: #14283d; color: #d6e5f4; border: 1px solid #315270; border-radius: 4px; padding: 6px 10px; }
QToolButton:hover { background: #1f4161; }
"""

# Use an actual vector arrow: CSS border triangles render as boxes in Qt on Windows.
_arrow_path = (Path(__file__).parent / "assets" / "chevron-down.svg").as_posix()
APP_STYLE += """
QComboBox { padding-right: 24px; }
QComboBox::drop-down { width: 22px; border: none; }
QComboBox::down-arrow { image: url("%s"); width: 12px; height: 8px; }
""" % _arrow_path
