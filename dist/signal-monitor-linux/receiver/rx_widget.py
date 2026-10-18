"""
接收机控制面板 (右侧 dock)
===========================
菜单驱动: 主窗口菜单选模式 -> set_mode() 切换本面板里显示的参数面板.
本面板只显示"当前模式"的参数, 不再用标签页.

布局:
    连接状态行 (只读, 显示当前 IP / 连接灯; 配置在弹出对话框里改)
    当前模式标题
    当前模式参数面板 (随菜单切换)
    应用 / 开始 / 停止
    状态日志
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .controller import ReceiverController, RxMode
from .mode_panels import (
    FFMPanel, PScanPanel, FScanPanel, MScanPanel, DDCPanel,
)
from .rx_config import RxConfig


PANEL_STYLE = """
QWidget { background: #0e1116; color: #d0d8e0; }
QGroupBox {
    border: 1px solid #2a3140; border-radius: 4px;
    margin-top: 10px; padding-top: 6px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton {
    background: #1c2330; border: 1px solid #2e3a4f;
    padding: 6px 10px; border-radius: 3px;
}
QPushButton:hover { background: #2a3447; }
QPushButton:pressed { background: #00d4ff; color: #000; }
QPushButton:disabled { color: #555; border-color: #222; }
QPushButton#primary { background: #00d4ff; color: #001018; font-weight: bold; }
QPushButton#primary:hover { background: #38e0ff; }
QPushButton#danger { background: #3a1c22; border-color: #5a2a33; }
QPushButton#danger:hover { background: #4a2430; }
QLineEdit, QComboBox {
    background: #141923; border: 1px solid #2a3140; padding: 3px 5px;
    border-radius: 2px;
}
QSpinBox, QDoubleSpinBox {
    background: #141923; border: 1px solid #2a3140;
    border-radius: 2px; padding: 3px 18px 3px 5px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 16px; border-left: 1px solid #2a3140;
    border-top-right-radius: 2px; background: #1c2330;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 16px; border-left: 1px solid #2a3140;
    border-bottom-right-radius: 2px; background: #1c2330;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #2a3447; }
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed { background: #00d4ff; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 6px solid #9fb0c0;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 6px solid #9fb0c0;
}
QTableWidget { background: #141923; gridline-color: #2a3140; }
QHeaderView::section { background: #1c2330; padding: 4px; border: none; }
QPlainTextEdit {
    background: #0a0d12; border: 1px solid #2a3140; color: #7fd4a0;
    font-family: Consolas, monospace; font-size: 11px;
}
QLabel#modeTitle { font-size: 14px; font-weight: bold; color: #00d4ff; }
"""


# 模式 -> (面板类, 标题)
MODE_META = {
    RxMode.FFM:    ("FFM 固定频点监测", FFMPanel),
    RxMode.PSCAN:  ("PSCAN 全景扫描", PScanPanel),
    RxMode.FPSCAN: ("FPSCAN 快速全景扫描", PScanPanel),
    RxMode.FSCAN:  ("FSCAN 扫频搜索", FScanPanel),
    RxMode.MSCAN:  ("MSCAN 存储信道扫描", MScanPanel),
    RxMode.DDC:    ("DDC 数字下变频", DDCPanel),
}


class ReceiverControlWidget(QtWidgets.QWidget):
    """菜单驱动的接收机控制面板, 放在右侧 dock"""

    def __init__(self, controller: ReceiverController = None, parent=None):
        super().__init__(parent)
        self.controller = controller or ReceiverController(self)
        self.cfg = RxConfig.load()

        self._mode: RxMode = None
        self._panels: dict = {}     # RxMode -> panel 实例 (懒创建并缓存)

        self.setStyleSheet(PANEL_STYLE)
        self._build_ui()
        self._wire()
        self._update_conn_ui(False)
        self.set_mode(RxMode.FFM)   # 默认显示 FFM

    # -------- UI --------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 连接状态行 (只读显示, 改配置走菜单的对话框)
        conn = QtWidgets.QGroupBox("连接状态")
        cl = QtWidgets.QHBoxLayout(conn)
        self.lbl_status = QtWidgets.QLabel("● 未连接")
        self.lbl_status.setStyleSheet("color:#ff5b5b; font-weight:bold;")
        self.lbl_target = QtWidgets.QLabel(
            f"{self.cfg.receiver_ip}:{self.cfg.tcp_port}")
        self.lbl_target.setStyleSheet("color:#8a96a5;")
        self.btn_disconnect = QtWidgets.QPushButton("断开")
        cl.addWidget(self.lbl_status)
        cl.addWidget(self.lbl_target, 1)
        cl.addWidget(self.btn_disconnect)
        root.addWidget(conn)

        # 当前模式标题
        self.lbl_mode = QtWidgets.QLabel("—")
        self.lbl_mode.setObjectName("modeTitle")
        root.addWidget(self.lbl_mode)

        # 模式参数面板容器 (用 QStackedWidget 缓存各面板)
        self.stack = QtWidgets.QStackedWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.stack)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        root.addWidget(scroll, 1)

        # 动作栏
        act = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("应用配置")
        self.btn_start = QtWidgets.QPushButton("开始")
        self.btn_start.setObjectName("primary")
        self.btn_stop = QtWidgets.QPushButton("停止")
        self.btn_stop.setObjectName("danger")
        act.addWidget(self.btn_apply)
        act.addWidget(self.btn_start)
        act.addWidget(self.btn_stop)
        root.addLayout(act)

        # 状态日志
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setFixedHeight(120)
        root.addWidget(self.log)

    # -------- 连线 --------

    def _wire(self):
        self.btn_disconnect.clicked.connect(self.controller.disconnect_receiver)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_start.clicked.connect(self.controller.start)
        self.btn_stop.clicked.connect(self.controller.stop)

        c = self.controller
        c.connectedChanged.connect(self._update_conn_ui)
        c.statusMessage.connect(self._log)
        c.errorOccurred.connect(lambda s: self._log("⚠ " + s))
        c.modeChanged.connect(lambda m: self._log(f"当前下发模式: {m.value}"))
        c.runningChanged.connect(self._update_run_ui)

    # -------- 菜单调用: 切换模式面板 --------

    def set_mode(self, mode: RxMode):
        """由主窗口菜单调用. 只切面板, 不下发 SCPI."""
        if mode not in self._panels:
            title, panel_cls = MODE_META[mode]
            panel = panel_cls()
            # PSCAN/FPSCAN 共用 PScanPanel, 用 fast 复选区分
            if mode == RxMode.FPSCAN and hasattr(panel, "set_fast"):
                panel.set_fast(True)
            self._panels[mode] = panel
            self.stack.addWidget(panel)
        self._mode = mode
        self.stack.setCurrentWidget(self._panels[mode])
        self.lbl_mode.setText("▶ " + MODE_META[mode][0])
        self._log(f"已切换到 {mode.value} 参数面板 (点[应用配置]下发)")

    # -------- 由主窗口对话框设置连接 --------

    def connect_with(self, receiver_ip: str, local_ip: str, tcp_port: int):
        self.cfg.receiver_ip = receiver_ip
        self.cfg.local_ip = local_ip
        self.cfg.tcp_port = tcp_port
        self.lbl_target.setText(f"{receiver_ip}:{tcp_port}")
        self.controller.connect_receiver(receiver_ip, local_ip, tcp_port)

    # -------- 应用配置 --------

    def _on_apply(self):
        if self._mode is None:
            return
        panel = self._panels[self._mode]
        cfg = panel.get_config()
        mode = self._mode
        # PScanPanel 的 fast 复选可能覆盖菜单选择
        if self._mode in (RxMode.PSCAN, RxMode.FPSCAN):
            mode = RxMode.FPSCAN if cfg.fast else RxMode.PSCAN
        self.controller.apply_mode(mode, cfg)

    # -------- 状态 --------

    def _update_conn_ui(self, connected: bool):
        if connected:
            self.lbl_status.setText("● 已连接")
            self.lbl_status.setStyleSheet("color:#5bff8c; font-weight:bold;")
        else:
            self.lbl_status.setText("● 未连接")
            self.lbl_status.setStyleSheet("color:#ff5b5b; font-weight:bold;")
        self.btn_disconnect.setEnabled(connected)
        self.btn_apply.setEnabled(connected)
        self.btn_start.setEnabled(connected)
        self.btn_stop.setEnabled(connected)

    def _update_run_ui(self, running: bool):
        self.btn_start.setEnabled(not running and self.controller.is_connected())
        self.btn_stop.setEnabled(running)

    def _log(self, msg: str):
        ts = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.log.appendPlainText(f"{ts}  {msg}")