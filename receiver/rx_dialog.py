"""
接收机配置对话框
=================
弹出式: 接收机 IP / 本机 IP / TCP 端口.
点"确定"保存到 MRConfig.ini 并发起连接.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .rx_config import RxConfig


DIALOG_STYLE = """
QDialog { background: #0e1116; color: #d0d8e0; }
QLabel { color: #d0d8e0; }
QLineEdit {
    background: #141923; border: 1px solid #2a3140; padding: 4px 6px;
    border-radius: 2px; color: #d0d8e0;
}
QSpinBox {
    background: #141923; border: 1px solid #2a3140;
    padding: 4px 18px 4px 6px; border-radius: 2px; color: #d0d8e0;
}
QSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 16px; border-left: 1px solid #2a3140; background: #1c2330;
}
QSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 16px; border-left: 1px solid #2a3140; background: #1c2330;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #2a3447; }
QSpinBox::up-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 6px solid #9fb0c0;
}
QSpinBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 6px solid #9fb0c0;
}
QPushButton {
    background: #1c2330; border: 1px solid #2e3a4f;
    padding: 6px 16px; border-radius: 3px; color: #d0d8e0;
}
QPushButton:hover { background: #2a3447; }
QPushButton#primary { background: #00d4ff; color: #001018; font-weight: bold; }
QPushButton#primary:hover { background: #38e0ff; }
"""


class ReceiverConfigDialog(QtWidgets.QDialog):
    """
    返回 (receiver_ip, local_ip, tcp_port).
    accepted 时已写回配置文件.
    """

    def __init__(self, cfg: RxConfig = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("接收机配置")
        self.setModal(True)
        self.setMinimumWidth(360)
        self.setStyleSheet(DIALOG_STYLE)

        self.cfg = cfg or RxConfig.load()

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setContentsMargins(16, 16, 16, 8)
        form.setSpacing(10)

        self.ed_rx_ip = QtWidgets.QLineEdit(self.cfg.receiver_ip)
        self.ed_local_ip = QtWidgets.QLineEdit(self.cfg.local_ip)
        self.sp_port = QtWidgets.QSpinBox()
        self.sp_port.setRange(1, 65535)
        self.sp_port.setValue(self.cfg.tcp_port)

        # IP 输入校验 (简单点分四段)
        ip_re = QtCore.QRegularExpression(
            r"^((25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(25[0-5]|2[0-4]\d|[01]?\d?\d)$")
        validator = QtGui_validator(ip_re)
        self.ed_rx_ip.setValidator(validator)
        self.ed_local_ip.setValidator(validator)

        form.addRow("接收机 IP", self.ed_rx_ip)
        form.addRow("本机 IP", self.ed_local_ip)
        form.addRow("TCP 端口", self.sp_port)

        hint = QtWidgets.QLabel("TCP 控制端口通常固定为 5025")
        hint.setStyleSheet("color:#6b7685; font-size:11px;")
        form.addRow("", hint)

        # 按钮
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_ok = QtWidgets.QPushButton("确定并连接")
        self.btn_ok.setObjectName("primary")
        self.btn_cancel = QtWidgets.QPushButton("取消")
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(btns)

        self.btn_ok.clicked.connect(self._on_ok)
        self.btn_cancel.clicked.connect(self.reject)

    def _on_ok(self):
        rx = self.ed_rx_ip.text().strip()
        local = self.ed_local_ip.text().strip()
        if not rx or not local:
            QtWidgets.QMessageBox.warning(self, "配置无效", "请填写完整的 IP 地址")
            return
        self.cfg.receiver_ip = rx
        self.cfg.local_ip = local
        self.cfg.tcp_port = self.sp_port.value()
        self.cfg.save()
        self.accept()

    def values(self):
        return self.cfg.receiver_ip, self.cfg.local_ip, self.cfg.tcp_port


# QtGui 延迟导入避免与 validator 命名冲突
def QtGui_validator(regex):
    from PyQt6 import QtGui
    return QtGui.QRegularExpressionValidator(regex)