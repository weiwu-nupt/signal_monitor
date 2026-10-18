"""
接收机配置对话框
=================
弹出式: 接收机 IP / 本机 IP / TCP 端口.
点"确定"保存到 MRConfig.ini 并发起连接.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from .rx_config import RxConfig
from .inputs import NumberInput
from .theme import APP_STYLE


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
        self.setStyleSheet(APP_STYLE)

        self.cfg = cfg or RxConfig.load()

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setContentsMargins(16, 16, 16, 8)
        form.setSpacing(10)

        self.ed_rx_ip = QtWidgets.QLineEdit(self.cfg.receiver_ip)
        self.ed_local_ip = QtWidgets.QLineEdit(self.cfg.local_ip)
        self.sp_port = NumberInput(1, 65535, self.cfg.tcp_port, name="控制端口", decimals=0)

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
        if not self.ed_rx_ip.hasAcceptableInput() or not self.ed_local_ip.hasAcceptableInput():
            QtWidgets.QMessageBox.warning(self, "配置无效", "请填写完整的 IP 地址")
            return
        try:
            port = self.sp_port.value()
        except ValueError as error:
            QtWidgets.QMessageBox.warning(self, "配置无效", str(error))
            return
        self.cfg.receiver_ip = rx
        self.cfg.local_ip = local
        self.cfg.tcp_port = port
        self.cfg.save()
        self.accept()

    def values(self):
        return self.cfg.receiver_ip, self.cfg.local_ip, self.cfg.tcp_port


# QtGui 延迟导入避免与 validator 命名冲突
def QtGui_validator(regex):
    from PyQt6 import QtGui
    return QtGui.QRegularExpressionValidator(regex)