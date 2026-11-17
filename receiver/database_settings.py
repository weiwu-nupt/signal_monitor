"""Local PostgreSQL connection editor for recognition storage."""
from PyQt6 import QtWidgets

from recognition_store import (RecognitionStore, database_settings,
                               postgres_url, save_database_settings)


class PostgreSQLSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('PostgreSQL 数据库设置')
        self.setMinimumWidth(480)
        values = database_settings()
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.host = QtWidgets.QLineEdit(str(values['host']))
        self.port = QtWidgets.QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(int(values['port']))
        self.database = QtWidgets.QLineEdit(str(values['database']))
        self.user = QtWidgets.QLineEdit(str(values['user']))
        self.password = QtWidgets.QLineEdit(str(values.get('password') or ''))
        self.password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        form.addRow('主机地址', self.host)
        form.addRow('端口', self.port)
        form.addRow('数据库名', self.database)
        form.addRow('用户名', self.user)
        form.addRow('密码', self.password)
        layout.addLayout(form)
        self.status = QtWidgets.QLabel('密码仅保存在本机 results/postgresql.json，不会写入程序或 Linux 包。')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.test_button = buttons.addButton('测试连接', QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        self.test_button.clicked.connect(self.test_connection)
        layout.addWidget(buttons)

    def values(self):
        return dict(host=self.host.text().strip(), port=self.port.value(),
                    database=self.database.text().strip(), user=self.user.text().strip(),
                    password=self.password.text())

    def test_connection(self):
        try:
            store = RecognitionStore(postgres_url(self.values()))
            store.close()
        except Exception as exc:
            self.status.setText(f'连接失败：{exc}')
        else:
            self.status.setText('连接成功，识别表结构已准备完成。')

    def save(self):
        try:
            save_database_settings(self.values())
        except (OSError, ValueError) as exc:
            self.status.setText(f'无法保存数据库设置：{exc}')
            return
        self.accept()
