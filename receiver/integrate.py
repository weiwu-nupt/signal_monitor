"""固定频点监测主界面：频谱、瀑布图与接收机参数。"""

from __future__ import annotations

import sys
from PyQt6 import QtCore, QtGui, QtWidgets

from .controller import ReceiverController, RxMode
from .mode_pages import FFMPage
from .rx_config import RxConfig
from .rx_dialog import ReceiverConfigDialog
from .theme import APP_STYLE


class ReceiverPagesManager(QtCore.QObject):
    """管理唯一的 FFM 页面、连接菜单和接收机状态。"""

    def __init__(self, main_window: QtWidgets.QMainWindow,
                 stack: QtWidgets.QStackedWidget,
                 controller: ReceiverController):
        super().__init__(main_window)
        self.win = main_window
        self.stack = stack
        self.controller = controller
        self.cfg = RxConfig.load()

        # 创建各模式页 (PSCAN/FPSCAN 共用)
        self.pages = {}        # RxMode -> page widget
        self._stack_index = {} # RxMode -> stack index
        self._create_pages()

        # 状态栏连接指示
        self._build_status_bar()
        # 菜单
        self._build_menus()

        # controller 状态 -> UI
        controller.connectedChanged.connect(self._on_connected)
        controller.runningChanged.connect(self._on_running)
        controller.statusMessage.connect(self._on_status_msg)
        controller.errorOccurred.connect(lambda s: self._flash_status("⚠ " + s))
        controller.modeChanged.connect(
            lambda m: self._flash_status("已应用接收机配置"))

        self._on_connected(False)

    def _create_pages(self):
        page = FFMPage(self.controller)
        self.pages[RxMode.FFM] = page
        self._stack_index[RxMode.FFM] = self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)

    # -------- 状态栏 --------

    def _build_status_bar(self):
        sb = self.win.statusBar()
        self.lbl_conn = QtWidgets.QLabel("● 未连接")
        self.lbl_conn.setStyleSheet("color:#ff5b5b; font-weight:bold;")
        self.lbl_target = QtWidgets.QLabel(
            f"{self.cfg.receiver_ip}:{self.cfg.tcp_port}")
        self.lbl_target.setStyleSheet("color:#8a96a5;")
        self.lbl_msg = QtWidgets.QLabel("")
        self.lbl_msg.setStyleSheet("color:#7fd4a0;")
        sb.addWidget(self.lbl_conn)
        sb.addWidget(self.lbl_target)
        sb.addWidget(self.lbl_msg, 1)

    # -------- 菜单 --------

    def _build_menus(self):
        mbar = self.win.menuBar()

        # 参数配置
        m_cfg = mbar.addMenu("参数配置")
        act_rx = m_cfg.addAction("接收机配置...")
        act_rx.triggered.connect(self._open_cfg_dialog)

        self.act_disconnect = m_cfg.addAction("断开接收机")
        self.act_disconnect.triggered.connect(self.controller.disconnect_receiver)

        m_file = mbar.addMenu("读取文件")
        m_file.addAction("打开 IQ 文件并循环回放…", self._open_iq_file)
        m_file.addAction("查看识别记录…", self._show_history)
        m_file.addAction("返回接收机", self._show_receiver)
        self.replay_page = None

    def _show_replay(self):
        if self.replay_page is None:
            from .file_replay import FileReplayPage
            self.replay_page = FileReplayPage()
            self.stack.addWidget(self.replay_page)
        self.stack.setCurrentWidget(self.replay_page)
        return self.replay_page

    def _open_iq_file(self):
        page = self._show_replay()
        if page.worker is None:
            page.choose_file()
        else:
            self._flash_status("请先停止当前文件回放，再选择其他文件")

    def _show_receiver(self):
        if self.replay_page is not None:
            self.replay_page.stop()
        self.stack.setCurrentWidget(self.pages[RxMode.FFM])

    def _show_history(self):
        from .recognition_history import RecognitionHistoryDialog
        from recognition_store import DEFAULT_DATABASE
        RecognitionHistoryDialog(DEFAULT_DATABASE, self.win).exec()

    def shutdown(self):
        if self.replay_page is not None:
            self.replay_page.shutdown()

    def _open_cfg_dialog(self):
        dlg = ReceiverConfigDialog(self.cfg, self.win)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            rx, local, port = dlg.values()
            self.lbl_target.setText(f"{rx}:{port}")
            self.controller.connect_receiver(rx, local, port)

    # -------- 状态回调 --------

    def _on_connected(self, connected: bool):
        if connected:
            self.lbl_conn.setText("● 已连接")
            self.lbl_conn.setStyleSheet("color:#5bff8c; font-weight:bold;")
        else:
            self.lbl_conn.setText("● 未连接")
            self.lbl_conn.setStyleSheet("color:#ff5b5b; font-weight:bold;")
        self.act_disconnect.setEnabled(connected)
        for page in set(self.pages.values()):
            page.set_enabled_actions(connected)

    def _on_running(self, running: bool):
        for page in set(self.pages.values()):
            page.btn_apply.setEnabled(self.controller.is_connected())
            page.btn_pause.setEnabled(running)
            page.btn_stop.setEnabled(running)
            if not running:
                page._set_display_paused(False)

    def _on_status_msg(self, msg: str):
        self._flash_status(msg)

    def _flash_status(self, msg: str):
        ts = QtCore.QTime.currentTime().toString("HH:mm:ss")
        self.lbl_msg.setText(f"{ts}  {msg}")


# ============================================================
# A) 挂到现有主窗口
# ============================================================

def attach_receiver_pages(main_window, fpga_widget=None,
                          controller: ReceiverController = None):
    """安装唯一的 FFM 页面，并保留宿主提供的旧部件供后台使用。"""
    controller = controller or ReceiverController(main_window)
    source = getattr(main_window, 'source', None)
    if source is not None and hasattr(source, 'set_bandwidth_filter'):
        controller.bind_recognition_source(source)
    previous = main_window.takeCentralWidget()
    if previous is not None:
        previous.setParent(main_window)
        previous.hide()
    stack = QtWidgets.QStackedWidget()
    main_window.setCentralWidget(stack)
    mgr = ReceiverPagesManager(main_window, stack, controller)
    main_window.receiver_manager = mgr
    return mgr


# ============================================================
# B) 独立运行
# ============================================================

class _StandaloneWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("接收机频谱监测")
        self.resize(1400, 860)
        self.setStyleSheet(APP_STYLE)

        import pyqtgraph as pg
        pg.setConfigOption("background", "#0e1116")
        pg.setConfigOption("foreground", "#d0d8e0")
        pg.setConfigOption("antialias", False)

        self.setFont(QtGui.QFont("Microsoft YaHei", 10))
        self.controller = ReceiverController(self)
        self.mgr = attach_receiver_pages(self, controller=self.controller)

    def closeEvent(self, event):
        self.mgr.shutdown()
        self.controller.disconnect_receiver()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = _StandaloneWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
