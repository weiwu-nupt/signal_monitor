"""
集成 / 独立运行入口 (页面式)
==============================
结构:
    主窗口中央 = QStackedWidget
        page 0  : 原 FPGA 频谱界面 (保留)
        page 1+ : 接收机各模式专用页 (左显示区 + 右参数)
    菜单:
        参数配置 ▾ -> 接收机配置...        弹出对话框
        接收机工作模式 ▾ -> FFM/PSCAN/FPSCAN/FSCAN/MSCAN/DDC
                            选中 -> 中央切到该模式页 (只切页, 点[应用配置]才下发)
        视图 ▾ -> FPGA 频谱界面            切回原界面
    连接状态: 放在状态栏 (持久, 各页共享)

A) 集成到现有 FPGA 程序:
    from receiver.integrate import attach_receiver_pages
    attach_receiver_pages(main_window, fpga_central_widget)

B) 独立运行 (无宿主, page0 用一个占位):
    python -m receiver.integrate
"""

from __future__ import annotations

import sys
from PyQt6 import QtCore, QtGui, QtWidgets

from .controller import ReceiverController, RxMode
from .mode_pages import PAGE_CLASSES
from .rx_config import RxConfig
from .rx_dialog import ReceiverConfigDialog


# 模式菜单顺序与标签
MODE_ORDER = [
    (RxMode.FFM, "FFM 固定频点"),
    (RxMode.PSCAN, "PSCAN 全景扫描"),
    (RxMode.FPSCAN, "FPSCAN 快速全景"),
    (RxMode.FSCAN, "FSCAN 扫频搜索"),
    (RxMode.MSCAN, "MSCAN 信道扫描"),
    (RxMode.DDC, "DDC 数字下变频"),
]


class ReceiverPagesManager(QtCore.QObject):
    """
    管理: 中央 QStackedWidget + 菜单 + 状态栏连接指示 + 各模式页.
    PSCAN 和 FPSCAN 共用一个 PScanPage (菜单两项都切到它, 并设 fast 复选).
    """

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
            lambda m: self._flash_status(f"已下发模式: {m.value}"))

        self._on_connected(False)

    def _create_pages(self):
        for mode, cls in PAGE_CLASSES.items():
            page = cls(self.controller)
            idx = self.stack.addWidget(page)
            self.pages[mode] = page
            self._stack_index[mode] = idx
        # FPSCAN 复用 PSCAN 页
        self.pages[RxMode.FPSCAN] = self.pages[RxMode.PSCAN]
        self._stack_index[RxMode.FPSCAN] = self._stack_index[RxMode.PSCAN]

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

        # 接收机工作模式
        m_mode = mbar.addMenu("接收机工作模式")
        self._mode_group = QtGui.QActionGroup(self.win)
        self._mode_group.setExclusive(True)
        self._mode_actions = {}
        for mode, label in MODE_ORDER:
            act = m_mode.addAction(label)
            act.setCheckable(True)
            self._mode_group.addAction(act)
            act.triggered.connect(lambda _c, m=mode: self.show_mode(m))
            self._mode_actions[mode] = act

        # 视图: 切回原 FPGA 界面 (page 0)
        m_view = mbar.addMenu("视图")
        act_fpga = m_view.addAction("FPGA 频谱界面")
        act_fpga.triggered.connect(self.show_fpga)

    def _open_cfg_dialog(self):
        dlg = ReceiverConfigDialog(self.cfg, self.win)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            rx, local, port = dlg.values()
            self.lbl_target.setText(f"{rx}:{port}")
            self.controller.connect_receiver(rx, local, port)

    # -------- 切页 --------

    def show_mode(self, mode: RxMode):
        """菜单选中模式: 中央切到该模式页 (不下发)"""
        page = self.pages[mode]
        self.stack.setCurrentIndex(self._stack_index[mode])
        # FPSCAN: 在共用页上勾选 fast
        if mode == RxMode.FPSCAN and hasattr(page.panel, "set_fast"):
            page.panel.set_fast(True)
        elif mode == RxMode.PSCAN and hasattr(page.panel, "set_fast"):
            page.panel.set_fast(False)
        if mode in self._mode_actions:
            self._mode_actions[mode].setChecked(True)
        self._flash_status(f"已切换到 {mode.value} 界面 (点[应用配置]下发)")

    def show_fpga(self):
        """切回原 FPGA 界面 (page 0)"""
        self.stack.setCurrentIndex(0)
        for a in self._mode_actions.values():
            a.setChecked(False)
        self._flash_status("已切换到 FPGA 频谱界面")

    # -------- 状态回调 --------

    def _on_connected(self, connected: bool):
        if connected:
            self.lbl_conn.setText("● 已连接")
            self.lbl_conn.setStyleSheet("color:#5bff8c; font-weight:bold;")
        else:
            self.lbl_conn.setText("● 未连接")
            self.lbl_conn.setStyleSheet("color:#ff5b5b; font-weight:bold;")
        for page in set(self.pages.values()):
            page.set_enabled_actions(connected)

    def _on_running(self, running: bool):
        for page in set(self.pages.values()):
            page.btn_start.setEnabled(not running and self.controller.is_connected())
            page.btn_stop.setEnabled(running)

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
    """
    把主窗口中央替换为 QStackedWidget:
        page 0 = fpga_widget (原 FPGA 界面; 不传则用主窗口当前 central)
        page 1+ = 接收机各模式页
    并加菜单 + 状态栏连接指示.

    返回 ReceiverPagesManager.
    """
    controller = controller or ReceiverController(main_window)

    # 取原中央部件作为 page 0
    if fpga_widget is None:
        fpga_widget = main_window.centralWidget()

    stack = QtWidgets.QStackedWidget()
    if fpga_widget is not None:
        stack.addWidget(fpga_widget)        # page 0
    else:
        placeholder = QtWidgets.QLabel("FPGA 频谱界面")
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        stack.addWidget(placeholder)

    main_window.setCentralWidget(stack)
    mgr = ReceiverPagesManager(main_window, stack, controller)
    return mgr


# ============================================================
# B) 独立运行
# ============================================================

class _StandaloneWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("接收机控制 (页面式 / 独立调试)")
        self.resize(1400, 860)
        self.setStyleSheet("QMainWindow{background:#0e1116;} "
                           "QStatusBar{background:#0a0d12;color:#aaa;} "
                           "QMenuBar{background:#141923;color:#d0d8e0;} "
                           "QMenuBar::item:selected{background:#1c2733;} "
                           "QMenu{background:#141923;color:#d0d8e0;} "
                           "QMenu::item:selected{background:#1c2733;color:#00d4ff;}")

        import pyqtgraph as pg
        pg.setConfigOption("background", "#0e1116")
        pg.setConfigOption("foreground", "#d0d8e0")
        pg.setConfigOption("antialias", False)

        # 占位 FPGA 界面 (page 0)
        fpga = QtWidgets.QLabel("（原 FPGA 频谱界面占位）\n通过菜单「接收机工作模式」切换到各模式页")
        fpga.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        fpga.setStyleSheet("color:#5a6573; font-size:16px;")

        self.controller = ReceiverController(self)
        self.mgr = attach_receiver_pages(self, fpga, self.controller)
        # 默认显示 FFM 页
        self.mgr.show_mode(RxMode.FFM)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = _StandaloneWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()