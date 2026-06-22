"""
接收机控制模块 (LS-TOOLS 协议)
================================
TCP (SCPI, 5025) 控制 + UDP 多端口数据接收, 支持 5 种模式:
    FFM / PSCAN(含 FPSCAN) / FSCAN / MSCAN / DDC

挂载到现有 FPGA 频谱程序 (PyQt6 + pyqtgraph).

子模块:
    rx_protocol  - 8字节报头 + 16字节 metadata + data 的数据协议解析
    scpi         - SCPI 命令字符串构造 (按板卡手册 2.x 原样)
    tcp_control  - TCP 5025 控制链路 (QThread)
    udp_receiver - UDP 数据链路 (每端口一个 QThread)
    controller   - 模式控制器, 串起 TCP+UDP, 对外发 Qt 信号
    mode_panels  - 各模式参数面板 (QWidget)
    rx_widget    - 顶层可挂载 QWidget (控制面板 + 信号出口)
"""

from .controller import ReceiverController, RxMode
from .rx_widget import ReceiverControlWidget

__all__ = ["ReceiverController", "RxMode", "ReceiverControlWidget"]