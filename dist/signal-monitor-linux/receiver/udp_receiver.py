"""
UDP 数据链路
=============
每个端口一个后台线程, 收 datagram -> DataProtocolParser -> 解析对象,
通过 Qt 信号 dataReceived(port, obj) 抛到主线程.

端口与数据通道 (来自配置文档/手册):
    8000 MEAS  频谱/测量/电平/ITU
    8001 AUD   音频
    8002 SCAN  PSCAN
    8004 IF/IQ
    8008 SCAN  FSCAN
    8010 SCAN  MSCAN
    8012 SCAN  FPSCAN
    8020 DDC
    8030 DDC 音频
"""

from __future__ import annotations

import socket
import threading
from typing import Dict, List, Optional

from PyQt6 import QtCore

from .rx_protocol import DataProtocolParser, AnyData


class UdpPort(QtCore.QObject):
    """单个 UDP 端口的接收线程"""

    dataReceived = QtCore.pyqtSignal(int, object)   # (port, AnyData)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self, port: int, bind_ip: str = "0.0.0.0",
                 parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.port = port
        self.bind_ip = bind_ip
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._parser = DataProtocolParser()

    def start(self) -> None:
        if self._running:
            return
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 大接收缓冲, 频谱/扫描帧较大
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            s.bind((self.bind_ip, self.port))
            s.settimeout(0.2)
        except OSError as e:
            self.errorOccurred.emit(f"UDP {self.port} 绑定失败: {e}")
            return
        self._sock = s
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"UDP-{self.port}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_stats(self) -> dict:
        return dict(self._parser.stats)

    def _loop(self) -> None:
        while self._running:
            try:
                data, _addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            for obj in self._parser.parse_datagram(data):
                self.dataReceived.emit(self.port, obj)


class UdpManager(QtCore.QObject):
    """管理一组 UDP 端口"""

    dataReceived = QtCore.pyqtSignal(int, object)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self, bind_ip: str = "0.0.0.0",
                 parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.bind_ip = bind_ip
        self._ports: Dict[int, UdpPort] = {}

    def open_ports(self, ports: List[int]) -> None:
        for p in ports:
            if p in self._ports:
                continue
            up = UdpPort(p, self.bind_ip)
            up.dataReceived.connect(self.dataReceived)
            up.errorOccurred.connect(self.errorOccurred)
            up.start()
            self._ports[p] = up

    def close_port(self, port: int) -> None:
        up = self._ports.pop(port, None)
        if up is not None:
            up.stop()

    def close_all(self) -> None:
        for up in list(self._ports.values()):
            up.stop()
        self._ports.clear()

    def active_ports(self) -> List[int]:
        return list(self._ports.keys())

    def get_stats(self) -> Dict[int, dict]:
        return {p: up.get_stats() for p, up in self._ports.items()}