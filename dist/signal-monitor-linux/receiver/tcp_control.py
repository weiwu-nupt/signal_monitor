"""
TCP 控制链路 (SCPI, 默认 5025)
================================
- 连接 / 断开
- send(cmd) 异步发送 (线程安全队列)
- query(cmd) 发送并等待一行返回
- 对外信号: connected / disconnected / errorOccurred / responseReceived
"""

from __future__ import annotations

import socket
import threading
import queue
from typing import Optional

from PyQt6 import QtCore

from . import scpi


class TcpControl(QtCore.QObject):
    connected = QtCore.pyqtSignal()
    disconnected = QtCore.pyqtSignal()
    errorOccurred = QtCore.pyqtSignal(str)
    responseReceived = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._sock: Optional[socket.socket] = None
        self._tx_q: "queue.Queue[bytes]" = queue.Queue()
        self._tx_thread: Optional[threading.Thread] = None
        self._rx_thread: Optional[threading.Thread] = None
        self._running = False
        self._rx_buf = bytearray()
        self._resp_q: "queue.Queue[str]" = queue.Queue()

    # -------- 连接管理 --------

    def connect_to(self, ip: str, port: int = 5025, timeout: float = 3.0) -> None:
        self.close()
        try:
            s = socket.create_connection((ip, port), timeout=timeout)
            s.settimeout(0.2)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as e:
            self.errorOccurred.emit(f"TCP 连接失败 {ip}:{port} - {e}")
            return
        self._sock = s
        self._running = True
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx_thread.start()
        self._rx_thread.start()
        self.connected.emit()

    def close(self) -> None:
        self._running = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        for t in (self._tx_thread, self._rx_thread):
            if t is not None:
                t.join(timeout=1.0)
        self._tx_thread = self._rx_thread = None
        # 清空队列
        with self._tx_q.mutex:
            self._tx_q.queue.clear()
        self.disconnected.emit()

    def is_connected(self) -> bool:
        return self._sock is not None and self._running

    # -------- 发送 / 查询 --------

    def send(self, cmd: str) -> None:
        """异步发送一条 SCPI 命令 (自动补 CR)"""
        if not self.is_connected():
            self.errorOccurred.emit("未连接, 无法发送命令")
            return
        self._tx_q.put(scpi.term(cmd))

    def send_many(self, cmds) -> None:
        for c in cmds:
            self.send(c)

    def query(self, cmd: str, timeout: float = 1.0) -> Optional[str]:
        """发送查询并等待一行响应 (阻塞调用方, 勿在 GUI 线程长用)"""
        if not self.is_connected():
            return None
        # 清掉旧响应
        while not self._resp_q.empty():
            try:
                self._resp_q.get_nowait()
            except queue.Empty:
                break
        self._tx_q.put(scpi.term(cmd))
        try:
            return self._resp_q.get(timeout=timeout)
        except queue.Empty:
            return None

    # -------- 线程 --------

    def _tx_loop(self) -> None:
        while self._running:
            try:
                data = self._tx_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._sock.sendall(data)
            except OSError as e:
                if self._running:
                    self.errorOccurred.emit(f"TCP 发送错误: {e}")
                break

    def _rx_loop(self) -> None:
        while self._running:
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError as e:
                if self._running:
                    self.errorOccurred.emit(f"TCP 接收错误: {e}")
                break
            if not chunk:
                if self._running:
                    self.errorOccurred.emit("TCP 连接被对端关闭")
                break
            self._rx_buf.extend(chunk)
            self._drain_lines()

    def _drain_lines(self) -> None:
        # SCPI 响应以 \r 或 \n 结束
        while True:
            nl = -1
            for i, b in enumerate(self._rx_buf):
                if b in (0x0A, 0x0D):
                    nl = i
                    break
            if nl < 0:
                break
            line = bytes(self._rx_buf[:nl]).decode("ascii", errors="ignore").strip()
            del self._rx_buf[:nl + 1]
            if line:
                self._resp_q.put(line)
                self.responseReceived.emit(line)