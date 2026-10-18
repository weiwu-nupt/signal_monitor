"""
数据源
=======
- DataSource: 抽象基类
- SimulatedSource: 走协议字节流的仿真器, 用于无 FPGA 时开发
- PCIeSource: 真实 PCIe 字符设备读取

输出统一为协议层的 SpectrumMessage / SignalMessage.
"""

from __future__ import annotations

import os
import time
import threading
import errno
from abc import ABC, abstractmethod
from queue import Queue, Full, Empty
from typing import Optional, Callable, Union

import numpy as np

from protocol import (
    ProtocolParser, SpectrumMessage, SignalMessage,
)


AnyMessage = Union[SpectrumMessage, SignalMessage]


# ============================================================
# 抽象基类
# ============================================================

class DataSource(ABC):
    """后台线程接收, 主线程通过 get_message 取消息"""

    def __init__(self, queue_size: int = 16):
        self._queue: Queue[AnyMessage] = Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._drop_count = 0

    def set_error_callback(self, cb: Callable[[Exception], None]) -> None:
        self._error_callback = cb

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_message(self, timeout: float = 0.0) -> Optional[AnyMessage]:
        try:
            return self._queue.get(timeout=timeout) if timeout > 0 \
                   else self._queue.get_nowait()
        except Empty:
            return None

    def get_drop_count(self) -> int:
        return self._drop_count

    def get_stats(self) -> dict:
        return {"drops": self._drop_count}

    def _push_message(self, msg: AnyMessage) -> None:
        try:
            self._queue.put_nowait(msg)
        except Full:
            try:
                self._queue.get_nowait()
                self._drop_count += 1
            except Empty:
                pass
            try:
                self._queue.put_nowait(msg)
            except Full:
                pass

    def _run_loop(self) -> None:
        try:
            self._loop_impl()
        except Exception as e:
            if self._error_callback:
                self._error_callback(e)
            else:
                raise

    @abstractmethod
    def _loop_impl(self) -> None: ...


# ============================================================
# 仿真数据源 (走完整协议字节流)
# ============================================================

class SimulatedSource(DataSource):
    """生成符合协议的字节流, 再用解析器解析, 输出与真实硬件一致的消息"""

    def __init__(self,
                 fft_points: int = 65535,
                 center_mhz: int = 1500,
                 bandwidth_mhz: int = 300,
                 frame_rate_hz: float = 1.0,
                 queue_size: int = 16):
        super().__init__(queue_size=queue_size)
        self.fft_points = fft_points
        self.center_mhz = center_mhz
        self.bandwidth_mhz = bandwidth_mhz
        self.frame_period = 1.0 / frame_rate_hz
        self._parser = ProtocolParser(samples_per_spectrum=fft_points)

    def _make_samples(self) -> np.ndarray:
        """生成 int16 频谱样点 (dB * 10)"""
        N = self.fft_points
        spec_db = -95.0 + np.random.randn(N).astype(np.float32) * 2.0
        n_sig = np.random.randint(3, 7)
        for _ in range(n_sig):
            pos = np.random.randint(int(N * 0.03), int(N * 0.97))
            amp = np.random.uniform(20, 55)
            width = np.random.randint(20, 200)
            idx = np.arange(N)
            spec_db += amp * np.exp(-((idx - pos) ** 2) / (2 * width ** 2))
        return (spec_db * 10).astype(np.int16)

    def get_stats(self) -> dict:
        s = super().get_stats()
        s.update(self._parser.get_stats())
        return s

    def _loop_impl(self) -> None:
        next_tick = time.monotonic()
        while self._running:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue
            next_tick += self.frame_period

            samples = self._make_samples()
            raw = ProtocolParser.build_spectrum_frame(
                self.center_mhz, self.bandwidth_mhz, samples,
            )
            self._parser.feed(raw)
            for msg in self._parser.parse_all():
                self._push_message(msg)


# ============================================================
# PCIe 真实数据源
# ============================================================

class PCIeSource(DataSource):
    """
    通过字符设备读取 PCIe 数据.
    典型路径: /dev/xdma0_c2h_0 (Card-to-Host channel 0)

    驱动准备:
      git clone https://github.com/Xilinx/dma_ip_drivers
      cd dma_ip_drivers/XDMA/linux-kernel/xdma && make && sudo make install
      sudo modprobe xdma
      lspci | grep -i xilinx
      ls /dev/xdma*
    """

    DEFAULT_READ_SIZE = 65536   # 一次读 64KB

    def __init__(self,
                 device_path: str = "/dev/xdma0_c2h_0",
                 fft_points: int = 65535,
                 read_size: int = DEFAULT_READ_SIZE,
                 verify_crc: bool = True,
                 queue_size: int = 16):
        super().__init__(queue_size=queue_size)
        self.device_path = device_path
        self.fft_points = fft_points
        self.read_size = read_size
        self._parser = ProtocolParser(
            samples_per_spectrum=fft_points,
            verify_crc=verify_crc,
        )
        self._bytes_read = 0

    def get_stats(self) -> dict:
        s = super().get_stats()
        s.update(self._parser.get_stats())
        s["bytes_read"] = self._bytes_read
        return s

    def _loop_impl(self) -> None:
        if not os.path.exists(self.device_path):
            raise FileNotFoundError(
                f"PCIe 设备 {self.device_path} 不存在. "
                f"请检查: \n"
                f"  1) lsmod | grep xdma  (驱动加载)\n"
                f"  2) lspci | grep -i xilinx  (硬件识别)\n"
                f"  3) ls /dev/xdma*  (设备节点)"
            )

        try:
            fd = os.open(self.device_path, os.O_RDONLY)
        except PermissionError as e:
            raise PermissionError(
                f"打开 {self.device_path} 权限不足. \n"
                f"用 sudo 运行, 或加 udev 规则:\n"
                f'  echo \'SUBSYSTEM=="xdma", MODE="0666"\' | '
                f'sudo tee /etc/udev/rules.d/60-xdma.rules\n'
                f"  sudo udevadm control --reload && sudo udevadm trigger"
            ) from e

        try:
            while self._running:
                try:
                    chunk = os.read(fd, self.read_size)
                except OSError as e:
                    if e.errno == errno.EAGAIN:
                        time.sleep(0.001)
                        continue
                    elif e.errno == errno.EINTR:
                        continue
                    else:
                        raise

                if not chunk:
                    time.sleep(0.001)
                    continue

                self._bytes_read += len(chunk)
                self._parser.feed(chunk)
                for msg in self._parser.parse_all():
                    self._push_message(msg)
        finally:
            os.close(fd)


# ============================================================
# 工厂
# ============================================================

def create_source(mode: str = "sim", **kwargs) -> DataSource:
    if mode == "sim":
        return SimulatedSource(**kwargs)
    elif mode == "pcie":
        return PCIeSource(**kwargs)
    else:
        raise ValueError(f"未知模式: {mode}")


if __name__ == "__main__":
    print("=== 测试 SimulatedSource ===")
    src = SimulatedSource(fft_points=8192, frame_rate_hz=5.0)
    src.start()
    for i in range(3):
        msg = src.get_message(timeout=2.0)
        if isinstance(msg, SpectrumMessage):
            print(f"  spectrum: center={msg.center_freq_hz/1e6:.0f}MHz, "
                  f"bw={msg.bandwidth_hz/1e6:.0f}MHz, "
                  f"points={msg.fft_points}, "
                  f"peak={msg.spectrum_db.max():.1f}dB")
    src.stop()
    print(f"  stats: {src.get_stats()}")