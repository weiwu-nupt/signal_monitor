"""
DMA 频谱数据源
================
基于 spectrum_reader.py 的 slot/ACK 协议, 通过 XDMA 字符设备
(/dev/xdma0_c2h_0 读, /dev/xdma0_h2c_0 写 ACK) 接收 8192 点频谱包,
在后台线程里持续轮询, 输出与 GUI 兼容的 SpectrumMessage.

与 data_source.py 中 PCIeSource 的区别:
  - PCIeSource 是"流式字节流 + ProtocolParser"模型 (int16 样点)
  - 本模块是"寄存器 status 轮询 + 绝对地址定长读 + 写 ACK"模型 (uint64 power)
本模块直接复用 spectrum_reader.py 里已验证过的 read_one_spectrum() 逻辑.

用法 (独立):
    src = DMASpectrumSource()
    src.start()
    msg = src.get_message(timeout=1.0)
    src.stop()

用法 (接入现有 main.py):
    在 create_source / 命令行里增加 mode="dma" 即可, 见文件末尾说明.
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from queue import Queue, Full, Empty
from typing import Optional, Callable

import numpy as np

# 复用 spectrum_reader.py 里已验证的底层协议实现
import spectrum_reader as sr

from protocol import SpectrumMessage


# ============================================================
# GUI 兼容的频谱消息
# ============================================================
# spectrum_reader 产出的是已经算好的 power_db (float64, 已 10*log10).
# 现有 protocol.SpectrumMessage 的 spectrum_db 属性是 samples * 0.1,
# 为了不改 main.py / waterfall.py, 这里做一个适配子类:
# 把 power_db 直接存进去, 覆写 spectrum_db 属性原样返回.

@dataclass
class DMASpectrumMessage(SpectrumMessage):
    """spectrum_reader 的 power_db 适配为 GUI 用的 SpectrumMessage"""

    _power_db: Optional[np.ndarray] = None

    @property
    def spectrum_db(self) -> np.ndarray:  # type: ignore[override]
        # power_db 已是 dB, 直接返回 float32
        if self._power_db is not None:
            return self._power_db
        return self.samples.astype(np.float32) * 0.1


# ============================================================
# DMA 频谱数据源
# ============================================================

class DMASpectrumSource:
    """
    后台线程持续调用 spectrum_reader.read_one_spectrum(),
    把每一帧封装成 DMASpectrumMessage 推进队列, 主线程 get_message 取.

    心跳节流: 没有新包时 sleep(interval), 不空转占满 CPU.
    错误处理: 单帧解析/校验失败默认丢弃并记 bad 计数, 不中断线程;
              设备级错误 (打开失败/读空) 通过 error_callback 抛给 GUI.
    """

    def __init__(self,
                 c2h_dev: str = sr.C2H_DEV,
                 h2c_dev: str = sr.H2C_DEV,
                 sync_words=sr.DEFAULT_ALLOWED_SYNC_WORDS,
                 poll_interval: float = 0.02,
                 raise_on_bad_packet: bool = False,
                 queue_size: int = 16):
        self.c2h_dev = c2h_dev
        self.h2c_dev = h2c_dev
        self.sync_words = tuple(sync_words)
        self.poll_interval = poll_interval
        self.raise_on_bad_packet = raise_on_bad_packet

        self._queue: "Queue[DMASpectrumMessage]" = Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None

        self._c2h_fd = -1
        self._h2c_fd = -1

        self._stats = {
            "spectrum_ok": 0,
            "signal_ok": 0,      # 该协议无信号消息, 恒为 0 (GUI 统计兼容)
            "bad_crc": 0,
            "bad_packet": 0,
            "empty_poll": 0,
            "drops": 0,
            "bytes_dropped": 0,  # GUI 统计兼容字段
            "last_seq": -1,
        }

    # ---- 与 DataSource 接口保持一致 ----

    def set_error_callback(self, cb: Callable[[Exception], None]) -> None:
        self._error_callback = cb

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="DMASpectrumRx", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_message(self, timeout: float = 0.0) -> Optional[DMASpectrumMessage]:
        try:
            return (self._queue.get(timeout=timeout) if timeout > 0
                    else self._queue.get_nowait())
        except Empty:
            return None

    def get_drop_count(self) -> int:
        return self._stats["drops"]

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ---- 内部 ----

    def _push_message(self, msg: DMASpectrumMessage) -> None:
        try:
            self._queue.put_nowait(msg)
        except Full:
            # 队列满: 丢最旧的一帧, 保留最新
            try:
                self._queue.get_nowait()
                self._stats["drops"] += 1
            except Empty:
                pass
            try:
                self._queue.put_nowait(msg)
            except Full:
                pass

    def _open_devices(self) -> None:
        if not os.path.exists(self.c2h_dev):
            raise FileNotFoundError(
                f"XDMA 读设备 {self.c2h_dev} 不存在. 请检查:\n"
                f"  lsmod | grep xdma   (驱动)\n"
                f"  ls /dev/xdma*       (设备节点)"
            )
        if not os.path.exists(self.h2c_dev):
            raise FileNotFoundError(
                f"XDMA 写设备 {self.h2c_dev} 不存在 (写 ACK 需要)."
            )
        try:
            self._c2h_fd = os.open(self.c2h_dev, os.O_RDONLY | os.O_SYNC)
            self._h2c_fd = os.open(self.h2c_dev, os.O_WRONLY | os.O_SYNC)
        except PermissionError as e:
            raise PermissionError(
                f"打开 XDMA 设备权限不足, 请 sudo 运行, 或加 udev 规则:\n"
                f'  echo \'SUBSYSTEM=="xdma", MODE="0666"\' | '
                f'sudo tee /etc/udev/rules.d/60-xdma.rules\n'
                f"  sudo udevadm control --reload && sudo udevadm trigger"
            ) from e

    def _close_devices(self) -> None:
        for fd_name in ("_c2h_fd", "_h2c_fd"):
            fd = getattr(self, fd_name)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, fd_name, -1)

    def _run_loop(self) -> None:
        try:
            self._open_devices()
            self._loop_impl()
        except Exception as e:
            if self._error_callback:
                self._error_callback(e)
            else:
                raise
        finally:
            self._close_devices()

    def _loop_impl(self) -> None:
        while self._running:
            try:
                result = sr.read_one_spectrum(
                    self._c2h_fd, self._h2c_fd, self.sync_words)
            except RuntimeError as e:
                # 单帧级错误: status magic 错 / 校验失败等
                self._stats["bad_packet"] += 1
                # 校验失败的细分: CRC
                if "crc" in str(e).lower():
                    self._stats["bad_crc"] += 1
                if self.raise_on_bad_packet:
                    raise
                time.sleep(self.poll_interval)
                continue

            if result is None:
                # 当前没有新 packet
                self._stats["empty_poll"] += 1
                time.sleep(self.poll_interval)
                continue

            msg = self._build_message(result)
            self._push_message(msg)

            self._stats["spectrum_ok"] += 1
            self._stats["last_seq"] = result["seq"]

    def _build_message(self, result: dict) -> DMASpectrumMessage:
        pkt = result["packet"]
        power_db = np.ascontiguousarray(result["power_db"], dtype=np.float32)

        # 单位: header 里 center/bw 字段单位是 MHz
        center_hz = float(pkt["center"]) * 1e6
        bw_hz = float(pkt["bw"]) * 1e6
        fft_n = int(pkt["fft_n"])

        return DMASpectrumMessage(
            center_freq_hz=center_hz,
            bandwidth_hz=bw_hz,
            fft_points=fft_n,
            samples=power_db,          # 占位 (spectrum_db 已覆写)
            crc_ok=pkt["checks"]["crc_ok"],
            _power_db=power_db,
        )


# ============================================================
# 工厂辅助: 接入 main.py 用
# ============================================================

def create_dma_source(**kwargs) -> DMASpectrumSource:
    return DMASpectrumSource(**kwargs)


if __name__ == "__main__":
    # 无硬件时这里会因设备不存在而走 error_callback / 抛异常,
    # 真实环境 sudo 运行可直接看到帧.
    import argparse

    ap = argparse.ArgumentParser(description="DMA 频谱源独立测试")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--interval", type=float, default=0.02)
    args = ap.parse_args()

    src = DMASpectrumSource(poll_interval=args.interval)

    def on_err(e):
        print(f"[ERROR] {type(e).__name__}: {e}")

    src.set_error_callback(on_err)
    src.start()

    got = 0
    t0 = time.time()
    while got < args.frames and time.time() - t0 < 10:
        msg = src.get_message(timeout=1.0)
        if msg is None:
            continue
        got += 1
        sdb = msg.spectrum_db
        print(f"frame {got}: center={msg.center_freq_hz/1e6:.0f}MHz "
              f"bw={msg.bandwidth_hz/1e6:.0f}MHz pts={msg.fft_points} "
              f"peak={sdb.max():.1f}dB crc_ok={msg.crc_ok}")
    src.stop()
    print("stats:", src.get_stats())