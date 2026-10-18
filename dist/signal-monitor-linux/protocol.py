"""
FPGA -> 主机 协议定义
======================

帧同步头: 4 字节 = 0x1ACFFC1D (CCSDS Attached Sync Marker, 大端)
所有多字节字段默认小端 (FPGA/x86)

------------------------------------------------------------
消息类型 0x01: 频谱消息
------------------------------------------------------------
 序号 | 字段       | 字节 | 说明
  1   | 帧同步头   |  4   | 0x1ACFFC1D
  2   | 消息类型   |  1   | 0x01
  3   | 消息长度   |  4   | 整帧长度 (含同步头与CRC)
  4   | 中心频率   |  2   | MHz
  5   | 信号带宽   |  2   | MHz
  6   | FFT点数    |  2   | 65535
  7   | 信号样点   |  N   | 频谱数据
  8   | CRC        |  2   | 校验范围: 字段4~6 (?待确认)

------------------------------------------------------------
消息类型 0x02: 信号消息 (描述若干检测到的信号)
------------------------------------------------------------
 序号 | 字段       | 字节   | 说明
  1   | 帧同步头   |  4     | 0x1ACFFC1D
  2   | 消息类型   |  1     | 0x02
  3   | 消息长度   |  4     | 整帧长度
  4   | 信号个数 K |  1     |
  5   | 信号编号   |  1     | 字段5~10 重复 K 次
  6   | 中心频率   |  2     | MHz
  7   | 信号带宽   |  2     | MHz
  8   | 采样频率   |  2     |
  9   | FFT点数    |  2     |
  10  | 信号样点   |  M     | 每个信号的样点数据
  11  | CRC        |  2     |
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np


# ============================================================
# 常量
# ============================================================

SYNC_HEADER: bytes = b"\x1A\xCF\xFC\x1D"   # 大端书写顺序
SYNC_HEADER_INT: int = 0x1ACFFC1D
SYNC_SIZE: int = 4

# 公共字段大小
MSG_TYPE_SIZE: int = 1
MSG_LEN_SIZE: int = 4
CRC_SIZE: int = 2

# 频谱消息固定部分 (从同步头到FFT点数字段, 不含样点和CRC)
SPECTRUM_HEADER_SIZE: int = SYNC_SIZE + MSG_TYPE_SIZE + MSG_LEN_SIZE + 2 + 2 + 2
# = 4 + 1 + 4 + 2 + 2 + 2 = 15 字节

# 信号消息固定头部 (同步头 + 消息类型 + 消息长度 + 信号个数)
SIGNAL_FIXED_HEADER_SIZE: int = SYNC_SIZE + MSG_TYPE_SIZE + MSG_LEN_SIZE + 1
# = 4 + 1 + 4 + 1 = 10 字节
SIGNAL_SUBHEADER_SIZE: int = 1 + 2 + 2 + 2 + 2   # 单个信号的头部 = 9 字节


class MsgType(IntEnum):
    SPECTRUM = 0x01
    SIGNAL = 0x02


# 单位 (待硬件方确认)
FREQ_UNIT_HZ: int = 1_000_000      # 中心频率/带宽/采样率字段单位都是 MHz

# 样点数据类型 (待硬件方确认) - 暂定 int16, 缩放 0.1 -> dB
SAMPLE_DTYPE = np.int16
SAMPLE_BYTES: int = 2
SAMPLE_SCALE: float = 0.1


# ============================================================
# CRC-16-CCITT (0x1021, init=0xFFFF) - 待硬件方确认是否一致
# ============================================================

def crc16_ccitt(data: bytes, init: int = 0xFFFF, poly: int = 0x1021) -> int:
    crc = init
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
    return crc


# ============================================================
# 解析结果数据类
# ============================================================

@dataclass
class SpectrumMessage:
    """0x01 频谱消息"""
    center_freq_hz: float
    bandwidth_hz: float
    fft_points: int
    samples: np.ndarray            # int16 原始样点
    crc_ok: bool = True

    @property
    def spectrum_db(self) -> np.ndarray:
        """转换为 dB 值"""
        return self.samples.astype(np.float32) * SAMPLE_SCALE

    @property
    def freq_axis_hz(self) -> np.ndarray:
        """生成频率轴"""
        start = self.center_freq_hz - self.bandwidth_hz / 2
        stop = self.center_freq_hz + self.bandwidth_hz / 2
        return np.linspace(start, stop, self.fft_points, dtype=np.float64)


@dataclass
class SignalEntry:
    """信号消息中的单个信号"""
    signal_id: int
    center_freq_hz: float
    bandwidth_hz: float
    sample_rate_hz: float
    fft_points: int
    samples: np.ndarray


@dataclass
class SignalMessage:
    """0x02 信号消息"""
    signals: List[SignalEntry] = field(default_factory=list)
    crc_ok: bool = True


# ============================================================
# 流式解析器
# ============================================================

class ProtocolParser:
    """
    流式协议解析器: 接受任意分段的字节流, 输出完整消息.

    使用:
        parser = ProtocolParser(samples_per_spectrum=65535)
        parser.feed(bytes_from_pcie)
        for msg in parser.parse_all():
            if isinstance(msg, SpectrumMessage): ...
            elif isinstance(msg, SignalMessage): ...
    """

    # 单帧最大长度 (用于丢弃太大的脏数据); 65535 点 * 2字节 + ~20字节头尾 ≈ 130KB
    MAX_FRAME_SIZE = 200 * 1024

    def __init__(self,
                 samples_per_spectrum: int = 65535,
                 verify_crc: bool = True):
        self.samples_per_spectrum = samples_per_spectrum
        self.verify_crc = verify_crc
        self._buf = bytearray()
        self._stats = {
            "spectrum_ok": 0,
            "signal_ok": 0,
            "bad_crc": 0,
            "bad_length": 0,
            "unknown_type": 0,
            "bytes_dropped": 0,
        }

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)
        # 防御: 如果 buffer 异常增大, 截断 (说明上游死循环或没解析出来)
        if len(self._buf) > self.MAX_FRAME_SIZE * 4:
            self._stats["bytes_dropped"] += len(self._buf) - self.MAX_FRAME_SIZE * 2
            del self._buf[:len(self._buf) - self.MAX_FRAME_SIZE * 2]

    def parse_all(self) -> list:
        """从缓冲中尽可能多地解析消息"""
        msgs = []
        while True:
            msg = self._try_parse_one()
            if msg is None:
                break
            if msg is not False:    # False 表示丢字节后重试
                msgs.append(msg)
        return msgs

    def get_stats(self) -> dict:
        return dict(self._stats)

    # -------- 内部 --------

    def _find_sync(self) -> int:
        """在 buffer 中找同步头, 返回偏移; 找不到返回 -1"""
        return self._buf.find(SYNC_HEADER)

    def _try_parse_one(self):
        """
        尝试解析一条消息.
        返回:
          - 一个消息对象 (成功)
          - False (丢了字节重新同步, 应继续尝试)
          - None (缓冲不足, 等更多数据)
        """
        # 1) 找同步头
        idx = self._find_sync()
        if idx < 0:
            # 没找到同步头, 保留最后3字节 (可能是不完整的同步头)
            if len(self._buf) > 3:
                self._stats["bytes_dropped"] += len(self._buf) - 3
                del self._buf[:-3]
            return None
        if idx > 0:
            self._stats["bytes_dropped"] += idx
            del self._buf[:idx]

        # 2) 至少需要 同步头(4) + 消息类型(1) + 消息长度(4) = 9 字节
        if len(self._buf) < SYNC_SIZE + MSG_TYPE_SIZE + MSG_LEN_SIZE:
            return None

        msg_type = self._buf[SYNC_SIZE]
        msg_len = struct.unpack(
            "<I",
            bytes(self._buf[SYNC_SIZE + MSG_TYPE_SIZE:
                            SYNC_SIZE + MSG_TYPE_SIZE + MSG_LEN_SIZE])
        )[0]

        # 合理性检查
        if msg_len < SYNC_SIZE + MSG_TYPE_SIZE + MSG_LEN_SIZE + CRC_SIZE \
                or msg_len > self.MAX_FRAME_SIZE:
            # 长度不合理: 丢同步头一个字节重新找
            self._stats["bad_length"] += 1
            self._stats["bytes_dropped"] += 1
            del self._buf[0]
            return False

        # 3) 等够整帧
        if len(self._buf) < msg_len:
            return None

        frame = bytes(self._buf[:msg_len])

        # 4) 校验 CRC
        crc_recv = struct.unpack("<H", frame[-CRC_SIZE:])[0]
        crc_ok = True
        if self.verify_crc:
            # CRC 范围 (待硬件方确认) - 这里实现为"除CRC本身外的所有字节"
            crc_calc = crc16_ccitt(frame[:-CRC_SIZE])
            if crc_calc != crc_recv:
                crc_ok = False
                self._stats["bad_crc"] += 1
                # CRC 错: 丢一个字节重试
                self._stats["bytes_dropped"] += 1
                del self._buf[0]
                return False

        # 5) 按类型解析
        if msg_type == MsgType.SPECTRUM:
            msg = self._parse_spectrum(frame, crc_ok)
            if msg is not None:
                self._stats["spectrum_ok"] += 1
                del self._buf[:msg_len]
                return msg
            else:
                self._stats["bytes_dropped"] += 1
                del self._buf[0]
                return False
        elif msg_type == MsgType.SIGNAL:
            msg = self._parse_signal(frame, crc_ok)
            if msg is not None:
                self._stats["signal_ok"] += 1
                del self._buf[:msg_len]
                return msg
            else:
                self._stats["bytes_dropped"] += 1
                del self._buf[0]
                return False
        else:
            self._stats["unknown_type"] += 1
            self._stats["bytes_dropped"] += 1
            del self._buf[0]
            return False

    def _parse_spectrum(self, frame: bytes, crc_ok: bool) -> Optional[SpectrumMessage]:
        # 布局: 同步(4) + 类型(1) + 长度(4) + 中心频率(2) + 带宽(2) + FFT点数(2) + 样点(N*2) + CRC(2)
        if len(frame) < SPECTRUM_HEADER_SIZE + CRC_SIZE:
            return None

        off = SYNC_SIZE + MSG_TYPE_SIZE + MSG_LEN_SIZE
        center_mhz, bw_mhz, fft_pts = struct.unpack(
            "<HHH", frame[off:off + 6]
        )

        sample_bytes_total = len(frame) - SPECTRUM_HEADER_SIZE - CRC_SIZE
        if sample_bytes_total != fft_pts * SAMPLE_BYTES:
            # 长度不匹配
            return None

        samples = np.frombuffer(
            frame[SPECTRUM_HEADER_SIZE:SPECTRUM_HEADER_SIZE + sample_bytes_total],
            dtype=SAMPLE_DTYPE,
        ).copy()

        return SpectrumMessage(
            center_freq_hz=center_mhz * FREQ_UNIT_HZ,
            bandwidth_hz=bw_mhz * FREQ_UNIT_HZ,
            fft_points=fft_pts,
            samples=samples,
            crc_ok=crc_ok,
        )

    def _parse_signal(self, frame: bytes, crc_ok: bool) -> Optional[SignalMessage]:
        # 布局: 同步(4) + 类型(1) + 长度(4) + 信号个数(1) + K * [信号子帧] + CRC(2)
        if len(frame) < SIGNAL_FIXED_HEADER_SIZE + CRC_SIZE:
            return None

        k = frame[SIGNAL_FIXED_HEADER_SIZE - 1]   # 信号个数
        signals = []
        cursor = SIGNAL_FIXED_HEADER_SIZE

        for _ in range(k):
            if cursor + SIGNAL_SUBHEADER_SIZE > len(frame) - CRC_SIZE:
                return None
            sig_id, center_mhz, bw_mhz, sr_mhz, fft_pts = struct.unpack(
                "<BHHHH", frame[cursor:cursor + SIGNAL_SUBHEADER_SIZE]
            )
            cursor += SIGNAL_SUBHEADER_SIZE
            sample_bytes_total = fft_pts * SAMPLE_BYTES
            if cursor + sample_bytes_total > len(frame) - CRC_SIZE:
                return None
            samples = np.frombuffer(
                frame[cursor:cursor + sample_bytes_total],
                dtype=SAMPLE_DTYPE,
            ).copy()
            cursor += sample_bytes_total

            signals.append(SignalEntry(
                signal_id=sig_id,
                center_freq_hz=center_mhz * FREQ_UNIT_HZ,
                bandwidth_hz=bw_mhz * FREQ_UNIT_HZ,
                sample_rate_hz=sr_mhz * FREQ_UNIT_HZ,
                fft_points=fft_pts,
                samples=samples,
            ))

        return SignalMessage(signals=signals, crc_ok=crc_ok)

    # ============================================================
    # 用于仿真/测试: 构造帧
    # ============================================================

    @staticmethod
    def build_spectrum_frame(center_mhz: int, bw_mhz: int,
                             samples: np.ndarray) -> bytes:
        """构造一个 0x01 频谱消息"""
        if samples.dtype != SAMPLE_DTYPE:
            samples = samples.astype(SAMPLE_DTYPE)
        samples = np.ascontiguousarray(samples)
        fft_pts = len(samples)

        total_len = (SPECTRUM_HEADER_SIZE + fft_pts * SAMPLE_BYTES + CRC_SIZE)

        body = (
            SYNC_HEADER
            + struct.pack("<B", MsgType.SPECTRUM)
            + struct.pack("<I", total_len)
            + struct.pack("<HHH", center_mhz, bw_mhz, fft_pts)
            + samples.tobytes()
        )
        crc = crc16_ccitt(body)
        return body + struct.pack("<H", crc)

    @staticmethod
    def build_signal_frame(signals: List[SignalEntry]) -> bytes:
        """构造一个 0x02 信号消息"""
        body_parts = [SYNC_HEADER, struct.pack("<B", MsgType.SIGNAL)]
        # 长度先占位
        sub_parts = [struct.pack("<B", len(signals))]
        for s in signals:
            samples = np.ascontiguousarray(s.samples.astype(SAMPLE_DTYPE))
            sub_parts.append(struct.pack(
                "<BHHHH",
                s.signal_id,
                int(s.center_freq_hz / FREQ_UNIT_HZ),
                int(s.bandwidth_hz / FREQ_UNIT_HZ),
                int(s.sample_rate_hz / FREQ_UNIT_HZ),
                s.fft_points,
            ))
            sub_parts.append(samples.tobytes())
        sub_bytes = b"".join(sub_parts)
        total_len = (SYNC_SIZE + MSG_TYPE_SIZE + MSG_LEN_SIZE
                     + len(sub_bytes) + CRC_SIZE)
        body = (SYNC_HEADER + struct.pack("<B", MsgType.SIGNAL)
                + struct.pack("<I", total_len) + sub_bytes)
        crc = crc16_ccitt(body)
        return body + struct.pack("<H", crc)


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    # 频谱帧
    fft_pts = 65535
    samples = (np.random.randn(fft_pts) * 30 - 700).astype(np.int16)
    raw = ProtocolParser.build_spectrum_frame(1500, 300, samples)
    print(f"频谱帧: {len(raw)} 字节 (FFT={fft_pts}点)")

    parser = ProtocolParser(samples_per_spectrum=fft_pts)
    # 模拟分3次到达
    parser.feed(raw[:1000])
    parser.feed(raw[1000:50000])
    parser.feed(raw[50000:])
    msgs = parser.parse_all()
    assert len(msgs) == 1
    m = msgs[0]
    assert isinstance(m, SpectrumMessage)
    print(f"  center={m.center_freq_hz/1e6:.0f}MHz, "
          f"bw={m.bandwidth_hz/1e6:.0f}MHz, "
          f"points={m.fft_points}, crc_ok={m.crc_ok}")
    assert np.array_equal(m.samples, samples)
    print("  样点一致")

    # 帧同步: 头部加垃圾
    print("\n测试帧同步 (前后加垃圾):")
    parser2 = ProtocolParser(samples_per_spectrum=fft_pts)
    parser2.feed(b"\xFF\xAA\x55" + raw + b"\x00\x11" + raw)
    msgs2 = parser2.parse_all()
    print(f"  解析出 {len(msgs2)} 帧 (应为 2)")
    print(f"  stats: {parser2.get_stats()}")

    # 信号帧
    print("\n信号帧测试:")
    sigs = [
        SignalEntry(signal_id=1, center_freq_hz=1.5e9, bandwidth_hz=10e6,
                    sample_rate_hz=20e6, fft_points=512,
                    samples=np.random.randint(-1000, 1000, 512, dtype=np.int16)),
        SignalEntry(signal_id=2, center_freq_hz=1.6e9, bandwidth_hz=5e6,
                    sample_rate_hz=10e6, fft_points=256,
                    samples=np.random.randint(-1000, 1000, 256, dtype=np.int16)),
    ]
    raw_sig = ProtocolParser.build_signal_frame(sigs)
    print(f"  信号帧: {len(raw_sig)} 字节, 2 个信号")
    parser3 = ProtocolParser()
    parser3.feed(raw_sig)
    msgs3 = parser3.parse_all()
    assert len(msgs3) == 1 and isinstance(msgs3[0], SignalMessage)
    print(f"  解析: {len(msgs3[0].signals)} 个信号")
    for s in msgs3[0].signals:
        print(f"    ID={s.signal_id}, "
              f"center={s.center_freq_hz/1e6:.0f}MHz, "
              f"bw={s.bandwidth_hz/1e6:.0f}MHz")