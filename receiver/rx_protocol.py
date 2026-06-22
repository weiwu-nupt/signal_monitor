"""
数据协议解析 (板卡手册第 3 章)
================================
报文结构:
    header (8B): version(uint32) + count(uint32)
    body * count:
        version 0/1: metadata(16B) = tag,ver,chan_cnt(1),chan_id(1),chan_state(2),size(4)
        version >=2: metadata(16B) = tag,ver,time_s,time_ns,size
        data (size - 16 字节)

tag 数据类别:
    1   IQ        2   Level     4   Spectrum   8 DFind   16 DFPan
    32  Audio     64  ITU       128 FastScan(ver32)/Scan-PScan(ver40)
    130 Scan-FScan  132 Scan-MScan
    256 TDOA      1024 SMS      2048 DDC       4096 GPS
    8192/8196/8200 DFC  16384 Compass  32768 MDDC

本模块把每个 body 解析成对应 dataclass. 重点实现 5 种模式要用到的:
    Spectrum / Scan / FastScan / Level / Audio / DDC 辅通道
其余给出原始 bytes, 留扩展点.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Union

import numpy as np


# ============================================================
# 常量: tag / version
# ============================================================

class Tag(IntEnum):
    IQ = 1
    LEVEL = 2
    SPECTRUM = 4
    DFIND = 8
    DFPAN = 16
    AUDIO = 32
    ITU = 64
    SCAN = 128          # FastScan(ver=32) 或 PScan(ver=40)
    FSCAN = 130         # Scan, ver=40
    MSCAN = 132         # Scan, ver=40
    TDOA = 256
    SMS = 1024
    DDC = 2048
    GPS = 4096
    DFC = 8192
    DFC_PAN = 8196
    DFC_SCAN = 8200
    COMPASS = 16384
    MDDC = 32768


# FastScan 用 tag=128 且 version=32 区分 (PScan 是 128/40)
FASTSCAN_VERSION = 32
PSCAN_VERSION = 40
SCAN_VERSION = 40

HEADER_SIZE = 8
METADATA_SIZE = 16

SPECTRUM_LEVEL_SCALE = 0.1   # 频谱/扫描数据单位 0.1 dBμV


# ============================================================
# 报头 / 元数据
# ============================================================

@dataclass
class Metadata:
    tag: int
    version: int
    size: int                 # 含本 16B metadata
    channel_count: int = 0    # version 0/1
    channel_id: int = 0
    channel_state: int = 0
    time_s: int = 0           # version >=2
    time_ns: int = 0


# ============================================================
# 各类数据体
# ============================================================

@dataclass
class SpectrumData:
    frequency_hz: int
    span_hz: int
    attenuation: int
    count: int
    levels_db: np.ndarray     # float32, 已 *0.1
    meta: Optional[Metadata] = None

    @property
    def freq_axis_hz(self) -> np.ndarray:
        start = self.frequency_hz - self.span_hz / 2
        stop = self.frequency_hz + self.span_hz / 2
        return np.linspace(start, stop, self.count, dtype=np.float64)


@dataclass
class ScanData:
    """PSCAN(NORMAL) / FSCAN / MSCAN 共用 (tag 128/130/132)"""
    start_hz: int
    stop_hz: int
    step_hz: int
    segment_index: int
    total: int
    offset: int
    count: int
    levels_db: np.ndarray
    scan_kind: str = "scan"   # "pscan"/"fscan"/"mscan"
    meta: Optional[Metadata] = None

    @property
    def freq_axis_hz(self) -> np.ndarray:
        if self.step_hz > 0:
            base = self.start_hz + self.offset * self.step_hz
            return base + np.arange(self.count) * self.step_hz
        return np.linspace(self.start_hz, self.stop_hz, self.count, dtype=np.float64)


@dataclass
class FastScanData:
    """FPSCAN (tag=128, ver=32). 800 点 信号/噪声/索引"""
    start_hz: int
    stop_hz: int
    step_hz: int
    segment_index: int
    count: int
    signal_db: np.ndarray       # float32
    noise_db: np.ndarray        # float32
    signal_index: np.ndarray    # int32, 在整段频谱内的索引
    meta: Optional[Metadata] = None

    @property
    def signal_freq_hz(self) -> np.ndarray:
        return self.start_hz + self.signal_index.astype(np.float64) * self.step_hz


@dataclass
class LevelData:
    frequency_hz: int
    bandwidth_hz: int
    attenuation: int
    level_db: float
    field_strength: float
    meta: Optional[Metadata] = None


@dataclass
class AudioData:
    frequency_hz: int
    bandwidth_hz: int
    sample_rate_hz: int
    count: int
    pcm: np.ndarray             # int16
    meta: Optional[Metadata] = None


@dataclass
class DDCSubChannel:
    """3.15 一个 data_ddc"""
    level: int
    spectrum_number: int
    occ_beta: int
    occ_db: int
    freq_offset: int
    demod_result: int
    spectrum_db: np.ndarray     # float32


@dataclass
class DDCData:
    used_chan_mask: int
    data_ddc_len: int
    channels: List[DDCSubChannel] = field(default_factory=list)
    meta: Optional[Metadata] = None


@dataclass
class RawData:
    """未专门解析的类型, 保留原始 bytes"""
    tag: int
    payload: bytes
    meta: Optional[Metadata] = None


AnyData = Union[SpectrumData, ScanData, FastScanData, LevelData,
                AudioData, DDCData, RawData]


# ============================================================
# 解析器: 一个 UDP datagram = 一个完整报文
# ============================================================

class DataProtocolParser:
    """
    UDP 是按数据报边界的, 一帧通常就是一个完整报文.
    parse_datagram() 解析 1 个 datagram -> List[AnyData].
    若上层是 TCP 流, 用 feed()/parse_all() (基于 size 切分).
    """

    def __init__(self):
        self._buf = bytearray()
        self.stats = {"frames": 0, "bodies": 0, "bad": 0, "unknown_tag": 0}

    # -------- UDP: 整报文 --------

    def parse_datagram(self, data: bytes) -> List[AnyData]:
        if len(data) < HEADER_SIZE:
            self.stats["bad"] += 1
            return []
        version, count = struct.unpack_from("<II", data, 0)
        out: List[AnyData] = []
        off = HEADER_SIZE
        for _ in range(count):
            body, off = self._parse_body(data, off, version)
            if body is None:
                break
            out.append(body)
        self.stats["frames"] += 1
        self.stats["bodies"] += len(out)
        return out

    # -------- TCP: 流式 --------

    def feed(self, data: bytes) -> None:
        self._buf.extend(data)

    def parse_all(self) -> List[AnyData]:
        out: List[AnyData] = []
        while True:
            if len(self._buf) < HEADER_SIZE:
                break
            version, count = struct.unpack_from("<II", self._buf, 0)
            # 试探整帧长度
            total = HEADER_SIZE
            ok = True
            for _ in range(count):
                if len(self._buf) < total + METADATA_SIZE:
                    ok = False
                    break
                size = struct.unpack_from("<I", self._buf, total + 12)[0]
                total += size
            if not ok or len(self._buf) < total:
                break
            frame = bytes(self._buf[:total])
            del self._buf[:total]
            out.extend(self.parse_datagram(frame))
        return out

    # -------- 内部 --------

    def _parse_meta(self, data: bytes, off: int, header_ver: int) -> Optional[Metadata]:
        if off + METADATA_SIZE > len(data):
            return None
        tag, ver = struct.unpack_from("<II", data, off)
        if header_ver >= 2:
            time_s, time_ns, size = struct.unpack_from("<III", data, off + 8)
            return Metadata(tag=tag, version=ver, size=size,
                            time_s=time_s, time_ns=time_ns)
        else:
            cc, cid, cstate, size = struct.unpack_from("<BBHI", data, off + 8)
            return Metadata(tag=tag, version=ver, size=size,
                            channel_count=cc, channel_id=cid,
                            channel_state=cstate)

    def _parse_body(self, data: bytes, off: int, header_ver: int):
        meta = self._parse_meta(data, off, header_ver)
        if meta is None or meta.size < METADATA_SIZE:
            self.stats["bad"] += 1
            return None, off
        body_start = off + METADATA_SIZE
        body_end = off + meta.size
        if body_end > len(data):
            self.stats["bad"] += 1
            return None, off
        payload = data[body_start:body_end]
        obj = self._dispatch(meta, payload)
        return obj, body_end

    def _dispatch(self, meta: Metadata, p: bytes) -> AnyData:
        tag = meta.tag
        try:
            if tag == Tag.SPECTRUM:
                return self._spectrum(meta, p)
            if tag == Tag.LEVEL:
                return self._level(meta, p)
            if tag == Tag.AUDIO:
                return self._audio(meta, p)
            if tag == Tag.SCAN:
                if meta.version == FASTSCAN_VERSION:
                    return self._fastscan(meta, p)
                return self._scan(meta, p, "pscan")
            if tag == Tag.FSCAN:
                return self._scan(meta, p, "fscan")
            if tag == Tag.MSCAN:
                return self._scan(meta, p, "mscan")
            if tag == Tag.DDC:
                return self._ddc(meta, p)
        except (struct.error, ValueError):
            self.stats["bad"] += 1
        self.stats["unknown_tag"] += 1
        return RawData(tag=tag, payload=p, meta=meta)

    # ---- 3.6 频谱 ----
    def _spectrum(self, meta, p):
        freq, span = struct.unpack_from("<qq", p, 0)
        att, count = struct.unpack_from("<iI", p, 16)
        raw = np.frombuffer(p, dtype="<i2", count=count, offset=24).copy()
        return SpectrumData(freq, span, att, count,
                            raw.astype(np.float32) * SPECTRUM_LEVEL_SCALE, meta)

    # ---- 3.9 扫描 ----
    def _scan(self, meta, p, kind):
        start, stop, step = struct.unpack_from("<qqq", p, 0)
        seg, total, offset, count = struct.unpack_from("<IIII", p, 24)
        raw = np.frombuffer(p, dtype="<i2", count=count, offset=40).copy()
        return ScanData(start, stop, step, seg, total, offset, count,
                        raw.astype(np.float32) * SPECTRUM_LEVEL_SCALE, kind, meta)

    # ---- 3.10 快速扫描 ----
    def _fastscan(self, meta, p):
        start, stop, step = struct.unpack_from("<qqq", p, 0)
        seg, count = struct.unpack_from("<II", p, 24)
        o = 32
        sig = np.frombuffer(p, dtype="<i2", count=count, offset=o).copy()
        o += count * 2
        noise = np.frombuffer(p, dtype="<i2", count=count, offset=o).copy()
        o += count * 2
        idx = np.frombuffer(p, dtype="<i4", count=count, offset=o).copy()
        return FastScanData(
            start, stop, step, seg, count,
            sig.astype(np.float32) * SPECTRUM_LEVEL_SCALE,
            noise.astype(np.float32) * SPECTRUM_LEVEL_SCALE,
            idx, meta)

    # ---- 3.5 电平 ----
    def _level(self, meta, p):
        freq, bw = struct.unpack_from("<qq", p, 0)
        att = struct.unpack_from("<i", p, 16)[0]
        level, fs = struct.unpack_from("<ff", p, 20)
        return LevelData(freq, bw, att, level, fs, meta)

    # ---- 3.7 音频 ----
    def _audio(self, meta, p):
        freq, bw, sr = struct.unpack_from("<qqq", p, 0)
        _, count = struct.unpack_from("<II", p, 24)
        pcm = np.frombuffer(p, dtype="<i2", count=count, offset=32).copy()
        return AudioData(freq, bw, sr, count, pcm, meta)

    # ---- 3.15 DDC 辅通道 ----
    def _ddc(self, meta, p):
        used, dlen = struct.unpack_from("<II", p, 0)
        n = bin(used).count("1")
        chans = []
        o = 8
        for _ in range(n):
            if o + dlen > len(p):
                break
            level, spec_num = struct.unpack_from("<hh", p, o)
            occ_b, occ_d = struct.unpack_from("<ii", p, o + 4)
            foff, demod = struct.unpack_from("<hh", p, o + 12)
            spec = np.frombuffer(p, dtype="<i2", count=spec_num, offset=o + 16).copy()
            chans.append(DDCSubChannel(
                level, spec_num, occ_b, occ_d, foff, demod,
                spec.astype(np.float32) * SPECTRUM_LEVEL_SCALE))
            o += dlen
        return DDCData(used, dlen, chans, meta)


# ============================================================
# 自检: 构造各类帧并回环解析
# ============================================================

def _build_frame(bodies: list, header_version: int = 0) -> bytes:
    """bodies: list of (tag, version, payload_bytes)"""
    out = struct.pack("<II", header_version, len(bodies))
    for tag, ver, payload in bodies:
        size = METADATA_SIZE + len(payload)
        if header_version >= 2:
            meta = struct.pack("<IIIII", tag, ver, 0, 0, size)
        else:
            meta = struct.pack("<IIBBHI", tag, ver, 0, 0, 0, size)
        out += meta + payload
    return out


if __name__ == "__main__":
    par = DataProtocolParser()

    # Spectrum
    count = 1024
    levels = (np.random.randn(count) * 50 - 600).astype("<i2")
    spec_payload = struct.pack("<qqiI", int(100e6), int(1e6), 0x0A05, count) + levels.tobytes()
    f = _build_frame([(Tag.SPECTRUM, 0, spec_payload)])
    objs = par.parse_datagram(f)
    assert isinstance(objs[0], SpectrumData) and objs[0].count == count
    print(f"[spectrum] count={objs[0].count} peak={objs[0].levels_db.max():.1f} "
          f"f0={objs[0].freq_axis_hz[0]/1e6:.3f}MHz")

    # FastScan (FPSCAN)
    c = 800
    sig = (np.random.randn(c) * 30 - 400).astype("<i2")
    noise = (np.random.randn(c) * 5 - 900).astype("<i2")
    idx = np.sort(np.random.randint(0, 320000, c)).astype("<i4")
    fp_payload = struct.pack("<qqqII", int(20e6), int(8000e6), 25000, 0, c)
    fp_payload += sig.tobytes() + noise.tobytes() + idx.tobytes()
    f = _build_frame([(Tag.SCAN, FASTSCAN_VERSION, fp_payload)])
    objs = par.parse_datagram(f)
    assert isinstance(objs[0], FastScanData) and objs[0].count == c
    print(f"[fastscan] count={objs[0].count} sigpeak={objs[0].signal_db.max():.1f} "
          f"f0={objs[0].signal_freq_hz[0]/1e6:.1f}MHz")

    # Scan (FSCAN)
    c = 500
    lv = (np.random.randn(c) * 40 - 500).astype("<i2")
    sc_payload = struct.pack("<qqqIIII", int(88e6), int(108e6), 25000, 0, c, 0, c) + lv.tobytes()
    f = _build_frame([(Tag.FSCAN, SCAN_VERSION, sc_payload)])
    objs = par.parse_datagram(f)
    assert isinstance(objs[0], ScanData) and objs[0].scan_kind == "fscan"
    print(f"[fscan] count={objs[0].count} step={objs[0].step_hz}Hz "
          f"f0={objs[0].freq_axis_hz[0]/1e6:.1f}MHz")

    # Level + version>=2 timestamp
    lvl_payload = struct.pack("<qqiff", int(101.7e6), int(120e3), 0x0A05, 55.3, 12.1)
    f = _build_frame([(Tag.LEVEL, 0, lvl_payload)], header_version=2)
    objs = par.parse_datagram(f)
    assert isinstance(objs[0], LevelData)
    print(f"[level] {objs[0].level_db:.1f}dBuV ts={objs[0].meta.time_s}")

    # multi-body frame
    f = _build_frame([(Tag.SPECTRUM, 0, spec_payload),
                      (Tag.LEVEL, 0, lvl_payload)])
    objs = par.parse_datagram(f)
    assert len(objs) == 2
    print(f"[multibody] {len(objs)} bodies")

    print("stats:", par.stats)
    print("ALL PROTOCOL TESTS PASSED")