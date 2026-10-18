"""Bounded-memory RF64/WAVE IQ replay and software spectrum/DDC processing."""
from dataclasses import dataclass
from pathlib import Path
import re
import struct

import numpy as np

from fpga_closed_loop import EnergyDetector
from ddc_rates import output_sample_rate


@dataclass(frozen=True)
class IQFileInfo:
    path: Path
    offset: int
    samples: int
    sample_rate: int
    center_hz: float | None


def inspect_iq_file(path) -> IQFileInfo:
    """Read chunk headers only; RF64 data length comes from ds64, not uint32."""
    path = Path(path)
    length = path.stat().st_size
    with path.open('rb') as f:
        header = f.read(12)
        if len(header) != 12 or header[:4] not in (b'RF64', b'RIFF') or header[8:] != b'WAVE':
            raise ValueError('请选择 RF64/WAVE IQ 文件')
        fmt = None
        data_size64 = None
        payload = None
        while f.tell() + 8 <= length:
            tag, size = struct.unpack('<4sI', f.read(8))
            offset = f.tell()
            if tag == b'ds64':
                if size < 28 or offset + size > length:
                    raise ValueError('ds64 数据损坏')
                _, data_size64, _, _ = struct.unpack('<QQQI', f.read(28))
            elif tag == b'fmt ':
                if size < 16 or size > 65536:
                    raise ValueError('无效的 fmt 块')
                raw = f.read(size)
                if len(raw) != size:
                    raise ValueError('fmt 块被截断')
                encoding, channels, rate, _, align, bits = struct.unpack_from('<HHIIHH', raw)
                if encoding == 0xfffe:
                    pcm_guid = bytes.fromhex('0100000000001000800000aa00389b71')
                    if size < 40 or raw[24:40] != pcm_guid or struct.unpack_from('<H', raw, 18)[0] != 16:
                        raise ValueError('仅支持 PCM int16 IQ')
                    encoding = 1
                if (encoding, channels, align, bits) != (1, 2, 4, 16) or rate <= 0:
                    raise ValueError('需要双通道 PCM int16，通道 0=I，通道 1=Q')
                fmt = rate
            if size == 0xffffffff:
                if tag != b'data' or data_size64 is None:
                    raise ValueError('RF64 缺少有效的 ds64 数据长度')
                size = data_size64
            if offset + size > length:
                raise ValueError('文件被截断：数据长度超过文件大小')
            if tag == b'data':
                if size == 0 or size % 4:
                    raise ValueError('IQ 数据长度必须为非零的 4 字节整数倍')
                payload = (offset, size // 4)
                if fmt is not None:
                    break
            f.seek(offset + size + (size & 1))
    if fmt is None or payload is None:
        raise ValueError('文件缺少 fmt 或 data 块')
    match = re.search(r'_fc(\d+(?:\.\d+)?)MHz', path.name, re.I)
    return IQFileInfo(path, *payload, fmt, float(match[1]) * 1e6 if match else None)


class IQReplayReader:
    def __init__(self, info, window_seconds=.01):
        self.info = info
        self.block_samples = max(1, round(info.sample_rate * window_seconds))
        self.file = info.path.open('rb')
        self.position = 0
        self.cycle = 1

    def close(self):
        self.file.close()

    def read(self):
        if self.position >= self.info.samples:
            self.position = 0
            self.cycle += 1
        start = self.position
        count = min(self.block_samples, self.info.samples - start)
        self.file.seek(self.info.offset + start * 4)
        raw = self.file.read(count * 4)
        if len(raw) != count * 4:
            raise ValueError('回放期间 IQ 文件被截断')
        self.position += count
        return np.frombuffer(raw, dtype='<i2').reshape(-1, 2), start


def spectrum_power(raw, nfft=32768):
    """Average Hann periodograms in linear power; units are dBFS per FFT bin."""
    iq = (raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)) / 32768
    if len(iq) < nfft:
        iq = np.pad(iq, (0, nfft - len(iq)))
    # Include the last partial block without allowing a large temporary FFT array.
    window = np.hanning(nfft)
    total = np.zeros(nfft)
    blocks = 0
    for start in range(0, len(iq), nfft):
        part = iq[start:start+nfft]
        if len(part) < nfft:
            part = np.pad(part, (0, nfft-len(part)))
        total += np.abs(np.fft.fft(part * window)) ** 2
        blocks += 1
    return np.fft.fftshift(total / (blocks * window.sum() ** 2))


def extract_channel(raw, sample_rate, center_hz, bandwidth_hz, fft=None,
                    output_rate_hz=None, input_start=0):
    """Band-limit before FFT resampling; preserve mixer phase across blocks."""
    count = len(raw)
    output_rate_hz = output_rate_hz or output_sample_rate(bandwidth_hz, sample_rate)
    if not 3*bandwidth_hz <= output_rate_hz <= min(4*bandwidth_hz, sample_rate):
        raise ValueError('IQ输出采样率必须为带宽3～4倍且不超过输入采样率')
    if fft is None:
        iq = raw[:, 0].astype(np.float32) + 1j * raw[:, 1].astype(np.float32)
        fft = np.fft.fft(iq)
    spacing = sample_rate / count
    width = round(output_rate_hz / spacing)
    if width < 2 or width > count or width % 2 or abs(width*spacing-output_rate_hz) > 1e-6:
        raise ValueError('DDC块时长与输出采样率不匹配')
    center_bin = round(center_hz / spacing)
    indices = (np.arange(width) - width // 2 + center_bin) % count
    residual_hz = center_hz-center_bin*spacing
    offsets = (np.arange(width)-width//2)*spacing-residual_hz
    # Retain occupied bandwidth; taper only the 25% capture margin. Keeping
    # the entire 3-4B output passband would also keep neighbouring signals.
    distance = np.abs(offsets)
    transition = np.clip((distance-bandwidth_hz/2)/(bandwidth_hz*.125), 0, 1)
    response = .5*(1+np.cos(np.pi*transition))
    physical = center_hz+offsets
    response[(physical < -sample_rate/2) | (physical >= sample_rate/2)] = 0
    bins = fft[indices]*response
    channel = np.fft.ifft(np.fft.ifftshift(bins)) * width / count
    phase = (center_hz*input_start/sample_rate) % 1.0
    channel *= np.exp(-2j*np.pi*(phase+residual_hz*np.arange(width)/output_rate_hz))
    result = np.column_stack((channel.real, channel.imag))
    return np.clip(np.rint(result), -32768, 32767).astype(np.int16), int(output_rate_hz)


def detect_signals(power, sample_rate, usable_hz, detector):
    axis = np.fft.fftshift(np.fft.fftfreq(len(power), 1 / sample_rate))
    mask = (axis >= -usable_hz / 2) & (axis < usable_hz / 2)
    selected = power[mask]
    step = sample_rate / len(power)
    center = axis[mask][0] + len(selected) * step / 2
    # Narrow mode finds contiguous emissions including sub-MHz signals.
    return detector.detect(selected, center, len(selected) * step, 'narrow', len(selected),
                           stream_id='file')


def read_channel_packet(info, start, target, cancelled=lambda: False, progress=None):
    """Read real successive file IQ until one 4 MiB DDC payload is assembled.

    Wrap at EOF, never pad/repeat a short DDC window to fake completion.
    Only one channel payload and one source block are resident at a time.
    """
    from fpga_closed_loop import IQ_BYTES
    wanted = IQ_BYTES // 4
    output = np.empty((wanted, 2), dtype=np.int16)
    file = info.path.open('rb')
    stride = round(info.sample_rate*.01)
    guard = round(info.sample_rate*.005)
    output_rate = int(target.sample_rate_hz)
    trim = round(output_rate*.005)
    logical_start = start-guard
    filled = 0
    try:
        while filled < wanted:
            if cancelled():
                return None
            remaining = stride+2*guard
            position = logical_start % info.samples
            pieces = []
            while remaining:
                if cancelled():
                    return None
                take = min(remaining, info.samples-position)
                file.seek(info.offset+position*4)
                data = file.read(take*4)
                if len(data) != take*4:
                    raise ValueError('DDC采集期间文件被截断')
                pieces.append(data)
                remaining -= take
                position = 0
            raw = np.frombuffer(b''.join(pieces), dtype='<i2').reshape(-1, 2)
            channel, rate = extract_channel(raw, info.sample_rate, target.center_hz,
                                          target.bandwidth_hz, output_rate_hz=output_rate,
                                          input_start=logical_start)
            # Keep the middle 10 ms; overlapping real input suppresses the
            # circular-filter edge transients when joining DDC blocks.
            channel = channel[trim:len(channel)-trim]
            logical_start += stride
            take = min(len(channel), wanted-filled)
            output[filled:filled+take] = channel[:take]
            filled += take
            if progress:
                progress(filled, wanted)
        return output, output_rate
    finally:
        file.close()


def collect_channel_batch(info, start, channels, ready, cancelled=lambda: False,
                          progress=None, accepts=lambda target: True, device=None):
    """Accumulate all active FPGA lanes against one continuous source timeline.

    Read/FFT each guarded source block once. Deliver each full payload immediately,
    in acquisition-time order, while slower lanes continue collecting. ``ready``
    may apply backpressure; no source samples or full payloads are discarded.
    Resident payload memory is bounded by one batch (128 narrow + 2 wide).
    """
    from fpga_closed_loop import IQ_BYTES
    from recognition_schedule import CAPACITY
    if any(sum(t.target_type == kind for t, _ in channels) > limit
           for kind, limit in CAPACITY.items()):
        raise ValueError('采集批次超过窄带128路/宽带2路限制')
    if device is not None and str(device).startswith('cuda'):
        from gpu_file_ddc import collect_gpu_batch
        return collect_gpu_batch(info, start, channels, ready, cancelled, progress, accepts, device)
    wanted = IQ_BYTES//4
    active = [(target, identifier, np.empty((wanted, 2), dtype=np.int16), 0)
              for target, identifier in channels if accepts(target)]
    stride = round(info.sample_rate*.01)
    guard = round(info.sample_rate*.005)
    logical_start = start-guard
    with info.path.open('rb') as file:
        while active:
            if cancelled():
                return
            active = [lane for lane in active if accepts(lane[0])]
            if not active:
                return
            remaining = stride+2*guard
            position = logical_start % info.samples
            pieces = []
            while remaining:
                if cancelled():
                    return
                take = min(remaining, info.samples-position)
                file.seek(info.offset+position*4)
                data = file.read(take*4)
                if len(data) != take*4:
                    raise ValueError('DDC采集期间文件被截断')
                pieces.append(data)
                remaining -= take
                position = 0
            raw = np.frombuffer(b''.join(pieces), dtype='<i2').reshape(-1, 2)
            iq = raw[:, 0].astype(np.float32)+1j*raw[:, 1].astype(np.float32)
            fft = np.fft.fft(iq)
            pending = []
            # Several lanes can fill within this 10 ms step. Emit the one whose
            # last sample occurs earlier first, irrespective of frequency/ID.
            for target, identifier, output, filled in sorted(active,
                    key=lambda lane: (wanted-lane[3])/lane[0].sample_rate_hz):
                if cancelled():
                    return
                rate = int(target.sample_rate_hz)
                channel, _ = extract_channel(raw, info.sample_rate, target.center_hz,
                    target.bandwidth_hz, fft=fft, output_rate_hz=rate, input_start=logical_start)
                trim = round(rate*.005)
                channel = channel[trim:len(channel)-trim]
                take = min(len(channel), wanted-filled)
                output[filled:filled+take] = channel[:take]
                filled += take
                if progress:
                    progress(identifier, filled, wanted)
                if filled == wanted:
                    ready(target, identifier, output, rate)
                else:
                    pending.append((target, identifier, output, filled))
            active = pending
            logical_start += stride
