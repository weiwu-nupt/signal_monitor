"""FPGA closed-loop spectrum detection and DDC IQ acquisition.

Protocol implementation based on the supplied FPGA message/register documents:

* C2H0: 32-byte header + 8192 uint64 wideband power values;
* C2H1: 32-byte header + 16384 uint64 narrowband power values (64 channels);
* H2C1: one atomic batch of 32-byte Type 0x01/0x02 target records;
* C2H2/C2H3: 32-byte header + 4 MiB {int16 I, int16 Q} DDC IQ packet.

No C2H channel is shared between threads or processes.  Start this module as
the application's single owner of C2H0..C2H3.
"""

from __future__ import annotations

import ctypes
import errno
import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Callable, Optional

import numpy as np

from protocol import SpectrumMessage


SYNC_WORD = 0x1ACF_FC1D
HEADER_BYTES = 32
WIDE_NFFT, NARROW_NFFT = 8192, 16384
WIDE_PACKET_BYTES = HEADER_BYTES + WIDE_NFFT * 8
NARROW_PACKET_BYTES = HEADER_BYTES + NARROW_NFFT * 8
IQ_BYTES = 4 * 1024 * 1024
IQ_PACKET_BYTES = HEADER_BYTES + IQ_BYTES
IQ_MESSAGE_LENGTH = 0x0040000D

WIDE_DEVICE = "/dev/xdma0_c2h_0"
NARROW_DEVICE = "/dev/xdma0_c2h_1"
WIDE_IQ_DEVICE = "/dev/xdma0_c2h_2"
NARROW_IQ_DEVICE = "/dev/xdma0_c2h_3"
H2C_TARGET_DEVICE = "/dev/xdma0_h2c_1"
USER_DEVICE = "/dev/xdma0_user"

TARGET_RECORD_BYTES = 32
TARGET_MESSAGE_LENGTH = 13
MAX_WIDE_TARGETS, MAX_NARROW_TARGETS = 2, 128
BASEBAND_LIMIT_HZ = 204_800_000
O_SYNC = getattr(os, "O_SYNC", 0)


@dataclass(frozen=True)
class Target:
    """A requested DDC target; centre frequency is signed baseband Hz."""

    target_type: str  # ``wide`` or ``narrow``
    sequence: int
    center_hz: int
    bandwidth_hz: int
    sample_rate_hz: int


@dataclass(frozen=True)
class TargetFilter:
    """Host-side bandwidth range for deciding which detections request IQ."""

    bandwidth_min_hz: Optional[float] = None
    bandwidth_max_hz: Optional[float] = None

    def allows(self, target: Target) -> bool:
        return (
            (self.bandwidth_min_hz is None or target.bandwidth_hz >= self.bandwidth_min_hz)
            and (self.bandwidth_max_hz is None or target.bandwidth_hz <= self.bandwidth_max_hz)
        )


@dataclass
class IQPacket:
    target_type: str
    sequence: int
    center_hz: int
    bandwidth_hz: int
    sample_rate_hz: int
    i_data: np.ndarray
    q_data: np.ndarray


@dataclass
class FPGASpectrumMessage(SpectrumMessage):
    """Spectrum message with source metadata needed by the target detector."""

    target_type: str = "wide"
    sequence: int = 0
    detected_targets: tuple[Target, ...] = ()
    selected_targets: tuple[Target, ...] = ()
    raw_power: Optional[np.ndarray] = None
    _power_db: Optional[np.ndarray] = None

    @property
    def spectrum_db(self) -> np.ndarray:  # type: ignore[override]
        return self._power_db if self._power_db is not None else self.samples.astype(np.float32)


def _parse_header(packet: bytes, expected_type: int, expected_length: int, expected_nfft: Optional[int] = None) -> dict[str, int]:
    if len(packet) != expected_length:
        raise RuntimeError(f"packet length={len(packet)}, expected={expected_length}")
    header = {
        "sync": struct.unpack_from("<I", packet, 0)[0],
        "type": packet[4],
        "msg_len": struct.unpack_from("<I", packet, 5)[0],
        "sequence": packet[9],
        "center_hz": struct.unpack_from("<i", packet, 10)[0],
        "bandwidth_hz": struct.unpack_from("<I", packet, 14)[0],
        "sample_rate_hz": struct.unpack_from("<I", packet, 18)[0],
        "fft_points": struct.unpack_from("<H", packet, 22)[0],
    }
    if header["sync"] != SYNC_WORD or header["type"] != expected_type:
        raise RuntimeError(f"invalid header sync/type: 0x{header['sync']:08X}/0x{header['type']:02X}")
    if header["msg_len"] != expected_length or any(packet[24:HEADER_BYTES]):
        raise RuntimeError("invalid message length or non-zero reserved header bytes")
    if expected_nfft is not None and header["fft_points"] != expected_nfft:
        raise RuntimeError(f"FFT points={header['fft_points']}, expected={expected_nfft}")
    return header


def parse_wide_packet(packet: bytes) -> tuple[dict[str, int], np.ndarray]:
    header = _parse_header(packet, 0x01, WIDE_PACKET_BYTES, WIDE_NFFT)
    return header, np.frombuffer(packet, dtype="<u8", count=WIDE_NFFT, offset=HEADER_BYTES).copy()


def parse_narrow_packet(packet: bytes) -> tuple[dict[str, int], np.ndarray]:
    header = _parse_header(packet, 0x02, NARROW_PACKET_BYTES, NARROW_NFFT)
    if not 0 <= header["sequence"] < 64:
        raise RuntimeError(f"narrow sequence={header['sequence']}, expected 0..63")
    return header, np.frombuffer(packet, dtype="<u8", count=NARROW_NFFT, offset=HEADER_BYTES).copy()


def parse_iq_packet(packet: bytes, target_type: str) -> IQPacket:
    if len(packet) != IQ_PACKET_BYTES:
        raise RuntimeError(f"IQ packet length={len(packet)}, expected={IQ_PACKET_BYTES}")
    # IQ header's message-length field is a protocol constant, not byte count.
    if struct.unpack_from("<I", packet, 0)[0] != SYNC_WORD or packet[4] != 0x04:
        raise RuntimeError("invalid IQ packet sync/type")
    if struct.unpack_from("<I", packet, 5)[0] != IQ_MESSAGE_LENGTH or any(packet[22:HEADER_BYTES]):
        raise RuntimeError("invalid IQ packet header")
    samples = np.frombuffer(packet, dtype="<i2", offset=HEADER_BYTES).reshape(-1, 2).copy()
    return IQPacket(
        target_type=target_type,
        sequence=packet[9],
        center_hz=struct.unpack_from("<i", packet, 10)[0],
        bandwidth_hz=struct.unpack_from("<I", packet, 14)[0],
        sample_rate_hz=struct.unpack_from("<I", packet, 18)[0],
        i_data=samples[:, 0], q_data=samples[:, 1],
    )


class StreamPacketReader:
    """Packet-size reads with sync recovery; one instance owns one C2H fd."""

    def __init__(self, device: str, packet_bytes: int, parser: Callable[[bytes], object], *, packet_mode: bool = False):
        self.device, self.packet_bytes, self.parser = device, packet_bytes, parser
        flags = (os.O_RDWR | os.O_TRUNC | O_SYNC) if packet_mode else (os.O_RDONLY | O_SYNC)
        self.fd = os.open(device, flags)
        self.buffer = bytearray()
        self._closed = False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            os.close(self.fd)

    def next(self):
        sync = struct.pack("<I", SYNC_WORD)
        while True:
            while len(self.buffer) < self.packet_bytes:
                try:
                    data = os.read(self.fd, max(self.packet_bytes, 32))
                except OSError as exc:
                    if exc.errno in (errno.EINTR, 512):
                        continue
                    raise
                if not data:
                    raise RuntimeError(f"{self.device} returned empty data")
                self.buffer.extend(data)
            offset = self.buffer.find(sync)
            if offset < 0:
                del self.buffer[:-3]
                continue
            if offset:
                del self.buffer[:offset]
                continue
            packet = bytes(self.buffer[:self.packet_bytes])
            try:
                result = self.parser(packet)
            except RuntimeError:
                del self.buffer[:1]
                continue
            del self.buffer[:self.packet_bytes]
            return result


class EnergyDetector:
    """Low-SNR detector with bandwidth-adaptive protection and smoothing.

    The running power average avoids dropping weak edge bins on one FFT frame;
    the protection band is calculated in Hz from the measured occupied
    bandwidth, then converted to the current spectrum's bin spacing.  This
    keeps the physical margin consistent between C2H0 and C2H1.
    """

    def __init__(
        self,
        snr_db: float = 4.0,
        min_guard_hz: float = 25_000.0,
        relative_guard: float = 0.10,
        max_guard_hz: float = 2_000_000.0,
        merge_gap_hz: float = 12_500.0,
        min_bins: int = 2,
        smoothing: float = 0.65,
    ):
        if (
            snr_db <= 0
            or min_guard_hz < 0
            or not 0 <= relative_guard <= 1
            or max_guard_hz < min_guard_hz
            or merge_gap_hz < 0
            or min_bins < 1
        ):
            raise ValueError("invalid energy-detector configuration")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.snr_db = snr_db
        self.min_guard_hz = min_guard_hz
        self.relative_guard = relative_guard
        self.max_guard_hz = max_guard_hz
        self.merge_gap_hz = merge_gap_hz
        self.min_bins = min_bins
        self.smoothing = smoothing
        self._smoothed: dict[object, np.ndarray] = {}
        self._settings_lock = threading.Lock()

    def set_snr_db(self, snr_db: float) -> None:
        """Update the relative energy threshold used by subsequent FFT frames."""
        if snr_db <= 0:
            raise ValueError("snr_db must be greater than zero")
        with self._settings_lock:
            self.snr_db = float(snr_db)

    def detect(self, power: np.ndarray, center_hz: float, bandwidth_hz: float, target_type: str, max_targets: int, stream_id: object = None) -> list[Target]:
        values = power.astype(np.float64)
        if stream_id is not None:
            previous = self._smoothed.get(stream_id)
            if previous is not None and previous.shape == values.shape:
                values = self.smoothing * previous + (1.0 - self.smoothing) * values
            self._smoothed[stream_id] = values.copy()
        with self._settings_lock:
            snr_db = self.snr_db
        floor = float(np.percentile(values, 35.0))
        threshold = floor * 10.0 ** (snr_db / 10.0)
        active = values >= threshold
        if not np.any(active):
            return []
        bin_hz = bandwidth_hz / len(values)
        indices = np.flatnonzero(active)
        groups: list[list[int]] = [[int(indices[0])]]
        for index in indices[1:]:
            if index - groups[-1][-1] <= int(np.ceil(self.merge_gap_hz / bin_hz)) + 1:
                groups[-1].append(int(index))
            else:
                groups.append([int(index)])
        candidates = []
        for group in groups:
            left, right = group[0], group[-1]
            if right - left + 1 < self.min_bins:
                continue
            occupied_hz = (right - left + 1) * bin_hz
            guard_hz = min(
                self.max_guard_hz,
                max(self.min_guard_hz, occupied_hz * self.relative_guard),
            )
            guard_bins = int(np.ceil(guard_hz / bin_hz))
            left, right = max(0, left - guard_bins), min(len(values) - 1, right + guard_bins)
            segment = values[left:right + 1]
            weighted_bin = (np.arange(left, right + 1) * segment).sum() / max(segment.sum(), 1.0)
            frequency = center_hz - bandwidth_hz / 2.0 + weighted_bin * bin_hz
            width = max(bin_hz, (right - left + 1) * bin_hz)
            candidates.append((float(segment.max()), frequency, width))
        candidates.sort(reverse=True, key=lambda item: item[0])
        targets = []
        for sequence, (_, frequency, width) in enumerate(candidates[:max_targets]):
            freq = int(round(frequency))
            if -BASEBAND_LIMIT_HZ <= freq < BASEBAND_LIMIT_HZ:
                targets.append(Target(target_type, sequence, freq, int(round(width)), int(round(bandwidth_hz))))
        return targets


def build_target_batch(targets: list[Target]) -> bytes:
    """Build the exact PCR1 H2C1 batch. Empty means one canonical Type-02 record."""
    if not targets:
        targets = [Target("narrow", 0, 0, 0, 0)]
    if len(targets) > MAX_WIDE_TARGETS + MAX_NARROW_TARGETS:
        raise ValueError("too many targets")
    records = []
    for target in targets:
        if target.target_type not in ("wide", "narrow"):
            raise ValueError("target_type must be wide or narrow")
        if not -BASEBAND_LIMIT_HZ <= target.center_hz < BASEBAND_LIMIT_HZ:
            raise ValueError("target centre outside baseband")
        if target.bandwidth_hz == 0:
            if target != Target("narrow", 0, 0, 0, 0):
                raise ValueError("only canonical empty target may have zero bandwidth")
        elif target.bandwidth_hz < 1 or (target.target_type == "narrow" and target.bandwidth_hz > 12_800_000):
            raise ValueError("invalid target bandwidth")
        record = bytearray(TARGET_RECORD_BYTES)
        struct.pack_into("<I", record, 0, SYNC_WORD)
        record[4] = 0x01 if target.target_type == "wide" else 0x02
        struct.pack_into("<I", record, 5, TARGET_MESSAGE_LENGTH)
        record[9] = target.sequence
        struct.pack_into("<i", record, 10, target.center_hz)
        struct.pack_into("<I", record, 14, target.bandwidth_hz)
        struct.pack_into("<I", record, 18, target.sample_rate_hz)
        records.append(bytes(record))
    return b"".join(records)


def send_target_batch(targets: list[Target], device: str = H2C_TARGET_DEVICE) -> int:
    """Write all targets exactly once; FPGA treats one write as one target set."""
    packet = build_target_batch(targets)
    allocation = ((len(packet) + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
    mapping = mmap.mmap(-1, allocation, access=mmap.ACCESS_WRITE)
    view = memoryview(mapping)[:len(packet)]
    try:
        mapping[:len(packet)] = packet
        fd = os.open(device, os.O_WRONLY | O_SYNC)
        try:
            written = os.write(fd, view)
        finally:
            os.close(fd)
        if written != len(packet):
            raise RuntimeError(f"H2C1 short write {written}/{len(packet)} bytes")
        return written
    finally:
        view.release()
        mapping.close()


class SpectrumDDCClosedLoop:
    """Single-process owner of spectrum input, target H2C1 and DDC IQ output."""

    def __init__(
        self,
        detector: Optional[EnergyDetector] = None,
        target_filter: Optional[TargetFilter] = None,
        target_interval_s: float = 0.25,
        queue_size: int = 64,
    ):
        self.detector = detector or EnergyDetector()
        self.target_filter = target_filter or TargetFilter()
        self.target_interval_s = target_interval_s
        self._queue: Queue[FPGASpectrumMessage] = Queue(maxsize=queue_size)
        self._iq_queue: Queue[IQPacket] = Queue(maxsize=16)
        self._running = False
        self._threads: list[threading.Thread] = []
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._iq_callback: Optional[Callable[[IQPacket], None]] = None
        self._latest_targets: dict[tuple[str, int], list[Target]] = {}
        self._latest_detected: dict[tuple[str, int], list[Target]] = {}
        self._last_sent: tuple[Target, ...] = ()
        self._last_send_s = 0.0
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._stats = {"spectrum_ok": 0, "signal_ok": 0, "bad_crc": 0, "drops": 0, "bytes_dropped": 0, "target_batches": 0, "iq_packets": 0}

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        self._error_callback = callback

    def set_iq_callback(self, callback: Callable[[IQPacket], None]) -> None:
        self._iq_callback = callback

    def set_detector_snr_db(self, snr_db: float) -> None:
        """Set the live energy-detection threshold; takes effect next frame."""
        self.detector.set_snr_db(snr_db)

    def set_bandwidth_filter(
        self,
        minimum_hz: Optional[float],
        maximum_hz: Optional[float],
    ) -> None:
        """Apply an IQ-request bandwidth range immediately to all detections."""
        if minimum_hz is not None and minimum_hz < 0:
            raise ValueError("minimum bandwidth must not be negative")
        if maximum_hz is not None and maximum_hz < 0:
            raise ValueError("maximum bandwidth must not be negative")
        if minimum_hz is not None and maximum_hz is not None and minimum_hz > maximum_hz:
            raise ValueError("minimum bandwidth must not exceed maximum bandwidth")
        with self._lock:
            self.target_filter = TargetFilter(minimum_hz, maximum_hz)
            self._latest_targets = {
                key: [target for target in detected if self.target_filter.allows(target)]
                for key, detected in self._latest_detected.items()
            }
        if self._running:
            self._send_current_targets()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        for name, fn in (("WideSpectrum", self._wide_loop), ("NarrowSpectrum", self._narrow_loop), ("WideIQ", lambda: self._iq_loop("wide")), ("NarrowIQ", lambda: self._iq_loop("narrow"))):
            thread = threading.Thread(target=self._guarded, args=(fn,), name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._running = False
        self._send_targets([])
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

    def get_message(self, timeout: float = 0.0) -> Optional[FPGASpectrumMessage]:
        try:
            return self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()
        except Empty:
            return None

    def get_iq_packet(self, timeout: float = 0.0) -> Optional[IQPacket]:
        try:
            return self._iq_queue.get(timeout=timeout) if timeout > 0 else self._iq_queue.get_nowait()
        except Empty:
            return None

    def get_stats(self) -> dict:
        return dict(self._stats)

    def _guarded(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            if self._running and self._error_callback:
                self._error_callback(exc)

    def _push(self, queue: Queue, item: object) -> None:
        try:
            queue.put_nowait(item)
        except Full:
            try:
                queue.get_nowait()
            except Empty:
                pass
            self._stats["drops"] += 1
            queue.put_nowait(item)

    def _wide_loop(self) -> None:
        reader = StreamPacketReader(WIDE_DEVICE, WIDE_PACKET_BYTES, parse_wide_packet)
        try:
            while self._running:
                header, power = reader.next()
                self._handle_spectrum(header, power, "wide")
        finally:
            reader.close()

    def _narrow_loop(self) -> None:
        reader = StreamPacketReader(NARROW_DEVICE, NARROW_PACKET_BYTES, parse_narrow_packet)
        try:
            while self._running:
                header, power = reader.next()
                self._handle_spectrum(header, power, "narrow")
        finally:
            reader.close()

    def _handle_spectrum(self, header: dict[str, int], power: np.ndarray, target_type: str) -> None:
        peak = max(float(power.max()), np.finfo(np.float64).tiny)
        db = np.maximum(10.0 * np.log10(np.maximum(power.astype(np.float64), np.finfo(np.float64).tiny) / peak), -120.0).astype(np.float32)
        maximum = MAX_WIDE_TARGETS if target_type == "wide" else MAX_NARROW_TARGETS
        detected = self.detector.detect(
            power, header["center_hz"], header["bandwidth_hz"], target_type,
            maximum, stream_id=(target_type, header["sequence"]),
        )
        selected = [target for target in detected if self.target_filter.allows(target)]
        with self._lock:
            self._latest_detected[(target_type, header["sequence"])] = detected
            self._latest_targets[(target_type, header["sequence"])] = selected
        # C2H0 is a coarse analysis path only.  GUI rendering receives C2H1.
        if target_type == "narrow":
            message = FPGASpectrumMessage(
                header["center_hz"], header["bandwidth_hz"], header["fft_points"],
                db, True, target_type, header["sequence"], tuple(detected),
                tuple(selected), power, db,
            )
            self._push(self._queue, message)
        self._stats["spectrum_ok"] += 1
        self._maybe_send_targets()

    def _maybe_send_targets(self) -> None:
        now = time.monotonic()
        if now - self._last_send_s >= self.target_interval_s:
            self._send_current_targets()

    def _send_current_targets(self) -> None:
        with self._lock:
            all_targets = [target for group in self._latest_targets.values() for target in group]
        # Global batch limits; score ordering was already done per spectrum.
        wide = [x for x in all_targets if x.target_type == "wide"][:MAX_WIDE_TARGETS]
        narrow = [
            Target("narrow", index, x.center_hz, x.bandwidth_hz, x.sample_rate_hz)
            for index, x in enumerate(x for x in all_targets if x.target_type == "narrow")
        ][:MAX_NARROW_TARGETS]
        self._send_targets(wide + narrow)

    def _send_targets(self, targets: list[Target]) -> None:
        with self._send_lock:
            canonical = tuple(targets)
            if canonical == self._last_sent:
                return
            send_target_batch(targets)
            self._last_sent, self._last_send_s = canonical, time.monotonic()
            self._stats["target_batches"] += 1
            self._stats["signal_ok"] = len(targets)

    def _iq_loop(self, target_type: str) -> None:
        device = WIDE_IQ_DEVICE if target_type == "wide" else NARROW_IQ_DEVICE
        reader = StreamPacketReader(device, IQ_PACKET_BYTES, lambda packet: parse_iq_packet(packet, target_type), packet_mode=True)
        try:
            while self._running:
                iq = reader.next()
                self._push(self._iq_queue, iq)
                self._stats["iq_packets"] += 1
                if self._iq_callback:
                    self._iq_callback(iq)
        finally:
            reader.close()
