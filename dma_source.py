"""XDMA C2H spectrum source for the replay-spectrum FPGA design.

The FPGA sends one contiguous packet on ``/dev/xdma0_c2h_0``:

* 64-byte header;
* 8192 little-endian ``uint64`` FFT-power values;
* 64-byte tail.

This is the protocol used by ``xdma_replay_spectrum(1).py``.  Unlike the
older ``spectrum_reader.py`` design, it does not use status-slot polling or
an H2C ACK packet.
"""

from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Callable, Optional

import numpy as np
from protocol import SpectrumMessage


NFFT = 8192
C2H_HEADER_BYTES = 64
C2H_PAYLOAD_BYTES = NFFT * 8
C2H_TAIL_BYTES = 64
C2H_PACKET_BYTES = C2H_HEADER_BYTES + C2H_PAYLOAD_BYTES + C2H_TAIL_BYTES

DEFAULT_C2H_DEV = "/dev/xdma0_c2h_0"
DEFAULT_SYNC_WORD = 0x1ACFFC1D


@dataclass
class DMASpectrumMessage(SpectrumMessage):
    """A GUI message whose samples are already represented in dB."""

    _power_db: Optional[np.ndarray] = None

    @property
    def spectrum_db(self) -> np.ndarray:  # type: ignore[override]
        if self._power_db is not None:
            return self._power_db
        return self.samples.astype(np.float32) * 0.1


def parse_c2h_packet(packet: bytes, *, sync_word: int = DEFAULT_SYNC_WORD) -> tuple[dict[str, int], np.ndarray]:
    """Validate and unpack a replay-spectrum C2H packet."""
    if len(packet) != C2H_PACKET_BYTES:
        raise RuntimeError(
            f"C2H packet length is {len(packet)}, expected {C2H_PACKET_BYTES}"
        )

    header = {
        "sync": struct.unpack_from("<I", packet, 0)[0],
        "msg_type": packet[4],
        "msg_len": struct.unpack_from("<I", packet, 5)[0],
        "center_mhz": struct.unpack_from("<H", packet, 9)[0],
        "bandwidth_mhz": struct.unpack_from("<H", packet, 11)[0],
        "fft_points": struct.unpack_from("<H", packet, 13)[0],
        "source": packet[15],
        "sequence": struct.unpack_from("<I", packet, 16)[0],
        "frame_id": struct.unpack_from("<I", packet, 20)[0],
    }
    if header["sync"] != sync_word:
        raise RuntimeError(
            f"unexpected C2H sync word 0x{header['sync']:08X}, "
            f"expected 0x{sync_word:08X}"
        )
    if header["fft_points"] != NFFT:
        raise RuntimeError(
            f"unexpected FFT size {header['fft_points']}, expected {NFFT}"
        )

    power = np.frombuffer(
        packet, dtype="<u8", count=NFFT, offset=C2H_HEADER_BYTES
    ).copy()
    return header, power


class DMASpectrumSource:
    """Read FPGA replay-spectrum packets in a background thread for the GUI.

    ``h2c_dev`` and ``poll_interval`` remain accepted for compatibility with
    the previous source API.  The current FPGA protocol neither opens H2C nor
    polls a status register; C2H reads block until a complete packet arrives.
    """

    def __init__(
        self,
        c2h_dev: str = DEFAULT_C2H_DEV,
        h2c_dev: Optional[str] = None,
        sync_words: object = None,
        poll_interval: float = 0.02,
        raise_on_bad_packet: bool = False,
        queue_size: int = 16,
        floor_db: float = -120.0,
    ) -> None:
        self.c2h_dev = c2h_dev
        self.h2c_dev = h2c_dev  # Retained for callers of the former API.
        self.poll_interval = poll_interval
        self.raise_on_bad_packet = raise_on_bad_packet
        self.floor_db = floor_db

        # The new protocol has one fixed sync word.  Accept a legacy iterable
        # only when it contains that word, so old call sites keep working.
        if sync_words is not None and DEFAULT_SYNC_WORD not in tuple(sync_words):
            raise ValueError(
                f"sync_words must include 0x{DEFAULT_SYNC_WORD:08X} for this FPGA design"
            )

        self._queue: Queue[DMASpectrumMessage] = Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._error_callback: Optional[Callable[[Exception], None]] = None
        self._c2h_fd = -1
        self._fd_lock = threading.Lock()
        self._stats = {
            "spectrum_ok": 0,
            "signal_ok": 0,
            "bad_crc": 0,       # No CRC is defined by the new C2H reader.
            "bad_packet": 0,
            "empty_poll": 0,
            "drops": 0,
            "bytes_dropped": 0,
            "last_seq": -1,
            "last_frame_id": -1,
        }

    def set_error_callback(self, cb: Callable[[Exception], None]) -> None:
        self._error_callback = cb

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="DMASpectrumRx", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._close_device()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_message(self, timeout: float = 0.0) -> Optional[DMASpectrumMessage]:
        try:
            return self._queue.get(timeout=timeout) if timeout > 0 else self._queue.get_nowait()
        except Empty:
            return None

    def get_drop_count(self) -> int:
        return self._stats["drops"]

    def get_stats(self) -> dict:
        return dict(self._stats)

    def _open_device(self) -> None:
        if not os.path.exists(self.c2h_dev):
            raise FileNotFoundError(
                f"XDMA C2H device {self.c2h_dev} does not exist; "
                "check the XDMA driver and /dev/xdma* nodes"
            )
        try:
            fd = os.open(self.c2h_dev, os.O_RDONLY | os.O_SYNC)
        except PermissionError as exc:
            raise PermissionError(
                f"permission denied opening {self.c2h_dev}; run with sufficient XDMA device permissions"
            ) from exc
        with self._fd_lock:
            self._c2h_fd = fd

    def _close_device(self) -> None:
        with self._fd_lock:
            fd, self._c2h_fd = self._c2h_fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    def _read_packet(self) -> bytes:
        """Accumulate exactly one packet; XDMA reads may be shorter than requested."""
        chunks: list[bytes] = []
        remaining = C2H_PACKET_BYTES
        while remaining:
            with self._fd_lock:
                fd = self._c2h_fd
            if fd < 0:
                raise RuntimeError("C2H device was closed")
            data = os.read(fd, remaining)
            if not data:
                raise RuntimeError("C2H returned EOF while reading a spectrum packet")
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _push_message(self, message: DMASpectrumMessage) -> None:
        try:
            self._queue.put_nowait(message)
        except Full:
            try:
                self._queue.get_nowait()
                self._stats["drops"] += 1
            except Empty:
                pass
            try:
                self._queue.put_nowait(message)
            except Full:
                pass

    def _run_loop(self) -> None:
        try:
            self._open_device()
            while self._running:
                try:
                    header, power = parse_c2h_packet(self._read_packet())
                except RuntimeError:
                    if not self._running:
                        break
                    self._stats["bad_packet"] += 1
                    if self.raise_on_bad_packet:
                        raise
                    continue
                self._push_message(self._build_message(header, power))
                self._stats["spectrum_ok"] += 1
                self._stats["last_seq"] = header["sequence"]
                self._stats["last_frame_id"] = header["frame_id"]
        except Exception as exc:
            if self._running and self._error_callback is not None:
                self._error_callback(exc)
        finally:
            self._close_device()

    def _build_message(self, header: dict[str, int], power: np.ndarray) -> DMASpectrumMessage:
        # Match xdma_replay_spectrum: display power normalized to the packet peak.
        power_float = power.astype(np.float64)
        peak = max(float(np.max(power_float)), np.finfo(np.float64).tiny)
        power_db = 10.0 * np.log10(np.maximum(power_float, np.finfo(np.float64).tiny) / peak)
        power_db = np.maximum(power_db, self.floor_db).astype(np.float32)
        return DMASpectrumMessage(
            center_freq_hz=header["center_mhz"] * 1e6,
            bandwidth_hz=header["bandwidth_mhz"] * 1e6,
            fft_points=header["fft_points"],
            samples=power_db,
            crc_ok=True,
            _power_db=power_db,
        )


def create_dma_source(**kwargs: object) -> DMASpectrumSource:
    return DMASpectrumSource(**kwargs)
