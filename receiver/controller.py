"""
模式控制器
===========
把 TCP(SCPI) + UDP(数据) 串起来, 对外提供:
    - connect_receiver(ip, local_ip, port)
    - apply_mode(mode, config)   下发 SCPI 并打开对应 UDP 端口
    - start() / stop()           TRAC ON/OFF
    - 信号: spectrumReady / scanReady / fastScanReady / levelReady /
            audioReady / ddcReady / statusMessage / errorOccurred

5 种模式与端口/命令的对应 (依据配置文档 + SCPI 手册):
    FFM   : MODE FIXed ; UDP 8000(MEAS) 8001(AUD) 8004(IF) ; MED LEVel,SPECtrum,ITU
    PSCAN : MODE PSCan ; PSCan:MODE NORMal ; UDP 8002(SCAN) ; MED SCAN
    FPSCAN: MODE PSCan ; PSCan:MODE FAST   ; UDP 8012(SCAN) ; MED SCAN
    FSCAN : MODE SWEep ; UDP 8008(SCAN) 8000 8001 ; MED SCAN
    MSCAN : MODE MSCan ; 频点表 MEMory:CONTents ; UDP 8010 8000 8001 ; MED SCAN
    DDC   : MODE FIXed + DDC:CONTents ; UDP 8020(DDC) 8030 8000 ; MED ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from PyQt6 import QtCore

from . import scpi
from .tcp_control import TcpControl
from .udp_receiver import UdpManager
from .rx_protocol import (
    SpectrumData, ScanData, FastScanData, LevelData, AudioData, DDCData,
)


# ============================================================
# 模式枚举
# ============================================================

class RxMode(Enum):
    FFM = "FFM"
    PSCAN = "PSCAN"
    FPSCAN = "FPSCAN"
    FSCAN = "FSCAN"
    MSCAN = "MSCAN"
    DDC = "DDC"


# ============================================================
# 各模式配置 dataclass
# ============================================================

@dataclass
class FFMConfig:
    center_mhz: float = 101.7
    span_khz: float = 1000.0
    bandwidth_khz: float = 120.0
    ref_level_dbuv: float = 80.0
    level_span_db: float = 10.0
    demod: str = "FM"
    squelch_on: bool = False
    threshold_dbuv: float = 10.0
    att_auto: bool = True
    if_att_db: float = 0.0
    rf_att_db: float = 0.0
    rf_mode: str = "NORMal"
    detector: str = "FAST"


@dataclass
class PScanConfig:
    start_mhz: float = 88.0
    stop_mhz: float = 108.0
    step_khz: float = 25.0
    ref_level_dbuv: float = 80.0
    level_span_db: float = 10.0
    if_att_db: float = 0.0
    rf_att_db: float = 0.0
    rf_mode: str = "NORMal"
    att_auto: bool = True
    fast: bool = False           # True -> FPSCAN


@dataclass
class FScanConfig:
    start_mhz: float = 88.0
    stop_mhz: float = 108.0
    step_khz: float = 25.0
    bandwidth_khz: float = 120.0
    ref_level_dbuv: float = 80.0
    level_span_db: float = 10.0
    threshold_dbuv: float = 10.0
    dwell_s: float = 0.1
    if_att_db: float = 0.0
    rf_att_db: float = 0.0
    demod: str = "FM"


@dataclass
class MScanChannel:
    idx: int
    freq_mhz: float
    bw_khz: float
    demod: str = "FM"


@dataclass
class MScanConfig:
    channels: List[MScanChannel] = field(default_factory=list)
    ref_level_dbuv: float = 80.0
    level_span_db: float = 10.0
    threshold_dbuv: float = 10.0
    dwell_s: float = 0.1
    if_att_db: float = 0.0
    rf_att_db: float = 0.0


@dataclass
class DDCChannel:
    idx: int
    freq_mhz: float
    bw_khz: float
    demod: str = "FM"
    media: List[str] = field(default_factory=lambda: ["LEV", "SPECtrum", "AUDio"])


@dataclass
class DDCConfig:
    center_mhz: float = 101.7
    span_khz: float = 1000.0
    channels: List[DDCChannel] = field(default_factory=list)
    ref_level_dbuv: float = 80.0
    level_span_db: float = 10.0


# ============================================================
# 模式 -> UDP 端口
# ============================================================

MODE_PORTS = {
    RxMode.FFM:    [8000, 8001, 8004],
    RxMode.PSCAN:  [8002],
    RxMode.FPSCAN: [8012],
    RxMode.FSCAN:  [8008, 8000, 8001],
    RxMode.MSCAN:  [8010, 8000, 8001],
    RxMode.DDC:    [8020, 8030, 8000],
}


# ============================================================
# 控制器
# ============================================================

class ReceiverController(QtCore.QObject):
    # 数据信号
    spectrumReady = QtCore.pyqtSignal(object)    # SpectrumData
    scanReady = QtCore.pyqtSignal(object)        # ScanData
    fastScanReady = QtCore.pyqtSignal(object)    # FastScanData
    levelReady = QtCore.pyqtSignal(object)       # LevelData
    audioReady = QtCore.pyqtSignal(object)       # AudioData
    ddcReady = QtCore.pyqtSignal(object)         # DDCData

    # 状态信号
    connectedChanged = QtCore.pyqtSignal(bool)
    modeChanged = QtCore.pyqtSignal(object)      # RxMode
    runningChanged = QtCore.pyqtSignal(bool)
    statusMessage = QtCore.pyqtSignal(str)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.tcp = TcpControl(self)
        self.udp = UdpManager(parent=self)

        self._local_ip = "192.168.0.120"
        self._tcp_port = 5025
        self._mode: Optional[RxMode] = None
        self._running = False

        self.tcp.connected.connect(lambda: self.connectedChanged.emit(True))
        self.tcp.disconnected.connect(lambda: self.connectedChanged.emit(False))
        self.tcp.errorOccurred.connect(self.errorOccurred)
        self.tcp.responseReceived.connect(
            lambda s: self.statusMessage.emit(f"[RX响应] {s}"))

        self.udp.dataReceived.connect(self._on_udp_data)
        self.udp.errorOccurred.connect(self.errorOccurred)

    # -------- 连接 --------

    def connect_receiver(self, ip: str, local_ip: str,
                         tcp_port: int = 5025) -> None:
        self._local_ip = local_ip
        self._tcp_port = tcp_port
        self.statusMessage.emit(f"连接接收机 {ip}:{tcp_port} ...")
        self.tcp.connect_to(ip, tcp_port)

    def disconnect_receiver(self) -> None:
        self.stop()
        self.udp.close_all()
        self.tcp.close()

    def is_connected(self) -> bool:
        return self.tcp.is_connected()

    @property
    def current_mode(self) -> Optional[RxMode]:
        return self._mode

    # -------- 模式下发 --------

    def apply_mode(self, mode: RxMode, config) -> None:
        if not self.is_connected():
            self.errorOccurred.emit("未连接接收机, 无法切换模式")
            return

        # 切模式前停止当前业务
        self._stop_internal()
        self.tcp.send(scpi.trace_udp_delete_all())

        # 端口差量处理: 关掉新模式不用的, 保留共用的, 打开新增的
        new_ports = set(MODE_PORTS[mode])
        for p in self.udp.active_ports():
            if p not in new_ports:
                self.udp.close_port(p)
        self.udp.open_ports(sorted(new_ports))

        if mode == RxMode.FFM:
            self._apply_ffm(config)
        elif mode in (RxMode.PSCAN, RxMode.FPSCAN):
            self._apply_pscan(mode, config)
        elif mode == RxMode.FSCAN:
            self._apply_fscan(config)
        elif mode == RxMode.MSCAN:
            self._apply_mscan(config)
        elif mode == RxMode.DDC:
            self._apply_ddc(config)

        self._mode = mode
        self.modeChanged.emit(mode)
        self.statusMessage.emit(f"已切换到 {mode.value} 模式")

    def _udp(self, port: int, channel: str) -> str:
        return scpi.trace_udp(self._local_ip, port, channel)

    # ---- FFM ----
    def _apply_ffm(self, c: FFMConfig) -> None:
        cmds = [
            scpi.freq_mode(scpi.MODE_FIXED),
            scpi.freq_center(c.center_mhz),
            scpi.freq_span(c.span_khz),
            scpi.bandwidth(c.bandwidth_khz),
            scpi.demodulation(c.demod),
            scpi.detector(c.detector),
            scpi.att_auto(c.att_auto),
            scpi.squelch(c.squelch_on),
            scpi.threshold(c.threshold_dbuv),
            self._udp(8000, "MEAS"),
            self._udp(8001, "AUD"),
            self._udp(8004, "IF"),
            scpi.trace_media(["LEVel", "SPECtrum", "ITU"]),
        ]
        if not c.att_auto:
            cmds += [scpi.att_if(c.if_att_db), scpi.att_rf(c.rf_att_db)]
        cmds.append(scpi.att_rf_mode(c.rf_mode))
        self.tcp.send_many(cmds)

    # ---- PSCAN / FPSCAN ----
    def _apply_pscan(self, mode: RxMode, c: PScanConfig) -> None:
        is_fast = (mode == RxMode.FPSCAN) or c.fast
        port = 8012 if is_fast else 8002
        cmds = [
            scpi.freq_mode(scpi.MODE_PSCAN),
            scpi.pscan_submode("FAST" if is_fast else "NORMal"),
            scpi.freq_start(c.start_mhz),
            scpi.freq_stop(c.stop_mhz),
            scpi.freq_step(c.step_khz),
            scpi.att_auto(c.att_auto),
        ]
        if not c.att_auto:
            cmds += [scpi.att_if(c.if_att_db), scpi.att_rf(c.rf_att_db)]
        cmds += [
            scpi.att_rf_mode(c.rf_mode),
            self._udp(port, "SCAN"),
            scpi.trace_media(["SCAN"]),
        ]
        self.tcp.send_many(cmds)

    # ---- FSCAN ----
    def _apply_fscan(self, c: FScanConfig) -> None:
        cmds = [
            scpi.freq_mode(scpi.MODE_SWEEP),
            scpi.freq_start(c.start_mhz),
            scpi.freq_stop(c.stop_mhz),
            scpi.freq_step(c.step_khz),
            scpi.bandwidth(c.bandwidth_khz),
            scpi.demodulation(c.demod),
            scpi.threshold(c.threshold_dbuv),
            scpi.dwell(c.dwell_s),
            scpi.att_if(c.if_att_db),
            scpi.att_rf(c.rf_att_db),
            self._udp(8008, "SCAN"),
            self._udp(8000, "MEAS"),
            self._udp(8001, "AUD"),
            scpi.trace_media(["SCAN", "LEVel", "SPECtrum"]),
        ]
        self.tcp.send_many(cmds)

    # ---- MSCAN ----
    def _apply_mscan(self, c: MScanConfig) -> None:
        cmds = [scpi.freq_mode(scpi.MODE_MSCAN)]
        for ch in c.channels:
            cmds.append(scpi.memory_contents(ch.idx, ch.freq_mhz, ch.bw_khz, ch.demod))
        cmds += [
            scpi.mscan_count(len(c.channels)),
            scpi.threshold(c.threshold_dbuv),
            scpi.dwell(c.dwell_s),
            scpi.att_if(c.if_att_db),
            scpi.att_rf(c.rf_att_db),
            self._udp(8010, "SCAN"),
            self._udp(8000, "MEAS"),
            self._udp(8001, "AUD"),
            scpi.trace_media(["SCAN", "LEVel"]),
        ]
        self.tcp.send_many(cmds)

    # ---- DDC ----
    def _apply_ddc(self, c: DDCConfig) -> None:
        cmds = [
            scpi.freq_mode(scpi.MODE_FIXED),    # DDC 在 FIX 下有效
            scpi.freq_center(c.center_mhz),
            scpi.freq_span(c.span_khz),
            scpi.ddc_close_all(),
        ]
        for ch in c.channels:
            cmds.append(scpi.ddc_contents(
                ch.idx, ch.freq_mhz, ch.bw_khz, ch.demod, ch.media))
        cmds += [
            self._udp(8020, "DDC"),
            self._udp(8030, "DDC"),
            self._udp(8000, "MEAS"),
            scpi.trace_media(["LEVel", "SPECtrum"]),
        ]
        self.tcp.send_many(cmds)

    # -------- 启停 --------

    def start(self) -> None:
        if not self.is_connected():
            self.errorOccurred.emit("未连接, 无法开始")
            return
        self.tcp.send(scpi.trace_on())
        self._running = True
        self.runningChanged.emit(True)
        self.statusMessage.emit("业务数据已开始 (TRAC ON)")

    def stop(self) -> None:
        self._stop_internal()

    def _stop_internal(self) -> None:
        if self.is_connected():
            self.tcp.send(scpi.trace_off())
        if self._running:
            self._running = False
            self.runningChanged.emit(False)
            self.statusMessage.emit("业务数据已停止 (TRAC OFF)")

    @property
    def running(self) -> bool:
        return self._running

    # -------- UDP 数据分发 --------

    @QtCore.pyqtSlot(int, object)
    def _on_udp_data(self, port: int, obj) -> None:
        if isinstance(obj, FastScanData):
            self.fastScanReady.emit(obj)
        elif isinstance(obj, ScanData):
            self.scanReady.emit(obj)
        elif isinstance(obj, SpectrumData):
            self.spectrumReady.emit(obj)
        elif isinstance(obj, LevelData):
            self.levelReady.emit(obj)
        elif isinstance(obj, AudioData):
            self.audioReady.emit(obj)
        elif isinstance(obj, DDCData):
            self.ddcReady.emit(obj)
        # RawData 等忽略

    def get_stats(self) -> dict:
        return {
            "connected": self.is_connected(),
            "mode": self._mode.value if self._mode else None,
            "running": self._running,
            "udp_ports": self.udp.active_ports(),
            "udp_stats": self.udp.get_stats(),
        }