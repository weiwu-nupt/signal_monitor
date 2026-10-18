"""
SCPI 命令构造 (板卡手册第 2 章, 原样实现)
==========================================
- 命令以 '\\r' 结尾 (手册: ASCII 回车结束)
- 频率默认单位 MHz, 这里统一用显式单位避免歧义
- 查询命令以 '?' 结尾

只负责"拼字符串", 不负责收发 (收发在 tcp_control).
所有方法返回 str (不含结尾 CR), send 时再补 CR.
"""

from __future__ import annotations

from typing import Iterable

CR = "\r"


def term(cmd: str) -> bytes:
    """补回车并编码"""
    return (cmd + CR).encode("ascii", errors="ignore")


# ============================================================
# 系统设置 (2.1)
# ============================================================

def idn() -> str:
    return "*IDN?"

def version() -> str:
    return "*VERsion?"

def restart() -> str:
    return "*RESTART"

def trace_udp(ip: str, port: int, channel: str) -> str:
    """注册一个 UDP 数据通道. channel: MEAS/SCAN/AUD/IF/DFIN/DDC/HW"""
    return f'TRAC:UDP "{ip}",{port},{channel}'

def trace_udp_delete(ip: str, port: int) -> str:
    return f'TRAC:UDP:DEL "{ip}",{port}'

def trace_udp_delete_all() -> str:
    return "TRAC:UDP:DEL ALL"

def trace_media(media: Iterable[str]) -> str:
    """设置请求的业务数据种类, 如 LEVel,SPECtrum"""
    return "TRACe:MED " + ",".join(media)

def trace_on() -> str:
    return "TRAC ON"

def trace_off() -> str:
    return "TRAC OFF"

def gps_enable(on: bool) -> str:
    return f"HARDware:GPS:ENABle {'ON' if on else 'OFF'}"

def compass_enable(on: bool) -> str:
    return f"HARDware:COMPass:ENABle {'ON' if on else 'OFF'}"


# ============================================================
# 频率 / 模式 (2.2)
# ============================================================

# FREQuency:MODE 取值
MODE_FIXED = "FIXed"
MODE_SWEEP = "SWEep"     # FSCAN
MODE_MSCAN = "MSCan"
MODE_PSCAN = "PSCan"
MODE_DDC = "DDC"
MODE_DFIND = "DFINd"
MODE_DPS = "DPS"
MODE_IQ = "IQ"

def freq_mode(mode: str) -> str:
    return f"FREQuency:MODE {mode}"

def freq_center(mhz: float) -> str:
    return f"FREQuency {mhz} MHz"

def freq_start(mhz: float) -> str:
    return f"FREQuency:STARt {mhz} MHz"

def freq_stop(mhz: float) -> str:
    return f"FREQuency:STOP {mhz} MHz"

def freq_step(khz: float) -> str:
    return f"FREQuency:STEP {khz} kHz"

def freq_span(khz: float) -> str:
    return f"FREQuency:SPAN {khz} kHz"

def freq_offset(khz: float) -> str:
    return f"FREQuency:OFFSet {khz} kHz"

def pscan_submode(mode: str) -> str:
    """NORMal|FAST|MULT"""
    return f"FREQuency:PSCan:MODE {mode}"


# ============================================================
# 带宽 / 衰减 / 解调
# ============================================================

def bandwidth(khz: float) -> str:
    return f"BANDwidth {khz} kHz"

def att_auto(on: bool) -> str:
    return f"ATTenuation:AUTomation {'ON' if on else 'OFF'}"

def att_if(db: float) -> str:
    return f"ATTenuation:IF {db}"

def att_rf(db: float) -> str:
    return f"ATTenuation:RF {db}"

def att_rf_mode(mode: str) -> str:
    """NORMal|LOWNoise|LOWDistort"""
    return f"ATTenuation:RF:MODE {mode}"

def demodulation(mode: str) -> str:
    """IQ|AM|FM|CW|LSB|USB|PULSE|PM|DMR|DPMR"""
    return f"DEModulation {mode}"


# ============================================================
# 测量参数
# ============================================================

def squelch(on: bool) -> str:
    return f"MEASure:SQUelch {'ON' if on else 'OFF'}"

def threshold(dbuv: float) -> str:
    return f"MEASure:THReshold {dbuv}"

def hold(seconds: float) -> str:
    return f"MEASure:HOLD {seconds} s"

def dwell(seconds: float) -> str:
    return f"MEASure:DWELl {seconds} s"

def meas_time(ms: float) -> str:
    return f"MEASure:TIME {ms} ms"

def meas_time_auto(on: bool) -> str:
    return f"MEASure:TIME:AUTomation {'ON' if on else 'OFF'}"


def bandwidth_xdb(value: float) -> str:
    """ITU XdB parameter, SCPI 2.2.22; valid range 0..100 dB."""
    return f"MEASure:BANDwidth:XDB {value}"


def bandwidth_beta(value: float) -> str:
    """ITU beta parameter; this application accepts 0 per user configuration."""
    return f"MEASure:BANDwidth:BETA {value}"

def detector(mode: str) -> str:
    """FAST|POSitive|AVERage|RMS"""
    return f"MEASure:DETector {mode}"


# ============================================================
# MSCAN 频点表 (2.2.24/25)
# ============================================================

def mscan_count(n: int) -> str:
    return f"MSCan:COUNt {n}"

def memory_contents(idx: int, freq_mhz: float, bw_khz: float, dem: str) -> str:
    return f"MEMory:CONTents {idx},{freq_mhz} MHz,{bw_khz} kHz,{dem}"


# ============================================================
# PSCAN 多段 (2.2.30/31)
# ============================================================

def pscan_count(n: int) -> str:
    return f"PSCan:COUNt {n}"

def pscan_contents(idx: int, start_mhz: float, stop_mhz: float, step_khz: float,
                   agc: bool, if_att: float, rf_att: float, rf_mode: str) -> str:
    return (f"PSCan:CONTents {idx},{start_mhz} MHz,{stop_mhz} MHz,"
            f"{step_khz} kHz,{'on' if agc else 'off'},{if_att},{rf_att},{rf_mode}")


# ============================================================
# DDC (2.2.26/27)
# ============================================================

def ddc_contents(idx: int, freq_mhz: float, bw_khz: float, dem: str,
                 media: Iterable[str]) -> str:
    """media 用 | 分隔, 如 LEV|SPECtrum|AUDio"""
    return (f"DDC:CONTents {idx},{freq_mhz} MHz,{bw_khz} kHz,{dem},"
            + "|".join(media))

def ddc_close_all() -> str:
    return "DDC:CLOSe:ALL"


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    samples = [
        freq_mode(MODE_PSCAN),
        freq_start(88), freq_stop(108), freq_step(25),
        pscan_submode("FAST"),
        trace_udp("192.168.0.120", 8012, "SCAN"),
        trace_media(["SCAN"]),
        bandwidth(120),
        memory_contents(1, 101.7, 120, "FM"),
        ddc_contents(1, 101.7, 120, "FM", ["LEV", "SPECtrum", "AUDio"]),
        trace_on(),
    ]
    for s in samples:
        print(repr(term(s)))
    print("SCPI builder OK")
