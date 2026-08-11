#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XCKU115 XDMA回放频谱接收程序

运行模式：
1. 单帧模式
加载一帧8192点IQ数据，单次回放并接收一包频谱数据

2. 连续模式
加载多帧完整IQ数据到DDR，开启循环回放，FPGA按配置间隔输出频谱包，按Ctrl+C停止
到达采集周期后，FPGA会抓取下一帧完整FFT数据

输入格式：
小端序int16类型I/Q交错数据
格式：I0,Q0,I1,Q1,...

使用示例：
#单帧输出
sudo -E python3 xdma_replay_spectrum.py ./real_300M_409p6Msps_iq16.iq --mode single --frame-index 0 --fs 409.6e6 --out-dir ./result_single

#每10ms输出一包，持续运行
sudo -E python3 xdma_replay_spectrum.py ./real_300M_409p6Msps_iq16.iq --mode continuous --period-ms 10 --fs 409.6e6 --out-dir ./result_10ms

#每10ms输出一包，接收2包后停止
sudo -E python3 xdma_replay_spectrum.py ./real_300M_409p6Msps_iq16.iq --mode continuous --period-ms 10 --max-packets 2 --fs 409.6e6 --out-dir ./result_10ms_2pkt

补充说明：
1.连续模式下--max-packets设为0表示不限制包数量
2.当前FPGA逻辑会立即输出第一包数据，后续数据包严格按照寄存器0x18配置周期执行
3.时钟250MHz下，10ms对应2500000个时钟周期，十六进制0x002625A0
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except (AttributeError, ValueError):
    pass

#数据与报文格式定义
NFFT = 8192
BYTES_PER_COMPLEX_SAMPLE = 4
IQ_FRAME_BYTES = NFFT * BYTES_PER_COMPLEX_SAMPLE

C2H_PACKET_BYTES = 65664
C2H_HEADER_BYTES = 64
C2H_PAYLOAD_BYTES = NFFT * 8
C2H_TAIL_BYTES = 64

#XDMA设备节点
XDMA_USER = "/dev/xdma0_user"
XDMA_H2C = "/dev/xdma0_h2c_0"
XDMA_C2H = "/dev/xdma0_c2h_0"

#AXI-Lite寄存器地址映射
REG_CMD = 0x00
REG_CONTROL = 0x04
REG_H2C_BASE = 0x08
REG_H2C_LEN = 0x0C
REG_REPLAY_BASE = 0x10
REG_REPLAY_LEN = 0x14
REG_CAPTURE_PERIOD = 0x18

REG_STATUS = 0x20
REG_H2C_WR_BYTES = 0x24
REG_H2C_DONE_CNT = 0x28
REG_H2C_ERR_CNT = 0x2C
REG_REPLAY_DONE_CNT = 0x30
REG_REPLAY_BYTES = 0x34
REG_C2H_PKT_CNT = 0x38
REG_C2H_DROP_CNT = 0x3C

#命令寄存器定义
CMD_CLEAR = 0x0000_0001
CMD_H2C_START = 0x0000_0002
CMD_REPLAY_START = 0x0000_0004

#控制寄存器定义
CONTROL_LOAD_ONLY = 0x0000_0006
CONTROL_SINGLE = 0x0000_0007
CONTROL_CONTINUOUS = 0x0000_000F
CONTROL_DISABLED = 0x0000_0000

#DDR地址与时钟参数
DDR_BASE = 0x0010_0000
DEFAULT_XDMA_CLK_HZ = 250_000_000.0

#检查root权限
def require_root() -> None:
    if os.geteuid() != 0:
        raise SystemExit(
            "ERROR: 需要root权限。\n请使用 sudo -E python3 执行。"
        )

#检查XDMA设备节点是否存在
def check_devices() -> None:
    for device in (XDMA_USER, XDMA_H2C, XDMA_C2H):
        if not Path(device).exists():
            raise SystemExit(
                f"ERROR: 设备节点不存在：{device}\n请先运行 start_xdma.sh。"
            )

#写入32位寄存器
def write_reg32(fd: int, offset: int, value: int) -> None:
    data = struct.pack("<I", value & 0xFFFF_FFFF)
    written = os.pwrite(fd, data, offset)
    if written != 4:
        raise RuntimeError(
            f"写寄存器0x{offset:02X}失败，返回{written}字节"
        )

#读取32位寄存器
def read_reg32(fd: int, offset: int) -> int:
    data = os.pread(fd, 4, offset)
    if len(data) != 4:
        raise RuntimeError(
            f"读寄存器0x{offset:02X}失败，返回{len(data)}字节"
        )
    return struct.unpack("<I", data)[0]

#读取所有状态寄存器
def read_status_registers(fd: int) -> dict[str, int]:
    register_list = [
        ("CONTROL", REG_CONTROL),
        ("CAPTURE_PERIOD", REG_CAPTURE_PERIOD),
        ("STATUS", REG_STATUS),
        ("H2C_WR_BYTES", REG_H2C_WR_BYTES),
        ("H2C_DONE_CNT", REG_H2C_DONE_CNT),
        ("H2C_ERR_CNT", REG_H2C_ERR_CNT),
        ("REPLAY_DONE_CNT", REG_REPLAY_DONE_CNT),
        ("REPLAY_BYTES", REG_REPLAY_BYTES),
        ("C2H_PKT_CNT", REG_C2H_PKT_CNT),
        ("C2H_DROP_CNT", REG_C2H_DROP_CNT),
    ]
    values: dict[str, int] = {}
    for name, offset in register_list:
        values[name] = read_reg32(fd, offset)
    return values

#打印寄存器状态
def print_status_registers(values: dict[str, int]) -> None:
    for name, value in values.items():
        print(f"{name:<18} = 0x{value:08X}")

#计算采集周期对应的时钟周期数
def calculate_capture_period(
    period_ms: float,
    xdma_clk_hz: float,
) -> tuple[int, float]:
    if period_ms <= 0:
        raise SystemExit("ERROR: --period-ms必须大于0")
    if xdma_clk_hz <= 0:
        raise SystemExit("ERROR: --xdma-clk-hz必须大于0")
    cycles = int(round(period_ms * xdma_clk_hz / 1000.0))
    if cycles < 1:
        raise SystemExit("ERROR: 捕获周期小于1个时钟周期")
    if cycles > 0xFFFF_FFFF:
        raise SystemExit("ERROR: 捕获周期超过32位寄存器范围")
    actual_period_s = cycles / xdma_clk_hz
    return cycles, actual_period_s

#配置采集周期寄存器并校验
def program_capture_period(
    fd: int,
    cycles: int,
    xdma_clk_hz: float,
) -> None:
    write_reg32(fd, REG_CAPTURE_PERIOD, cycles)
    readback = read_reg32(fd, REG_CAPTURE_PERIOD)
    if readback != cycles:
        raise RuntimeError(
            f"CAPTURE_PERIOD回读不一致：写入0x{cycles:08X}，回读0x{readback:08X}"
        )
    period_s = readback / xdma_clk_hz
    print("\n===== 采集周期配置 =====")
    print(f"xdma_axi_aclk      = {xdma_clk_hz:.3f} Hz")
    print(f"时钟周期数         = {readback}")
    print(f"十六进制值         = 0x{readback:08X}")
    print(f"周期时长           = {period_s * 1000.0:.9f} ms")
    print(f"输出帧率           = {1.0 / period_s:.9f} packet/s")

#校验IQ文件合法性
def validate_iq_file(path: Path) -> tuple[int, int]:
    file_size = path.stat().st_size
    if file_size < IQ_FRAME_BYTES:
        raise SystemExit(
            f"ERROR: IQ文件只有{file_size}字节，至少需要一帧{IQ_FRAME_BYTES}字节"
        )
    if file_size % BYTES_PER_COMPLEX_SAMPLE != 0:
        raise SystemExit(
            f"ERROR: IQ文件大小{file_size}不是4字节整数倍"
        )
    complete_frames = file_size // IQ_FRAME_BYTES
    trailing_bytes = file_size % IQ_FRAME_BYTES
    return complete_frames, trailing_bytes

#单帧模式：提取指定帧IQ数据
def prepare_single_frame(
    input_path: Path,
    frame_index: int,
    out_dir: Path,
) -> tuple[Path, int, np.ndarray]:
    complete_frames, _ = validate_iq_file(input_path)
    if frame_index < 0 or frame_index >= complete_frames:
        raise SystemExit(
            f"ERROR: frame-index={frame_index}无效；文件完整帧范围为0到{complete_frames - 1}"
        )
    with input_path.open("rb") as input_file:
        input_file.seek(frame_index * IQ_FRAME_BYTES)
        frame_bytes = input_file.read(IQ_FRAME_BYTES)
    if len(frame_bytes) != IQ_FRAME_BYTES:
        raise RuntimeError("读取单帧IQ失败")
    replay_path = out_dir / "replay_single_frame.iq"
    replay_path.write_bytes(frame_bytes)
    iq_i16 = np.frombuffer(
        frame_bytes,
        dtype="<i2",
    ).reshape(NFFT, 2)
    iq_complex = (
        iq_i16[:, 0].astype(np.float64)
        + 1j * iq_i16[:, 1].astype(np.float64)
    )
    return replay_path, IQ_FRAME_BYTES, iq_complex

#连续模式：准备多帧回放数据
def prepare_continuous_replay(
    input_path: Path,
    start_frame: int,
    replay_frames: int,
    out_dir: Path,
) -> tuple[Path, int, int]:
    complete_frames, trailing_bytes = validate_iq_file(input_path)
    if start_frame < 0 or start_frame >= complete_frames:
        raise SystemExit(
            f"ERROR: start-frame={start_frame}无效；文件完整帧范围为0到{complete_frames - 1}"
        )
    available_frames = complete_frames - start_frame
    if replay_frames == 0:
        selected_frames = available_frames
    else:
        if replay_frames < 1:
            raise SystemExit("ERROR: --replay-frames必须为0或正整数")
        selected_frames = min(replay_frames, available_frames)
    replay_bytes = selected_frames * IQ_FRAME_BYTES
    if replay_bytes > 0xFFFF_FFFF:
        raise SystemExit(
            "ERROR: Replay长度超过32位寄存器范围"
        )
    if DDR_BASE + replay_bytes > 0x1_0000_0000:
        raise SystemExit(
            "ERROR: DDR_BASE + Replay长度超过32位地址空间"
        )
    source_offset = start_frame * IQ_FRAME_BYTES
    use_original = (
        source_offset == 0
        and replay_bytes == input_path.stat().st_size
        and trailing_bytes == 0
    )
    if use_original:
        replay_path = input_path
    else:
        replay_path = out_dir / "replay_continuous_frames.iq"
        with input_path.open("rb") as source, replay_path.open("wb") as target:
            source.seek(source_offset)
            remaining = replay_bytes
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("生成Replay文件时提前到达文件末尾")
                target.write(chunk)
                remaining -= len(chunk)
    actual_size = replay_path.stat().st_size
    if actual_size != replay_bytes:
        raise RuntimeError(
            f"Replay文件大小错误：{actual_size}，期望{replay_bytes}"
        )
    return replay_path, replay_bytes, selected_frames

#通过H2C通道下发回放数据
def write_replay_data_to_h2c(
    replay_path: Path,
    byte_len: int,
) -> None:
    result = subprocess.run(
        [
            "dd",
            f"if={replay_path}",
            f"of={XDMA_H2C}",
            f"bs={byte_len}",
            "count=1",
            "iflag=fullblock",
            "status=none",
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"H2C dd失败，返回码={result.returncode}"
        )

#解析C2H返回的频谱报文
def parse_c2h_packet(
    packet: bytes,
) -> tuple[dict[str, int], np.ndarray]:
    if len(packet) != C2H_PACKET_BYTES:
        raise RuntimeError(
            f"C2H包长度错误：{len(packet)}，期望{C2H_PACKET_BYTES}"
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
    if header["sync"] != 0x1ACFFC1D:
        raise RuntimeError(
            f"同步字错误：0x{header['sync']:08X}"
        )
    if header["fft_points"] != NFFT:
        raise RuntimeError(
            f"FFT点数错误：{header['fft_points']}"
        )
    power = np.frombuffer(
        packet,
        dtype="<u8",
        count=NFFT,
        offset=C2H_HEADER_BYTES,
    ).copy()
    return header, power

#将FFT频点索引转换为实际频率
def fft_bin_to_frequency(
    bin_index: int,
    sample_rate_hz: float,
) -> float:
    if bin_index < NFFT // 2:
        return bin_index * sample_rate_hz / NFFT
    return (bin_index - NFFT) * sample_rate_hz / NFFT

#单帧模式：软硬件FFT结果对比并保存文件
def save_single_comparison(
    reference_iq: np.ndarray,
    hardware_power_u64: np.ndarray,
    sample_rate_hz: float,
    out_dir: Path,
    floor_db: float,
) -> None:
    hardware_power = hardware_power_u64.astype(np.float64)
    software_power = np.abs(np.fft.fft(reference_iq, n=NFFT)) ** 2
    hardware_peak_bin = int(np.argmax(hardware_power))
    software_peak_bin = int(np.argmax(software_power))
    denominator = float(np.dot(software_power, software_power))
    gain = (
        float(np.dot(hardware_power, software_power) / denominator)
        if denominator > 0.0
        else 0.0
    )
    error = hardware_power - gain * software_power
    hardware_energy = float(np.dot(hardware_power, hardware_power))
    nmse = (
        float(np.dot(error, error) / hardware_energy)
        if hardware_energy > 0.0
        else 0.0
    )
    frequency_hz = np.fft.fftfreq(
        NFFT,
        d=1.0 / sample_rate_hz,
    )
    tiny = np.finfo(np.float64).tiny
    hardware_db = 10.0 * np.log10(
        np.maximum(hardware_power, tiny)
        / max(float(np.max(hardware_power)), tiny)
    )
    software_db = 10.0 * np.log10(
        np.maximum(software_power, tiny)
        / max(float(np.max(software_power)), tiny)
    )
    hardware_db = np.maximum(hardware_db, floor_db)
    software_db = np.maximum(software_db, floor_db)
    csv_path = out_dir / "single_spectrum_compare.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "bin",
                "frequency_hz",
                "hardware_power",
                "software_power",
                "hardware_db_normalized",
                "software_db_normalized",
            ]
        )
        for index in range(NFFT):
            writer.writerow(
                [
                    index,
                    f"{frequency_hz[index]:.9f}",
                    int(hardware_power_u64[index]),
                    f"{software_power[index]:.12e}",
                    f"{hardware_db[index]:.6f}",
                    f"{software_db[index]:.6f}",
                ]
            )
    print("\n===== 单包频谱对比 =====")
    print(f"硬件峰值频点       = {hardware_peak_bin}")
    print(f"软件峰值频点       = {software_peak_bin}")
    print(
        f"硬件峰值频率       = {fft_bin_to_frequency(hardware_peak_bin, sample_rate_hz) / 1e6:.6f} MHz"
    )
    print(
        f"软件峰值频率       = {fft_bin_to_frequency(software_peak_bin, sample_rate_hz) / 1e6:.6f} MHz"
    )
    print(f"拟合后归一化均方误差 = {nmse:.6e}")
    print(f"对比数据文件       = {csv_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = np.fft.fftshift(np.arange(NFFT))
        centered_frequency_mhz = np.fft.fftshift(frequency_hz) / 1e6
        figure = plt.figure(figsize=(12, 6))
        axis = figure.add_subplot(111)
        axis.plot(
            centered_frequency_mhz,
            hardware_db[order],
            label="硬件FFT",
        )
        axis.plot(
            centered_frequency_mhz,
            software_db[order],
            label="NumPy FFT",
        )
        axis.set_xlabel("频率(MHz)")
        axis.set_ylabel("归一化功率(dB)")
        axis.set_title("硬件FFT 与 NumPy FFT 对比")
        axis.set_ylim(floor_db, 3.0)
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        plot_path = out_dir / "single_spectrum_compare.png"
        figure.savefig(plot_path, dpi=160)
        plt.close(figure)
        print(f"对比波形图         = {plot_path}")
    except Exception as exc:
        print(f"警告: 未生成PNG图片：{exc}")

#延时启动回放线程
def start_replay_after_delay(
    control_value: int,
    delay_s: float,
    started_event: threading.Event,
    start_info: dict[str, int | str],
) -> None:
    try:
        print(
            f"[REPLAY] 等待{delay_s:.3f}秒后启动Replay...",
            flush=True,
        )
        time.sleep(delay_s)
        fd = os.open(XDMA_USER, os.O_RDWR)
        try:
            print(
                f"[REPLAY] 写CONTROL = 0x{control_value:08X}",
                flush=True,
            )
            write_reg32(fd, REG_CONTROL, control_value)
            time.sleep(0.02)
            start_info["monotonic_ns"] = time.monotonic_ns()
            print(
                f"[REPLAY] 写CMD_REPLAY_START = 0x{CMD_REPLAY_START:08X}",
                flush=True,
            )
            write_reg32(fd, REG_CMD, CMD_REPLAY_START)
            print(
                "[REPLAY] Replay启动命令已发送",
                flush=True,
            )
        finally:
            os.close(fd)
    except Exception as exc:
        start_info["error"] = str(exc)
        print(
            f"[REPLAY] 启动失败：{exc}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        started_event.set()

#持续接收C2H频谱包
def receive_packets(
    mode: str,
    max_packets: int,
    sample_rate_hz: float,
    out_dir: Path,
    save_raw_stream: bool,
    save_each_packet: bool,
    print_every: int,
    reference_iq: np.ndarray | None,
    floor_db: float,
    control_value: int,
) -> int:
    latest_packet_path = out_dir / "latest_c2h_fft_packet.bin"
    single_packet_path = out_dir / "c2h_fft_packet.bin"
    summary_path = out_dir / "packet_summary.csv"
    raw_stream_path = out_dir / "c2h_fft_stream.bin"
    raw_stream_file = (
        raw_stream_path.open("wb")
        if save_raw_stream
        else None
    )
    received_count = 0
    previous_completion_ns: int | None = None
    started_event = threading.Event()
    start_info: dict[str, int | str] = {}
    starter = threading.Thread(
        target=start_replay_after_delay,
        args=(
            control_value,
            0.5,
            started_event,
            start_info,
        ),
        name="replay-starter",
        daemon=True,
    )
    c2h_fd = os.open(XDMA_C2H, os.O_RDONLY)
    try:
        with summary_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as summary_file:
            fieldnames = [
                "packet_index",
                "host_time_utc",
                "completion_monotonic_ns",
                "interval_from_previous_ms",
                "sequence",
                "frame_id",
                "source",
                "peak_bin",
                "peak_frequency_hz",
                "peak_power",
            ]
            writer = csv.DictWriter(
                summary_file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            summary_file.flush()
            starter.start()
            print("\n===== C2H数据接收 =====")
            if mode == "continuous" and max_packets == 0:
                print("连续输出模式: 按Ctrl+C停止")
            else:
                print(f"目标接收包数     = {max_packets}")
            while True:
                if max_packets > 0 and received_count >= max_packets:
                    break
                if received_count == 0:
                    print(
                        "[C2H] 已提交阻塞读取请求，等待FPGA发送第一包...",
                        flush=True,
                    )
                packet = os.read(
                    c2h_fd,
                    C2H_PACKET_BYTES,
                )
                completion_ns = time.monotonic_ns()
                if len(packet) != C2H_PACKET_BYTES:
                    raise RuntimeError(
                        f"第{received_count + 1}包长度为{len(packet)}，期望{C2H_PACKET_BYTES}"
                    )
                header, power = parse_c2h_packet(packet)
                peak_bin = int(np.argmax(power))
                peak_frequency_hz = fft_bin_to_frequency(
                    peak_bin,
                    sample_rate_hz,
                )
                peak_power = int(power[peak_bin])
                interval_ms: float | str = ""
                if previous_completion_ns is not None:
                    interval_ms = (
                        completion_ns - previous_completion_ns
                    ) / 1e6
                previous_completion_ns = completion_ns
                received_count += 1
                temp_latest = out_dir / ".latest_c2h_fft_packet.tmp"
                temp_latest.write_bytes(packet)
                os.replace(temp_latest, latest_packet_path)
                if mode == "single":
                    single_packet_path.write_bytes(packet)
                if raw_stream_file is not None:
                    raw_stream_file.write(packet)
                    raw_stream_file.flush()
                if save_each_packet:
                    packet_path = (
                        out_dir
                        / f"c2h_fft_packet_{received_count:08d}.bin"
                    )
                    packet_path.write_bytes(packet)
                writer.writerow(
                    {
                        "packet_index": received_count,
                        "host_time_utc": (
                            dt.datetime.now(dt.timezone.utc)
                            .isoformat(timespec="milliseconds")
                        ),
                        "completion_monotonic_ns": completion_ns,
                        "interval_from_previous_ms": interval_ms,
                        "sequence": header["sequence"],
                        "frame_id": header["frame_id"],
                        "source": header["source"],
                        "peak_bin": peak_bin,
                        "peak_frequency_hz": (
                            f"{peak_frequency_hz:.9f}"
                        ),
                        "peak_power": peak_power,
                    }
                )
                summary_file.flush()
                if (
                    received_count == 1
                    or received_count % print_every == 0
                ):
                    interval_text = (
                        "首包"
                        if interval_ms == ""
                        else f"{float(interval_ms):.6f} ms"
                    )
                    print(
                        f"包序号={received_count}, 序列号={header['sequence']}, "
                        f"帧ID={header['frame_id']}, 间隔={interval_text}, "
                        f"峰值频点={peak_bin}, 峰值频率={peak_frequency_hz / 1e6:.6f} MHz",
                        flush=True,
                    )
                if mode == "single":
                    assert reference_iq is not None
                    save_single_comparison(
                        reference_iq=reference_iq,
                        hardware_power_u64=power,
                        sample_rate_hz=sample_rate_hz,
                        out_dir=out_dir,
                        floor_db=floor_db,
                    )
                    break
    finally:
        if raw_stream_file is not None:
            raw_stream_file.close()
        os.close(c2h_fd)
    return received_count

#寄存器配置并加载回放数据到DDR
def configure_and_load_replay(
    replay_path: Path,
    replay_bytes: int,
    capture_cycles: int,
    xdma_clk_hz: float,
) -> None:
    user_fd = os.open(XDMA_USER, os.O_RDWR)
    try:
        print("\n===== 初始化与配置 =====")
        write_reg32(
            user_fd,
            REG_CONTROL,
            CONTROL_LOAD_ONLY,
        )
        write_reg32(
            user_fd,
            REG_CMD,
            CMD_CLEAR,
        )
        time.sleep(0.2)
        program_capture_period(
            user_fd,
            capture_cycles,
            xdma_clk_hz,
        )
        write_reg32(user_fd, REG_H2C_BASE, DDR_BASE)
        write_reg32(user_fd, REG_H2C_LEN, replay_bytes)
        write_reg32(user_fd, REG_REPLAY_BASE, DDR_BASE)
        write_reg32(user_fd, REG_REPLAY_LEN, replay_bytes)
        write_reg32(
            user_fd,
            REG_CMD,
            CMD_H2C_START,
        )
        print("\n===== H2C数据加载 =====")
        print(f"回放文件路径     = {replay_path.resolve()}")
        print(f"数据字节数       = {replay_bytes}")
        print(f"字节数(十六进制) = 0x{replay_bytes:08X}")
        write_replay_data_to_h2c(
            replay_path,
            replay_bytes,
        )
        print("\n===== H2C加载后寄存器状态 =====")
        values = read_status_registers(user_fd)
        print_status_registers(values)
        if values["H2C_WR_BYTES"] != replay_bytes:
            raise RuntimeError(
                f"H2C_WR_BYTES错误：{values['H2C_WR_BYTES']}，期望{replay_bytes}"
            )
        if values["H2C_DONE_CNT"] != 1:
            raise RuntimeError(
                f"H2C_DONE_CNT错误：{values['H2C_DONE_CNT']}，期望1"
            )
        if values["H2C_ERR_CNT"] != 0:
            raise RuntimeError(
                f"H2C_ERR_CNT非零：{values['H2C_ERR_CNT']}"
            )
    finally:
        os.close(user_fd)

#读取最终寄存器状态
def print_final_status() -> dict[str, int]:
    user_fd = os.open(XDMA_USER, os.O_RDWR)
    try:
        print("\n===== 最终寄存器状态 =====")
        values = read_status_registers(user_fd)
        print_status_registers(values)
        if values["C2H_DROP_CNT"] != 0:
            print(
                "警告: C2H存在丢包，原因可能是数据包未及时释放或系统读取速度不足"
            )
        return values
    finally:
        os.close(user_fd)

#停止FPGA回放与采集逻辑
def stop_fpga_runtime() -> None:
    try:
        user_fd = os.open(XDMA_USER, os.O_RDWR)
        try:
            write_reg32(
                user_fd,
                REG_CONTROL,
                CONTROL_DISABLED,
            )
            time.sleep(0.01)
            write_reg32(
                user_fd,
                REG_CMD,
                CMD_CLEAR,
            )
        finally:
            os.close(user_fd)
    except Exception as exc:
        print(f"警告: 停止FPGA运行状态失败：{exc}")

#解析命令行参数
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于DDR回放的单帧/连续频谱C2H输出工具"
    )
    parser.add_argument(
        "iq_file",
        type=Path,
        help="小端序int16类型I/Q交错IQ文件",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "continuous"),
        default="single",
        help="single:单帧回放接收单包;continuous:循环回放持续接收",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=409.6e6,
        help="采样率(Hz)，默认409.6e6",
    )
    parser.add_argument(
        "--period-ms",
        type=float,
        default=10.0,
        help="周期性输出间隔(ms)，默认10",
    )
    parser.add_argument(
        "--xdma-clk-hz",
        type=float,
        default=DEFAULT_XDMA_CLK_HZ,
        help="xdma_axi_aclk时钟频率，默认250000000",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="单帧模式：选择的8192点帧序号",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="连续模式：载入DDR的起始帧序号",
    )
    parser.add_argument(
        "--replay-frames",
        type=int,
        default=0,
        help="连续模式：循环回放帧数，0代表使用全部可用帧",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="连续模式：最大接收包数，0代表不限制",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./xdma_replay_spectrum_result"),
        help="结果输出目录",
    )
    parser.add_argument(
        "--save-raw-stream",
        action="store_true",
        help="将所有原始数据包追加保存到c2h_fft_stream.bin",
    )
    parser.add_argument(
        "--save-each-packet",
        action="store_true",
        help="将每个数据包单独保存为bin文件",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="每接收N包打印一次状态",
    )
    parser.add_argument(
        "--floor-db",
        type=float,
        default=-120.0,
        help="单帧模式频谱归一化下限(dB)",
    )
    return parser

#主函数
def main() -> int:
    print("===== 程序启动 =====", flush=True)
    print(f"脚本路径         = {Path(__file__).resolve()}", flush=True)
    print(f"Python版本       = {sys.version.split()[0]}", flush=True)
    args = build_parser().parse_args()
    if not args.iq_file.exists():
        raise SystemExit(
            f"ERROR: IQ文件不存在：{args.iq_file}"
        )
    if args.fs <= 0:
        raise SystemExit("ERROR: --fs必须大于0")
    if args.max_packets < 0:
        raise SystemExit("ERROR: --max-packets不能为负数")
    if args.print_every < 1:
        raise SystemExit("ERROR: --print-every至少为1")
    require_root()
    check_devices()
    capture_cycles, actual_period_s = calculate_capture_period(
        args.period_ms,
        args.xdma_clk_hz,
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_iq: np.ndarray | None = None
    if args.mode == "single":
        replay_path, replay_bytes, reference_iq = prepare_single_frame(
            input_path=args.iq_file,
            frame_index=args.frame_index,
            out_dir=out_dir,
        )
        control_value = CONTROL_SINGLE
        max_packets = 1
        selected_frames = 1
    else:
        replay_path, replay_bytes, selected_frames = (
            prepare_continuous_replay(
                input_path=args.iq_file,
                start_frame=args.start_frame,
                replay_frames=args.replay_frames,
                out_dir=out_dir,
            )
        )
        control_value = CONTROL_CONTINUOUS
        max_packets = args.max_packets
    print("===== 运行配置 =====")
    print(f"运行模式         = {args.mode}")
    print(f"IQ文件路径       = {args.iq_file.resolve()}")
    print(f"采样率           = {args.fs:.3f} Hz")
    print(f"FFT点数          = {NFFT}")
    print(f"频点间隔         = {args.fs / NFFT:.6f} Hz")
    print(f"选中帧数         = {selected_frames}")
    print(f"回放数据总字节数 = {replay_bytes}")
    print(f"配置周期         = {actual_period_s * 1000.0:.9f} ms")
    print(f"周期对应时钟数   = {capture_cycles}")
    print(f"周期十六进制值    = 0x{capture_cycles:08X}")
    if args.mode == "continuous":
        if max_packets == 0:
            print("接收包数限制     = 无限制，按Ctrl+C停止")
        else:
            print(f"接收包数限制     = {max_packets}")
    configure_and_load_replay(
        replay_path=replay_path,
        replay_bytes=replay_bytes,
        capture_cycles=capture_cycles,
        xdma_clk_hz=args.xdma_clk_hz,
    )
    interrupted = False
    try:
        received_count = receive_packets(
            mode=args.mode,
            max_packets=max_packets,
            sample_rate_hz=args.fs,
            out_dir=out_dir,
            save_raw_stream=args.save_raw_stream,
            save_each_packet=args.save_each_packet,
            print_every=args.print_every,
            reference_iq=reference_iq,
            floor_db=args.floor_db,
            control_value=control_value,
        )
        print(f"\n已接收数据包数   = {received_count}")
    except KeyboardInterrupt:
        interrupted = True
        print("\n检测到Ctrl+C，停止连续回放...")
    finally:
        try:
            print_final_status()
        finally:
            stop_fpga_runtime()
    if interrupted:
        print("\n已停止: 连续输出正常结束。")
    else:
        print("\n执行完成: 回放与频谱C2H输出流程结束。")
    print(f"结果目录         = {out_dir}")
    print(
        "最新数据包文件    = "
        f"{out_dir / 'latest_c2h_fft_packet.bin'}"
    )
    print(
        "数据包汇总表      = "
        f"{out_dir / 'packet_summary.csv'}"
    )
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        stop_fpga_runtime()
        raise SystemExit(1)
