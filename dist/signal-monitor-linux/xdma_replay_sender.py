#!/usr/bin/env python3
"""Load IQ replay data and start FPGA replay without consuming C2H data.

Use this helper with ``main.py --mode dma``.  The original
``xdma_replay_spectrum.py`` also opens ``/dev/xdma0_c2h_0`` to receive and
analyse packets, so it cannot run alongside ``DMASpectrumSource``.

Typical single-frame test (terminal 1)::

    sudo -E python3 ./xdma_replay_sender.py \
        ./real_300M_409p6Msps_iq16.iq --mode single --frame-index 0 \
        --fs 409.6e6 --out-dir ./result_single

During the default 10-second delay, start the GUI in terminal 2::

    python3 ./main.py --mode dma

Only this helper opens XDMA user/H2C devices.  ``main.py`` is the sole C2H
consumer.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import xdma_replay_spectrum as replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load replay IQ data and start FPGA replay; C2H is left for main.py."
    )
    parser.add_argument("iq_file", type=Path, help="little-endian int16 interleaved I/Q file")
    parser.add_argument("--mode", choices=("single", "continuous"), default="single")
    parser.add_argument("--fs", type=float, default=409.6e6, help="sample rate in Hz (display only)")
    parser.add_argument("--period-ms", type=float, default=10.0, help="continuous replay packet period")
    parser.add_argument(
        "--xdma-clk-hz", type=float, default=replay.DEFAULT_XDMA_CLK_HZ,
        help="xdma_axi_aclk frequency in Hz",
    )
    parser.add_argument("--frame-index", type=int, default=0, help="frame index for single mode")
    parser.add_argument("--start-frame", type=int, default=0, help="first frame for continuous mode")
    parser.add_argument(
        "--replay-frames", type=int, default=0,
        help="continuous frames to load; 0 means all remaining frames",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./xdma_replay_sender_result"))
    parser.add_argument(
        "--start-delay", type=float, default=10.0,
        help="seconds to wait after loading, so main.py can open C2H (default: 10)",
    )
    return parser


def start_replay(control_value: int) -> None:
    """Enable the requested replay mode and issue the replay-start command."""
    user_fd = os.open(replay.XDMA_USER, os.O_RDWR)
    try:
        replay.write_reg32(user_fd, replay.REG_CONTROL, control_value)
        time.sleep(0.02)
        replay.write_reg32(user_fd, replay.REG_CMD, replay.CMD_REPLAY_START)
    finally:
        os.close(user_fd)


def main() -> int:
    args = build_parser().parse_args()
    if not args.iq_file.is_file():
        raise SystemExit(f"ERROR: IQ file does not exist: {args.iq_file}")
    if args.fs <= 0:
        raise SystemExit("ERROR: --fs must be greater than zero")
    if args.start_delay < 0:
        raise SystemExit("ERROR: --start-delay must not be negative")

    replay.require_root()
    replay.check_devices()
    capture_cycles, actual_period_s = replay.calculate_capture_period(
        args.period_ms, args.xdma_clk_hz
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        replay_path, replay_bytes, _ = replay.prepare_single_frame(
            args.iq_file, args.frame_index, out_dir
        )
        control_value = replay.CONTROL_SINGLE
        selected_frames = 1
    else:
        replay_path, replay_bytes, selected_frames = replay.prepare_continuous_replay(
            args.iq_file, args.start_frame, args.replay_frames, out_dir
        )
        control_value = replay.CONTROL_CONTINUOUS

    print("===== Replay sender configuration =====")
    print(f"mode                 = {args.mode}")
    print(f"sample rate          = {args.fs:.3f} Hz")
    print(f"selected frames      = {selected_frames}")
    print(f"replay bytes         = {replay_bytes}")
    print(f"capture period       = {actual_period_s * 1000.0:.6f} ms")
    print(f"capture cycles       = {capture_cycles} (0x{capture_cycles:08X})")

    # This writes only XDMA user/H2C.  Do not call receive_packets() here.
    replay.configure_and_load_replay(
        replay_path, replay_bytes, capture_cycles, args.xdma_clk_hz
    )
    print(
        f"Replay data loaded. Start 'python3 ./main.py --mode dma' within "
        f"{args.start_delay:.1f} seconds."
    )
    time.sleep(args.start_delay)
    start_replay(control_value)
    print("Replay started. C2H packets are now owned by main.py / DMASpectrumSource.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        replay.stop_fpga_runtime()
        raise SystemExit(1)
