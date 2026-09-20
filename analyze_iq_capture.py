"""Offline spectrum and EnergyDetector inspection for fixed 4 MiB IQ captures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fpga_closed_loop import EnergyDetector


def _downsample_mean(values: np.ndarray, points: int) -> np.ndarray:
    trim = values.size - values.size % points
    return values[:trim].reshape(points, -1).mean(axis=1)


def analyse(
    path: Path,
    fs_hz: float,
    snr_db: float,
    output_dir: Path,
    analysis_low_hz: float = -150e6,
    analysis_high_hz: float = 150e6,
) -> tuple[Path, Path, int]:
    """Count emissions only within the configured usable RF band.

    The spectrum outside this interval is a front-end/filter transition band,
    not evidence of either a signal or a low noise floor.  In particular, the
    deep stopband of this test file must not make the detector believe that
    the in-band floor is -70 dB.
    """
    if not -fs_hz / 2 <= analysis_low_hz < analysis_high_hz <= fs_hz / 2:
        raise ValueError("analysis band must be inside [-fs/2, fs/2]")
    raw = np.fromfile(path, dtype="<i2")
    if raw.size % 2:
        raise ValueError("IQ file has an odd count of int16 values")
    iq = raw.reshape(-1, 2).astype(np.float32)
    samples = iq[:, 0] + 1j * iq[:, 1]
    power = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2 / samples.size
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(samples.size, d=1.0 / fs_hz))
    # C2H0 is an 8192-bin wideband spectrum.  Analyse this raw capture at
    # precisely that resolution, rather than turning individual FFT noise
    # fluctuations into hundreds of false "signals".
    detector_bins = 8192
    detector_power = _downsample_mean(power, detector_bins)
    detector_frequency_hz = _downsample_mean(frequency_hz, detector_bins)
    in_analysis_band = (
        (detector_frequency_hz >= analysis_low_hz)
        & (detector_frequency_hz < analysis_high_hz)
    )
    detector_power = detector_power[in_analysis_band]
    detector_frequency_hz = detector_frequency_hz[in_analysis_band]
    analysis_bandwidth_hz = analysis_high_hz - analysis_low_hz
    detector = EnergyDetector(snr_db=snr_db)
    noise_floor_db = detector.estimate_noise_floor_db(
        detector_power, exclude_filter_stopband=True
    )
    reference_db = float(10.0 * np.log10(detector_power.max()))
    noise_floor_relative_db = noise_floor_db - reference_db
    targets = detector.detect(
        detector_power,
        (analysis_low_hz + analysis_high_hz) / 2.0,
        analysis_bandwidth_hz,
        "wide",
        max_targets=128,
        stream_id="offline-iq",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "real_300m_energy_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "noise_floor_relative_db", f"{noise_floor_relative_db:.6f}",
            "analysis_band_hz", f"[{analysis_low_hz:.0f}, {analysis_high_hz:.0f})",
            "wide_measurement", "99_percent_occupied_bandwidth",
        ))
        writer.writerow(("id", "center_hz", "bandwidth_hz", "center_mhz", "bandwidth_mhz"))
        for index, target in enumerate(targets, start=1):
            writer.writerow((index, target.center_hz, target.bandwidth_hz,
                             f"{target.center_hz / 1e6:.6f}", f"{target.bandwidth_hz / 1e6:.6f}"))

    shown_power = detector_power
    shown_frequency = detector_frequency_hz / 1e6
    shown_db = 10.0 * np.log10(shown_power / shown_power.max() + 1e-15)
    figure, (spectrum_ax, candidates_ax) = plt.subplots(
        2, 1, figsize=(15, 9), constrained_layout=True,
        gridspec_kw={"height_ratios": (3, 2)},
    )
    spectrum_ax.plot(shown_frequency, shown_db, color="#00a6d6", linewidth=0.7)
    spectrum_ax.set_title(
        "Signals in "
        f"[{analysis_low_hz / 1e6:.0f}, {analysis_high_hz / 1e6:.0f}) MHz "
        f"— in-band noise {noise_floor_relative_db:.1f} dB, 99% OBW"
    )
    spectrum_ax.set_xlabel("Baseband frequency (MHz)")
    spectrum_ax.set_ylabel("Relative power (dB)")
    spectrum_ax.grid(alpha=0.25)
    spectrum_ax.set_xlim(analysis_low_hz / 1e6, analysis_high_hz / 1e6)

    # Draw every candidate.  Labels are limited to broad regions because 123
    # labels would obscure the spectrum; the CSV contains all IDs.
    for index, target in enumerate(targets, start=1):
        left = (target.center_hz - target.bandwidth_hz / 2) / 1e6
        right = (target.center_hz + target.bandwidth_hz / 2) / 1e6
        spectrum_ax.axvspan(left, right, color="#ff8c42", alpha=0.12)
        if target.bandwidth_hz >= 1_000_000:
            spectrum_ax.text(
                target.center_hz / 1e6, -3.0, f"#{index}\n{target.bandwidth_hz / 1e6:.1f}M",
                ha="center", va="top", fontsize=8, color="#9b3b00",
            )

    ordered = sorted(enumerate(targets, start=1), key=lambda item: item[1].center_hz)
    for row, (index, target) in enumerate(ordered):
        left = (target.center_hz - target.bandwidth_hz / 2) / 1e6
        width = target.bandwidth_hz / 1e6
        candidates_ax.broken_barh([(left, width)], (row - 0.35, 0.7), facecolors="#ff8c42")
    candidates_ax.set_title(f"All {len(targets)} EnergyDetector candidates (IDs in CSV)")
    candidates_ax.set_xlabel("Baseband frequency (MHz)")
    candidates_ax.set_ylabel("Candidate, frequency-sorted")
    candidates_ax.set_xlim(analysis_low_hz / 1e6, analysis_high_hz / 1e6)
    candidates_ax.set_ylim(-1, len(ordered))
    candidates_ax.grid(axis="x", alpha=0.25)
    candidates_ax.set_yticks([])

    image_path = output_dir / "real_300m_energy_detection.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    return image_path, csv_path, len(targets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("iq_file", type=Path)
    parser.add_argument("--fs", type=float, default=409.6e6)
    parser.add_argument("--snr-db", type=float, default=4.0)
    parser.add_argument("--out-dir", type=Path, default=Path("result_iq_analysis"))
    parser.add_argument(
        "--analysis-low-mhz", type=float, default=-150.0,
        help="lower edge of the usable band; filter response below it is ignored",
    )
    parser.add_argument(
        "--analysis-high-mhz", type=float, default=150.0,
        help="upper edge of the usable band; filter response above it is ignored",
    )
    args = parser.parse_args()
    image, table, count = analyse(
        args.iq_file,
        args.fs,
        args.snr_db,
        args.out_dir,
        args.analysis_low_mhz * 1e6,
        args.analysis_high_mhz * 1e6,
    )
    print(f"candidates={count}\nplot={image}\ncsv={table}")


if __name__ == "__main__":
    main()
