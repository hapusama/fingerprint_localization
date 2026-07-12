#!/usr/bin/env python3
"""Project 20 MHz chirp CIRs into expected LoRa bin[-2,+2] windows.

The chirp captures are treated as wideband complex channel observations.  For
each measured corridor/location, this script extracts a short complex CIR
around the detected chirp peak, samples its frequency response over the LoRa
125 kHz subband, applies that response to an ideal LoRa preamble upchirp, and
then dechirps/FFTs to synthesize the expected LoRa bin window.  The synthetic
window is compared with the measured LoRa preamble FFT bin[-2,+2] window.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[3]
MULTIPATH_ROOT = ROOT / "INFOCOM_origin_data/multipath_data"
SEGMENT_INPUT = (
    ROOT
    / "v2_output/20260623_from_raw/step6a_chirp_original_project_method"
    / "04_original_three_threshold_segment_peaks.csv"
)
LORA_INPUT = (
    ROOT
    / "v2_output/20260623_from_raw/data_processing"
    / "lora_frequency_s17_54points.csv"
)
DEFAULT_OUT = ROOT / "v2_output/20260626_chirp_lora_bin_projection"

FS_CHIRP = 20e6
CHIRP_BW = 18e6
CHIRP_DURATION = 1e-3
PERIOD = 20e-3
LORA_BW = 125e3
SF = 11
FFT_SIZE = 1 << SF
BIN_OFFSETS = np.arange(-2, 3, dtype=int)
EPS = 1e-12

FILE_RE = re.compile(
    r"^(?P<experiment>[^_\-]+)[_\-](?P<corridor>[^_\-]+)"
    r"[_\-](?P<location>[^_\-]+)(?:[_\-].*)?$"
)


@dataclass(frozen=True)
class ChirpProjection:
    file_name: str
    corridor_id: int
    location_id: int
    segment: int
    corr_score: float
    bins_complex: np.ndarray
    bins_mag: np.ndarray
    bins_rel_db: np.ndarray


def make_lfm_template() -> np.ndarray:
    count = max(1, int(round(FS_CHIRP * CHIRP_DURATION)))
    time = np.arange(count, dtype=np.float64) / FS_CHIRP
    sweep_rate = CHIRP_BW / CHIRP_DURATION
    phase = 2.0 * np.pi * (
        (-CHIRP_BW / 2.0) * time + 0.5 * sweep_rate * time * time
    )
    return (np.exp(1j * phase) * np.hanning(count)).astype(np.complex64)


def fft_correlate_valid(segment: np.ndarray, reference: np.ndarray) -> np.ndarray:
    valid_length = segment.size - reference.size + 1
    if valid_length <= 0:
        return np.empty(0, dtype=np.complex64)
    correlation_length = segment.size + reference.size - 1
    fft_length = 1 << int(np.ceil(np.log2(correlation_length)))
    kernel = np.conj(reference[::-1])
    correlation = np.fft.ifft(
        np.fft.fft(segment, fft_length) * np.fft.fft(kernel, fft_length)
    )
    return correlation[reference.size - 1 : reference.size - 1 + valid_length]


def parse_chirp_name(file_name: str) -> tuple[int, int]:
    match = FILE_RE.match(Path(file_name).stem)
    if not match:
        raise ValueError(f"Cannot parse chirp file name: {file_name}")
    groups = match.groupdict()
    return int(groups["corridor"]), int(groups["location"])


def lora_waveforms() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.arange(FFT_SIZE, dtype=np.float64)
    upchirp = np.exp(1j * np.pi * n * (n / FFT_SIZE - 1.0))
    downchirp = np.conj(upchirp)
    freqs = np.fft.fftfreq(FFT_SIZE, d=1.0 / LORA_BW)
    return upchirp.astype(np.complex128), downchirp.astype(np.complex128), freqs


def project_cir_to_lora_bins(
    cir: np.ndarray,
    tap_delays_sec: np.ndarray,
    upchirp: np.ndarray,
    downchirp: np.ndarray,
    lora_freqs: np.ndarray,
) -> np.ndarray:
    # Remove arbitrary common phase and scale; the bin shape is what is tested.
    main = int(np.argmax(np.abs(cir)))
    cir = cir * np.exp(-1j * np.angle(cir[main]))
    cir = cir / (np.max(np.abs(cir)) + EPS)

    response = np.exp(
        -2j * np.pi * np.outer(lora_freqs, tap_delays_sec)
    ) @ cir
    tx_spec = np.fft.fft(upchirp)
    rx = np.fft.ifft(tx_spec * response)
    dechirped = rx * downchirp
    spec = np.fft.fft(dechirped)
    peak = int(np.argmax(np.abs(spec)))
    return np.asarray([spec[(peak + off) % FFT_SIZE] for off in BIN_OFFSETS])


def relative_db(mag: np.ndarray) -> np.ndarray:
    ref = float(mag[BIN_OFFSETS.tolist().index(0)])
    return 20.0 * np.log10((mag + EPS) / (ref + EPS))


def load_chirp_segments(max_segments: int, threshold_db: float) -> pd.DataFrame:
    segments = pd.read_csv(SEGMENT_INPUT)
    segments = segments[segments["threshold_db"] == threshold_db].copy()
    segments = segments[~segments["file_name"].str.contains("_fail", regex=False)]
    segments = segments.sort_values(["file_name", "segment"])
    if max_segments > 0:
        segments = segments.groupby("file_name", as_index=False).head(max_segments)
    return segments.reset_index(drop=True)


def synthesize_chirp_bins(
    segments: pd.DataFrame,
    half_window_us: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = make_lfm_template()
    period_length = int(round(FS_CHIRP * PERIOD))
    half_window = max(1, int(round(half_window_us * 1e-6 * FS_CHIRP)))
    tap_offsets = np.arange(-half_window, half_window + 1, dtype=np.int64)
    tap_delays_sec = tap_offsets / FS_CHIRP
    upchirp, downchirp, lora_freqs = lora_waveforms()

    long_records: list[dict] = []
    segment_records: list[dict] = []
    for file_name, group in segments.groupby("file_name", sort=True):
        path = MULTIPATH_ROOT / file_name
        if not path.exists():
            continue
        samples = np.memmap(path, dtype=np.complex64, mode="r")
        corridor_id, location_id = parse_chirp_name(file_name)
        for _, row in group.iterrows():
            segment_id = int(row["segment"])
            start = segment_id * period_length
            stop = min(samples.size, start + period_length + reference.size - 1)
            segment = np.asarray(samples[start:stop], dtype=np.complex64)
            correlation = fft_correlate_valid(segment, reference)
            if correlation.size == 0:
                continue
            detect_index = int(row["detect_index"])
            lo = detect_index - half_window
            hi = detect_index + half_window + 1
            if lo < 0 or hi > correlation.size:
                continue
            cir = np.asarray(correlation[lo:hi], dtype=np.complex128)
            bins = project_cir_to_lora_bins(
                cir, tap_delays_sec, upchirp, downchirp, lora_freqs
            )
            mag = np.abs(bins)
            rel = relative_db(mag)
            item = ChirpProjection(
                file_name=file_name,
                corridor_id=corridor_id,
                location_id=location_id,
                segment=segment_id,
                corr_score=float(row["corr_score"]),
                bins_complex=bins,
                bins_mag=mag,
                bins_rel_db=rel,
            )
            segment_records.append(
                {
                    "file_name": item.file_name,
                    "corridor_id": item.corridor_id,
                    "location_id": item.location_id,
                    "segment": item.segment,
                    "corr_score": item.corr_score,
                    **{
                        f"synth_mag_bin_{off:+d}": float(value)
                        for off, value in zip(BIN_OFFSETS, item.bins_mag)
                    },
                    **{
                        f"synth_rel_db_bin_{off:+d}": float(value)
                        for off, value in zip(BIN_OFFSETS, item.bins_rel_db)
                    },
                    **{
                        f"synth_phase_bin_{off:+d}": float(np.angle(value))
                        for off, value in zip(BIN_OFFSETS, item.bins_complex)
                    },
                }
            )
            for off, value, mag_value, rel_value in zip(
                BIN_OFFSETS, item.bins_complex, item.bins_mag, item.bins_rel_db
            ):
                long_records.append(
                    {
                        "source": "chirp_synth",
                        "file_name": item.file_name,
                        "corridor_id": item.corridor_id,
                        "location_id": item.location_id,
                        "segment": item.segment,
                        "bin_offset": int(off),
                        "mag": float(mag_value),
                        "rel_db_to_center": float(rel_value),
                        "phase_rad": float(np.angle(value)),
                    }
                )
    return pd.DataFrame(segment_records), pd.DataFrame(long_records)


def summarize_synth_segments(segment_bins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (corridor_id, location_id), group in segment_bins.groupby(
        ["corridor_id", "location_id"], sort=True
    ):
        row = {
            "corridor_id": int(corridor_id),
            "location_id": int(location_id),
            "chirp_segment_count": int(len(group)),
            "chirp_corr_score_mean": float(group["corr_score"].mean()),
        }
        for off in BIN_OFFSETS:
            row[f"synth_mag_bin_{off:+d}_mean"] = float(
                group[f"synth_mag_bin_{off:+d}"].mean()
            )
            row[f"synth_rel_db_bin_{off:+d}_mean"] = float(
                group[f"synth_rel_db_bin_{off:+d}"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_measured_lora() -> pd.DataFrame:
    lora = pd.read_csv(LORA_INPUT)
    rows = []
    for (corridor_id, location_id), group in lora.groupby(
        ["corridor_id", "position_id"], sort=True
    ):
        row = {
            "corridor_id": int(corridor_id),
            "location_id": int(location_id),
            "lora_packet_count": int(len(group)),
            "lora_detect_score_db_mean": float(group["detect_score_db"].mean()),
        }
        mag_stack = []
        for off in BIN_OFFSETS:
            col = f"preamble_fft_mag_bin_{off:+d}"
            value = group[col].astype(float).mean()
            row[f"meas_mag_bin_{off:+d}_mean"] = float(value)
            mag_stack.append(float(value))
        rel = relative_db(np.asarray(mag_stack, dtype=np.float64))
        for off, value in zip(BIN_OFFSETS, rel):
            row[f"meas_rel_db_bin_{off:+d}_mean"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def vector_from_row(row: pd.Series, prefix: str, suffix: str) -> np.ndarray:
    return np.asarray(
        [float(row[f"{prefix}_{off:+d}{suffix}"]) for off in BIN_OFFSETS],
        dtype=np.float64,
    )


def compare_windows(synth: pd.DataFrame, measured: pd.DataFrame) -> pd.DataFrame:
    joined = synth.merge(measured, on=["corridor_id", "location_id"], how="inner")
    records = []
    for _, row in joined.iterrows():
        synth_rel = vector_from_row(row, "synth_rel_db_bin", "_mean")
        meas_rel = vector_from_row(row, "meas_rel_db_bin", "_mean")
        synth_mag = vector_from_row(row, "synth_mag_bin", "_mean")
        meas_mag = vector_from_row(row, "meas_mag_bin", "_mean")
        synth_mag_norm = synth_mag / (np.linalg.norm(synth_mag) + EPS)
        meas_mag_norm = meas_mag / (np.linalg.norm(meas_mag) + EPS)
        rel_rmse = float(np.sqrt(np.mean((synth_rel - meas_rel) ** 2)))
        mag_cosine = float(np.dot(synth_mag_norm, meas_mag_norm))
        pearson = float(pearsonr(synth_rel, meas_rel).statistic)
        spearman = float(spearmanr(synth_rel, meas_rel).statistic)
        record = row.to_dict()
        record.update(
            {
                "rel_db_rmse": rel_rmse,
                "mag_shape_cosine": mag_cosine,
                "rel_db_pearson": pearson,
                "rel_db_spearman": spearman,
            }
        )
        records.append(record)
    return pd.DataFrame(records).sort_values(["corridor_id", "location_id"])


def write_long_measured(measured: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in measured.iterrows():
        for off in BIN_OFFSETS:
            rows.append(
                {
                    "source": "lora_measured",
                    "file_name": "",
                    "corridor_id": int(row["corridor_id"]),
                    "location_id": int(row["location_id"]),
                    "segment": "",
                    "bin_offset": int(off),
                    "mag": float(row[f"meas_mag_bin_{off:+d}_mean"]),
                    "rel_db_to_center": float(
                        row[f"meas_rel_db_bin_{off:+d}_mean"]
                    ),
                    "phase_rad": "",
                }
            )
    return pd.DataFrame(rows)


def write_summary(path: Path, comparison: pd.DataFrame, args: argparse.Namespace) -> None:
    metrics = {
        "synthesized_point_count": int(len(comparison)),
        "assumption": "LoRa 125 kHz subband is centered in the 20 MHz chirp baseband.",
        "chirp_half_window_us": args.half_window_us,
        "max_chirp_segments_per_file": args.max_segments,
        "threshold_db": args.threshold_db,
    }
    if len(comparison):
        metrics.update(
            {
                "median_rel_db_rmse": float(comparison["rel_db_rmse"].median()),
                "mean_rel_db_rmse": float(comparison["rel_db_rmse"].mean()),
                "median_mag_shape_cosine": float(
                    comparison["mag_shape_cosine"].median()
                ),
                "mean_mag_shape_cosine": float(
                    comparison["mag_shape_cosine"].mean()
                ),
                "median_rel_db_pearson": float(
                    comparison["rel_db_pearson"].median()
                ),
                "mean_rel_db_pearson": float(comparison["rel_db_pearson"].mean()),
            }
        )
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-segments", type=int, default=40)
    parser.add_argument("--half-window-us", type=float, default=8.0)
    parser.add_argument("--threshold-db", type=float, default=-25.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments = load_chirp_segments(args.max_segments, args.threshold_db)
    segment_bins, synth_long = synthesize_chirp_bins(
        segments, half_window_us=args.half_window_us
    )
    synth_summary = summarize_synth_segments(segment_bins)
    measured = summarize_measured_lora()
    comparison = compare_windows(synth_summary, measured)
    measured_long = write_long_measured(measured)
    long = pd.concat([synth_long, measured_long], ignore_index=True)

    segment_bins.to_csv(args.output_dir / "01_chirp_synth_segment_bins.csv", index=False)
    synth_summary.to_csv(args.output_dir / "02_chirp_synth_point_bins.csv", index=False)
    measured.to_csv(args.output_dir / "03_lora_measured_point_bins.csv", index=False)
    comparison.to_csv(args.output_dir / "04_chirp_lora_bin_window_comparison.csv", index=False)
    long.to_csv(args.output_dir / "05_chirp_synth_vs_lora_long.csv", index=False)
    write_summary(args.output_dir / "analysis_summary.json", comparison, args)

    readme = [
        "# Chirp-to-LoRa bin projection",
        "",
        "This run extracts complex CIR windows from the 20 MHz chirp captures, "
        "projects the centered 125 kHz LoRa subband through the CIR, synthesizes "
        "dechirped LoRa FFT bin[-2,+2], and compares the point-level mean shape "
        "with measured LoRa preamble bin[-2,+2].",
        "",
        f"- Assumption: LoRa subband center equals chirp baseband center.",
        f"- Compared points: {len(comparison)}",
        f"- Median rel-dB RMSE: {comparison['rel_db_rmse'].median() if len(comparison) else ''}",
        f"- Median magnitude-shape cosine: {comparison['mag_shape_cosine'].median() if len(comparison) else ''}",
        "",
    ]
    (args.output_dir / "README.md").write_text(
        "\n".join(readme), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
