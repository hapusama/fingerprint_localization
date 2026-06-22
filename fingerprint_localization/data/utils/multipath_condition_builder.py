from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch


PEAK_RE = re.compile(r"(?P<delay>-?\d+(?:\.\d+)?)us:(?P<power>-?\d+(?:\.\d+)?)dB")
FILE_RE = re.compile(r"^(?P<experiment>[^_\-]+)[_\-](?P<corridor>[^_\-]+)[_\-](?P<position>[^_\-]+)(?:[_\-].*)?$")
EPS = 1e-12


def clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return text


def parse_file_meta(path: Path) -> dict:
    match = FILE_RE.match(path.stem)
    if not match:
        return {"experiment_id": "", "corridor_id": "", "position_id": path.stem}
    groups = match.groupdict()
    return {
        "experiment_id": groups["experiment"],
        "corridor_id": groups["corridor"],
        "position_id": groups["position"],
    }


def collect_bin_files(inputs: Sequence[Path], glob_pattern: str) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        if input_path.is_file() and input_path.suffix.lower() == ".bin":
            files.append(input_path)
        elif input_path.is_dir():
            files.extend(sorted(input_path.rglob(glob_pattern)))
        else:
            raise FileNotFoundError(input_path)

    cleaned = []
    for path in files:
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        cleaned.append(path.resolve())
    unique = sorted(set(cleaned))
    if not unique:
        raise FileNotFoundError(f"No .bin files matched {inputs} / {glob_pattern}")
    return unique


def make_lfm_template(fs: float, chirp_bw: float, chirp_duration: float, amplitude: float = 1.0) -> np.ndarray:
    n = max(1, int(round(fs * chirp_duration)))
    t = np.arange(n, dtype=np.float64) / fs
    sweep_rate = chirp_bw / chirp_duration
    phase = 2.0 * np.pi * ((-chirp_bw / 2.0) * t + 0.5 * sweep_rate * t * t)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def fft_correlate_valid(segment: np.ndarray, ref: np.ndarray) -> np.ndarray:
    valid_len = segment.size - ref.size + 1
    if valid_len <= 0:
        return np.empty(0, dtype=np.complex64)
    n_corr = segment.size + ref.size - 1
    n_fft = 1 << int(np.ceil(np.log2(n_corr)))
    kernel = np.conj(ref[::-1])
    corr = np.fft.ifft(np.fft.fft(segment, n_fft) * np.fft.fft(kernel, n_fft))
    return corr[ref.size - 1 : ref.size - 1 + valid_len]


def local_energy_valid(segment: np.ndarray, ref_len: int) -> np.ndarray:
    power = np.abs(segment).astype(np.float64) ** 2
    prefix = np.concatenate(([0.0], np.cumsum(power)))
    return prefix[ref_len:] - prefix[:-ref_len]


def pick_peaks(values: np.ndarray, floor: float, min_gap: int, max_count: int) -> list[int]:
    if values.size < 3:
        return []
    candidates = np.flatnonzero(
        (values[1:-1] >= floor)
        & (values[1:-1] >= values[:-2])
        & (values[1:-1] >= values[2:])
    ) + 1
    if candidates.size == 0:
        return []
    candidates = candidates[np.argsort(values[candidates])[::-1]]
    selected: list[int] = []
    for idx in candidates:
        if all(abs(int(idx) - old) >= min_gap for old in selected):
            selected.append(int(idx))
            if len(selected) >= max_count:
                break
    return sorted(selected)


def parse_taps(peaks_text: str) -> tuple[np.ndarray, np.ndarray]:
    delays = []
    powers_db = []
    for match in PEAK_RE.finditer(str(peaks_text)):
        delays.append(float(match.group("delay")))
        powers_db.append(float(match.group("power")))
    if not delays:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.asarray(delays, dtype=np.float64), np.asarray(powers_db, dtype=np.float64)


def tap_metrics(peaks_text: str) -> dict:
    delays_us, powers_db = parse_taps(peaks_text)
    if delays_us.size == 0:
        return {
            "multipath_tap_count": 0.0,
            "multipath_excess_delay_us": 0.0,
            "multipath_rms_delay_us": 0.0,
            "multipath_power_spread_db": 0.0,
            "multipath_secondary_power_db": -60.0,
        }

    weights = 10.0 ** (powers_db / 10.0)
    weights = weights / max(float(weights.sum()), EPS)
    mean_delay = float(np.sum(weights * delays_us))
    rms_delay = float(np.sqrt(np.sum(weights * (delays_us - mean_delay) ** 2)))
    secondary_db = float(np.max(powers_db[1:])) if powers_db.size > 1 else -60.0
    return {
        "multipath_tap_count": float(delays_us.size),
        "multipath_excess_delay_us": float(np.max(delays_us) - np.min(delays_us)),
        "multipath_rms_delay_us": rms_delay,
        "multipath_power_spread_db": float(np.max(powers_db) - np.min(powers_db)),
        "multipath_secondary_power_db": secondary_db,
    }


def analyze_file(path: Path, args: argparse.Namespace, ref: np.ndarray) -> tuple[pd.DataFrame, dict]:
    x = np.memmap(path, dtype=np.complex64, mode="r")
    fs = float(args.fs)
    period_len = int(round(fs * args.period))
    if period_len <= ref.size:
        raise ValueError("period must be longer than chirp-duration")

    max_segments = x.size // period_len
    if args.max_segments > 0:
        max_segments = min(max_segments, args.max_segments)

    min_gap = max(1, int(round(args.min_gap_us * 1e-6 * fs)))
    pre = max(0, int(round(args.pre_delay_us * 1e-6 * fs)))
    post = max(1, int(round(args.max_delay_us * 1e-6 * fs)))
    threshold_linear = 10.0 ** (args.threshold_db / 20.0)
    ref_energy = float(np.sum(np.abs(ref) ** 2))

    rows = []
    score_values = []
    trusted = 0
    meta = parse_file_meta(path)

    for seg_id in range(max_segments):
        start = seg_id * period_len
        stop = min(x.size, start + period_len + ref.size - 1)
        segment = np.asarray(x[start:stop], dtype=np.complex64)
        corr = fft_correlate_valid(segment, ref)
        if corr.size == 0:
            continue
        local_energy = local_energy_valid(segment, ref.size)
        energy_floor = max(float(np.max(local_energy)) * 1e-6, 1e-20)
        valid_energy = local_energy >= energy_floor
        score = np.zeros(corr.size, dtype=np.float64)
        denom = np.sqrt(local_energy[valid_energy] * ref_energy)
        score[valid_energy] = np.abs(corr[valid_energy]) / denom
        detect_index = int(np.argmax(score))
        detect_score = float(score[detect_index])
        score_values.append(detect_score)

        peaks_text = ""
        peak_count = 0
        if detect_score >= args.corr_gate:
            lo = max(0, detect_index - pre)
            hi = min(corr.size, detect_index + post + 1)
            pdp = np.abs(corr[lo:hi])
            if pdp.size:
                pdp = pdp / max(float(np.max(pdp)), EPS)
                peaks = pick_peaks(pdp, threshold_linear, min_gap, args.max_peaks)
                tap_text = []
                for peak in peaks:
                    rel_delay_us = (lo + peak - detect_index) / fs * 1e6
                    amp_db = 20.0 * np.log10(float(pdp[peak]) + EPS)
                    tap_text.append(f"{rel_delay_us:.3f}us:{amp_db:.2f}dB")
                peaks_text = ";".join(tap_text)
                peak_count = len(peaks)
                trusted += 1

        row = {
            "file_name": path.name,
            "segment": seg_id,
            "time_sec": start / fs,
            "corr_score": detect_score,
            "detect_index": detect_index,
            "significant_peak_count": peak_count,
            "peaks": peaks_text,
            **meta,
            **tap_metrics(peaks_text),
        }
        rows.append(row)

    summary = {
        "file_name": path.name,
        "segments": int(max_segments),
        "trusted_segments": int(trusted),
        "mean_corr_score": float(np.mean(score_values)) if score_values else 0.0,
        "max_corr_score": float(np.max(score_values)) if score_values else 0.0,
        **meta,
    }
    return pd.DataFrame(rows), summary


def normalize_columns(df: pd.DataFrame, columns: Sequence[str]) -> tuple[pd.DataFrame, list[str], dict]:
    df = df.copy()
    normalized_columns = []
    stats = {}
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").fillna(0.0).astype(float)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std <= 0:
            std = 1.0
        out_col = f"{column}_norm"
        df[out_col] = ((values - mean) / (std + EPS)).astype(np.float32)
        normalized_columns.append(out_col)
        stats[column] = {"mean": mean, "std": std}
    return df, normalized_columns, stats


def write_augmented_location_vector(
    location_vector_path: Path,
    output_path: Path,
    condition_df: pd.DataFrame,
    condition_columns: Sequence[str],
    label_mode: str,
) -> None:
    loc_df = pd.read_csv(location_vector_path)
    loc_df["_join_key"] = loc_df["idx"].map(lambda x: clean_id(x)) if label_mode == "idx" else loc_df["location_id"].map(clean_id)
    cond = condition_df.copy()
    cond["_join_key"] = cond["label_key"].map(clean_id)
    merged = loc_df.merge(cond[["_join_key", *condition_columns]], on="_join_key", how="left")
    for column in condition_columns:
        merged[column] = merged[column].fillna(0.0)
    merged = merged.drop(columns=["_join_key"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


def build_condition_table(segment_df: pd.DataFrame, label_mode: str) -> pd.DataFrame:
    df = segment_df.copy()
    df["label_key"] = df["position_id"].map(clean_id)

    metric_columns = [
        "multipath_tap_count",
        "multipath_excess_delay_us",
        "multipath_rms_delay_us",
        "multipath_power_spread_db",
        "multipath_secondary_power_db",
        "corr_score",
    ]
    grouped = df.groupby("label_key", as_index=False)[metric_columns].mean()
    grouped = grouped.rename(columns={"corr_score": "multipath_corr_score"})
    return grouped


def save_condition_artifacts(
    segment_df: pd.DataFrame,
    summaries: list[dict],
    args: argparse.Namespace,
) -> None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    segment_df.to_csv(args.output_csv, index=False)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    condition_df = build_condition_table(segment_df, args.label_mode)
    raw_condition_columns = [
        "multipath_tap_count",
        "multipath_excess_delay_us",
        "multipath_rms_delay_us",
        "multipath_power_spread_db",
        "multipath_secondary_power_db",
        "multipath_corr_score",
    ]
    condition_df, normalized_columns, stats = normalize_columns(condition_df, raw_condition_columns)
    condition_df.to_csv(args.condition_csv, index=False)

    metadata = {
        "source_bins": [str(path) for path in collect_bin_files(args.input, args.glob)],
        "label_mode": args.label_mode,
        "raw_condition_columns": raw_condition_columns,
        "condition_columns": normalized_columns,
        "normalization": stats,
    }
    Path(args.condition_metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_augmented_location_vector(
        args.location_vector,
        args.output_location_vector,
        condition_df,
        normalized_columns,
        args.label_mode,
    )

    print(f"Wrote segment metrics     -> {args.output_csv}")
    print(f"Wrote per-label condition -> {args.condition_csv}")
    print(f"Wrote augmented vector    -> {args.output_location_vector}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze wideband chirp multipath .bin captures and append condition features to location vectors."
    )
    parser.add_argument("--input", type=Path, nargs="+", default=[Path("../dong/data_analysis")])
    parser.add_argument("--glob", default="*.bin")
    parser.add_argument("--location-vector", type=Path, default=Path("model/v1/output/location_vector_v2.csv"))
    parser.add_argument("--output-location-vector", type=Path, default=Path("model/v1/output/location_vector_multipath.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processedData/multipath_segments.csv"))
    parser.add_argument("--condition-csv", type=Path, default=Path("data/processedData/multipath_conditions.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("data/processedData/multipath_summaries.json"))
    parser.add_argument("--condition-metadata", type=Path, default=Path("data/processedData/multipath_condition_metadata.json"))
    parser.add_argument("--label-mode", choices=("idx", "location_id"), default="location_id")
    parser.add_argument("--fs", type=float, default=20e6)
    parser.add_argument("--chirp-bw", type=float, default=18e6)
    parser.add_argument("--chirp-duration", type=float, default=1e-3)
    parser.add_argument("--period", type=float, default=20e-3)
    parser.add_argument("--corr-gate", type=float, default=0.08)
    parser.add_argument("--threshold-db", type=float, default=-20.0)
    parser.add_argument("--min-gap-us", type=float, default=0.10)
    parser.add_argument("--pre-delay-us", type=float, default=1.0)
    parser.add_argument("--max-delay-us", type=float, default=8.0)
    parser.add_argument("--max-peaks", type=int, default=8)
    parser.add_argument("--max-segments", type=int, default=100, help="0 means all full periods.")
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bin_files = collect_bin_files(args.input, args.glob)
    if args.max_files > 0:
        bin_files = bin_files[: args.max_files]

    ref = make_lfm_template(args.fs, args.chirp_bw, args.chirp_duration)
    frames = []
    summaries = []
    for index, path in enumerate(bin_files, start=1):
        print(f"[{index}/{len(bin_files)}] analyzing {path.name}")
        frame, summary = analyze_file(path, args, ref)
        frames.append(frame)
        summaries.append(summary)
        print(
            f"  trusted={summary['trusted_segments']}/{summary['segments']} "
            f"max_corr={summary['max_corr_score']:.3f}"
        )

    segment_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if segment_df.empty:
        raise ValueError("No multipath segments were analyzed.")
    save_condition_artifacts(segment_df, summaries, args)


if __name__ == "__main__":
    main()
