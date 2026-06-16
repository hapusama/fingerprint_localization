#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched-filter analysis for LFM chirp sounding captures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DONG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = DONG_ROOT / "outputs" / "captures" / "chime_test_rx_fc32.bin"
DEFAULT_CSV = DONG_ROOT / "outputs" / "analysis" / "chime_test_paths.csv"
DEFAULT_JSON = DONG_ROOT / "outputs" / "analysis" / "chime_test_summary.json"


def make_lfm_template(
    fs: float,
    chirp_bw: float,
    chirp_duration: float,
    amplitude: float = 1.0,
) -> np.ndarray:
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


def pick_peaks(
    values: np.ndarray,
    floor: float,
    min_gap: int,
    max_count: int,
) -> list[int]:
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


def analyze(args: argparse.Namespace) -> dict:
    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(infile)

    x = np.memmap(infile, dtype=np.complex64, mode="r")
    fs = float(args.fs)
    period_len = int(round(fs * args.period))
    ref = make_lfm_template(fs, args.chirp_bw, args.chirp_duration)
    ref_energy = float(np.sum(np.abs(ref) ** 2))
    if period_len <= ref.size:
        raise ValueError("period must be longer than chirp_duration")

    max_segments = x.size // period_len
    if args.max_segments > 0:
        max_segments = min(max_segments, args.max_segments)

    min_gap = max(1, int(round(args.min_gap_us * 1e-6 * fs)))
    pre = max(0, int(round(args.pre_delay_us * 1e-6 * fs)))
    post = max(1, int(round(args.max_delay_us * 1e-6 * fs)))
    threshold_linear = 10.0 ** (args.threshold_db / 20.0)

    rows = []
    good_count = 0
    peak_counts = []
    score_values = []

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

        lo = max(0, detect_index - pre)
        hi = min(corr.size, detect_index + post + 1)
        pdp = np.abs(corr[lo:hi])
        if pdp.size == 0 or detect_score < args.corr_gate:
            peak_counts.append(0)
            rows.append([seg_id, start / fs, detect_score, detect_index, 0, ""])
            continue

        pdp = pdp / max(float(np.max(pdp)), 1e-30)
        peaks = pick_peaks(pdp, threshold_linear, min_gap, args.max_peaks)
        good_count += 1
        peak_counts.append(len(peaks))
        tap_text = []
        for peak in peaks:
            rel_delay_us = (lo + peak - detect_index) / fs * 1e6
            amp_db = 20.0 * np.log10(float(pdp[peak]) + 1e-30)
            tap_text.append(f"{rel_delay_us:.3f}us:{amp_db:.2f}dB")
        rows.append([seg_id, start / fs, detect_score, detect_index, len(peaks), ";".join(tap_text)])

    out_csv = Path(args.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["segment", "time_sec", "corr_score", "detect_index", "significant_peak_count", "peaks"])
        writer.writerows(rows)

    summary = {
        "infile": str(infile),
        "fs_hz": fs,
        "chirp_bw_hz": args.chirp_bw,
        "chirp_duration_s": args.chirp_duration,
        "period_s": args.period,
        "segments": int(max_segments),
        "trusted_segments": int(good_count),
        "corr_gate": args.corr_gate,
        "threshold_db": args.threshold_db,
        "mean_corr_score": float(np.mean(score_values)) if score_values else 0.0,
        "max_corr_score": float(np.max(score_values)) if score_values else 0.0,
        "mean_significant_peak_count": float(np.mean(peak_counts)) if peak_counts else 0.0,
        "peak_count_histogram": {
            str(k): int(np.count_nonzero(np.asarray(peak_counts) == k))
            for k in range(0, (max(peak_counts) if peak_counts else 0) + 1)
        },
        "csv_out": str(out_csv),
    }
    out_json = Path(args.json_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LFM chirp sounding IQ")
    parser.add_argument("--infile", default=str(DEFAULT_CAPTURE))
    parser.add_argument("--fs", type=float, default=20e6)
    parser.add_argument("--chirp-bw", type=float, default=18e6)
    parser.add_argument("--chirp-duration", type=float, default=1e-3)
    parser.add_argument("--period", type=float, default=20e-3)
    parser.add_argument("--corr-gate", type=float, default=0.25)
    parser.add_argument("--threshold-db", type=float, default=-20.0)
    parser.add_argument("--min-gap-us", type=float, default=0.10)
    parser.add_argument("--pre-delay-us", type=float, default=1.0)
    parser.add_argument("--max-delay-us", type=float, default=8.0)
    parser.add_argument("--max-peaks", type=int, default=8)
    parser.add_argument("--max-segments", type=int, default=0, help="0 means process all full periods")
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(f"[analyze] segments: {summary['segments']}")
    print(f"[analyze] trusted: {summary['trusted_segments']}")
    print(f"[analyze] max corr: {summary['max_corr_score']:.3f}")
    print(f"[analyze] mean significant peaks: {summary['mean_significant_peak_count']:.2f}")
    print(f"[analyze] csv: {summary['csv_out']}")


if __name__ == "__main__":
    main()
