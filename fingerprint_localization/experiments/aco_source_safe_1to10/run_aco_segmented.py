#!/usr/bin/env python3
"""Run segmented-preamble ACO on the existing 1:10 augmented split.

The baseline ACO treats each packet as a 16-layer path, one layer per preamble
symbol. This script groups adjacent preamble symbols into 8 or 4 segment layers,
rebuilds segment observations, and reruns the same ACO search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
PKG_ROOT = PROJECT_ROOT / "fingerprint_localization"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import run_experiment as re  # noqa: E402
from model.v3 import aco_packet_path as aco  # noqa: E402


def read_split(path: Path) -> dict[str, list[tuple[str, int]]]:
    split = {"train": [], "val": [], "test": []}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split[row["split"]].append((row["file_stem"], int(row["packet_index"])))
    return split


def median(values: Sequence[float]) -> float:
    return aco.median(list(values))


def mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def segment_bounds(n_symbols: int, segment_count: int) -> list[tuple[int, int]]:
    if segment_count <= 0:
        raise ValueError("segment_count must be positive")
    if n_symbols < segment_count:
        raise ValueError(f"Cannot split {n_symbols} symbols into {segment_count} nonempty segments")
    bounds = []
    for idx in range(segment_count):
        start = round(idx * n_symbols / segment_count)
        end = round((idx + 1) * n_symbols / segment_count)
        if end <= start:
            end = start + 1
        bounds.append((start, min(end, n_symbols)))
    return bounds


def rebuild_segmented_samples(
    samples: Sequence[aco.PacketSample],
    q4_offsets: Sequence[float],
    segment_count: int,
    auto_peak_quantile: float,
    auto_q4_dev_quantile: float,
    q4_peak_offset_max: float,
    q4_peak_to_side_threshold: float,
) -> tuple[list[aco.PacketSample], dict]:
    provisional = []
    all_centers = []
    all_devs = []

    for sample in samples:
        symbols = sorted(sample.symbols, key=lambda item: item.symbol_index)
        segments = []
        for segment_index, (start, end) in enumerate(segment_bounds(len(symbols), segment_count)):
            chunk = symbols[start:end]
            raw_bins = [
                median(symbol.raw_bins[offset_idx] for symbol in chunk)
                for offset_idx in range(len(aco.RAW_OFFSETS))
            ]
            q4_curve = [
                median(symbol.q4_curve[offset_idx] for symbol in chunk)
                for offset_idx in range(len(q4_offsets))
            ]
            peak_idx = max(range(len(q4_curve)), key=lambda idx: q4_curve[idx])
            side_db = [db for db, offset in zip(q4_curve, q4_offsets) if abs(offset) >= 1.0]
            peak_to_side = max(q4_curve) - mean(side_db) if side_db else 0.0
            segments.append(
                {
                    "symbol_index": segment_index,
                    "raw_bins": raw_bins,
                    "zw": aco.raw_structure(raw_bins),
                    "q4_curve": q4_curve,
                    "q4_peak_offset": q4_offsets[peak_idx],
                    "q4_peak_to_side_db": peak_to_side,
                    "source_symbol_start": chunk[0].symbol_index,
                    "source_symbol_end": chunk[-1].symbol_index,
                }
            )
            all_centers.append(raw_bins[2])

        packet_median = [
            median(segment["q4_curve"][offset_idx] for segment in segments)
            for offset_idx in range(len(q4_offsets))
        ]
        for segment in segments:
            dev = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(segment["q4_curve"], packet_median))
                / len(q4_offsets)
            )
            segment["q4_dev_from_packet"] = dev
            all_devs.append(dev)
        provisional.append((sample, segments))

    peak_threshold = aco.quantile(all_centers, auto_peak_quantile)
    q4_dev_threshold = aco.quantile(all_devs, auto_q4_dev_quantile)

    segmented_samples = []
    for sample, segments in provisional:
        out_symbols = []
        for segment in segments:
            reliable = (
                abs(segment["q4_peak_offset"]) < q4_peak_offset_max
                and segment["raw_bins"][2] > peak_threshold
                and segment["q4_peak_to_side_db"] > q4_peak_to_side_threshold
                and segment["q4_dev_from_packet"] < q4_dev_threshold
            )
            out_symbols.append(
                aco.SymbolObservation(
                    symbol_index=segment["symbol_index"],
                    raw_bins=segment["raw_bins"],
                    zw=segment["zw"],
                    q4_curve=segment["q4_curve"],
                    q4_peak_offset=segment["q4_peak_offset"],
                    q4_peak_to_side_db=segment["q4_peak_to_side_db"],
                    q4_dev_from_packet=segment["q4_dev_from_packet"],
                    q4_reliable=reliable,
                )
            )
        segmented_samples.append(
            aco.PacketSample(
                key=sample.key,
                file_name=sample.file_name,
                packet_index=sample.packet_index,
                label=sample.label,
                rssi_plus=sample.rssi_plus,
                symbols=out_symbols,
            )
        )

    thresholds = {
        "segment_count": segment_count,
        "source_symbol_count": len(samples[0].symbols) if samples else 0,
        "symbols_per_segment": (len(samples[0].symbols) / segment_count) if samples else 0,
        "peak_threshold": peak_threshold,
        "q4_dev_threshold": q4_dev_threshold,
        "q4_peak_offset_max": q4_peak_offset_max,
        "q4_peak_to_side_threshold": q4_peak_to_side_threshold,
    }
    return segmented_samples, thresholds


def metric_fields(rows: Sequence[dict]) -> list[str]:
    preferred = [
        "method",
        "split",
        "segment_count",
        "source_symbol_count",
        "symbols_per_segment",
        "ants",
        "packet_count",
        "location_count",
        "top_k",
        "rssi_top1_accuracy",
        "rssi_topk_recall",
        "aco_path_mode_accuracy",
        "aco_pheromone_accuracy",
        "aco_vote_accuracy",
        "q4_reliable_symbol_rate",
        "peak_threshold",
        "q4_dev_threshold",
        "q4_peak_offset_max",
        "q4_peak_to_side_threshold",
        "symbol_count",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    return preferred + extras


def run_segment_count(segment_count: int, base_samples, q4_offsets, split_indices, chirp_priors, args):
    segmented_samples, thresholds = rebuild_segmented_samples(
        base_samples,
        q4_offsets,
        segment_count=segment_count,
        auto_peak_quantile=0.10,
        auto_q4_dev_quantile=0.75,
        q4_peak_offset_max=0.50,
        q4_peak_to_side_threshold=6.0,
    )
    segmented_indices = re.indices_for_split(segmented_samples, split_indices)
    aco_args = re.aco_args_from_cli(args)

    out_dir = EXPERIMENT_DIR / "results" / f"aco_segments{segment_count}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for split_name in ["train_loocv", "val", "test"]:
        train_idx = segmented_indices["train"]
        eval_idx = segmented_indices["train"] if split_name == "train_loocv" else segmented_indices[split_name]
        metrics, predictions = re.evaluate_aco_fixed_train(
            segmented_samples,
            q4_offsets,
            chirp_priors,
            thresholds,
            train_idx,
            eval_idx,
            aco_args,
        )
        metrics = {
            "method": f"aco_segments{segment_count}",
            "split": split_name,
            "segment_count": segment_count,
            "source_symbol_count": thresholds["source_symbol_count"],
            "symbols_per_segment": thresholds["symbols_per_segment"],
            "ants": args.ants,
            **metrics,
        }
        summary.append(metrics)
        with (out_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        if predictions:
            re.write_csv(out_dir / f"{split_name}_predictions.csv", predictions, list(predictions[0].keys()))

    re.write_csv(out_dir / f"aco_segments{segment_count}_summary.csv", summary, metric_fields(summary))
    with (out_dir / "segment_config.json").open("w", encoding="utf-8") as f:
        json.dump({"segment_count": segment_count, "thresholds": thresholds}, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", default="8,4")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--aco-iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    paths = {
        "rssi": EXPERIMENT_DIR / "data" / "noisy_rssi_plus_packet_level_54points.csv",
        "raw": EXPERIMENT_DIR / "data" / "noisy_lora_frequency_s17_54points.csv",
        "spectrum": EXPERIMENT_DIR / "data" / "noisy_subbin_spectrum_long.csv",
    }
    base_samples, q4_offsets, _base_thresholds = re.build_aco_samples(paths, args)
    split = read_split(EXPERIMENT_DIR / "data" / "split_assignments.csv")
    chirp_priors = aco.read_chirp_priors(re.DEFAULT_INPUTS["chirp"])

    all_summary = []
    for segment_count in [int(item.strip()) for item in args.segments.split(",") if item.strip()]:
        all_summary.extend(run_segment_count(segment_count, base_samples, q4_offsets, split, chirp_priors, args))

    compare_dir = EXPERIMENT_DIR / "results" / "aco_segment_count_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    re.write_csv(compare_dir / "aco_segment_count_compare_summary.csv", all_summary, metric_fields(all_summary))
    print(json.dumps(all_summary, indent=2, ensure_ascii=False))
    print(f"Wrote {compare_dir}")


if __name__ == "__main__":
    main()
