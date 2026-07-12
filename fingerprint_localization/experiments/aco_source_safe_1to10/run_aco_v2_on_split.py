#!/usr/bin/env python3
"""Run model/v3 ACO 2.0 on the 1:10 Gaussian-noise train/val/test split.

This script does not regenerate noisy data or a new split. It consumes the
already-created files under this folder's `data/` directory and writes a
separate `results/aco_v2/` output so the numbers can be compared with the
earlier Gaussian-noise ACO runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
MODEL_V3_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3"
if str(MODEL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_V3_DIR))

import aco_packet_path_v2 as aco2  # noqa: E402


DEFAULT_DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_RESULT_DIR = EXPERIMENT_DIR / "results"
DEFAULT_RSSI_CSV = DEFAULT_DATA_DIR / "noisy_rssi_plus_packet_level_54points.csv"
DEFAULT_SPECTRUM_CSV = DEFAULT_DATA_DIR / "noisy_subbin_spectrum_long.csv"
DEFAULT_SPLIT_CSV = DEFAULT_DATA_DIR / "split_assignments.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "aco_v2"
DEFAULT_METHOD_SUMMARY = DEFAULT_RESULT_DIR / "method_summary.csv"


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_from_split_row(row: dict) -> tuple[str, int]:
    return row["file_stem"], int(float(row["packet_index"]))


def load_split_indices(samples: Sequence[aco2.SegmentPacket], split_csv: Path) -> dict[str, list[int]]:
    split_keys: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for row in read_csv(split_csv):
        split_keys[row["split"]].append(key_from_split_row(row))

    key_to_index = {sample.key: idx for idx, sample in enumerate(samples)}
    missing = {
        name: [key for key in keys if key not in key_to_index]
        for name, keys in split_keys.items()
    }
    missing_count = sum(len(keys) for keys in missing.values())
    if missing_count:
        details = ", ".join(f"{name}={len(keys)}" for name, keys in missing.items() if keys)
        raise ValueError(f"{missing_count} split keys were not found in ACO 2.0 samples: {details}")

    return {
        name: sorted(key_to_index[key] for key in keys)
        for name, keys in split_keys.items()
    }


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        rssi_csv=args.rssi_csv,
        spectrum_csv=args.spectrum_csv,
        chirp_template_csv=args.chirp_template_csv,
        chirp_structure_csv=args.chirp_structure_csv,
        location_csv=args.location_csv,
        output_dir=args.output_dir,
        top_k=args.top_k,
        rssi_class_k=args.rssi_class_k,
        segment_count=args.segment_count,
        ants=args.ants,
        iterations=args.iterations,
        elite_ants=args.elite_ants,
        seed=args.seed,
        rssi_weight=args.rssi_weight,
        bin_weight=args.bin_weight,
        energy_weight=args.energy_weight,
        raw_weight=args.raw_weight,
        q4_weight=args.q4_weight,
        shrinkage_lambda=args.shrinkage_lambda,
        phy_var_c0=args.phy_var_c0,
        phy_var_c1=args.phy_var_c1,
        phy_var_c2=args.phy_var_c2,
        sigma0_sq=args.sigma0_sq,
        min_variance=args.min_variance,
        huber_delta=args.huber_delta,
        logdet_weight=args.logdet_weight,
        normalize_bin_cost=args.normalize_bin_cost,
        garbage_cost=args.garbage_cost,
        lambda0_switch=args.lambda0_switch,
        switch_eta=args.switch_eta,
        lambda_div=args.lambda_div,
        lambda_g=args.lambda_g,
        max_garbage=args.max_garbage,
        garbage_overuse_penalty=args.garbage_overuse_penalty,
        lambda_c=args.lambda_c,
        tau_stay=args.tau_stay,
        tau_switch=args.tau_switch,
        pheromone_power=args.pheromone_power,
        heuristic_power=args.heuristic_power,
        evaporation=args.evaporation,
        min_pheromone=args.min_pheromone,
        aco_temperature=args.aco_temperature,
        q4_shift_grid=args.q4_shift_grid,
        peak_threshold=args.peak_threshold,
        auto_peak_quantile=args.auto_peak_quantile,
        q4_dev_threshold=args.q4_dev_threshold,
        auto_q4_dev_quantile=args.auto_q4_dev_quantile,
        q4_peak_offset_max=args.q4_peak_offset_max,
        q4_peak_to_side_threshold=args.q4_peak_to_side_threshold,
        audit_templates=False,
        audit_template_limit=0,
    )


def evaluate_split(
    samples: Sequence[aco2.SegmentPacket],
    q4_offsets: Sequence[float],
    chirp_shapes,
    chirp_struct,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    split_name: str,
    args: argparse.Namespace,
    leave_one_out_prototypes: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    rng = random.Random(args.seed)
    correct = Counter()
    topk_contains = 0
    predictions = []
    candidate_rows = []
    segment_rows = []
    template_cache = {}
    prototype_cache = {}

    def training_state(indices: Sequence[int]):
        key = tuple(indices)
        if key not in template_cache:
            template_cache[key] = aco2.build_templates(samples, labels, indices, chirp_shapes, chirp_struct, args)
            prototype_cache[key] = aco2.build_segment_prototypes(samples, labels, indices)
        return template_cache[key], prototype_cache[key]

    fixed_templates, fixed_prototypes = training_state(train_indices)
    for test_index in eval_indices:
        sample = samples[test_index]
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        rssi_ranked = aco2.base.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        if leave_one_out_prototypes and test_index in train_indices:
            templates, prototypes = training_state(effective_train)
        else:
            templates, prototypes = fixed_templates, fixed_prototypes
        rssi_pred = candidates[0]
        rssi_costs = {label: score for label, score in rssi_ranked if label in candidates}
        obs_costs, rows = aco2.build_observation_costs_v2(
            sample,
            candidates,
            rssi_costs,
            templates,
            prototypes,
            q4_offsets,
            args,
        )
        result = aco2.run_aco_v2_for_packet(obs_costs, candidates, templates, args, rng)

        topk_contains += int(sample.label in candidates)
        correct["rssi"] += int(rssi_pred == sample.label)
        for key in ["path_mode", "pheromone", "vote"]:
            correct[key] += int(result[f"{key}_label"] == sample.label)

        for row in rows:
            row.update(
                {
                    "split": split_name,
                    "sample_index": test_index,
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                }
            )
            segment_rows.append(row)
        for label in candidates:
            tmpl = templates[label]
            candidate_rows.append(
                {
                    "split": split_name,
                    "sample_index": test_index,
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                    "candidate_label": label,
                    "self_pheromone": result["self_pheromone"].get(label, 0.0),
                    "elite_vote": result["elite_vote"].get(label, 0.0),
                    "template_reliability": tmpl.reliability,
                    "alpha_shrink": tmpl.alpha_shrink,
                    "chirp_source": tmpl.chirp_source,
                }
            )
        predictions.append(
            {
                "split": split_name,
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "true_display": aco2.point_display(sample.label),
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": int(rssi_pred == sample.label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "aco_path_mode_label": result["path_mode_label"],
                "aco_path_mode_correct": int(result["path_mode_label"] == sample.label),
                "aco_pheromone_label": result["pheromone_label"],
                "aco_pheromone_correct": int(result["pheromone_label"] == sample.label),
                "aco_vote_label": result["vote_label"],
                "aco_vote_correct": int(result["vote_label"] == sample.label),
                "best_path_cost": result["best_cost"],
                "best_path_labels": ";".join(result["best_path_labels"]),
                "best_path_garbage_count": result["garbage_count"],
            }
        )

    n = len(predictions)
    segment_total = sum(len(samples[idx].segment_q4_reliable) for idx in eval_indices)
    q4_reliable = sum(sum(1 for flag in samples[idx].segment_q4_reliable if flag) for idx in eval_indices)
    metrics = {
        "method": "aco_v2",
        "split": split_name,
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "segment_count": args.segment_count,
        "rssi_class_k": args.rssi_class_k,
        "rssi_top1_correct": correct["rssi"],
        "rssi_top1_accuracy": correct["rssi"] / n if n else 0.0,
        "rssi_topk_contains_true": topk_contains,
        "rssi_topk_recall": topk_contains / n if n else 0.0,
        "aco_path_mode_correct": correct["path_mode"],
        "aco_path_mode_accuracy": correct["path_mode"] / n if n else 0.0,
        "aco_pheromone_correct": correct["pheromone"],
        "aco_pheromone_accuracy": correct["pheromone"] / n if n else 0.0,
        "aco_vote_correct": correct["vote"],
        "aco_vote_accuracy": correct["vote"] / n if n else 0.0,
        "garbage_state_usage_mean": sum(int(row["best_path_garbage_count"]) for row in predictions) / n if n else 0.0,
        "q4_reliable_segment_count": q4_reliable,
        "segment_count_total": segment_total,
        "q4_reliable_segment_rate": q4_reliable / segment_total if segment_total else 0.0,
    }
    return metrics, predictions, candidate_rows, segment_rows


def write_split_outputs(
    output_dir: Path,
    split_name: str,
    metrics: dict,
    predictions: list[dict],
    candidate_rows: list[dict],
    segment_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / f"{split_name}_predictions.csv", predictions, list(predictions[0].keys()))
    write_csv(output_dir / f"{split_name}_candidate_scores.csv", candidate_rows, list(candidate_rows[0].keys()))
    preferred = [
        "split",
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "segment_index",
        "candidate_label",
        "C_obs",
        "C_R",
        "C_bin",
        "C_bin_raw",
        "C_E",
        "C_W",
        "C_Q",
        "q4_reliable",
    ]
    fields = preferred + sorted({key for row in segment_rows for key in row} - set(preferred))
    write_csv(output_dir / f"{split_name}_segment_costs.csv", segment_rows, fields)


def write_summary_with_aco_v2(result_dir: Path, summary_rows: list[dict], method_summary: Path) -> None:
    existing = read_csv(method_summary) if method_summary.exists() else []
    combined = [row for row in existing if not (row.get("method") == "aco_v2")]
    combined.extend(summary_rows)
    preferred = [
        "method",
        "split",
        "packet_count",
        "location_count",
        "top_k",
        "segment_count",
        "rssi_class_k",
        "rssi_top1_correct",
        "rssi_top1_accuracy",
        "rssi_topk_contains_true",
        "rssi_topk_recall",
        "aco_path_mode_correct",
        "aco_path_mode_accuracy",
        "aco_pheromone_correct",
        "aco_pheromone_accuracy",
        "aco_vote_correct",
        "aco_vote_accuracy",
        "garbage_state_usage_mean",
        "q4_reliable_segment_count",
        "q4_reliable_segment_rate",
    ]
    fields = preferred + sorted({key for row in combined for key in row} - set(preferred))
    write_csv(result_dir / "method_summary_with_aco_v2.csv", combined, fields)


def run(args: argparse.Namespace) -> dict:
    aco_args = build_args(args)
    rssi_packets = aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, thresholds = aco2.base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = aco2.prepare_chirp_fields(aco_args, labels)

    eval_plan = [
        ("train_loocv", split_indices["train"], split_indices["train"]),
        ("val", split_indices["train"], split_indices["val"]),
        ("test", split_indices["train"], split_indices["test"]),
    ]
    summary_rows = []
    for split_name, train_indices, eval_indices in eval_plan:
        metrics, predictions, candidate_rows, segment_rows = evaluate_split(
            samples,
            q4_offsets,
            chirp_shapes,
            chirp_struct,
            train_indices,
            eval_indices,
            split_name,
            aco_args,
            leave_one_out_prototypes=args.leave_one_out_prototypes,
        )
        metrics.update(thresholds)
        write_split_outputs(args.output_dir, split_name, metrics, predictions, candidate_rows, segment_rows)
        summary_rows.append(metrics)

    summary_fields = list(summary_rows[0].keys())
    write_csv(args.output_dir / "aco_v2_summary.csv", summary_rows, summary_fields)
    write_summary_with_aco_v2(args.result_dir, summary_rows, args.method_summary)
    metadata = {
        "method": "ACO 2.0 on gaussian_noise_1to10_split",
        "data_policy": "Consumes existing 1:10 Gaussian-noise augmented CSVs and split_assignments.csv; does not regenerate noise or split.",
        "inputs": {
            "rssi_csv": str(args.rssi_csv),
            "spectrum_csv": str(args.spectrum_csv),
            "split_csv": str(args.split_csv),
            "chirp_template_csv": str(args.chirp_template_csv),
            "chirp_structure_csv": str(args.chirp_structure_csv),
            "location_csv": str(args.location_csv),
        },
        "sample_counts": {
            "aligned": len(samples),
            "train": len(split_indices["train"]),
            "val": len(split_indices["val"]),
            "test": len(split_indices["test"]),
            "locations": len(set(labels)),
        },
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(aco_args).items()
        },
        "symbol_thresholds": thresholds,
        "chirp_template_field": chirp_metadata,
        "summary": summary_rows,
    }
    with (args.output_dir / "aco_v2_split_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method-summary", type=Path, default=DEFAULT_METHOD_SUMMARY)
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--chirp-template-csv", type=Path, default=aco2.DEFAULT_CHIRP_TEMPLATE_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=aco2.DEFAULT_CHIRP_STRUCT_CSV)
    parser.add_argument("--location-csv", type=Path, default=aco2.DEFAULT_LOCATION_CSV)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--segment-count", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--rssi-weight", type=float, default=0.45)
    parser.add_argument("--bin-weight", type=float, default=0.02)
    parser.add_argument("--energy-weight", type=float, default=0.20)
    parser.add_argument("--raw-weight", type=float, default=0.55)
    parser.add_argument("--q4-weight", type=float, default=0.0)
    parser.add_argument("--shrinkage-lambda", type=float, default=8.0)
    parser.add_argument("--phy-var-c0", type=float, default=0.05)
    parser.add_argument("--phy-var-c1", type=float, default=0.50)
    parser.add_argument("--phy-var-c2", type=float, default=1.0)
    parser.add_argument("--sigma0-sq", type=float, default=0.02)
    parser.add_argument("--min-variance", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--logdet-weight", type=float, default=0.05)
    parser.add_argument("--normalize-bin-cost", action="store_true", default=True)
    parser.add_argument("--garbage-cost", type=float, default=1.0)
    parser.add_argument("--lambda0-switch", type=float, default=0.70)
    parser.add_argument("--switch-eta", type=float, default=0.20)
    parser.add_argument("--lambda-div", type=float, default=0.20)
    parser.add_argument("--lambda-g", type=float, default=0.50)
    parser.add_argument("--max-garbage", type=int, default=2)
    parser.add_argument("--garbage-overuse-penalty", type=float, default=4.0)
    parser.add_argument("--lambda-c", type=float, default=0.15)
    parser.add_argument("--tau-stay", type=float, default=1.4)
    parser.add_argument("--tau-switch", type=float, default=0.35)
    parser.add_argument("--pheromone-power", type=float, default=1.0)
    parser.add_argument("--heuristic-power", type=float, default=1.4)
    parser.add_argument("--evaporation", type=float, default=0.25)
    parser.add_argument("--min-pheromone", type=float, default=1e-4)
    parser.add_argument("--aco-temperature", type=float, default=None)
    parser.add_argument("--q4-shift-grid", default="-0.25,0,0.25")
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    parser.add_argument("--leave-one-out-prototypes", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata["sample_counts"], indent=2, ensure_ascii=False))
    for row in metadata["summary"]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {metadata['args']['output_dir']}")


if __name__ == "__main__":
    main()
