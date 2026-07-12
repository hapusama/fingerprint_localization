#!/usr/bin/env python3
"""Run MFR-ACO on the fixed 1:10 Gaussian-noise train/val/test split."""

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

import mfr_aco_global_multipath as mfr  # noqa: E402


DEFAULT_DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_RESULT_DIR = EXPERIMENT_DIR / "results"
DEFAULT_RSSI_CSV = DEFAULT_DATA_DIR / "noisy_rssi_plus_packet_level_54points.csv"
DEFAULT_SPECTRUM_CSV = DEFAULT_DATA_DIR / "noisy_subbin_spectrum_long.csv"
DEFAULT_SPLIT_CSV = DEFAULT_DATA_DIR / "split_assignments.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_DIR / "mfr_aco"
EPS = 1e-12


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def key_from_split_row(row: dict) -> tuple[str, int]:
    return row["file_stem"], int(float(row["packet_index"]))


def load_split_indices(samples: Sequence[mfr.base.PacketSample], split_csv: Path) -> dict[str, list[int]]:
    split_keys: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for row in read_csv(split_csv):
        split_keys[row["split"]].append(key_from_split_row(row))
    key_to_index = {sample.key: idx for idx, sample in enumerate(samples)}
    missing = {name: [key for key in keys if key not in key_to_index] for name, keys in split_keys.items()}
    missing_count = sum(len(keys) for keys in missing.values())
    if missing_count:
        details = ", ".join(f"{name}={len(keys)}" for name, keys in missing.items() if keys)
        raise ValueError(f"{missing_count} split keys were not found in MFR-ACO samples: {details}")
    return {name: sorted(key_to_index[key] for key in keys) for name, keys in split_keys.items()}


def candidate_scores_for_packet(
    test_index: int,
    candidates: Sequence[str],
    rssi_ranked: Sequence[tuple[str, float]],
    raw_prototypes: dict,
    evidence: Sequence[mfr.PacketEvidence],
    field: dict[str, mfr.MultipathPoint],
    temperatures: dict[str, float],
    args: argparse.Namespace,
) -> tuple[dict, dict, dict, dict, dict, dict, dict, dict]:
    rssi_scores = dict(rssi_ranked)
    t_r = temperatures["t_r"]
    t_w = temperatures["t_w"]
    t_q = temperatures["t_q"]
    t_m = temperatures["t_m"]
    eta_r = {label: math.exp(-rssi_scores[label] / (t_r + EPS)) for label in candidates}
    raw_distances = {
        label: mfr.raw_distance(evidence[test_index].w_pkt, raw_prototypes[label])
        for label in candidates
        if label in raw_prototypes
    }
    eta_w = {label: math.exp(-raw_distances.get(label, t_w) / (t_w + EPS)) for label in candidates}
    q_w = math.exp(-evidence[test_index].q_raw_sum / (t_q + EPS))
    mp = mfr.candidate_multipath_quantities(candidates, field, t_m)
    g_m = {label: q_w * mp[label]["rel"] * mp[label]["sep"] for label in candidates}
    k_m = {
        (left, right): mfr.multipath_kernel(left, right, field, t_m)
        for left in candidates
        for right in candidates
    }
    stage = args.ablation_stage
    if stage < 1:
        eta_w = {label: 1.0 for label in candidates}
        raw_weight = {label: 0.0 for label in candidates}
    elif stage == 1:
        raw_weight = {label: 1.0 for label in candidates}
    elif stage == 2:
        raw_weight = {label: q_w for label in candidates}
    else:
        raw_weight = {label: 1.0 + args.beta * g_m[label] for label in candidates}
    candidate_boost = {
        label: (1.0 + args.beta * g_m[label]) if stage >= 3 else 1.0
        for label in candidates
    }
    transition_boost = {}
    for left in candidates:
        for right in candidates:
            if stage >= 5:
                transition_boost[(left, right)] = (1.0 + g_m[left]) if left == right else k_m[(left, right)]
            else:
                transition_boost[(left, right)] = 1.0
    return eta_r, eta_w, raw_weight, candidate_boost, transition_boost, k_m, mp, {"q_w": q_w, "g_m": g_m}


def evaluate_split(
    samples: Sequence[mfr.base.PacketSample],
    evidence: Sequence[mfr.PacketEvidence],
    field: dict[str, mfr.MultipathPoint],
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    split_name: str,
    args: argparse.Namespace,
    temperatures: dict[str, float],
) -> tuple[dict, list[dict], list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    fixed_raw_prototypes = mfr.build_raw_prototypes(evidence, labels, train_indices)
    rng = random.Random(args.seed)
    predictions = []
    candidate_rows = []
    correct = Counter()
    topk_contains = 0

    for test_index in eval_indices:
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        raw_prototypes = (
            mfr.build_raw_prototypes(evidence, labels, effective_train)
            if split_name == "train_loocv" and test_index in train_indices
            else fixed_raw_prototypes
        )
        rssi_ranked = mfr.base.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        eta_r, eta_w, raw_weight, candidate_boost, transition_boost, k_m, mp, aux = candidate_scores_for_packet(
            test_index,
            candidates,
            rssi_ranked,
            raw_prototypes,
            evidence,
            field,
            temperatures,
            args,
        )
        result = mfr.run_mfr_aco_for_packet(
            candidates,
            eta_r,
            eta_w,
            raw_weight,
            candidate_boost,
            transition_boost,
            k_m,
            args,
            rng,
        )
        true_label = labels[test_index]
        rssi_pred = candidates[0]
        topk_contains += int(true_label in candidates)
        outputs = {
            "rssi_top1": rssi_pred,
            "path_mode": result["path_mode_label"],
            "pheromone": result["pheromone_label"],
            "vote": result["vote_label"],
            "physical": result["physical_label"],
        }
        for name, pred in outputs.items():
            correct[name] += int(pred == true_label)
        predictions.append(
            {
                "split": split_name,
                "sample_index": test_index,
                "file_name": samples[test_index].file_name,
                "packet_index": samples[test_index].packet_index,
                "true_label": true_label,
                "true_display": mfr.point_display(true_label),
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": int(rssi_pred == true_label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(true_label in candidates),
                "q_w": aux["q_w"],
                "aco_path_mode_label": result["path_mode_label"],
                "aco_path_mode_correct": int(result["path_mode_label"] == true_label),
                "aco_pheromone_label": result["pheromone_label"],
                "aco_pheromone_correct": int(result["pheromone_label"] == true_label),
                "aco_vote_label": result["vote_label"],
                "aco_vote_correct": int(result["vote_label"] == true_label),
                "mfr_physical_label": result["physical_label"],
                "mfr_physical_correct": int(result["physical_label"] == true_label),
                "best_cost": result["best_cost"],
                "best_path_labels": result["best_path_labels"],
            }
        )
        for label in candidates:
            item = field.get(label)
            candidate_rows.append(
                {
                    "split": split_name,
                    "sample_index": test_index,
                    "true_label": true_label,
                    "candidate_label": label,
                    "eta_r": eta_r[label],
                    "eta_w": eta_w[label],
                    "q_w": aux["q_w"],
                    "rel_m": mp[label]["rel"],
                    "sep_m": mp[label]["sep"],
                    "g_m": aux["g_m"][label],
                    "raw_weight": raw_weight[label],
                    "candidate_boost": candidate_boost[label],
                    "self_pheromone": result["self_pheromone"][label],
                    "elite_vote": result["elite_vote"][label],
                    "physical_score": result["physical_score"][label],
                    "multipath_confidence": item.confidence if item else 0.0,
                    "multipath_source": item.source if item else "",
                }
            )

    n = len(predictions)
    metrics = {
        "method": "mfr_aco",
        "split": split_name,
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "search_depth": args.search_depth,
        "raw_segments": args.raw_segments,
        "ablation_stage": args.ablation_stage,
        "rssi_class_k": args.rssi_class_k,
        "rssi_top1_correct": correct["rssi_top1"],
        "rssi_top1_accuracy": correct["rssi_top1"] / n if n else 0.0,
        "rssi_topk_contains_true": topk_contains,
        "rssi_topk_recall": topk_contains / n if n else 0.0,
        "aco_path_mode_correct": correct["path_mode"],
        "aco_path_mode_accuracy": correct["path_mode"] / n if n else 0.0,
        "aco_pheromone_correct": correct["pheromone"],
        "aco_pheromone_accuracy": correct["pheromone"] / n if n else 0.0,
        "aco_vote_correct": correct["vote"],
        "aco_vote_accuracy": correct["vote"] / n if n else 0.0,
        "mfr_physical_correct": correct["physical"],
        "mfr_physical_accuracy": correct["physical"] / n if n else 0.0,
        **temperatures,
        "kappa_r": args.kappa_r,
        "kappa_w": args.kappa_w,
        "beta": args.beta,
        "gamma": args.gamma,
        "lambda_m": args.lambda_m,
        "lambda_div": args.lambda_div,
    }
    return metrics, predictions, candidate_rows


def write_split_outputs(output_dir: Path, split_name: str, metrics: dict, predictions: list[dict], candidates: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / f"{split_name}_predictions.csv", predictions, list(predictions[0].keys()))
    write_csv(output_dir / f"{split_name}_candidate_scores.csv", candidates, list(candidates[0].keys()))


def summary_fields(rows: Sequence[dict]) -> list[str]:
    preferred = [
        "method",
        "split",
        "packet_count",
        "location_count",
        "top_k",
        "search_depth",
        "raw_segments",
        "ablation_stage",
        "rssi_class_k",
        "rssi_top1_accuracy",
        "rssi_topk_recall",
        "aco_path_mode_accuracy",
        "aco_pheromone_accuracy",
        "aco_vote_accuracy",
        "mfr_physical_accuracy",
        "t_r",
        "t_w",
        "t_q",
        "t_m",
        "kappa_r",
        "kappa_w",
        "beta",
        "gamma",
        "lambda_m",
        "lambda_div",
    ]
    return preferred + sorted({key for row in rows for key in row} - set(preferred))


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rssi_packets = mfr.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, _q4_offsets, symbol_thresholds = mfr.base.read_symbol_packets(args.spectrum_csv, args)
    samples = mfr.base.align_samples(rssi_packets, symbol_packets)
    if not samples:
        raise RuntimeError("No aligned RSSI+/symbol packet samples found.")
    split_indices = load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    evidence = mfr.build_all_packet_evidence(samples, args.raw_segments)
    field, field_metadata = mfr.build_multipath_field(args.location_csv, args.chirp_csv)
    temperatures = {
        "t_r": mfr.compute_rssi_temperature([sample.rssi_plus for sample in samples], labels, split_indices["train"], args.rssi_class_k),
        "t_w": mfr.compute_raw_temperature(evidence, labels, split_indices["train"]),
        "t_q": mfr.compute_t_q(evidence, split_indices["train"]),
        "t_m": args.multipath_temperature if args.multipath_temperature and args.multipath_temperature > EPS else mfr.compute_multipath_temperature(field, labels),
    }

    mfr.write_multipath_field(args.output_dir / "global_multipath_field.csv", field)
    eval_plan = [
        ("train_loocv", split_indices["train"], split_indices["train"]),
        ("val", split_indices["train"], split_indices["val"]),
        ("test", split_indices["train"], split_indices["test"]),
    ]
    summary = []
    for split_name, train_indices, eval_indices in eval_plan:
        metrics, predictions, candidate_rows = evaluate_split(
            samples,
            evidence,
            field,
            train_indices,
            eval_indices,
            split_name,
            args,
            temperatures,
        )
        metrics.update(symbol_thresholds)
        write_split_outputs(args.output_dir, split_name, metrics, predictions, candidate_rows)
        summary.append(metrics)

    write_csv(args.output_dir / "mfr_aco_split_summary.csv", summary, summary_fields(summary))
    metadata = {
        "method": "MFR-ACO on gaussian_noise_1to10_split",
        "data_policy": "Consumes existing 1:10 Gaussian-noise augmented CSVs and split_assignments.csv; does not regenerate noise or split.",
        "inputs": {
            "rssi_csv": str(args.rssi_csv),
            "spectrum_csv": str(args.spectrum_csv),
            "split_csv": str(args.split_csv),
            "chirp_csv": str(args.chirp_csv),
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
            for key, value in vars(args).items()
        },
        "symbol_thresholds": symbol_thresholds,
        "temperatures": temperatures,
        "multipath_field": field_metadata,
        "summary": summary,
    }
    with (args.output_dir / "mfr_aco_split_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--chirp-csv", type=Path, default=mfr.DEFAULT_CHIRP_CSV)
    parser.add_argument("--location-csv", type=Path, default=mfr.DEFAULT_LOCATION_CSV)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--search-depth", type=int, default=4)
    parser.add_argument("--raw-segments", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--tau-stay", type=float, default=1.4)
    parser.add_argument("--tau-switch", type=float, default=0.35)
    parser.add_argument("--pheromone-power", type=float, default=1.0)
    parser.add_argument("--evaporation", type=float, default=0.25)
    parser.add_argument("--min-pheromone", type=float, default=1e-4)
    parser.add_argument("--aco-temperature", type=float, default=None)
    parser.add_argument("--multipath-temperature", type=float, default=None)
    parser.add_argument("--kappa-r", type=float, default=1.0)
    parser.add_argument("--kappa-w", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-m", type=float, default=1.0)
    parser.add_argument("--lambda-div", type=float, default=0.2)
    parser.add_argument("--ablation-stage", type=int, default=5, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata["sample_counts"], indent=2, ensure_ascii=False))
    for row in metadata["summary"]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {metadata['args']['output_dir']}")


if __name__ == "__main__":
    main()
