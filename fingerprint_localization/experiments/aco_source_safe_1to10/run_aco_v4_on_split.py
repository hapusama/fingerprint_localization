#!/usr/bin/env python3
"""Run ACO 4.0 on the existing 1:10 Gaussian-noise split."""

from __future__ import annotations

import argparse
import csv
import json
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
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import aco_packet_path_v4 as aco4  # noqa: E402
import run_aco_v2_on_split as split_runner  # noqa: E402


DEFAULT_OUTPUT_DIR = split_runner.DEFAULT_RESULT_DIR / "aco_v4"
DEFAULT_METHOD_SUMMARY = split_runner.DEFAULT_RESULT_DIR / "method_summary_with_aco_v4.csv"
DEFAULT_ACO_V2_DIR = split_runner.DEFAULT_RESULT_DIR / "aco_v2"


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


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    out = split_runner.build_args(args)
    out.lambda_rssi_prior = args.lambda_rssi_prior
    out.lambda_raw_prior = args.lambda_raw_prior
    out.lambda_veto = args.lambda_veto
    out.lambda_q_switch = args.lambda_q_switch
    out.t_seg = args.t_seg
    out.t_seg_resolved = args.t_seg
    out.t_seg_quantile = args.t_seg_quantile
    out.garbage_cost_min = args.garbage_cost_min
    out.lambda_garbage_stability = args.lambda_garbage_stability
    out.lambda_score_vote = args.lambda_score_vote
    out.lambda_score_cost = args.lambda_score_cost
    return out


def estimate_t_seg(
    samples: Sequence[aco4.aco2.SegmentPacket],
    q4_offsets: Sequence[float],
    chirp_shapes,
    chirp_struct,
    train_indices: Sequence[int],
    args: argparse.Namespace,
) -> float:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    templates = aco4.aco2.build_templates(samples, labels, train_indices, chirp_shapes, chirp_struct, args)
    prototypes = aco4.aco2.build_segment_prototypes(samples, labels, train_indices)
    values = []
    old_t = args.t_seg_resolved
    args.t_seg_resolved = 1.0
    for test_index in train_indices:
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        ranked = aco4.aco2.base.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in ranked[: args.top_k]]
        if not candidates or any(label not in templates for label in candidates):
            continue
        rssi_costs = {label: score for label, score in ranked if label in candidates}
        _obs, _rows, meta = aco4.build_observation_costs_v4(
            samples[test_index],
            candidates,
            rssi_costs,
            templates,
            prototypes,
            q4_offsets,
            args,
        )
        values.append(meta["segment_cost_std"])
    args.t_seg_resolved = old_t
    if not values:
        return 1.0
    values = sorted(values)
    quantile = min(1.0, max(0.0, args.t_seg_quantile))
    value = values[round((len(values) - 1) * quantile)]
    return value if value > aco4.EPS else 1.0


def evaluate_split(
    samples: Sequence[aco4.aco2.SegmentPacket],
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
            template_cache[key] = aco4.aco2.build_templates(samples, labels, indices, chirp_shapes, chirp_struct, args)
            prototype_cache[key] = aco4.aco2.build_segment_prototypes(samples, labels, indices)
        return template_cache[key], prototype_cache[key]

    fixed_templates, fixed_prototypes = training_state(train_indices)
    for test_index in eval_indices:
        sample = samples[test_index]
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        ranked = aco4.aco2.base.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in ranked[: args.top_k]]
        if not candidates:
            continue
        if leave_one_out_prototypes and test_index in train_indices:
            templates, prototypes = training_state(effective_train)
        else:
            templates, prototypes = fixed_templates, fixed_prototypes
        rssi_pred = candidates[0]
        rssi_costs = {label: score for label, score in ranked if label in candidates}
        obs_costs, rows, meta = aco4.build_observation_costs_v4(
            sample,
            candidates,
            rssi_costs,
            templates,
            prototypes,
            q4_offsets,
            args,
        )
        result = aco4.run_aco_v4_for_packet(obs_costs, candidates, templates, meta, args, rng)

        topk_contains += int(sample.label in candidates)
        correct["rssi"] += int(rssi_pred == sample.label)
        for key in ["path_mode", "pheromone", "vote", "score4"]:
            correct[key] += int(result[f"{key}_label"] == sample.label)

        baseline_label = result["vote_label"]
        final_label = result["score4_label"]
        baseline_correct = baseline_label == sample.label
        final_correct = final_label == sample.label

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
                    "score4": result["score4"].get(label, 0.0),
                    "candidate_mean_obs": meta["candidate_mean_obs"].get(label, 0.0),
                    "candidate_cost_norm": meta["candidate_cost_norm"].get(label, 0.0),
                    "cost_veto": meta["veto"].get(label, 1.0),
                    "is_rssi_top1": int(label == meta["rssi_top1"]),
                    "is_raw_winner": int(label == meta["raw_winner"]),
                    "raw_margin": meta["raw_margin"],
                    "Q_seg": meta["q_seg"],
                    "segment_cost_std": meta["segment_cost_std"],
                    "template_reliability": templates[label].reliability,
                    "alpha_shrink": templates[label].alpha_shrink,
                    "chirp_source": templates[label].chirp_source,
                }
            )
        predictions.append(
            {
                "split": split_name,
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "true_display": aco4.aco2.point_display(sample.label),
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": int(rssi_pred == sample.label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "raw_winner_label": meta["raw_winner"],
                "raw_margin": meta["raw_margin"],
                "Q_seg": meta["q_seg"],
                "segment_cost_std": meta["segment_cost_std"],
                "aco_path_mode_label": result["path_mode_label"],
                "aco_path_mode_correct": int(result["path_mode_label"] == sample.label),
                "aco_pheromone_label": result["pheromone_label"],
                "aco_pheromone_correct": int(result["pheromone_label"] == sample.label),
                "aco_vote_label": result["vote_label"],
                "aco_vote_correct": int(result["vote_label"] == sample.label),
                "aco_score4_label": result["score4_label"],
                "aco_score4_correct": int(result["score4_label"] == sample.label),
                "score4_changed_from_vote": int(final_label != baseline_label),
                "score4_W2R_from_vote": int((not baseline_correct) and final_correct),
                "score4_R2W_from_vote": int(baseline_correct and (not final_correct)),
                "best_path_cost": result["best_cost"],
                "best_path_labels": ";".join(result["best_path_labels"]),
                "best_path_garbage_count": result["garbage_count"],
            }
        )

    n = len(predictions)
    segment_total = sum(len(samples[idx].segment_q4_reliable) for idx in eval_indices)
    q4_reliable = sum(sum(1 for flag in samples[idx].segment_q4_reliable if flag) for idx in eval_indices)
    changed = sum(int(row["score4_changed_from_vote"]) for row in predictions)
    w2r = sum(int(row["score4_W2R_from_vote"]) for row in predictions)
    r2w = sum(int(row["score4_R2W_from_vote"]) for row in predictions)
    metrics = {
        "method": "aco_v4",
        "split": split_name,
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "segment_count": args.segment_count,
        "rssi_class_k": args.rssi_class_k,
        "T_seg": args.t_seg_resolved,
        "lambda_R": args.lambda_rssi_prior,
        "lambda_W": args.lambda_raw_prior,
        "lambda_V": args.lambda_veto,
        "lambda_Q": args.lambda_q_switch,
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
        "aco_score4_correct": correct["score4"],
        "aco_score4_accuracy": correct["score4"] / n if n else 0.0,
        "score4_change_count_from_vote": changed,
        "score4_W2R_from_vote": w2r,
        "score4_R2W_from_vote": r2w,
        "score4_net_from_vote": w2r - r2w,
        "garbage_state_usage_mean": sum(int(row["best_path_garbage_count"]) for row in predictions) / n if n else 0.0,
        "q4_reliable_segment_count": q4_reliable,
        "segment_count_total": segment_total,
        "q4_reliable_segment_rate": q4_reliable / segment_total if segment_total else 0.0,
        "Q_seg_mean": sum(float(row["Q_seg"]) for row in predictions) / n if n else 0.0,
        "segment_cost_std_mean": sum(float(row["segment_cost_std"]) for row in predictions) / n if n else 0.0,
    }
    return metrics, predictions, candidate_rows, segment_rows


def write_split_outputs(output_dir: Path, split_name: str, metrics: dict, predictions: list[dict], candidate_rows: list[dict], segment_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / f"{split_name}_predictions.csv", predictions, list(predictions[0].keys()))
    write_csv(output_dir / f"{split_name}_candidate_scores.csv", candidate_rows, list(candidate_rows[0].keys()))
    preferred = [
        "split", "sample_index", "file_name", "packet_index", "true_label", "segment_index", "candidate_label",
        "C_obs", "C_seg_base", "C_R", "C_bin", "C_bin_raw", "C_E", "C_W", "C_Q", "Q_seg", "segment_cost_std", "T_seg",
        "q4_reliable", "rssi_top1_label", "raw_winner_label", "raw_margin",
    ]
    fields = preferred + sorted({key for row in segment_rows for key in row} - set(preferred))
    write_csv(output_dir / f"{split_name}_segment_costs.csv", segment_rows, fields)


def compare_with_aco_v2(aco_v2_dir: Path, output_dir: Path, split_name: str, predictions: list[dict]) -> dict:
    baseline_path = aco_v2_dir / f"{split_name}_predictions.csv"
    if not baseline_path.exists():
        return {}
    baseline = {row["sample_index"]: row for row in read_csv(baseline_path)}
    rows = []
    changed = 0
    w2r = 0
    r2w = 0
    for pred in predictions:
        sample_index = pred["sample_index"]
        if str(sample_index) not in baseline:
            continue
        base_pred = baseline[str(sample_index)]
        old_label = base_pred["aco_vote_label"]
        new_label = pred["aco_score4_label"]
        true_label = pred["true_label"]
        old_correct = old_label == true_label
        new_correct = new_label == true_label
        if old_label == new_label:
            continue
        changed += 1
        w2r += int((not old_correct) and new_correct)
        r2w += int(old_correct and (not new_correct))
        rows.append(
            {
                "split": split_name,
                "sample_index": sample_index,
                "file_name": pred["file_name"],
                "packet_index": pred["packet_index"],
                "true_label": true_label,
                "rssi_top1_label": pred["rssi_top1_label"],
                "rssi_topk_candidates": pred["rssi_topk_candidates"],
                "aco_v2_vote_label": old_label,
                "aco_v2_vote_correct": int(old_correct),
                "aco_v4_score4_label": new_label,
                "aco_v4_score4_correct": int(new_correct),
                "change_type": (
                    "W2R"
                    if ((not old_correct) and new_correct)
                    else ("R2W" if (old_correct and (not new_correct)) else "changed_same_correctness")
                ),
                "Q_seg": pred["Q_seg"],
                "segment_cost_std": pred["segment_cost_std"],
                "raw_winner_label": pred["raw_winner_label"],
                "raw_margin": pred["raw_margin"],
            }
        )
    fields = list(rows[0].keys()) if rows else [
        "split", "sample_index", "file_name", "packet_index", "true_label", "rssi_top1_label",
        "rssi_topk_candidates", "aco_v2_vote_label", "aco_v2_vote_correct", "aco_v4_score4_label",
        "aco_v4_score4_correct", "change_type", "Q_seg", "segment_cost_std", "raw_winner_label", "raw_margin",
    ]
    write_csv(output_dir / f"{split_name}_changes_vs_aco_v2.csv", rows, fields)
    return {
        "aco_v4_change_count_vs_aco_v2": changed,
        "aco_v4_W2R_vs_aco_v2": w2r,
        "aco_v4_R2W_vs_aco_v2": r2w,
        "aco_v4_net_vs_aco_v2": w2r - r2w,
    }


def write_summary(result_dir: Path, summary_rows: list[dict], method_summary: Path) -> None:
    existing = read_csv(method_summary) if method_summary.exists() else []
    combined = [row for row in existing if row.get("method") != "aco_v4"]
    combined.extend(summary_rows)
    fields = [
        "method", "split", "packet_count", "location_count", "top_k", "segment_count", "rssi_class_k",
        "T_seg", "lambda_R", "lambda_W", "lambda_V", "lambda_Q",
        "rssi_top1_accuracy", "rssi_topk_recall", "aco_vote_accuracy", "aco_score4_accuracy",
        "score4_change_count_from_vote", "score4_W2R_from_vote", "score4_R2W_from_vote", "score4_net_from_vote",
        "aco_v4_change_count_vs_aco_v2", "aco_v4_W2R_vs_aco_v2", "aco_v4_R2W_vs_aco_v2", "aco_v4_net_vs_aco_v2",
        "garbage_state_usage_mean", "Q_seg_mean", "segment_cost_std_mean",
    ]
    fields = fields + sorted({key for row in combined for key in row} - set(fields))
    write_csv(result_dir / "method_summary_with_aco_v4.csv", combined, fields)


def run(args: argparse.Namespace) -> dict:
    aco_args = build_args(args)
    rssi_packets = aco4.aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, thresholds = aco4.aco2.base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = aco4.aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco4.aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = split_runner.load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = aco4.aco2.prepare_chirp_fields(aco_args, labels)
    if aco_args.t_seg_resolved is None:
        aco_args.t_seg_resolved = estimate_t_seg(samples, q4_offsets, chirp_shapes, chirp_struct, split_indices["train"], aco_args)

    full_eval_plan = [
        ("train_loocv", split_indices["train"], split_indices["train"]),
        ("val", split_indices["train"], split_indices["val"]),
        ("test", split_indices["train"], split_indices["test"]),
    ]
    requested_splits = [name.strip() for name in args.splits.split(",") if name.strip()]
    valid_splits = {name for name, _, _ in full_eval_plan}
    unknown_splits = set(requested_splits) - valid_splits
    if not requested_splits or unknown_splits:
        raise ValueError(
            f"--splits must be a comma-separated subset of {sorted(valid_splits)}; "
            f"received {args.splits!r}"
        )
    eval_plan = [item for item in full_eval_plan if item[0] in requested_splits]
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
        metrics.update(compare_with_aco_v2(args.aco_v2_dir, args.output_dir, split_name, predictions))
        write_split_outputs(args.output_dir, split_name, metrics, predictions, candidate_rows, segment_rows)
        summary_rows.append(metrics)

    write_csv(args.output_dir / "aco_v4_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_summary(args.result_dir, summary_rows, args.method_summary)
    metadata = {
        "method": "ACO 4.0 evidence-reliability aware ACO on gaussian_noise_1to10_split",
        "source": "fingerprint_localization/docs/mainline_202607/PAPER_MAINLINE_ALGORITHM_DOC.md",
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
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(aco_args).items()},
        "symbol_thresholds": thresholds,
        "chirp_template_field": chirp_metadata,
        "summary": summary_rows,
    }
    with (args.output_dir / "aco_v4_split_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=split_runner.DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--method-summary", type=Path, default=DEFAULT_METHOD_SUMMARY)
    parser.add_argument("--aco-v2-dir", type=Path, default=DEFAULT_ACO_V2_DIR)
    parser.add_argument("--rssi-csv", type=Path, default=split_runner.DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=split_runner.DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=split_runner.DEFAULT_SPLIT_CSV)
    parser.add_argument("--chirp-template-csv", type=Path, default=aco4.aco2.DEFAULT_CHIRP_TEMPLATE_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=aco4.aco2.DEFAULT_CHIRP_STRUCT_CSV)
    parser.add_argument("--location-csv", type=Path, default=aco4.aco2.DEFAULT_LOCATION_CSV)
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
    parser.add_argument("--garbage-cost-min", type=float, default=0.35)
    parser.add_argument("--lambda-garbage-stability", type=float, default=0.35)
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
    parser.add_argument("--lambda-rssi-prior", type=float, default=0.2)
    parser.add_argument("--lambda-raw-prior", type=float, default=0.1)
    parser.add_argument("--lambda-veto", type=float, default=0.5)
    parser.add_argument("--lambda-q-switch", type=float, default=1.0)
    parser.add_argument("--t-seg", type=float, default=None)
    parser.add_argument("--t-seg-quantile", type=float, default=0.95)
    parser.add_argument("--lambda-score-vote", type=float, default=1.0)
    parser.add_argument("--lambda-score-cost", type=float, default=0.15)
    parser.add_argument("--q4-shift-grid", default="-0.25,0,0.25")
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    parser.add_argument("--leave-one-out-prototypes", action="store_true")
    parser.add_argument(
        "--splits",
        default="train_loocv,val,test",
        help="Comma-separated evaluation splits. Use train_loocv,val for tuning and test only after selection.",
    )
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata["sample_counts"], indent=2, ensure_ascii=False))
    for row in metadata["summary"]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {metadata['args']['output_dir']}")


if __name__ == "__main__":
    main()
