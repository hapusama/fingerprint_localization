#!/usr/bin/env python3
"""Validation-selected ACO 4.1 sweep on the 1:10 6:2:2 split."""

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

import aco_packet_path_v41 as aco41  # noqa: E402
import run_aco_v2_on_split as split_runner  # noqa: E402


DEFAULT_OUTPUT_DIR = split_runner.DEFAULT_RESULT_DIR / "aco_v41_sweep"
DEFAULT_FINAL_OUTPUT_DIR = split_runner.DEFAULT_RESULT_DIR / "aco_v41_best"
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


def parse_float_list(value: str, default: Sequence[float]) -> list[float]:
    if not value.strip():
        return list(default)
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def build_args(args: argparse.Namespace) -> argparse.Namespace:
    out = split_runner.build_args(args)
    for key in [
        "garbage_cost_min",
        "lambda_garbage_stability",
        "lambda_rssi_prior",
        "lambda_raw_prior",
        "lambda_veto",
        "lambda_q_switch",
        "lambda_score_vote",
        "lambda_score_cost",
        "lambda_vr",
        "lambda_rel",
        "t_seg",
        "t_seg_resolved",
        "t_seg_quantile",
        "t_d",
        "t_d_resolved",
        "t_d_quantile",
        "t_c",
        "t_c_resolved",
        "t_c_quantile",
        "min_segment_reliability",
        "ws_bias",
        "ws_a1",
        "ws_a2",
        "ws_a3",
        "ws_a4",
        "ws_a5",
        "ws_omega_template",
        "ws_omega_struct",
        "ws_sep_scale",
    ]:
        setattr(out, key, getattr(args, key))
    return out


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * min(1.0, max(0.0, q)))
    value = ordered[idx]
    return value if value > aco41.EPS else 1.0


def estimate_scales(samples, q4_offsets, chirp_shapes, chirp_struct, train_indices, args) -> tuple[float, float, float]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    templates = aco41.aco2.build_templates(samples, labels, train_indices, chirp_shapes, chirp_struct, args)
    prototypes = aco41.aco2.build_segment_prototypes(samples, labels, train_indices)
    tseg_values = []
    td_values = []
    tc_values = []
    old = (args.t_seg_resolved, args.t_d_resolved, args.t_c_resolved)
    args.t_seg_resolved = 1.0
    args.t_d_resolved = 1.0
    args.t_c_resolved = 1.0
    for test_index in train_indices:
        effective_train = [idx for idx in train_indices if idx != test_index]
        ranked = aco41.aco2.base.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in ranked[: args.top_k]]
        if not candidates:
            continue
        rssi_costs = {label: score for label, score in ranked if label in candidates}
        _obs, _rows, meta = aco41.build_observation_costs_v41(
            samples[test_index],
            candidates,
            rssi_costs,
            templates,
            chirp_struct,
            prototypes,
            q4_offsets,
            args,
        )
        tseg_values.append(meta["segment_cost_std"])
        td_values.extend(meta["segment_deviation"])
        tc_values.extend(meta["segment_cost_min"])
    args.t_seg_resolved, args.t_d_resolved, args.t_c_resolved = old
    return (
        quantile(tseg_values, args.t_seg_quantile),
        quantile(td_values, args.t_d_quantile),
        quantile(tc_values, args.t_c_quantile),
    )


def training_state(samples, labels, chirp_shapes, chirp_struct, args, template_cache, prototype_cache, indices):
    key = tuple(indices)
    if key not in template_cache:
        template_cache[key] = aco41.aco2.build_templates(samples, labels, indices, chirp_shapes, chirp_struct, args)
        prototype_cache[key] = aco41.aco2.build_segment_prototypes(samples, labels, indices)
    return template_cache[key], prototype_cache[key]


def evaluate_split(
    samples,
    q4_offsets,
    chirp_shapes,
    chirp_struct,
    train_indices,
    eval_indices,
    split_name: str,
    args,
    write_details: bool = False,
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
    fixed_templates, fixed_prototypes = training_state(
        samples, labels, chirp_shapes, chirp_struct, args, template_cache, prototype_cache, train_indices
    )
    for test_index in eval_indices:
        sample = samples[test_index]
        effective_train = [idx for idx in train_indices if idx != test_index]
        ranked = aco41.aco2.base.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in ranked[: args.top_k]]
        if not candidates:
            continue
        templates, prototypes = fixed_templates, fixed_prototypes
        rssi_costs = {label: score for label, score in ranked if label in candidates}
        obs_costs, rows, meta = aco41.build_observation_costs_v41(
            sample, candidates, rssi_costs, templates, chirp_struct, prototypes, q4_offsets, args
        )
        result = aco41.run_aco_v41_for_packet(obs_costs, candidates, templates, meta, args, rng)
        rssi_pred = candidates[0]
        topk_contains += int(sample.label in candidates)
        correct["rssi"] += int(rssi_pred == sample.label)
        for key in ["path_mode", "pheromone", "vote", "score4"]:
            correct[key] += int(result[f"{key}_label"] == sample.label)
        baseline_label = result["vote_label"]
        final_label = result["score4_label"]
        baseline_correct = baseline_label == sample.label
        final_correct = final_label == sample.label
        if write_details:
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
                    "score41": result["score4"].get(label, 0.0),
                    "candidate_mean_obs": meta["candidate_mean_obs"].get(label, 0.0),
                    "candidate_cost_norm": meta["candidate_cost_norm"].get(label, 0.0),
                    "cost_veto": meta["veto"].get(label, 1.0),
                    "R_ws": meta["R_ws"].get(label, 0.0),
                    "Sep_ws": meta["Sep_ws"].get(label, 0.0),
                    "Sep_gate": meta["Sep_gate"].get(label, 0.0),
                    "is_rssi_top1": int(label == meta["rssi_top1"]),
                    "is_raw_winner": int(label == meta["raw_winner"]),
                    "raw_margin": meta["raw_margin"],
                    "Q_pkt": meta["q_pkt"],
                    "q_s_mean": sum(meta["q_s"]) / len(meta["q_s"]),
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
                "true_display": aco41.aco2.point_display(sample.label),
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": int(rssi_pred == sample.label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "raw_winner_label": meta["raw_winner"],
                "raw_margin": meta["raw_margin"],
                "Q_pkt": meta["q_pkt"],
                "q_s_mean": sum(meta["q_s"]) / len(meta["q_s"]),
                "segment_cost_std": meta["segment_cost_std"],
                "aco_path_mode_label": result["path_mode_label"],
                "aco_path_mode_correct": int(result["path_mode_label"] == sample.label),
                "aco_pheromone_label": result["pheromone_label"],
                "aco_pheromone_correct": int(result["pheromone_label"] == sample.label),
                "aco_vote_label": result["vote_label"],
                "aco_vote_correct": int(result["vote_label"] == sample.label),
                "aco_score41_label": result["score4_label"],
                "aco_score41_correct": int(result["score4_label"] == sample.label),
                "score41_changed_from_vote": int(final_label != baseline_label),
                "score41_W2R_from_vote": int((not baseline_correct) and final_correct),
                "score41_R2W_from_vote": int(baseline_correct and (not final_correct)),
                "best_path_cost": result["best_cost"],
                "best_path_labels": ";".join(result["best_path_labels"]),
                "best_path_garbage_count": result["garbage_count"],
            }
        )

    n = len(predictions)
    segment_total = sum(len(samples[idx].segment_q4_reliable) for idx in eval_indices)
    q4_reliable = sum(sum(1 for flag in samples[idx].segment_q4_reliable if flag) for idx in eval_indices)
    changed = sum(int(row["score41_changed_from_vote"]) for row in predictions)
    w2r = sum(int(row["score41_W2R_from_vote"]) for row in predictions)
    r2w = sum(int(row["score41_R2W_from_vote"]) for row in predictions)
    metrics = {
        "method": "aco_v41",
        "split": split_name,
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "segment_count": args.segment_count,
        "rssi_class_k": args.rssi_class_k,
        "T_seg": args.t_seg_resolved,
        "T_d": args.t_d_resolved,
        "T_c": args.t_c_resolved,
        "lambda_R": args.lambda_rssi_prior,
        "lambda_W": args.lambda_raw_prior,
        "lambda_V": args.lambda_veto,
        "lambda_Q": args.lambda_q_switch,
        "lambda_VR": args.lambda_vr,
        "lambda_rel": args.lambda_rel,
        "ws_omega_struct": args.ws_omega_struct,
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
        "aco_score41_correct": correct["score4"],
        "aco_score41_accuracy": correct["score4"] / n if n else 0.0,
        "score41_change_count_from_vote": changed,
        "score41_W2R_from_vote": w2r,
        "score41_R2W_from_vote": r2w,
        "score41_net_from_vote": w2r - r2w,
        "garbage_state_usage_mean": sum(int(row["best_path_garbage_count"]) for row in predictions) / n if n else 0.0,
        "q4_reliable_segment_count": q4_reliable,
        "segment_count_total": segment_total,
        "q4_reliable_segment_rate": q4_reliable / segment_total if segment_total else 0.0,
        "Q_pkt_mean": sum(float(row["Q_pkt"]) for row in predictions) / n if n else 0.0,
        "q_s_mean": sum(float(row["q_s_mean"]) for row in predictions) / n if n else 0.0,
        "segment_cost_std_mean": sum(float(row["segment_cost_std"]) for row in predictions) / n if n else 0.0,
    }
    return metrics, predictions, candidate_rows, segment_rows


def compare_with_aco_v2(aco_v2_dir: Path, output_dir: Path, split_name: str, predictions: list[dict]) -> dict:
    baseline_path = aco_v2_dir / f"{split_name}_predictions.csv"
    if not baseline_path.exists():
        return {}
    baseline = {row["sample_index"]: row for row in read_csv(baseline_path)}
    rows = []
    changed = w2r = r2w = 0
    for pred in predictions:
        key = str(pred["sample_index"])
        if key not in baseline:
            continue
        old_label = baseline[key]["aco_vote_label"]
        new_label = pred["aco_score41_label"]
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
                "sample_index": pred["sample_index"],
                "file_name": pred["file_name"],
                "packet_index": pred["packet_index"],
                "true_label": true_label,
                "rssi_top1_label": pred["rssi_top1_label"],
                "rssi_topk_candidates": pred["rssi_topk_candidates"],
                "aco_v2_vote_label": old_label,
                "aco_v2_vote_correct": int(old_correct),
                "aco_v41_score_label": new_label,
                "aco_v41_score_correct": int(new_correct),
                "change_type": "W2R" if ((not old_correct) and new_correct) else ("R2W" if old_correct and (not new_correct) else "changed_same_correctness"),
                "Q_pkt": pred["Q_pkt"],
                "q_s_mean": pred["q_s_mean"],
                "segment_cost_std": pred["segment_cost_std"],
                "raw_winner_label": pred["raw_winner_label"],
                "raw_margin": pred["raw_margin"],
            }
        )
    fields = list(rows[0].keys()) if rows else [
        "split", "sample_index", "file_name", "packet_index", "true_label", "rssi_top1_label",
        "rssi_topk_candidates", "aco_v2_vote_label", "aco_v2_vote_correct", "aco_v41_score_label",
        "aco_v41_score_correct", "change_type", "Q_pkt", "q_s_mean", "segment_cost_std", "raw_winner_label", "raw_margin",
    ]
    write_csv(output_dir / f"{split_name}_changes_vs_aco_v2.csv", rows, fields)
    return {
        "aco_v41_change_count_vs_aco_v2": changed,
        "aco_v41_W2R_vs_aco_v2": w2r,
        "aco_v41_R2W_vs_aco_v2": r2w,
        "aco_v41_net_vs_aco_v2": w2r - r2w,
    }


def write_outputs(output_dir: Path, split_name: str, metrics: dict, predictions: list[dict], candidate_rows: list[dict], segment_rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / f"{split_name}_predictions.csv", predictions, list(predictions[0].keys()))
    write_csv(output_dir / f"{split_name}_candidate_scores.csv", candidate_rows, list(candidate_rows[0].keys()))
    if segment_rows:
        preferred = [
            "split", "sample_index", "file_name", "packet_index", "true_label", "segment_index", "candidate_label",
            "C_obs", "C_seg_base", "C_R", "C_bin", "C_bin_raw", "C_E", "C_W", "C_Q",
            "Q_pkt", "q_s", "R_ws", "Sep_ws", "Sep_gate", "segment_deviation", "segment_cost_min", "segment_cost_std",
        ]
        fields = preferred + sorted({key for row in segment_rows for key in row} - set(preferred))
        write_csv(output_dir / f"{split_name}_segment_costs.csv", segment_rows, fields)


def metric_key(row: dict) -> tuple:
    return (
        row["aco_score41_accuracy"],
        row["aco_vote_accuracy"],
        row.get("aco_v41_net_vs_aco_v2", 0),
        -row.get("aco_v41_R2W_vs_aco_v2", 0),
        -row["score41_R2W_from_vote"],
    )


def config_key(config: dict) -> tuple:
    return tuple(round(config[key], 12) for key in ["T_seg", "T_d_scale", "T_c_scale", "lambda_VR", "lambda_rel", "ws_omega_struct"])


def apply_config(args, config, base_scales):
    args.t_seg_resolved = config["T_seg"]
    args.t_d_resolved = base_scales["T_d"] * config["T_d_scale"]
    args.t_c_resolved = base_scales["T_c"] * config["T_c_scale"]
    args.lambda_vr = config["lambda_VR"]
    args.lambda_rel = config["lambda_rel"]
    args.ws_omega_struct = config["ws_omega_struct"]


def run(args: argparse.Namespace) -> dict:
    aco_args = build_args(args)
    rssi_packets = aco41.aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, thresholds = aco41.aco2.base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = aco41.aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco41.aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = split_runner.load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = aco41.aco2.prepare_chirp_fields(aco_args, labels)
    auto_t_seg, auto_t_d, auto_t_c = estimate_scales(samples, q4_offsets, chirp_shapes, chirp_struct, split_indices["train"], aco_args)
    base_scales = {
        "T_seg": args.t_seg if args.t_seg is not None else auto_t_seg,
        "T_d": args.t_d if args.t_d is not None else auto_t_d,
        "T_c": args.t_c if args.t_c is not None else auto_t_c,
    }

    base_config = {
        "T_seg": base_scales["T_seg"],
        "T_d_scale": 1.0,
        "T_c_scale": 1.0,
        "lambda_VR": args.lambda_vr,
        "lambda_rel": args.lambda_rel,
        "ws_omega_struct": args.ws_omega_struct,
    }
    configs = []
    seen = set()

    def add(stage: str, config: dict) -> None:
        key = config_key(config)
        if key not in seen:
            seen.add(key)
            configs.append({"stage": stage, **config})

    for value in parse_float_list(args.sweep_t_seg, [base_scales["T_seg"], 0.035, 0.05, 0.075, 0.10, 0.15]):
        add("T_seg", {**base_config, "T_seg": value})
    for value in parse_float_list(args.sweep_t_d_scale, [0.5, 1.0, 2.0]):
        add("T_d", {**base_config, "T_d_scale": value})
    for value in parse_float_list(args.sweep_t_c_scale, [0.5, 1.0, 2.0]):
        add("T_c", {**base_config, "T_c_scale": value})
    for value in parse_float_list(args.sweep_lambda_vr, [0.0, 0.05, 0.1, 0.2]):
        add("lambda_VR", {**base_config, "lambda_VR": value})
    for value in parse_float_list(args.sweep_lambda_rel, [0.0, 0.05, 0.1, 0.2]):
        add("lambda_rel", {**base_config, "lambda_rel": value})
    for value in parse_float_list(args.sweep_ws_omega_struct, [0.0, 0.2, 0.5, 1.0]):
        add("ws_omega_struct", {**base_config, "ws_omega_struct": value})

    sweep_rows = []
    best_row = None
    for idx, config in enumerate(configs, start=1):
        apply_config(aco_args, config, base_scales)
        metrics, predictions, _candidate_rows, _segment_rows = evaluate_split(
            samples,
            q4_offsets,
            chirp_shapes,
            chirp_struct,
            split_indices["train"],
            split_indices["val"],
            "val",
            aco_args,
            write_details=False,
        )
        metrics.update(thresholds)
        metrics.update(compare_with_aco_v2(args.aco_v2_dir, args.output_dir, "val", predictions))
        metrics.update(
            {
                "sweep_index": idx,
                "stage": config["stage"],
                "T_d_scale": config["T_d_scale"],
                "T_c_scale": config["T_c_scale"],
            }
        )
        sweep_rows.append(metrics)
        if best_row is None or metric_key(metrics) > metric_key(best_row):
            best_row = metrics
        print(
            f"[{idx}/{len(configs)}] {config['stage']} "
            f"T={aco_args.t_seg_resolved:.6g} Td={aco_args.t_d_resolved:.6g} Tc={aco_args.t_c_resolved:.6g} "
            f"VR={aco_args.lambda_vr:.3g} rel={aco_args.lambda_rel:.3g} omegaS={aco_args.ws_omega_struct:.3g} "
            f"val_score41={metrics['aco_score41_accuracy']:.6f}",
            flush=True,
        )

    assert best_row is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sweep_index", "stage", "split", "packet_count", "T_seg", "T_d", "T_c", "T_d_scale", "T_c_scale",
        "lambda_VR", "lambda_rel", "ws_omega_struct", "rssi_topk_recall", "aco_vote_accuracy", "aco_score41_accuracy",
        "aco_v41_change_count_vs_aco_v2", "aco_v41_W2R_vs_aco_v2", "aco_v41_R2W_vs_aco_v2", "aco_v41_net_vs_aco_v2",
    ]
    fields += sorted({key for row in sweep_rows for key in row} - set(fields))
    write_csv(args.output_dir / "aco_v41_val_sweep_summary.csv", sweep_rows, fields)
    with (args.output_dir / "aco_v41_val_sweep_best.json").open("w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2, ensure_ascii=False)

    final_config = {
        "T_seg": best_row["T_seg"],
        "T_d_scale": best_row["T_d"] / base_scales["T_d"],
        "T_c_scale": best_row["T_c"] / base_scales["T_c"],
        "lambda_VR": best_row["lambda_VR"],
        "lambda_rel": best_row["lambda_rel"],
        "ws_omega_struct": best_row["ws_omega_struct"],
    }
    apply_config(aco_args, final_config, base_scales)
    final_rows = []
    for split_name, eval_indices in [
        ("train_loocv", split_indices["train"]),
        ("val", split_indices["val"]),
        ("test", split_indices["test"]),
    ]:
        metrics, predictions, candidate_rows, segment_rows = evaluate_split(
            samples,
            q4_offsets,
            chirp_shapes,
            chirp_struct,
            split_indices["train"],
            eval_indices,
            split_name,
            aco_args,
            write_details=True,
        )
        metrics.update(thresholds)
        metrics.update(compare_with_aco_v2(args.aco_v2_dir, args.final_output_dir, split_name, predictions))
        write_outputs(args.final_output_dir, split_name, metrics, predictions, candidate_rows, segment_rows)
        final_rows.append(metrics)

    write_csv(args.final_output_dir / "aco_v41_summary.csv", final_rows, list(final_rows[0].keys()))
    metadata = {
        "selection_policy": "ACO 4.1 parameters selected by validation split; test split evaluated once with selected config.",
        "base_scales": base_scales,
        "best_val": best_row,
        "summary": final_rows,
        "chirp_template_field": chirp_metadata,
    }
    with (args.output_dir / "aco_v41_sweep_and_final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with (args.final_output_dir / "aco_v41_split_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-output-dir", type=Path, default=DEFAULT_FINAL_OUTPUT_DIR)
    parser.add_argument("--aco-v2-dir", type=Path, default=DEFAULT_ACO_V2_DIR)
    parser.add_argument("--rssi-csv", type=Path, default=split_runner.DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=split_runner.DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=split_runner.DEFAULT_SPLIT_CSV)
    parser.add_argument("--chirp-template-csv", type=Path, default=aco41.aco2.DEFAULT_CHIRP_TEMPLATE_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=aco41.aco2.DEFAULT_CHIRP_STRUCT_CSV)
    parser.add_argument("--location-csv", type=Path, default=aco41.aco2.DEFAULT_LOCATION_CSV)
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
    parser.add_argument("--lambda-score-vote", type=float, default=1.0)
    parser.add_argument("--lambda-score-cost", type=float, default=0.15)
    parser.add_argument("--lambda-vr", type=float, default=0.05)
    parser.add_argument("--lambda-rel", type=float, default=0.05)
    parser.add_argument("--t-seg", type=float, default=None)
    parser.add_argument("--t-seg-resolved", type=float, default=None)
    parser.add_argument("--t-seg-quantile", type=float, default=0.95)
    parser.add_argument("--t-d", type=float, default=None)
    parser.add_argument("--t-d-resolved", type=float, default=None)
    parser.add_argument("--t-d-quantile", type=float, default=0.95)
    parser.add_argument("--t-c", type=float, default=None)
    parser.add_argument("--t-c-resolved", type=float, default=None)
    parser.add_argument("--t-c-quantile", type=float, default=0.95)
    parser.add_argument("--min-segment-reliability", type=float, default=0.05)
    parser.add_argument("--ws-bias", type=float, default=0.0)
    parser.add_argument("--ws-a1", type=float, default=1.0)
    parser.add_argument("--ws-a2", type=float, default=1.0)
    parser.add_argument("--ws-a3", type=float, default=1.0)
    parser.add_argument("--ws-a4", type=float, default=1.0)
    parser.add_argument("--ws-a5", type=float, default=0.5)
    parser.add_argument("--ws-omega-template", type=float, default=1.0)
    parser.add_argument("--ws-omega-struct", type=float, default=0.2)
    parser.add_argument("--ws-sep-scale", type=float, default=1.0)
    parser.add_argument("--sweep-t-seg", default="")
    parser.add_argument("--sweep-t-d-scale", default="")
    parser.add_argument("--sweep-t-c-scale", default="")
    parser.add_argument("--sweep-lambda-vr", default="")
    parser.add_argument("--sweep-lambda-rel", default="")
    parser.add_argument("--sweep-ws-omega-struct", default="")
    parser.add_argument("--q4-shift-grid", default="-0.25,0,0.25")
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "summary": payload["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
