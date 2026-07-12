#!/usr/bin/env python3
"""ACO 2.x optimization trials on the fixed 1:10 Gaussian-noise split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
MODEL_V3_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(MODEL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_V3_DIR))

import aco_packet_path_v2 as aco2  # noqa: E402
import run_aco_v2_ablation_on_split as ablation  # noqa: E402
import run_aco_v2_on_split as split_runner  # noqa: E402


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "aco_v2_2x_trials"
DEFAULT_ACO_V2_DIR = EXPERIMENT_DIR / "results" / "aco_v2"
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


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def minmax_scores(costs: dict[str, float]) -> dict[str, float]:
    if not costs:
        return {}
    lo = min(costs.values())
    hi = max(costs.values())
    if hi - lo <= EPS:
        return {label: 0.0 for label in costs}
    return {label: (value - lo) / (hi - lo) for label, value in costs.items()}


def softmax_from_costs(norm_costs: dict[str, float]) -> dict[str, float]:
    values = {label: math.exp(-cost) for label, cost in norm_costs.items()}
    total = sum(values.values()) or 1.0
    return {label: value / total for label, value in values.items()}


def rank_labels_from_costs(costs: dict[str, float]) -> list[str]:
    return sorted(costs, key=lambda label: (costs[label], aco2.natural_label_key(label)))


def rank_labels_from_scores(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda label: (-scores[label], aco2.natural_label_key(label)))


def margin_from_costs(costs: dict[str, float]) -> float:
    ordered = sorted(costs.values())
    if len(ordered) < 2:
        return 0.0
    return max(0.0, (ordered[1] - ordered[0]) / (abs(ordered[1]) + EPS))


def margin_from_scores(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return max(0.0, (ordered[0] - ordered[1]) / (abs(ordered[0]) + EPS))


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def load_predictions(aco_v2_dir: Path, split: str) -> dict[int, dict]:
    return {int(row["sample_index"]): row for row in read_csv(aco_v2_dir / f"{split}_predictions.csv")}


def load_candidate_scores(aco_v2_dir: Path, split: str) -> dict[int, dict[str, dict]]:
    grouped: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in read_csv(aco_v2_dir / f"{split}_candidate_scores.csv"):
        grouped[int(row["sample_index"])][row["candidate_label"]] = row
    return grouped


def load_segment_cost_sums(aco_v2_dir: Path, split: str) -> dict[int, dict[str, dict[str, float]]]:
    sums: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for row in read_csv(aco_v2_dir / f"{split}_segment_costs.csv"):
        if row["candidate_label"] == aco2.GARBAGE_LABEL:
            continue
        sample_index = int(row["sample_index"])
        label = row["candidate_label"]
        for cost_name in ["C_R", "C_E", "C_W", "C_Q"]:
            if row.get(cost_name, "") != "":
                sums[sample_index][label][cost_name] += parse_float(row[cost_name])
    return sums


def build_packet_bin_costs(args: argparse.Namespace) -> tuple[dict[int, dict], dict]:
    aco_args = split_runner.build_args(args)
    rssi_packets = aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, _q4_offsets, _thresholds = aco2.base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = split_runner.load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, _chirp_metadata = aco2.prepare_chirp_fields(aco_args, labels)
    config_by_version = {config.version: config for config in ablation.ABLATIONS}
    raw_config = config_by_version["V2.1"]
    chirp_config = config_by_version["V2.2"]
    raw_templates = ablation.build_templates_for_ablation(
        samples, labels, split_indices["train"], chirp_shapes, chirp_struct, aco_args, raw_config
    )
    chirp_templates = ablation.build_templates_for_ablation(
        samples, labels, split_indices["train"], chirp_shapes, chirp_struct, aco_args, chirp_config
    )

    by_sample: dict[int, dict] = {}
    for split in ["train_loocv", "val", "test"]:
        preds = load_predictions(args.aco_v2_dir, split)
        for sample_index, pred in preds.items():
            candidates = [label for label in pred["rssi_topk_candidates"].split(";") if label]
            raw_costs = {}
            chirp_costs = {}
            for label in candidates:
                raw_costs[label] = sum(
                    ablation.bin_cost(shape, raw_templates[label], aco_args, raw_config)
                    for shape in samples[sample_index].segment_shapes
                )
                chirp_costs[label] = sum(
                    ablation.bin_cost(shape, chirp_templates[label], aco_args, chirp_config)
                    for shape in samples[sample_index].segment_shapes
                )
            raw_ranked = rank_labels_from_costs(raw_costs)
            chirp_ranked = rank_labels_from_costs(chirp_costs)
            raw_norm = minmax_scores(raw_costs)
            by_sample[sample_index] = {
                "split": split,
                "candidates": candidates,
                "raw_costs": raw_costs,
                "raw_norm_costs": raw_norm,
                "raw_scores": softmax_from_costs(raw_norm),
                "raw_winner": raw_ranked[0] if raw_ranked else "",
                "chirp_costs": chirp_costs,
                "chirp_winner": chirp_ranked[0] if chirp_ranked else "",
                "m_bin": margin_from_costs(raw_costs),
                "raw_chirp_agree": int(bool(raw_ranked) and bool(chirp_ranked) and raw_ranked[0] == chirp_ranked[0]),
            }
    metadata = {
        "sample_count": len(samples),
        "train": len(split_indices["train"]),
        "val": len(split_indices["val"]),
        "test": len(split_indices["test"]),
    }
    return by_sample, metadata


def build_records(args: argparse.Namespace) -> tuple[dict[str, list[dict]], dict]:
    bin_costs, metadata = build_packet_bin_costs(args)
    records_by_split: dict[str, list[dict]] = {}
    for split in ["train_loocv", "val", "test"]:
        preds = load_predictions(args.aco_v2_dir, split)
        candidate_scores = load_candidate_scores(args.aco_v2_dir, split)
        cost_sums = load_segment_cost_sums(args.aco_v2_dir, split)
        records = []
        for sample_index, pred in preds.items():
            true_label = pred["true_label"]
            candidates = [label for label in pred["rssi_topk_candidates"].split(";") if label]
            votes = {
                label: parse_float(candidate_scores[sample_index][label].get("elite_vote"))
                for label in candidates
            }
            vote_sum = sum(votes.values()) or 1.0
            s_aco = {label: votes[label] / vote_sum for label in candidates}
            aco_winner = rank_labels_from_scores(votes)[0]
            bin_row = bin_costs[sample_index]
            raw_norm = bin_row["raw_norm_costs"]
            s_bin = bin_row["raw_scores"]
            m_aco = margin_from_scores(votes)
            m_bin = bin_row["m_bin"]
            records.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "file_name": pred["file_name"],
                    "packet_index": int(pred["packet_index"]),
                    "true_label": true_label,
                    "candidates": candidates,
                    "rssi_top1": pred["rssi_top1_label"],
                    "aco_winner": aco_winner,
                    "base_correct": int(aco_winner == true_label),
                    "votes": votes,
                    "s_aco": s_aco,
                    "m_aco": m_aco,
                    "raw_winner": bin_row["raw_winner"],
                    "chirp_winner": bin_row["chirp_winner"],
                    "raw_chirp_agree": bin_row["raw_chirp_agree"],
                    "m_bin": m_bin,
                    "raw_norm_costs": raw_norm,
                    "s_bin": s_bin,
                    "cost_sums": cost_sums[sample_index],
                }
            )
        records_by_split[split] = records
    return records_by_split, metadata


def evaluate_predictions(records: Sequence[dict], pred_key: str, method: str, params: dict) -> dict:
    n = len(records)
    base_correct = sum(record["base_correct"] for record in records)
    final_correct = sum(int(record[pred_key] == record["true_label"]) for record in records)
    trigger_count = sum(int(record[pred_key] != record["aco_winner"]) for record in records)
    w2r = sum(
        int(record["aco_winner"] != record["true_label"] and record[pred_key] == record["true_label"])
        for record in records
    )
    r2w = sum(
        int(record["aco_winner"] == record["true_label"] and record[pred_key] != record["true_label"])
        for record in records
    )
    return {
        "method": method,
        "split": records[0]["split"] if records else "",
        "packet_count": n,
        "base_correct": base_correct,
        "base_acc": base_correct / n if n else 0.0,
        "trigger_count": trigger_count,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "final_correct": final_correct,
        "final_acc": final_correct / n if n else 0.0,
        **params,
    }


def apply_2_1(records: Sequence[dict], theta_a: float, theta_b: float) -> list[dict]:
    out = []
    for record in records:
        pred = record["aco_winner"]
        if record["m_aco"] < theta_a and record["m_bin"] > theta_b and record["raw_chirp_agree"]:
            pred = record["raw_winner"]
        row = dict(record)
        row["pred_2_1"] = pred
        out.append(row)
    return out


def apply_2_2(records: Sequence[dict], base_w_bin: float) -> list[dict]:
    out = []
    for record in records:
        consistency = record["raw_chirp_agree"]
        w_pkt = base_w_bin * record["m_bin"] * (1.0 - record["m_aco"]) * consistency
        c_r_norm = minmax_scores(
            {label: record["cost_sums"][label].get("C_R", 0.0) for label in record["candidates"]}
        )
        c_e_norm = minmax_scores(
            {label: record["cost_sums"][label].get("C_E", 0.0) for label in record["candidates"]}
        )
        c_w_norm = minmax_scores(
            {label: record["cost_sums"][label].get("C_W", 0.0) for label in record["candidates"]}
        )
        scores = {}
        for label in record["candidates"]:
            base_cost = (
                0.45 * c_r_norm.get(label, 0.0)
                + 0.20 * c_e_norm.get(label, 0.0)
                + 0.55 * c_w_norm.get(label, 0.0)
            )
            scores[label] = base_cost + w_pkt * record["raw_norm_costs"].get(label, 0.0)
        pred = rank_labels_from_costs(scores)[0]
        row = dict(record)
        row["pred_2_2"] = pred
        row["w_bin_pkt"] = w_pkt
        out.append(row)
    return out


def apply_2_3(records: Sequence[dict], lambda_bin: float, consistency_gate: bool) -> list[dict]:
    out = []
    for record in records:
        gate = record["raw_chirp_agree"] if consistency_gate else 1
        scale = lambda_bin * record["m_bin"] * (1.0 - record["m_aco"]) * gate
        scores = {
            label: record["s_aco"].get(label, 0.0) + scale * record["s_bin"].get(label, 0.0)
            for label in record["candidates"]
        }
        pred = rank_labels_from_scores(scores)[0]
        row = dict(record)
        row["pred_2_3"] = pred
        row["lambda_scale"] = scale
        out.append(row)
    return out


def apply_2_4(records: Sequence[dict], theta_a: float, theta_b: float) -> list[dict]:
    out = []
    for record in records:
        pred = record["aco_winner"]
        if (
            record["m_aco"] < theta_a
            and record["m_bin"] > theta_b
            and record["raw_chirp_agree"]
            and record["raw_winner"] != record["aco_winner"]
        ):
            pred = record["raw_winner"]
        row = dict(record)
        row["pred_2_4"] = pred
        out.append(row)
    return out


def selector_features(record: dict) -> list[float]:
    bin_winner = record["raw_winner"]
    aco_winner = record["aco_winner"]
    delta_s = record["s_bin"].get(bin_winner, 0.0) - record["s_aco"].get(aco_winner, 0.0)
    return [
        record["m_aco"],
        record["m_bin"],
        float(record["raw_chirp_agree"]),
        float(bin_winner == aco_winner),
        delta_s,
    ]


def selector_target(record: dict) -> int:
    return int(record["raw_winner"] != record["aco_winner"] and record["raw_winner"] == record["true_label"])


def fit_logistic_selector(
    records: Sequence[dict],
    positive_weight: float,
    l2: float,
    epochs: int = 900,
    lr: float = 0.15,
) -> dict:
    x_raw = [selector_features(record) for record in records]
    y = [selector_target(record) for record in records]
    dim = len(x_raw[0])
    means = [sum(row[j] for row in x_raw) / len(x_raw) for j in range(dim)]
    stds = []
    for j in range(dim):
        variance = sum((row[j] - means[j]) ** 2 for row in x_raw) / len(x_raw)
        stds.append(math.sqrt(variance) if variance > EPS else 1.0)
    x = [[(row[j] - means[j]) / stds[j] for j in range(dim)] for row in x_raw]
    weights = [0.0] * (dim + 1)
    total_weight = sum(positive_weight if label else 1.0 for label in y) or 1.0
    for _epoch in range(epochs):
        grad = [0.0] * (dim + 1)
        for row, label in zip(x, y):
            z = weights[0] + sum(weights[j + 1] * row[j] for j in range(dim))
            p = sigmoid(z)
            sample_weight = positive_weight if label else 1.0
            error = (p - label) * sample_weight
            grad[0] += error
            for j in range(dim):
                grad[j + 1] += error * row[j]
        for j in range(1, dim + 1):
            grad[j] += l2 * weights[j]
        for j in range(dim + 1):
            weights[j] -= lr * grad[j] / total_weight
    return {"weights": weights, "means": means, "stds": stds}


def selector_probability(record: dict, model: dict) -> float:
    features = selector_features(record)
    z = model["weights"][0]
    for j, value in enumerate(features):
        z += model["weights"][j + 1] * ((value - model["means"][j]) / model["stds"][j])
    return sigmoid(z)


def apply_logistic_selector(
    records: Sequence[dict],
    model: dict,
    threshold: float,
    require_agreement: bool,
) -> list[dict]:
    out = []
    for record in records:
        score = selector_probability(record, model)
        can_replace = record["raw_winner"] != record["aco_winner"]
        if require_agreement:
            can_replace = can_replace and bool(record["raw_chirp_agree"])
        pred = record["raw_winner"] if can_replace and score >= threshold else record["aco_winner"]
        row = dict(record)
        row["pred_2_4_logreg"] = pred
        row["selector_score"] = score
        out.append(row)
    return out


def choose_best(rows: Sequence[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            row["final_acc"],
            row["net_gain"],
            -row["R2W"],
            -row["trigger_count"],
        ),
    )


def compact_prediction_rows(records: Sequence[dict], pred_key: str, method: str, params: dict) -> list[dict]:
    rows = []
    for record in records:
        rows.append(
            {
                "method": method,
                "split": record["split"],
                "sample_index": record["sample_index"],
                "file_name": record["file_name"],
                "packet_index": record["packet_index"],
                "true_label": record["true_label"],
                "aco_winner": record["aco_winner"],
                "final_label": record[pred_key],
                "base_correct": record["base_correct"],
                "final_correct": int(record[pred_key] == record["true_label"]),
                "triggered": int(record[pred_key] != record["aco_winner"]),
                "m_aco": record["m_aco"],
                "m_bin": record["m_bin"],
                "raw_winner": record["raw_winner"],
                "chirp_winner": record["chirp_winner"],
                "raw_chirp_agree": record["raw_chirp_agree"],
                **params,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_by_split, metadata = build_records(args)
    summary_rows = []
    sweep_rows = []
    prediction_rows = []

    # ACO 2.1: fixed first-pass low-confidence bin arbitration.
    theta_a_fixed = args.theta_a_fixed
    theta_b_fixed = args.theta_b_fixed
    for split, records in records_by_split.items():
        recs = apply_2_1(records, theta_a_fixed, theta_b_fixed)
        params = {"theta_a": theta_a_fixed, "theta_b": theta_b_fixed, "selection": "fixed"}
        summary_rows.append(evaluate_predictions(recs, "pred_2_1", "ACO2.1_low_conf_bin_arbitration", params))
        prediction_rows.extend(compact_prediction_rows(recs, "pred_2_1", "ACO2.1_low_conf_bin_arbitration", params))

    # ACO 2.2: normalized cost fusion with adaptive packet-level bin weight.
    w_grid = [float(item) for item in args.w_bin_grid.split(",") if item.strip()]
    val_2_2 = []
    for w_bin in w_grid:
        recs = apply_2_2(records_by_split["val"], w_bin)
        row = evaluate_predictions(recs, "pred_2_2", "ACO2.2_norm_cost_adaptive_bin", {"base_w_bin": w_bin})
        val_2_2.append(row)
        sweep_rows.append({**row, "sweep_for": "ACO2.2_norm_cost_adaptive_bin"})
    best_2_2 = choose_best(val_2_2)
    for split, records in records_by_split.items():
        recs = apply_2_2(records, best_2_2["base_w_bin"])
        params = {"base_w_bin": best_2_2["base_w_bin"], "selection": "best_on_val"}
        summary_rows.append(evaluate_predictions(recs, "pred_2_2", "ACO2.2_norm_cost_adaptive_bin", params))
        prediction_rows.extend(compact_prediction_rows(recs, "pred_2_2", "ACO2.2_norm_cost_adaptive_bin", params))

    # ACO 2.3: candidate-level posterior reranking.
    lambda_grid = [float(item) for item in args.lambda_bin_grid.split(",") if item.strip()]
    val_2_3 = []
    for lambda_bin in lambda_grid:
        for gate in [False, True]:
            recs = apply_2_3(records_by_split["val"], lambda_bin, gate)
            row = evaluate_predictions(
                recs,
                "pred_2_3",
                "ACO2.3_candidate_posterior_rerank",
                {"lambda_bin": lambda_bin, "consistency_gate": int(gate)},
            )
            val_2_3.append(row)
            sweep_rows.append({**row, "sweep_for": "ACO2.3_candidate_posterior_rerank"})
    best_2_3 = choose_best(val_2_3)
    for split, records in records_by_split.items():
        recs = apply_2_3(records, best_2_3["lambda_bin"], bool(int(best_2_3["consistency_gate"])))
        params = {
            "lambda_bin": best_2_3["lambda_bin"],
            "consistency_gate": best_2_3["consistency_gate"],
            "selection": "best_on_val",
        }
        summary_rows.append(evaluate_predictions(recs, "pred_2_3", "ACO2.3_candidate_posterior_rerank", params))
        prediction_rows.extend(compact_prediction_rows(recs, "pred_2_3", "ACO2.3_candidate_posterior_rerank", params))

    # ACO 2.4: validation-selected threshold selector.
    theta_a_grid = [float(item) for item in args.theta_a_grid.split(",") if item.strip()]
    theta_b_grid = [float(item) for item in args.theta_b_grid.split(",") if item.strip()]
    val_2_4 = []
    for theta_a in theta_a_grid:
        for theta_b in theta_b_grid:
            recs = apply_2_4(records_by_split["val"], theta_a, theta_b)
            row = evaluate_predictions(
                recs,
                "pred_2_4",
                "ACO2.4_val_grid_selector",
                {"theta_a": theta_a, "theta_b": theta_b},
            )
            val_2_4.append(row)
            sweep_rows.append({**row, "sweep_for": "ACO2.4_val_grid_selector"})
    best_2_4 = choose_best(val_2_4)
    for split, records in records_by_split.items():
        recs = apply_2_4(records, best_2_4["theta_a"], best_2_4["theta_b"])
        params = {"theta_a": best_2_4["theta_a"], "theta_b": best_2_4["theta_b"], "selection": "best_on_val"}
        summary_rows.append(evaluate_predictions(recs, "pred_2_4", "ACO2.4_val_grid_selector", params))
        prediction_rows.extend(compact_prediction_rows(recs, "pred_2_4", "ACO2.4_val_grid_selector", params))

    # ACO 2.4 alternative: a tiny logistic selector trained on train-loocv and selected on validation.
    thresholds = [float(item) for item in args.selector_threshold_grid.split(",") if item.strip()]
    pos_weights = [float(item) for item in args.selector_pos_weight_grid.split(",") if item.strip()]
    l2_values = [float(item) for item in args.selector_l2_grid.split(",") if item.strip()]
    val_2_4_logreg = []
    selector_models = {}
    for positive_weight in pos_weights:
        for l2 in l2_values:
            model_key = (positive_weight, l2)
            selector_models[model_key] = fit_logistic_selector(records_by_split["train_loocv"], positive_weight, l2)
            for threshold in thresholds:
                for require_agreement in [False, True]:
                    recs = apply_logistic_selector(
                        records_by_split["val"],
                        selector_models[model_key],
                        threshold,
                        require_agreement,
                    )
                    params = {
                        "positive_weight": positive_weight,
                        "l2": l2,
                        "selector_threshold": threshold,
                        "require_agreement": int(require_agreement),
                    }
                    row = evaluate_predictions(recs, "pred_2_4_logreg", "ACO2.4_light_selector_logreg", params)
                    val_2_4_logreg.append(row)
                    sweep_rows.append({**row, "sweep_for": "ACO2.4_light_selector_logreg"})
    best_2_4_logreg = choose_best(val_2_4_logreg)
    best_model = selector_models[(best_2_4_logreg["positive_weight"], best_2_4_logreg["l2"])]
    for split, records in records_by_split.items():
        recs = apply_logistic_selector(
            records,
            best_model,
            best_2_4_logreg["selector_threshold"],
            bool(int(best_2_4_logreg["require_agreement"])),
        )
        params = {
            "positive_weight": best_2_4_logreg["positive_weight"],
            "l2": best_2_4_logreg["l2"],
            "selector_threshold": best_2_4_logreg["selector_threshold"],
            "require_agreement": best_2_4_logreg["require_agreement"],
            "selection": "best_on_val",
        }
        summary_rows.append(evaluate_predictions(recs, "pred_2_4_logreg", "ACO2.4_light_selector_logreg", params))
        prediction_rows.extend(compact_prediction_rows(recs, "pred_2_4_logreg", "ACO2.4_light_selector_logreg", params))

    write_csv(args.output_dir / "aco_v2_2x_trial_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "aco_v2_2x_sweeps.csv", sweep_rows, sorted({key for row in sweep_rows for key in row}))
    write_csv(args.output_dir / "aco_v2_2x_predictions.csv", prediction_rows, sorted({key for row in prediction_rows for key in row}))
    payload = {
        "method": "ACO 2.1-2.4 optimization trials on gaussian_noise_1to10_split",
        "source": "external_design_notes/蚁群算法2.x试验.md",
        "data_policy": "Uses the existing 1:10 Gaussian-noise augmented fixed split.",
        "metadata": metadata,
        "selected": {
            "ACO2.2_norm_cost_adaptive_bin": best_2_2,
            "ACO2.3_candidate_posterior_rerank": best_2_3,
            "ACO2.4_val_grid_selector": best_2_4,
            "ACO2.4_light_selector_logreg": best_2_4_logreg,
        },
        "selector_logreg": {
            "feature_names": ["m_ACO", "m_bin", "raw_chirp_agree", "bin_equals_ACO", "delta_S"],
            "best_model": best_model,
        },
        "summary": summary_rows,
    }
    with (args.output_dir / "aco_v2_2x_trial_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aco-v2-dir", type=Path, default=DEFAULT_ACO_V2_DIR)
    parser.add_argument("--rssi-csv", type=Path, default=split_runner.DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=split_runner.DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=split_runner.DEFAULT_SPLIT_CSV)
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
    parser.add_argument("--theta-a-fixed", type=float, default=0.20)
    parser.add_argument("--theta-b-fixed", type=float, default=0.10)
    parser.add_argument("--w-bin-grid", default="0,0.25,0.5,1,2,4,8,16,32")
    parser.add_argument("--lambda-bin-grid", default="0,0.25,0.5,1,2,4,8,16,32")
    parser.add_argument("--theta-a-grid", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.01")
    parser.add_argument("--theta-b-grid", default="0,0.01,0.02,0.05,0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument("--selector-threshold-grid", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95")
    parser.add_argument("--selector-pos-weight-grid", default="1,2,4,8,12,16,24,32")
    parser.add_argument("--selector-l2-grid", default="0,0.001,0.01,0.1")
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["selected"], indent=2, ensure_ascii=False))
    for row in payload["summary"]:
        if row["split"] == "test":
            print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
