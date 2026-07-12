#!/usr/bin/env python3
"""ML-ACO 2.0 candidate reranking on 1:10 Gaussian-noise data.

The experiment follows `external_design_notes/蚁群算法2.0+ML.md`.
It trains a small packet-wise softmax logistic ranker over RSSI+ Top-3
candidates using ACO 2.0 evidence as features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ML_DIR = Path(__file__).resolve().parent
MODEL_V3_DIR = ML_DIR.parent
PROJECT_ROOT = ML_DIR.parents[3]
GAUSSIAN_DIR = (
    PROJECT_ROOT / "fingerprint_localization" / "experiments" / "aco_source_safe_1to10"
)
if str(MODEL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_V3_DIR))
if str(GAUSSIAN_DIR) not in sys.path:
    sys.path.insert(0, str(GAUSSIAN_DIR))

import aco_packet_path_v2 as aco2  # noqa: E402
import run_aco_v2_on_split as split_runner  # noqa: E402
import run_aco_v2_2x_trials as trials  # noqa: E402


DEFAULT_OUTPUT_DIR = ML_DIR / "output_gaussian_noise_1to10_group_safe"
DEFAULT_SPLIT_CSV = DEFAULT_OUTPUT_DIR / "group_safe_split_assignments.csv"
DEFAULT_ACO_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "aco_v2_features"
DEFAULT_SOURCE_SPLIT = GAUSSIAN_DIR / "data" / "split_assignments.csv"
EPS = 1e-12


FEATURE_GROUPS = {
    "rssi": {
        "rssi_rank_inv",
        "rssi_rank_norm",
        "is_rssi_top1",
        "C_R_mean",
        "C_R_norm_low",
        "C_R_delta_top1",
        "rssi_margin12",
    },
    "aco": {
        "elite_vote",
        "vote_share",
        "vote_norm_high",
        "vote_rank_inv",
        "self_pheromone",
        "pheromone_norm_high",
        "pheromone_rank_inv",
        "is_aco_vote",
        "is_aco_pheromone",
        "is_aco_path_mode",
        "aco_vote_margin",
        "template_reliability",
    },
    "cost": {
        "C_obs_mean",
        "C_obs_norm_low",
        "C_E_mean",
        "C_E_norm_low",
        "C_W_mean",
        "C_W_norm_low",
        "C_Q_mean",
        "C_Q_norm_low",
        "segment_cost_min",
        "segment_cost_max",
        "segment_cost_std",
    },
    "raw_bin": {
        "C_bin_mean",
        "C_bin_norm_low",
        "C_bin_raw_mean",
        "C_bin_raw_norm_low",
        "raw_norm_cost",
        "raw_score",
        "raw_rank_inv",
        "is_raw_winner",
        "raw_margin",
    },
    "chirp_physical": {
        "is_chirp_winner",
        "raw_chirp_agree",
        "raw_chirp_both_support",
        "alpha_shrink",
        "chirp_source_measured_count",
        "chirp_source_nearest_count",
        "chirp_source_fallback_count",
    },
    "garbage_path": {
        "best_path_garbage_count",
        "best_path_garbage_rate",
        "candidate_in_best_path_count",
        "candidate_in_best_path_rate",
    },
}


FEATURE_SETS = [
    ("E1_rssi_only", ["rssi"]),
    ("E1_rssi_aco", ["rssi", "aco"]),
    ("E1_rssi_aco_cost", ["rssi", "aco", "cost"]),
    ("E1_rssi_aco_cost_raw", ["rssi", "aco", "cost", "raw_bin"]),
    ("E1_full_logistic_ranker", ["rssi", "aco", "cost", "raw_bin", "chirp_physical", "garbage_path"]),
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    rows = list(rows)
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


def strip_aug(stem_or_name: str) -> str:
    stem = Path(stem_or_name).stem
    return re.sub(r"_aug\d+$", "", stem)


def base_packet_key(file_name_or_stem: str, packet_index: object) -> str:
    return f"{strip_aug(file_name_or_stem)}::{int(float(packet_index))}"


def normalize_low(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo <= EPS:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def normalize_high(values: dict[str, float]) -> dict[str, float]:
    low = normalize_low(values)
    return {key: 1.0 - value for key, value in low.items()}


def rank_inv_from_scores(values: dict[str, float], high_better: bool) -> dict[str, float]:
    labels = list(values)
    if high_better:
        ordered = sorted(labels, key=lambda label: (-values[label], aco2.natural_label_key(label)))
    else:
        ordered = sorted(labels, key=lambda label: (values[label], aco2.natural_label_key(label)))
    denom = max(1, len(ordered) - 1)
    return {label: 1.0 - (rank / denom) for rank, label in enumerate(ordered)}


def margin_high(values: dict[str, float]) -> float:
    ordered = sorted(values.values(), reverse=True)
    if len(ordered) < 2:
        return 0.0
    return (ordered[0] - ordered[1]) / (abs(ordered[0]) + EPS)


def margin_low(values: dict[str, float]) -> float:
    ordered = sorted(values.values())
    if len(ordered) < 2:
        return 0.0
    return (ordered[1] - ordered[0]) / (abs(ordered[1]) + EPS)


def leakage_audit(split_rows: Sequence[dict]) -> dict:
    by_group = defaultdict(set)
    for row in split_rows:
        by_group[base_packet_key(row["file_stem"], row["packet_index"])].add(row["split"])
    leaky = {key: splits for key, splits in by_group.items() if len(splits) > 1}
    combo_counts = Counter(";".join(sorted(splits)) for splits in leaky.values())
    return {
        "row_count": len(split_rows),
        "base_packet_count": len(by_group),
        "multi_split_base_packet_count": len(leaky),
        "multi_split_combo_counts": dict(combo_counts),
    }


def make_group_safe_split(source_split_csv: Path, output_csv: Path, seed: int) -> dict:
    source_rows = read_csv(source_split_csv)
    grouped = defaultdict(list)
    for row in source_rows:
        grouped[(row["position_key"], base_packet_key(row["file_stem"], row["packet_index"]))].append(row)

    by_label = defaultdict(list)
    for (label, group_key), rows in grouped.items():
        by_label[label].append((group_key, rows))

    rng = random.Random(seed)
    out_rows = []
    for label in sorted(by_label, key=aco2.natural_label_key):
        groups = by_label[label]
        rng.shuffle(groups)
        n = len(groups)
        if n == 1:
            n_train, n_val = 1, 0
        elif n == 2:
            n_train, n_val = 1, 0
        elif n == 3:
            n_train, n_val = 1, 1
        else:
            n_train = max(2, int(round(n * 0.6)))
            n_val = max(1, int(round(n * 0.2)))
            if n_train + n_val >= n:
                n_val = 1
                n_train = n - 2
        assignments = [
            ("train", groups[:n_train]),
            ("val", groups[n_train : n_train + n_val]),
            ("test", groups[n_train + n_val :]),
        ]
        for split, assigned_groups in assignments:
            for _group_key, rows in assigned_groups:
                for row in rows:
                    out_rows.append(
                        {
                            "split": split,
                            "position_key": row["position_key"],
                            "file_stem": row["file_stem"],
                            "packet_index": row["packet_index"],
                        }
                    )
    out_rows.sort(key=lambda row: (row["split"], aco2.natural_label_key(row["position_key"]), row["file_stem"], int(row["packet_index"])))
    write_csv(output_csv, out_rows, ["split", "position_key", "file_stem", "packet_index"])
    return {
        "source_split_audit": leakage_audit(source_rows),
        "group_safe_split_audit": leakage_audit(out_rows),
        "split_counts": dict(Counter(row["split"] for row in out_rows)),
    }


def build_aco_args(args: argparse.Namespace, split_csv: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        result_dir=args.output_dir,
        output_dir=output_dir,
        method_summary=args.output_dir / "method_summary_group_safe.csv",
        rssi_csv=args.rssi_csv,
        spectrum_csv=args.spectrum_csv,
        split_csv=split_csv,
        chirp_template_csv=args.chirp_template_csv,
        chirp_structure_csv=args.chirp_structure_csv,
        location_csv=args.location_csv,
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
        leave_one_out_prototypes=args.leave_one_out_prototypes,
    )


def ensure_aco_features(args: argparse.Namespace, split_csv: Path) -> dict:
    aco_output = args.aco_output_dir
    expected = aco_output / "test_candidate_scores.csv"
    if args.force_aco or not expected.exists():
        aco_output.mkdir(parents=True, exist_ok=True)
        split_runner.run(build_aco_args(args, split_csv, aco_output))
    with (aco_output / "aco_v2_split_metrics.json").open(encoding="utf-8") as f:
        return json.load(f)


def build_trial_args(args: argparse.Namespace, split_csv: Path) -> argparse.Namespace:
    trial_args = build_aco_args(args, split_csv, args.aco_output_dir)
    trial_args.aco_v2_dir = args.aco_output_dir
    trial_args.output_dir = args.output_dir
    return trial_args


def load_predictions(aco_dir: Path, split: str) -> dict[int, dict]:
    return {int(row["sample_index"]): row for row in read_csv(aco_dir / f"{split}_predictions.csv")}


def load_candidate_scores(aco_dir: Path, split: str) -> dict[int, dict[str, dict]]:
    out = defaultdict(dict)
    for row in read_csv(aco_dir / f"{split}_candidate_scores.csv"):
        out[int(row["sample_index"])][row["candidate_label"]] = row
    return out


def aggregate_segment_costs(aco_dir: Path, split: str) -> dict[int, dict[str, dict]]:
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in read_csv(aco_dir / f"{split}_segment_costs.csv"):
        label = row["candidate_label"]
        if label == aco2.GARBAGE_LABEL:
            continue
        sample_index = int(row["sample_index"])
        for key in ["C_obs", "C_R", "C_bin", "C_bin_raw", "C_E", "C_W", "C_Q"]:
            if row.get(key, "") != "":
                values[sample_index][label][key].append(parse_float(row[key]))

    out = defaultdict(dict)
    for sample_index, by_label in values.items():
        for label, by_key in by_label.items():
            stats = {}
            for key, vals in by_key.items():
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                variance = sum((value - mean) ** 2 for value in vals) / len(vals)
                stats[f"{key}_mean"] = mean
                stats[f"{key}_sum"] = sum(vals)
                stats[f"{key}_min"] = min(vals)
                stats[f"{key}_max"] = max(vals)
                stats[f"{key}_std"] = math.sqrt(variance)
            out[sample_index][label] = stats
    return out


def path_count_for_label(path_labels: str, label: str) -> int:
    return sum(1 for item in path_labels.split(";") if item == label)


def chirp_source_counts(source: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in source.split(";") if part.strip()]
    measured = sum(1 for part in parts if "measured" in part)
    nearest = sum(1 for part in parts if "nearest" in part)
    fallback = sum(1 for part in parts if "fallback" in part)
    return measured, nearest, fallback


def feature_names_for_groups(group_names: Sequence[str]) -> list[str]:
    names = []
    for group in group_names:
        names.extend(sorted(FEATURE_GROUPS[group]))
    return names


def build_candidate_dataset(args: argparse.Namespace, split_csv: Path) -> tuple[dict[str, list[dict]], dict, list[str]]:
    records_by_split, metadata = trials.build_records(build_trial_args(args, split_csv))
    datasets = {}
    all_feature_names = sorted(set().union(*FEATURE_GROUPS.values()))
    for split, records in records_by_split.items():
        preds = load_predictions(args.aco_output_dir, split)
        candidate_scores = load_candidate_scores(args.aco_output_dir, split)
        seg_stats = aggregate_segment_costs(args.aco_output_dir, split)
        rows = []
        for record in records:
            sample_index = record["sample_index"]
            pred = preds[sample_index]
            candidates = record["candidates"]
            true_label = record["true_label"]
            c_r = {
                label: seg_stats[sample_index][label].get("C_R_mean", 0.0)
                for label in candidates
            }
            c_r_norm = normalize_low(c_r)
            c_r_top = min(c_r.values()) if c_r else 0.0
            rssi_margin = margin_low(c_r)
            vote_values = {
                label: parse_float(candidate_scores[sample_index][label].get("elite_vote"))
                for label in candidates
            }
            vote_norm = normalize_high(vote_values)
            vote_rank = rank_inv_from_scores(vote_values, high_better=True)
            pheromone_values = {
                label: parse_float(candidate_scores[sample_index][label].get("self_pheromone"))
                for label in candidates
            }
            pheromone_norm = normalize_high(pheromone_values)
            pheromone_rank = rank_inv_from_scores(pheromone_values, high_better=True)
            raw_rank = rank_inv_from_scores(record["raw_norm_costs"], high_better=False)
            cost_norms = {}
            for cost_key in ["C_obs", "C_bin", "C_bin_raw", "C_E", "C_W", "C_Q"]:
                values = {
                    label: seg_stats[sample_index][label].get(f"{cost_key}_mean", 0.0)
                    for label in candidates
                }
                cost_norms[cost_key] = normalize_low(values)
            best_path_labels = pred.get("best_path_labels", "")
            best_path_gc = parse_float(pred.get("best_path_garbage_count"))
            for rank, label in enumerate(candidates):
                score_row = candidate_scores[sample_index][label]
                measured, nearest, fallback = chirp_source_counts(score_row.get("chirp_source", ""))
                stats = seg_stats[sample_index][label]
                candidate_path_count = path_count_for_label(best_path_labels, label)
                features = {
                    "rssi_rank_inv": 1.0 - rank / max(1, len(candidates) - 1),
                    "rssi_rank_norm": rank / max(1, len(candidates) - 1),
                    "is_rssi_top1": float(rank == 0),
                    "C_R_mean": c_r[label],
                    "C_R_norm_low": c_r_norm[label],
                    "C_R_delta_top1": c_r[label] - c_r_top,
                    "rssi_margin12": rssi_margin,
                    "elite_vote": vote_values[label],
                    "vote_share": record["s_aco"].get(label, 0.0),
                    "vote_norm_high": vote_norm[label],
                    "vote_rank_inv": vote_rank[label],
                    "self_pheromone": pheromone_values[label],
                    "pheromone_norm_high": pheromone_norm[label],
                    "pheromone_rank_inv": pheromone_rank[label],
                    "is_aco_vote": float(label == pred.get("aco_vote_label")),
                    "is_aco_pheromone": float(label == pred.get("aco_pheromone_label")),
                    "is_aco_path_mode": float(label == pred.get("aco_path_mode_label")),
                    "aco_vote_margin": record["m_aco"],
                    "template_reliability": parse_float(score_row.get("template_reliability")),
                    "C_obs_mean": stats.get("C_obs_mean", 0.0),
                    "C_obs_norm_low": cost_norms["C_obs"][label],
                    "C_E_mean": stats.get("C_E_mean", 0.0),
                    "C_E_norm_low": cost_norms["C_E"][label],
                    "C_W_mean": stats.get("C_W_mean", 0.0),
                    "C_W_norm_low": cost_norms["C_W"][label],
                    "C_Q_mean": stats.get("C_Q_mean", 0.0),
                    "C_Q_norm_low": cost_norms["C_Q"][label],
                    "segment_cost_min": stats.get("C_obs_min", 0.0),
                    "segment_cost_max": stats.get("C_obs_max", 0.0),
                    "segment_cost_std": stats.get("C_obs_std", 0.0),
                    "C_bin_mean": stats.get("C_bin_mean", 0.0),
                    "C_bin_norm_low": cost_norms["C_bin"][label],
                    "C_bin_raw_mean": stats.get("C_bin_raw_mean", 0.0),
                    "C_bin_raw_norm_low": cost_norms["C_bin_raw"][label],
                    "raw_norm_cost": record["raw_norm_costs"].get(label, 0.0),
                    "raw_score": record["s_bin"].get(label, 0.0),
                    "raw_rank_inv": raw_rank.get(label, 0.0),
                    "is_raw_winner": float(label == record["raw_winner"]),
                    "raw_margin": record["m_bin"],
                    "is_chirp_winner": float(label == record["chirp_winner"]),
                    "raw_chirp_agree": float(record["raw_chirp_agree"]),
                    "raw_chirp_both_support": float(label == record["raw_winner"] and label == record["chirp_winner"]),
                    "alpha_shrink": parse_float(score_row.get("alpha_shrink")),
                    "chirp_source_measured_count": float(measured),
                    "chirp_source_nearest_count": float(nearest),
                    "chirp_source_fallback_count": float(fallback),
                    "best_path_garbage_count": best_path_gc,
                    "best_path_garbage_rate": best_path_gc / max(1, args.segment_count),
                    "candidate_in_best_path_count": float(candidate_path_count),
                    "candidate_in_best_path_rate": candidate_path_count / max(1, args.segment_count),
                }
                rows.append(
                    {
                        "split": split,
                        "sample_index": sample_index,
                        "base_packet_key": base_packet_key(record["file_name"], record["packet_index"]),
                        "file_name": record["file_name"],
                        "packet_index": record["packet_index"],
                        "true_label": true_label,
                        "candidate_label": label,
                        "target": int(label == true_label),
                        "true_in_top3": int(true_label in candidates),
                        "aco_vote_label": pred.get("aco_vote_label"),
                        "rssi_top1_label": pred.get("rssi_top1_label"),
                        "features": features,
                    }
                )
        datasets[split] = rows
    return datasets, metadata, all_feature_names


def group_candidate_rows(rows: Sequence[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sample_index"], {"rows": [], "true_in_top3": row["true_in_top3"]})
        grouped[row["sample_index"]]["rows"].append(row)
    groups = []
    for sample_index, item in grouped.items():
        rows_sorted = sorted(item["rows"], key=lambda row: row["candidate_label"])
        true_label = rows_sorted[0]["true_label"]
        aco_vote = rows_sorted[0]["aco_vote_label"]
        groups.append(
            {
                "sample_index": sample_index,
                "split": rows_sorted[0]["split"],
                "file_name": rows_sorted[0]["file_name"],
                "packet_index": rows_sorted[0]["packet_index"],
                "base_packet_key": rows_sorted[0]["base_packet_key"],
                "true_label": true_label,
                "true_in_top3": item["true_in_top3"],
                "aco_vote_label": aco_vote,
                "aco_vote_correct": int(aco_vote == true_label),
                "rows": rows_sorted,
            }
        )
    return sorted(groups, key=lambda group: group["sample_index"])


def standardizer(groups: Sequence[dict], feature_names: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    vals = {name: [] for name in feature_names}
    for group in groups:
        if not group["true_in_top3"]:
            continue
        for row in group["rows"]:
            for name in feature_names:
                vals[name].append(row["features"].get(name, 0.0))
    means = {}
    stds = {}
    for name in feature_names:
        data = vals[name] or [0.0]
        mean = sum(data) / len(data)
        variance = sum((value - mean) ** 2 for value in data) / len(data)
        means[name] = mean
        std = math.sqrt(variance)
        stds[name] = std if std > EPS else 1.0
    return means, stds


def vectorize(row: dict, feature_names: Sequence[str], means: dict[str, float], stds: dict[str, float]) -> list[float]:
    return [(row["features"].get(name, 0.0) - means[name]) / stds[name] for name in feature_names]


def softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(score - m) for score in scores]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def train_logistic_ranker(
    train_groups: Sequence[dict],
    feature_names: Sequence[str],
    l2: float,
    epochs: int,
    lr: float,
) -> dict:
    train_groups = [group for group in train_groups if group["true_in_top3"]]
    means, stds = standardizer(train_groups, feature_names)
    dim = len(feature_names)
    weights = [0.0] * dim
    bias = 0.0
    vector_groups = []
    for group in train_groups:
        xs = [vectorize(row, feature_names, means, stds) for row in group["rows"]]
        targets = [row["target"] for row in group["rows"]]
        vector_groups.append((xs, targets))
    for epoch in range(epochs):
        grad = [0.0] * dim
        grad_b = 0.0
        loss = 0.0
        group_count = 0
        for xs, targets in vector_groups:
            scores = [bias + sum(weights[j] * x[j] for j in range(dim)) for x in xs]
            probs = softmax(scores)
            group_count += 1
            for prob, target, x in zip(probs, targets, xs):
                if target:
                    loss -= math.log(max(prob, EPS))
                err = prob - target
                grad_b += err
                for j in range(dim):
                    grad[j] += err * x[j]
        denom = max(1, group_count)
        for j in range(dim):
            grad[j] = grad[j] / denom + l2 * weights[j]
            weights[j] -= lr * grad[j]
        bias -= lr * grad_b / denom
        if epoch > 30 and loss / denom < 0.01:
            break
    return {"feature_names": list(feature_names), "means": means, "stds": stds, "weights": weights, "bias": bias, "l2": l2}


def score_group(group: dict, model: dict) -> list[dict]:
    names = model["feature_names"]
    rows = []
    for row in group["rows"]:
        x = vectorize(row, names, model["means"], model["stds"])
        score = model["bias"] + sum(model["weights"][j] * x[j] for j in range(len(names)))
        rows.append({"row": row, "score": score})
    rows.sort(key=lambda item: (-item["score"], aco2.natural_label_key(item["row"]["candidate_label"])))
    return rows


def evaluate_groups(groups: Sequence[dict], model: dict | None, method: str, conservative_theta: float | None = None) -> tuple[dict, list[dict]]:
    prediction_rows = []
    n = len(groups)
    top3_n = sum(group["true_in_top3"] for group in groups)
    base_correct = sum(group["aco_vote_correct"] for group in groups)
    final_correct = 0
    top3_correct = 0
    trigger_count = 0
    w2r = 0
    r2w = 0
    for group in groups:
        true_label = group["true_label"]
        aco_label = group["aco_vote_label"]
        if model is None:
            final_label = aco_label
            margin = 0.0
            score_rows = []
        else:
            score_rows = score_group(group, model)
            ml_label = score_rows[0]["row"]["candidate_label"]
            margin = score_rows[0]["score"] - (score_rows[1]["score"] if len(score_rows) > 1 else score_rows[0]["score"])
            if conservative_theta is not None and margin < conservative_theta:
                final_label = aco_label
            else:
                final_label = ml_label
        final_ok = int(final_label == true_label)
        base_ok = int(aco_label == true_label)
        final_correct += final_ok
        top3_correct += int(group["true_in_top3"] and final_ok)
        trigger_count += int(final_label != aco_label)
        w2r += int((not base_ok) and final_ok)
        r2w += int(base_ok and not final_ok)
        prediction_rows.append(
            {
                "method": method,
                "split": group["split"],
                "sample_index": group["sample_index"],
                "file_name": group["file_name"],
                "packet_index": group["packet_index"],
                "base_packet_key": group["base_packet_key"],
                "true_label": true_label,
                "true_in_top3": group["true_in_top3"],
                "aco_vote_label": aco_label,
                "final_label": final_label,
                "base_correct": base_ok,
                "final_correct": final_ok,
                "triggered": int(final_label != aco_label),
                "ml_margin": margin,
                "ml_top1_label": score_rows[0]["row"]["candidate_label"] if score_rows else "",
                "ml_top1_score": score_rows[0]["score"] if score_rows else "",
                "ml_top2_label": score_rows[1]["row"]["candidate_label"] if len(score_rows) > 1 else "",
                "ml_top2_score": score_rows[1]["score"] if len(score_rows) > 1 else "",
            }
        )
    metrics = {
        "method": method,
        "split": groups[0]["split"] if groups else "",
        "packet_count": n,
        "true_in_top3_count": top3_n,
        "base_correct": base_correct,
        "base_accuracy": base_correct / n if n else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / n if n else 0.0,
        "top3_inner_correct": top3_correct,
        "top3_inner_accuracy": top3_correct / top3_n if top3_n else 0.0,
        "trigger_count": trigger_count,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "conservative_theta": "" if conservative_theta is None else conservative_theta,
    }
    return metrics, prediction_rows


def choose_best(rows: Sequence[dict]) -> dict:
    return max(rows, key=lambda row: (row["final_accuracy"], row["net_gain"], -row["R2W"], -row["trigger_count"]))


def feature_importance_rows(model: dict) -> list[dict]:
    rows = []
    for name, weight in zip(model["feature_names"], model["weights"]):
        rows.append({"feature": name, "weight": weight, "abs_weight": abs(weight)})
    rows.sort(key=lambda row: (-row["abs_weight"], row["feature"]))
    return rows


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_info = make_group_safe_split(args.source_split_csv, args.group_safe_split_csv, args.seed)
    aco_metadata = ensure_aco_features(args, args.group_safe_split_csv)
    datasets, record_metadata, all_feature_names = build_candidate_dataset(args, args.group_safe_split_csv)
    candidate_rows_flat = []
    for split_rows in datasets.values():
        for row in split_rows:
            flat = {key: value for key, value in row.items() if key != "features"}
            flat.update(row["features"])
            candidate_rows_flat.append(flat)
    candidate_fields = [
        "split",
        "sample_index",
        "base_packet_key",
        "file_name",
        "packet_index",
        "true_label",
        "candidate_label",
        "target",
        "true_in_top3",
        "aco_vote_label",
        "rssi_top1_label",
    ] + all_feature_names
    write_csv(args.output_dir / "ml_candidate_features.csv", candidate_rows_flat, candidate_fields)

    groups_by_split = {split: group_candidate_rows(rows) for split, rows in datasets.items()}
    summary_rows = []
    prediction_rows = []

    for split, groups in groups_by_split.items():
        metrics, preds = evaluate_groups(groups, None, "E0_aco_v2_baseline")
        summary_rows.append(metrics)
        prediction_rows.extend(preds)

    selected = {}
    trained_models = {}
    ablation_rows = []
    for method, group_names in FEATURE_SETS:
        feature_names = feature_names_for_groups(group_names)
        val_rows = []
        for l2 in [float(item) for item in args.l2_grid.split(",") if item.strip()]:
            model = train_logistic_ranker(groups_by_split["train_loocv"], feature_names, l2, args.epochs, args.learning_rate)
            metrics, _preds = evaluate_groups(groups_by_split["val"], model, method)
            metrics["l2"] = l2
            metrics["feature_group"] = "+".join(group_names)
            val_rows.append(metrics)
            ablation_rows.append({**metrics, "selection": "candidate_l2_on_val"})
        best_val = choose_best(val_rows)
        model = train_logistic_ranker(
            groups_by_split["train_loocv"],
            feature_names,
            best_val["l2"],
            args.epochs,
            args.learning_rate,
        )
        trained_models[method] = model
        selected[method] = best_val
        for split, groups in groups_by_split.items():
            metrics, preds = evaluate_groups(groups, model, method)
            metrics["l2"] = best_val["l2"]
            metrics["feature_group"] = "+".join(group_names)
            metrics["selection"] = "best_l2_on_val"
            summary_rows.append(metrics)
            prediction_rows.extend(preds)

    full_method = "E1_full_logistic_ranker"
    full_model = trained_models[full_method]
    theta_rows = []
    for theta in [float(item) for item in args.theta_grid.split(",") if item.strip()]:
        metrics, _preds = evaluate_groups(groups_by_split["val"], full_model, "E4_logistic_conservative_margin", theta)
        theta_rows.append(metrics)
        ablation_rows.append({**metrics, "selection": "candidate_theta_on_val"})
    best_theta = choose_best(theta_rows)
    selected["E4_logistic_conservative_margin"] = best_theta
    for split, groups in groups_by_split.items():
        metrics, preds = evaluate_groups(
            groups,
            full_model,
            "E4_logistic_conservative_margin",
            float(best_theta["conservative_theta"]),
        )
        metrics["l2"] = selected[full_method]["l2"]
        metrics["feature_group"] = "full"
        metrics["selection"] = "best_theta_on_val"
        summary_rows.append(metrics)
        prediction_rows.extend(preds)

    write_csv(args.output_dir / "ml_aco_v2_ranker_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "ml_aco_v2_ablation_sweep.csv", ablation_rows, sorted({key for row in ablation_rows for key in row}))
    write_csv(args.output_dir / "ml_aco_v2_predictions.csv", prediction_rows, sorted({key for row in prediction_rows for key in row}))
    write_csv(args.output_dir / "ml_aco_v2_feature_importance.csv", feature_importance_rows(full_model), ["feature", "weight", "abs_weight"])

    payload = {
        "method": "ML-ACO 2.0 softmax logistic candidate reranker",
        "source": "external_design_notes/蚁群算法2.0+ML.md",
        "data_policy": "Uses 1:10 Gaussian-noise augmented data with a group-safe split: all augmented copies of the same original packet stay in one split.",
        "split_info": split_info,
        "aco_feature_generation": {
            "output_dir": str(args.aco_output_dir),
            "metadata": aco_metadata.get("sample_counts", {}),
        },
        "record_metadata": record_metadata,
        "unavailable_models": {
            "LightGBM/XGBoost": "not installed in current Python environment",
            "MLP": "not run in first pass; no numpy/torch stack available",
        },
        "selected": selected,
        "summary": summary_rows,
    }
    with (args.output_dir / "ml_aco_v2_ranker_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-safe-split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--aco-output-dir", type=Path, default=DEFAULT_ACO_OUTPUT_DIR)
    parser.add_argument("--source-split-csv", type=Path, default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--rssi-csv", type=Path, default=split_runner.DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=split_runner.DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--chirp-template-csv", type=Path, default=aco2.DEFAULT_CHIRP_TEMPLATE_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=aco2.DEFAULT_CHIRP_STRUCT_CSV)
    parser.add_argument("--location-csv", type=Path, default=aco2.DEFAULT_LOCATION_CSV)
    parser.add_argument("--force-aco", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--segment-count", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260627)
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
    parser.add_argument("--l2-grid", default="0,0.001,0.01,0.1")
    parser.add_argument("--theta-grid", default="0,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--learning-rate", type=float, default=0.18)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["split_info"], indent=2, ensure_ascii=False))
    for row in payload["summary"]:
        if row["split"] == "test":
            print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
