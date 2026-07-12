#!/usr/bin/env python3
"""ACO 4.5 reliability-aware ensemble selector.

This experiment starts from the ACO4.4 method candidate table and adds the
pieces that the oracle analysis suggested were missing:

* candidate-vs-base deltas for ACO Top3 evidence;
* smoothed empirical reliability for sources and source families;
* explicit base-protection features.

The protocol remains train_loocv -> choose threshold on val -> evaluate test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_aco_v42_reranker_on_split as base42  # noqa: E402


RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_CANDIDATE_PATH = RESULTS_DIR / "aco_v44_method_ensemble_selector" / "method_candidate_features.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "aco_v45_reliability_selector"
EPS = 1e-12

STATIC_COLUMNS = {
    "split",
    "sample_index",
    "true_label",
    "candidate_label",
    "target",
    "base_label",
    "base_correct",
    "source_names",
}

DELTA_KEYS = [
    "source_hit_count",
    "source_hit_frac",
    "aco4_hit_count",
    "aco2_hit_count",
    "ablation_hit_count",
    "weak_hit_count",
    "raw_chirp_source_hit",
    "consensus_margin",
    "in_aco_top3",
    "is_raw_chirp_agreed_packet",
    "is_raw_winner_packet",
    "is_chirp_winner_packet",
    "top3_candidate_mean_obs_norm",
    "top3_candidate_cost_norm",
    "top3_cost_veto",
    "top3_cost_veto41",
    "top3_elite_vote_norm",
    "top3_score4_norm",
    "top3_score41",
    "top3_self_pheromone",
    "top3_template_reliability_norm",
    "top3_rssi_rank_inv",
    "top3_is_rssi_top1",
    "top3_is_v2_vote",
    "top3_raw_chirp_agree_winner",
    "top3_raw_chirp_agree_x_low_cost",
    "top3_raw_chirp_min_margin_x_agree_winner",
    "top3_v2_vote_x_low_cost",
    "top3_v2_vote_x_raw_winner",
]

BASE_PROTECT_KEYS = [
    "source_hit_count",
    "source_hit_frac",
    "aco4_hit_count",
    "aco2_hit_count",
    "ablation_hit_count",
    "weak_hit_count",
    "raw_chirp_source_hit",
    "consensus_margin",
    "in_aco_top3",
    "top3_candidate_mean_obs_norm",
    "top3_cost_veto41",
    "top3_elite_vote_norm",
    "top3_score4_norm",
    "top3_self_pheromone",
    "top3_template_reliability_norm",
    "top3_is_rssi_top1",
    "top3_is_v2_vote",
    "top3_raw_chirp_agree_winner",
    "top3_v2_vote_x_low_cost",
]


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


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


def label_distance(a: str, b: str) -> tuple[float, float, float]:
    if not a or not b:
        return 0.0, 1.0, 1.0
    ca, la = base42.parse_label(a)
    cb, lb = base42.parse_label(b)
    same_corridor = float(ca == cb)
    loc_delta = abs(la - lb)
    corridor_delta = abs(ca - cb)
    return same_corridor, loc_delta / 54.0, corridor_delta


def source_family(source: str) -> str:
    if source.startswith("src_"):
        source = source[4:]
    if source.startswith("ab_"):
        return "ablation"
    if source in {"aco_v2"}:
        return "aco2"
    if source in {"knn", "mfr_prev"}:
        return "weak"
    if source.startswith("expert_extra_"):
        return "raw_chirp"
    return "aco4"


def source_signature(row: dict) -> tuple[str, ...]:
    names = [name for name in row.get("source_names", "").split(";") if name]
    families = sorted({source_family(name) for name in names})
    return tuple(families)


def source_pattern(row: dict) -> str:
    sig = source_signature(row)
    return "+".join(sig) if sig else "none"


def numeric_columns(rows: Sequence[dict]) -> list[str]:
    names = []
    for name in rows[0]:
        if name in STATIC_COLUMNS:
            continue
        ok = True
        for row in rows[:200]:
            try:
                float(row.get(name, 0.0))
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            names.append(name)
    return sorted(names)


def group_rows(rows: Sequence[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by[row["split"]].append(row)
    return by


def group_samples(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_index"])].append(row)
    out = []
    for sample_index, sample_rows in sorted(grouped.items()):
        sample_rows.sort(key=lambda row: base42.natural_label_key(row["candidate_label"]))
        base_rows = [row for row in sample_rows if row["candidate_label"] == row["base_label"]]
        base_row = base_rows[0] if base_rows else sample_rows[0]
        out.append(
            {
                "split": sample_rows[0]["split"],
                "sample_index": sample_index,
                "true_label": sample_rows[0]["true_label"],
                "base_label": sample_rows[0]["base_label"],
                "base_correct": int(sample_rows[0]["base_correct"]),
                "true_in_candidates": int(any(int(row["target"]) for row in sample_rows)),
                "base_row": base_row,
                "rows": sample_rows,
            }
        )
    return out


def smoothed_rate(correct: float, total: float, prior: float, strength: float) -> float:
    return (correct + prior * strength) / max(EPS, total + strength)


def build_reliability_tables(train_rows: Sequence[dict]) -> dict:
    total = len(train_rows)
    correct = sum(int(row["target"]) for row in train_rows)
    global_rate = correct / max(1, total)

    source_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    family_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    pattern_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    label_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    base_to_candidate_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    source_label_counts: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])

    for row in train_rows:
        target = int(row["target"])
        names = [name for name in row.get("source_names", "").split(";") if name]
        pattern = source_pattern(row)
        pattern_counts[pattern][0] += target
        pattern_counts[pattern][1] += 1.0
        label_counts[row["candidate_label"]][0] += target
        label_counts[row["candidate_label"]][1] += 1.0
        transition = f"{row['base_label']}->{row['candidate_label']}"
        base_to_candidate_counts[transition][0] += target
        base_to_candidate_counts[transition][1] += 1.0
        for name in names:
            source_counts[name][0] += target
            source_counts[name][1] += 1.0
            family = source_family(name)
            family_counts[family][0] += target
            family_counts[family][1] += 1.0
            key = f"{name}|{row['candidate_label']}"
            source_label_counts[key][0] += target
            source_label_counts[key][1] += 1.0

    return {
        "global_rate": global_rate,
        "source": source_counts,
        "family": family_counts,
        "pattern": pattern_counts,
        "label": label_counts,
        "transition": base_to_candidate_counts,
        "source_label": source_label_counts,
    }


def table_rate(table: dict[str, list[float]], key: str, prior: float, strength: float) -> tuple[float, float]:
    correct, total = table.get(key, [0.0, 0.0])
    return smoothed_rate(correct, total, prior, strength), total


def add_engineered_features(rows: Sequence[dict], rel: dict, num_cols: Sequence[str]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], int(row["sample_index"]))].append(dict(row))

    engineered = []
    global_rate = rel["global_rate"]
    for (_split, _sample_index), sample_rows in sorted(grouped.items()):
        base_rows = [row for row in sample_rows if row["candidate_label"] == row["base_label"]]
        base_row = base_rows[0] if base_rows else sample_rows[0]

        label_counts = Counter(row["candidate_label"] for row in sample_rows for _ in range(int(round(parse_float(row["source_hit_count"], 1.0)))))
        base_label = base_row["candidate_label"]
        top_count = max(label_counts.values()) if label_counts else 0
        base_count = label_counts.get(base_label, 0)

        base_strength = (
            0.50 * parse_float(base_row.get("top3_score4_norm"))
            + 0.35 * parse_float(base_row.get("top3_elite_vote_norm"))
            + 0.25 * parse_float(base_row.get("top3_template_reliability_norm"))
            + 0.20 * parse_float(base_row.get("top3_is_rssi_top1"))
            + 0.20 * parse_float(base_row.get("top3_is_v2_vote"))
            + 0.05 * parse_float(base_row.get("source_hit_count"))
        )

        for row in sample_rows:
            row = dict(row)
            candidate_label = row["candidate_label"]
            same_corridor, loc_delta, corridor_delta = label_distance(candidate_label, base_label)
            row["cand_vs_base_same_corridor"] = same_corridor
            row["cand_vs_base_loc_delta"] = loc_delta
            row["cand_vs_base_corridor_delta"] = corridor_delta
            row["cand_is_base"] = float(candidate_label == base_label)
            row["base_vote_count"] = float(base_count)
            row["base_vote_gap_vs_top"] = float(base_count - top_count)
            row["base_strength_score"] = base_strength

            for key in BASE_PROTECT_KEYS:
                row[f"base_{key}"] = parse_float(base_row.get(key))
            for key in DELTA_KEYS:
                row[f"delta_{key}"] = parse_float(row.get(key)) - parse_float(base_row.get(key))

            names = [name for name in row.get("source_names", "").split(";") if name]
            source_rates = []
            source_totals = []
            source_label_rates = []
            for name in names:
                rate, total = table_rate(rel["source"], name, global_rate, 20.0)
                source_rates.append(rate)
                source_totals.append(total)
                label_rate, _total = table_rate(rel["source_label"], f"{name}|{candidate_label}", rate, 8.0)
                source_label_rates.append(label_rate)
            if source_rates:
                row["source_rel_mean"] = sum(source_rates) / len(source_rates)
                row["source_rel_max"] = max(source_rates)
                row["source_rel_min"] = min(source_rates)
                row["source_rel_weighted"] = sum(rate * math.log1p(total) for rate, total in zip(source_rates, source_totals)) / max(
                    EPS, sum(math.log1p(total) for total in source_totals)
                )
                row["source_label_rel_mean"] = sum(source_label_rates) / len(source_label_rates)
                row["source_label_rel_max"] = max(source_label_rates)
            else:
                row["source_rel_mean"] = global_rate
                row["source_rel_max"] = global_rate
                row["source_rel_min"] = global_rate
                row["source_rel_weighted"] = global_rate
                row["source_label_rel_mean"] = global_rate
                row["source_label_rel_max"] = global_rate

            families = list(source_signature(row))
            family_rates = [table_rate(rel["family"], family, global_rate, 15.0)[0] for family in families]
            row["family_rel_mean"] = sum(family_rates) / len(family_rates) if family_rates else global_rate
            row["family_rel_max"] = max(family_rates) if family_rates else global_rate

            pattern = source_pattern(row)
            row["source_pattern_rel"], row["source_pattern_count"] = table_rate(rel["pattern"], pattern, global_rate, 10.0)
            row["candidate_label_rel"], row["candidate_label_count"] = table_rate(rel["label"], candidate_label, global_rate, 8.0)
            row["transition_rel"], row["transition_count"] = table_rate(rel["transition"], f"{base_label}->{candidate_label}", global_rate, 4.0)
            row["nonbase_reliability_score"] = (
                0.25 * parse_float(row["source_rel_weighted"])
                + 0.20 * parse_float(row["source_pattern_rel"])
                + 0.20 * parse_float(row["source_label_rel_mean"])
                + 0.15 * parse_float(row["transition_rel"])
                + 0.10 * parse_float(row["candidate_label_rel"])
                + 0.10 * parse_float(row["family_rel_mean"])
            )
            row["rescue_pressure"] = (
                0.60 * parse_float(row["nonbase_reliability_score"])
                + 0.20 * max(0.0, parse_float(row["delta_top3_candidate_mean_obs_norm"]))
                + 0.20 * max(0.0, parse_float(row["delta_top3_cost_veto41"]))
                - 0.18 * parse_float(row["base_strength_score"])
                - 0.12 * parse_float(row["cand_vs_base_loc_delta"])
            )
            engineered.append(row)
    return engineered


def feature_names_for(rows: Sequence[dict], feature_set: str) -> list[str]:
    all_names = numeric_columns(rows)
    if feature_set == "compact":
        keep = {
            "bias_feature",
            "cand_is_base",
            "cand_vs_base_same_corridor",
            "cand_vs_base_loc_delta",
            "cand_vs_base_corridor_delta",
            "source_hit_count",
            "source_hit_frac",
            "aco4_hit_count",
            "aco2_hit_count",
            "ablation_hit_count",
            "weak_hit_count",
            "raw_chirp_source_hit",
            "in_aco_top3",
            "is_raw_chirp_agreed_packet",
            "raw_chirp_min_margin_packet",
            "top3_candidate_mean_obs_norm",
            "top3_cost_veto41",
            "top3_elite_vote_norm",
            "top3_score4_norm",
            "top3_self_pheromone",
            "top3_template_reliability_norm",
            "top3_is_rssi_top1",
            "top3_is_v2_vote",
            "delta_source_hit_count",
            "delta_aco4_hit_count",
            "delta_aco2_hit_count",
            "delta_ablation_hit_count",
            "delta_weak_hit_count",
            "delta_raw_chirp_source_hit",
            "delta_top3_candidate_mean_obs_norm",
            "delta_top3_cost_veto41",
            "delta_top3_elite_vote_norm",
            "delta_top3_score4_norm",
            "delta_top3_self_pheromone",
            "delta_top3_template_reliability_norm",
            "delta_top3_is_rssi_top1",
            "delta_top3_is_v2_vote",
            "base_source_hit_count",
            "base_aco4_hit_count",
            "base_aco2_hit_count",
            "base_ablation_hit_count",
            "base_weak_hit_count",
            "base_raw_chirp_source_hit",
            "base_top3_is_rssi_top1",
            "base_top3_is_v2_vote",
            "base_top3_score4_norm",
            "base_top3_elite_vote_norm",
            "base_strength_score",
            "source_rel_mean",
            "source_rel_max",
            "source_rel_weighted",
            "source_label_rel_mean",
            "family_rel_mean",
            "source_pattern_rel",
            "candidate_label_rel",
            "transition_rel",
            "nonbase_reliability_score",
            "rescue_pressure",
        }
        return [name for name in all_names if name in keep]
    if feature_set == "reliability_only":
        return [
            name
            for name in all_names
            if name in {"bias_feature", "cand_is_base", "cand_vs_base_loc_delta", "base_strength_score"}
            or name.startswith("source_rel")
            or name.startswith("source_label_rel")
            or name.startswith("family_rel")
            or name.startswith("source_pattern")
            or name.startswith("candidate_label")
            or name.startswith("transition")
            or name in {"nonbase_reliability_score", "rescue_pressure"}
        ]
    if feature_set == "no_label":
        return [name for name in all_names if name not in {"candidate_corridor", "candidate_location_scaled"}]
    return all_names


def compute_scaler(groups: Sequence[dict], names: Sequence[str]) -> tuple[list[float], list[float]]:
    values = [[parse_float(row.get(name)) for name in names] for group in groups for row in group["rows"]]
    means = []
    stds = []
    for j in range(len(names)):
        col = [row[j] for row in values]
        mean = sum(col) / max(1, len(col))
        var = sum((value - mean) ** 2 for value in col) / max(1, len(col))
        means.append(mean)
        stds.append(math.sqrt(var) if var > EPS else 1.0)
    return means, stds


def vectorize(row: dict, names: Sequence[str], means: Sequence[float], stds: Sequence[float]) -> list[float]:
    return [(parse_float(row.get(name)) - means[j]) / stds[j] for j, name in enumerate(names)]


def train_softmax_selector(groups: Sequence[dict], names: Sequence[str], l2: float, epochs: int, lr: float, seed: int) -> dict:
    train_groups = [group for group in groups if group["true_in_candidates"]]
    means, stds = compute_scaler(train_groups, names)
    train_vectors = []
    for group in train_groups:
        targets = [int(row["target"]) for row in group["rows"]]
        pos = sum(targets)
        if pos <= 0:
            continue
        train_vectors.append(
            (
                [vectorize(row, names, means, stds) for row in group["rows"]],
                [target / pos for target in targets],
            )
        )
    rng = random.Random(seed)
    weights = [0.0 for _ in names]
    bias = 0.0
    for _epoch in range(epochs):
        shuffled = list(train_vectors)
        rng.shuffle(shuffled)
        grad = [0.0 for _ in names]
        grad_b = 0.0
        for xs, targets in shuffled:
            logits = [bias + sum(weights[j] * x[j] for j in range(len(weights))) for x in xs]
            max_logit = max(logits)
            exps = [math.exp(value - max_logit) for value in logits]
            denom = sum(exps) or 1.0
            probs = [value / denom for value in exps]
            for idx, target in enumerate(targets):
                err = probs[idx] - target
                grad_b += err
                x = xs[idx]
                for j in range(len(weights)):
                    grad[j] += err * x[j]
        denom = max(1, len(shuffled))
        for j in range(len(weights)):
            grad[j] = grad[j] / denom + l2 * weights[j]
            weights[j] -= lr * grad[j]
        bias -= lr * grad_b / denom
    return {"feature_names": list(names), "means": means, "stds": stds, "weights": weights, "bias": bias, "l2": l2}


def score_rows(group: dict, model: dict) -> list[dict]:
    names = model["feature_names"]
    rows = []
    for row in group["rows"]:
        x = vectorize(row, names, model["means"], model["stds"])
        score = model["bias"] + sum(model["weights"][j] * x[j] for j in range(len(names)))
        if model.get("protect_base") and row["candidate_label"] != group["base_label"]:
            score -= model["protect_base"] * parse_float(row.get("base_strength_score"))
        rows.append({"row": row, "score": score})
    rows.sort(key=lambda item: (-item["score"], base42.natural_label_key(item["row"]["candidate_label"])))
    return rows


def theta_grid(groups: Sequence[dict], model: dict) -> list[float]:
    margins = []
    for group in groups:
        scored = score_rows(group, model)
        top_label = scored[0]["row"]["candidate_label"]
        if top_label == group["base_label"]:
            continue
        score_by_label = {item["row"]["candidate_label"]: item["score"] for item in scored}
        base_score = score_by_label.get(group["base_label"], scored[-1]["score"])
        margins.append(scored[0]["score"] - base_score)
    if not margins:
        return [1e9]
    ordered = sorted(margins)
    qs = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0]
    values = sorted(set(round(ordered[round((len(ordered) - 1) * q)], 8) for q in qs))
    return [-1e9] + values + [value + 1e-6 for value in values] + [1e9]


def evaluate(groups: Sequence[dict], model: dict | None, theta: float | None, method: str) -> tuple[dict, list[dict]]:
    n = len(groups)
    base_correct = sum(group["base_correct"] for group in groups)
    final_correct = trigger = w2r = r2w = oracle = 0
    rows = []
    for group in groups:
        true_label = group["true_label"]
        base_label = group["base_label"]
        oracle += int(any(int(row["target"]) for row in group["rows"]))
        if model is None:
            final_label = base_label
            selector_label = ""
            selector_margin = 0.0
            top_score = ""
            base_score = ""
            source_names = ""
        else:
            scored = score_rows(group, model)
            selector_label = scored[0]["row"]["candidate_label"]
            source_names = scored[0]["row"].get("source_names", "")
            score_by_label = {item["row"]["candidate_label"]: item["score"] for item in scored}
            top_score = scored[0]["score"]
            base_score = score_by_label.get(base_label, scored[-1]["score"])
            selector_margin = top_score - base_score
            if theta is not None and (selector_label == base_label or selector_margin < theta):
                final_label = base_label
            else:
                final_label = selector_label
        base_ok = int(base_label == true_label)
        final_ok = int(final_label == true_label)
        final_correct += final_ok
        trigger += int(final_label != base_label)
        w2r += int((not base_ok) and final_ok)
        r2w += int(base_ok and not final_ok)
        rows.append(
            {
                "method": method,
                "split": group["split"],
                "sample_index": group["sample_index"],
                "true_label": true_label,
                "true_in_candidates": group["true_in_candidates"],
                "base_label": base_label,
                "base_correct": base_ok,
                "selector_label": selector_label,
                "selector_source_names": source_names,
                "final_label": final_label,
                "final_correct": final_ok,
                "triggered": int(final_label != base_label),
                "selector_margin": selector_margin,
                "selector_top_score": top_score,
                "base_score": base_score,
                "W2R": int((not base_ok) and final_ok),
                "R2W": int(base_ok and not final_ok),
            }
        )
    return {
        "method": method,
        "split": groups[0]["split"] if groups else "",
        "packet_count": n,
        "candidate_oracle_correct": oracle,
        "candidate_oracle_accuracy": oracle / n if n else 0.0,
        "base_correct": base_correct,
        "base_accuracy": base_correct / n if n else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / n if n else 0.0,
        "trigger_count": trigger,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "theta": "" if theta is None else theta,
    }, rows


def feature_importance(model: dict) -> list[dict]:
    rows = [
        {"feature": name, "weight": weight, "abs_weight": abs(weight)}
        for name, weight in zip(model["feature_names"], model["weights"])
    ]
    rows.sort(key=lambda row: (-row["abs_weight"], row["feature"]))
    return rows


def run(args: argparse.Namespace) -> dict:
    raw_rows = read_csv(args.candidate_path)
    by_split = group_rows(raw_rows)
    rel = build_reliability_tables(by_split["train_loocv"])
    engineered_rows = add_engineered_features(raw_rows, rel, numeric_columns(raw_rows))
    groups = {
        split: group_samples([row for row in engineered_rows if row["split"] == split])
        for split in ["train_loocv", "val", "test"]
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_fieldnames = ["split", "sample_index", "true_label", "candidate_label", "target", "base_label", "base_correct", "source_names"]
    all_fieldnames += [name for name in sorted({key for row in engineered_rows for key in row}) if name not in set(all_fieldnames)]
    write_csv(args.output_dir / "method_candidate_features_v45.csv", engineered_rows, all_fieldnames)

    summary_rows = []
    for split, split_groups in groups.items():
        metrics, _ = evaluate(split_groups, None, None, "v43_raw_chirp_base")
        summary_rows.append(metrics)

    best = None
    best_model = None
    for feature_set in [part.strip() for part in args.feature_sets.split(",") if part.strip()]:
        names = feature_names_for(engineered_rows, feature_set)
        for l2 in [float(part) for part in args.l2_grid.split(",") if part.strip()]:
            print(f"training feature_set={feature_set} l2={l2}", flush=True)
            model0 = train_softmax_selector(groups["train_loocv"], names, l2, args.epochs, args.learning_rate, args.seed)
            for protect_base in [float(part) for part in args.protect_base_grid.split(",") if part.strip()]:
                model = dict(model0)
                model["protect_base"] = protect_base
                raw_metrics, _ = evaluate(groups["val"], model, None, f"aco_v45_{feature_set}_raw")
                raw_metrics["feature_set"] = feature_set
                raw_metrics["l2"] = l2
                raw_metrics["protect_base"] = protect_base
                raw_metrics["selection"] = "raw_selector"
                summary_rows.append(raw_metrics)
                for theta in theta_grid(groups["val"], model):
                    metrics, _ = evaluate(groups["val"], model, theta, f"aco_v45_{feature_set}_gate")
                    metrics["feature_set"] = feature_set
                    metrics["l2"] = l2
                    metrics["protect_base"] = protect_base
                    metrics["selection"] = "theta_on_val"
                    summary_rows.append(metrics)
                    score_tuple = (
                        metrics["final_accuracy"],
                        metrics["net_gain"],
                        -metrics["R2W"],
                        metrics["W2R"],
                        -metrics["trigger_count"],
                    )
                    if best is None or score_tuple > (
                        best["final_accuracy"],
                        best["net_gain"],
                        -best["R2W"],
                        best["W2R"],
                        -best["trigger_count"],
                    ):
                        best = metrics
                        best_model = model
    assert best is not None and best_model is not None

    final_rows = []
    for split, split_groups in groups.items():
        metrics, preds = evaluate(split_groups, best_model, parse_float(best["theta"]), "aco_v45_selected")
        metrics["feature_set"] = best["feature_set"]
        metrics["l2"] = best["l2"]
        metrics["protect_base"] = best["protect_base"]
        final_rows.append(metrics)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))

    write_csv(args.output_dir / "aco_v45_selection_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "aco_v45_final_summary.csv", final_rows, list(final_rows[0].keys()))
    write_csv(args.output_dir / "aco_v45_feature_importance.csv", feature_importance(best_model), ["feature", "weight", "abs_weight"])

    payload = {
        "protocol": "Train reliability-aware selector on train_loocv, choose threshold on val, evaluate test once.",
        "best_val": best,
        "final": final_rows,
        "model": {
            "feature_names": best_model["feature_names"],
            "weights": best_model["weights"],
            "bias": best_model["bias"],
            "means": best_model["means"],
            "stds": best_model["stds"],
            "l2": best_model["l2"],
            "protect_base": best_model.get("protect_base", 0.0),
        },
    }
    with (args.output_dir / "aco_v45_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-sets", default="compact,reliability_only,no_label,all")
    parser.add_argument("--l2-grid", default="0.0003,0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--protect-base-grid", default="0,0.05,0.1,0.2,0.35")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "final": payload["final"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
