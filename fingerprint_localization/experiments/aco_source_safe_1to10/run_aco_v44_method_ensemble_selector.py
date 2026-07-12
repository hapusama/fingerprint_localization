#!/usr/bin/env python3
"""ACO 4.4 method-ensemble selector.

The ACO4.x and raw/chirp experiments leave complementary errors with KNN/MFR
and ACO2 ablation variants.  This script treats each method prediction as a
candidate label, trains a sample-wise softmax selector on train_loocv, chooses
a conservative replacement threshold on validation, and evaluates test once.
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
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "aco_v44_method_ensemble_selector"
EPS = 1e-12


METHOD_SPECS = [
    ("v43_raw_chirp", "aco_v43_reranker_raw_chirp_features", "final_label", "aco4"),
    ("v42_interactions", "aco_v42_reranker_interactions", "final_label", "aco4"),
    ("v43_challenger", "aco_v43_raw_chirp_challenger", "final_label", "aco4"),
    ("v42_reranker", "aco_v42_reranker", "final_label", "aco4"),
    ("v4_best", "aco_v4_best", "aco_vote_label", "aco4"),
    ("v41_softgate", "aco_v41_softgate_best", "aco_vote_label", "aco4"),
    ("v4_q4w005", "aco_v4_q4w005", "aco_vote_label", "aco4"),
    ("v4_tseg005", "aco_v4_tseg_0p05", "aco_vote_label", "aco4"),
    ("aco_v2", "aco_v2", "aco_vote_label", "aco2"),
    ("knn", "knn", "pred_label", "weak"),
    ("mfr_prev", "mfr_aco_prev_best", "aco_vote_label", "weak"),
]

ABLATION_SPECS = [
    ("ab_v1_vote", "v1_0", "aco_vote_label", "ablation"),
    ("ab_v1_pheromone", "v1_0", "aco_pheromone_label", "ablation"),
    ("ab_v2_1_vote", "v2_1", "aco_vote_label", "ablation"),
    ("ab_v2_2_vote", "v2_2", "aco_vote_label", "ablation"),
    ("ab_v2_5_vote", "v2_5", "aco_vote_label", "ablation"),
    ("ab_v2_7_vote", "v2_7", "aco_vote_label", "ablation"),
]

TOP3_FEATURE_KEYS = [
    "candidate_mean_obs",
    "candidate_mean_obs_norm",
    "candidate_cost_norm",
    "score4_norm",
    "score41",
    "cost_veto",
    "cost_veto41",
    "template_reliability_norm",
    "self_pheromone",
    "elite_vote_norm",
    "rssi_rank_inv",
    "is_rssi_top1",
    "is_raw_winner",
    "is_v2_vote",
    "v2_disagrees_v4_supports_candidate",
    "v2_vote_x_raw_winner",
    "v2_vote_x_low_cost",
    "raw_chirp_agree_winner",
    "raw_chirp_any_winner",
    "raw_chirp_winner_count",
    "raw_chirp_agree_x_low_cost",
    "raw_chirp_min_margin_x_agree_winner",
    "v21_raw_gaussian_winner",
    "v22_chirp_shrink_winner",
    "v21_raw_margin_x_winner",
    "v22_chirp_margin_x_winner",
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
    return base42.parse_float(value, default)


def label_distance(a: str, b: str) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 1.0
    ca, la = base42.parse_label(a)
    cb, lb = base42.parse_label(b)
    return float(ca == cb), abs(la - lb) / 54.0


def load_method_predictions(results_dir: Path, split: str) -> dict[str, dict[int, dict]]:
    out = {}
    for name, folder, label_col, family in METHOD_SPECS:
        path = results_dir / folder / f"{split}_predictions.csv"
        rows = read_csv(path)
        out[name] = {
            int(row["sample_index"]): {
                "label": row[label_col],
                "true_label": row["true_label"],
                "family": family,
            }
            for row in rows
        }
    return out


def load_ablation_predictions(results_dir: Path, split: str) -> dict[str, dict[int, dict]]:
    out = {}
    for name, version, label_col, family in ABLATION_SPECS:
        path = results_dir / "aco_v2_ablation" / version / f"{split}_predictions.csv"
        rows = read_csv(path)
        out[name] = {
            int(row["sample_index"]): {
                "label": row[label_col],
                "true_label": row["true_label"],
                "family": family,
            }
            for row in rows
        }
    return out


def load_top3_features(results_dir: Path) -> dict[str, dict[int, dict[str, dict]]]:
    path = results_dir / "aco_v43_reranker_raw_chirp_features" / "aco_v43_candidate_features.csv"
    grouped: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in read_csv(path):
        grouped[row["split"]][int(row["sample_index"])][row["candidate_label"]] = row
    return grouped


def load_raw_chirp_packets(results_dir: Path) -> dict[str, dict[int, dict]]:
    path = results_dir / "aco_v43_raw_chirp_challenger" / "packet_challenger_features.csv"
    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in read_csv(path):
        grouped[row["split"]][int(row["sample_index"])] = row
    return grouped


def build_groups(args: argparse.Namespace, split: str, top3_features: dict, raw_packets: dict) -> list[dict]:
    method_preds = load_method_predictions(args.results_dir, split)
    ablation_preds = load_ablation_predictions(args.results_dir, split)
    sources = {**method_preds, **ablation_preds}
    sample_ids = sorted(set.intersection(*(set(rows) for rows in sources.values())))
    groups = []
    for sample_index in sample_ids:
        true_label = next(iter(sources.values()))[sample_index]["true_label"]
        labels_by_source = {name: rows[sample_index]["label"] for name, rows in sources.items()}
        family_by_source = {name: rows[sample_index]["family"] for name, rows in sources.items()}
        packet = raw_packets.get(split, {}).get(sample_index, {})
        raw_winner = packet.get("raw_winner", "")
        chirp_winner = packet.get("chirp_winner", "")
        agreed_winner = packet.get("agreed_winner", "")
        for extra_label in [raw_winner, chirp_winner, agreed_winner]:
            if extra_label:
                labels_by_source.setdefault(f"expert_extra_{extra_label}", extra_label)
                family_by_source.setdefault(f"expert_extra_{extra_label}", "raw_chirp")
        label_counts = Counter(labels_by_source.values())
        top_count = max(label_counts.values()) if label_counts else 0
        ordered_counts = sorted(label_counts.values(), reverse=True)
        second_count = ordered_counts[1] if len(ordered_counts) > 1 else 0
        v43_label = labels_by_source.get("v43_raw_chirp", "")
        v42_label = labels_by_source.get("v42_interactions", "")
        v2_label = labels_by_source.get("aco_v2", "")
        knn_label = labels_by_source.get("knn", "")
        mfr_label = labels_by_source.get("mfr_prev", "")
        top3_by_label = top3_features.get(split, {}).get(sample_index, {})
        rssi_label = ""
        if top3_by_label:
            rssi_label = next(iter(top3_by_label.values())).get("rssi_top1_label", "")
        rows = []
        for label in sorted(label_counts, key=base42.natural_label_key):
            hit_sources = [name for name, pred_label in labels_by_source.items() if pred_label == label]
            family_counts = Counter(family_by_source[name] for name in hit_sources)
            same_v43, delta_v43 = label_distance(label, v43_label)
            same_v42, delta_v42 = label_distance(label, v42_label)
            same_v2, delta_v2 = label_distance(label, v2_label)
            same_rssi, delta_rssi = label_distance(label, rssi_label)
            same_knn, delta_knn = label_distance(label, knn_label)
            same_mfr, delta_mfr = label_distance(label, mfr_label)
            candidate_features = top3_by_label.get(label, {})
            features = {
                "bias_feature": 1.0,
                "source_hit_count": float(len(hit_sources)),
                "source_hit_frac": len(hit_sources) / max(1, len(labels_by_source)),
                "is_consensus_top": float(label_counts[label] == top_count),
                "consensus_margin": float(label_counts[label] - second_count),
                "aco4_hit_count": float(family_counts["aco4"]),
                "aco2_hit_count": float(family_counts["aco2"]),
                "ablation_hit_count": float(family_counts["ablation"]),
                "weak_hit_count": float(family_counts["weak"]),
                "raw_chirp_source_hit": float(family_counts["raw_chirp"]),
                "same_corridor_as_v43": same_v43,
                "abs_loc_delta_v43": delta_v43,
                "same_corridor_as_v42": same_v42,
                "abs_loc_delta_v42": delta_v42,
                "same_corridor_as_v2": same_v2,
                "abs_loc_delta_v2": delta_v2,
                "same_corridor_as_rssi": same_rssi,
                "abs_loc_delta_rssi": delta_rssi,
                "same_corridor_as_knn": same_knn,
                "abs_loc_delta_knn": delta_knn,
                "same_corridor_as_mfr": same_mfr,
                "abs_loc_delta_mfr": delta_mfr,
                "is_raw_winner_packet": float(label == raw_winner),
                "is_chirp_winner_packet": float(label == chirp_winner),
                "is_raw_chirp_agreed_packet": float(bool(agreed_winner) and label == agreed_winner),
                "raw_margin_v21": parse_float(packet.get("raw_margin_v21"), 0.0),
                "chirp_margin_v22": parse_float(packet.get("chirp_margin_v22"), 0.0),
                "raw_chirp_min_margin_packet": min(
                    parse_float(packet.get("raw_margin_v21"), 0.0),
                    parse_float(packet.get("chirp_margin_v22"), 0.0),
                ),
                "in_aco_top3": float(label in top3_by_label),
            }
            for name in METHOD_SPECS:
                method_name = name[0]
                features[f"src_{method_name}"] = float(labels_by_source.get(method_name) == label)
            for name in ABLATION_SPECS:
                method_name = name[0]
                features[f"src_{method_name}"] = float(labels_by_source.get(method_name) == label)
            for key in TOP3_FEATURE_KEYS:
                features[f"top3_{key}"] = parse_float(candidate_features.get(key), 0.0)
            ca, la = base42.parse_label(label)
            features["candidate_corridor"] = float(ca)
            features["candidate_location_scaled"] = la / 54.0
            rows.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "true_label": true_label,
                    "candidate_label": label,
                    "target": int(label == true_label),
                    "base_label": v43_label,
                    "base_correct": int(v43_label == true_label),
                    "features": features,
                    "source_names": ";".join(hit_sources),
                }
            )
        groups.append(
            {
                "split": split,
                "sample_index": sample_index,
                "true_label": true_label,
                "base_label": v43_label,
                "base_correct": int(v43_label == true_label),
                "true_in_candidates": int(true_label in label_counts),
                "rows": rows,
            }
        )
    return groups


def feature_names_for(groups: Sequence[dict], feature_set: str) -> list[str]:
    all_names = sorted({name for group in groups for row in group["rows"] for name in row["features"]})
    if feature_set == "no_label":
        return [name for name in all_names if name not in {"candidate_corridor", "candidate_location_scaled"}]
    if feature_set == "consensus":
        keep_prefixes = ("src_",)
        keep = {
            "bias_feature",
            "source_hit_count",
            "source_hit_frac",
            "is_consensus_top",
            "consensus_margin",
            "aco4_hit_count",
            "aco2_hit_count",
            "ablation_hit_count",
            "weak_hit_count",
            "raw_chirp_source_hit",
            "same_corridor_as_v43",
            "abs_loc_delta_v43",
            "same_corridor_as_v42",
            "abs_loc_delta_v42",
            "same_corridor_as_v2",
            "abs_loc_delta_v2",
            "same_corridor_as_rssi",
            "abs_loc_delta_rssi",
            "in_aco_top3",
            "is_raw_chirp_agreed_packet",
            "raw_chirp_min_margin_packet",
        }
        return [name for name in all_names if name in keep or name.startswith(keep_prefixes)]
    return all_names


def row_passes_source_policy(row: dict, policy: str) -> bool:
    if policy == "all":
        return True
    features = row["features"]
    if row["candidate_label"] == row["base_label"]:
        return True
    weak_only = features.get("weak_hit_count", 0.0) > 0.0 and (
        features.get("aco4_hit_count", 0.0)
        + features.get("aco2_hit_count", 0.0)
        + features.get("ablation_hit_count", 0.0)
        + features.get("raw_chirp_source_hit", 0.0)
    ) <= 0.0
    if policy == "no_weak_only" and weak_only:
        return False
    if policy == "weak_requires_top3_support" and weak_only:
        top3_support = (
            features.get("in_aco_top3", 0.0) > 0.0
            and (
                features.get("top3_candidate_mean_obs_norm", 0.0) >= 0.5
                or features.get("top3_raw_chirp_agree_winner", 0.0) > 0.0
                or features.get("top3_is_v2_vote", 0.0) > 0.0
            )
        )
        if not top3_support:
            return False
    if policy == "weak_requires_consensus" and weak_only:
        return False
    return True


def compute_scaler(groups: Sequence[dict], names: Sequence[str]) -> tuple[list[float], list[float]]:
    values = [[row["features"].get(name, 0.0) for name in names] for group in groups for row in group["rows"]]
    means = []
    stds = []
    for j in range(len(names)):
        col = [row[j] for row in values]
        mean = sum(col) / len(col)
        var = sum((value - mean) ** 2 for value in col) / len(col)
        means.append(mean)
        stds.append(math.sqrt(var) if var > EPS else 1.0)
    return means, stds


def vectorize(row: dict, names: Sequence[str], means: Sequence[float], stds: Sequence[float]) -> list[float]:
    return [(row["features"].get(name, 0.0) - means[j]) / stds[j] for j, name in enumerate(names)]


def train_softmax_selector(groups: Sequence[dict], names: Sequence[str], l2: float, epochs: int, lr: float, seed: int) -> dict:
    train_groups = [group for group in groups if group["true_in_candidates"]]
    means, stds = compute_scaler(train_groups, names)
    train_vectors = []
    for group in train_groups:
        targets = [row["target"] for row in group["rows"]]
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
        if not row_passes_source_policy(row, model.get("source_policy", "all")):
            continue
        x = vectorize(row, names, model["means"], model["stds"])
        score = model["bias"] + sum(model["weights"][j] * x[j] for j in range(len(names)))
        rows.append({"row": row, "score": score})
    if not rows:
        for row in group["rows"]:
            if row["candidate_label"] == group["base_label"]:
                return [{"row": row, "score": 0.0}]
        return [{"row": group["rows"][0], "score": 0.0}]
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
        return [-1e9, 1e9]
    ordered = sorted(margins)
    qs = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 1.0]
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
        oracle += int(any(row["candidate_label"] == true_label for row in group["rows"]))
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
            source_names = scored[0]["row"]["source_names"]
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    top3_features = load_top3_features(args.results_dir)
    raw_packets = load_raw_chirp_packets(args.results_dir)
    groups = {split: build_groups(args, split, top3_features, raw_packets) for split in ["train_loocv", "val", "test"]}

    candidate_rows = []
    for split_groups in groups.values():
        for group in split_groups:
            for row in group["rows"]:
                flat = {key: value for key, value in row.items() if key not in {"features"}}
                flat.update(row["features"])
                candidate_rows.append(flat)
    static = {"split", "sample_index", "true_label", "candidate_label", "target", "base_label", "base_correct", "source_names"}
    feature_names_all = sorted({key for row in candidate_rows for key in row if key not in static})
    write_csv(
        args.output_dir / "method_candidate_features.csv",
        candidate_rows,
        ["split", "sample_index", "true_label", "candidate_label", "target", "base_label", "base_correct", "source_names"] + feature_names_all,
    )

    summary_rows = []
    for split, split_groups in groups.items():
        metrics, _ = evaluate(split_groups, None, None, "v43_raw_chirp_base")
        summary_rows.append(metrics)

    best = None
    best_model = None
    for feature_set in [part.strip() for part in args.feature_sets.split(",") if part.strip()]:
        names = feature_names_for(groups["train_loocv"], feature_set)
        for l2 in [float(part) for part in args.l2_grid.split(",") if part.strip()]:
            print(f"training feature_set={feature_set} l2={l2}", flush=True)
            base_model = train_softmax_selector(groups["train_loocv"], names, l2, args.epochs, args.learning_rate, args.seed)
            for source_policy in [part.strip() for part in args.source_policies.split(",") if part.strip()]:
                model = dict(base_model)
                model["source_policy"] = source_policy
                raw_metrics, _ = evaluate(groups["val"], model, None, f"aco_v44_{feature_set}_{source_policy}_raw")
                raw_metrics["feature_set"] = feature_set
                raw_metrics["l2"] = l2
                raw_metrics["source_policy"] = source_policy
                raw_metrics["selection"] = "raw_selector"
                summary_rows.append(raw_metrics)
                for theta in theta_grid(groups["val"], model):
                    metrics, _ = evaluate(groups["val"], model, theta, f"aco_v44_{feature_set}_{source_policy}_gate")
                    metrics["feature_set"] = feature_set
                    metrics["l2"] = l2
                    metrics["source_policy"] = source_policy
                    metrics["selection"] = "theta_on_val"
                    summary_rows.append(metrics)
                    if best is None or (
                        metrics["final_accuracy"],
                        metrics["net_gain"],
                        -metrics["R2W"],
                        metrics["W2R"],
                        -metrics["trigger_count"],
                    ) > (
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
        metrics, preds = evaluate(split_groups, best_model, parse_float(best["theta"], 0.0), "aco_v44_selected")
        metrics["feature_set"] = best["feature_set"]
        metrics["l2"] = best["l2"]
        metrics["source_policy"] = best["source_policy"]
        final_rows.append(metrics)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))
    write_csv(args.output_dir / "aco_v44_selection_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "aco_v44_final_summary.csv", final_rows, list(final_rows[0].keys()))
    write_csv(args.output_dir / "aco_v44_feature_importance.csv", feature_importance(best_model), ["feature", "weight", "abs_weight"])
    payload = {
        "protocol": "Train method-ensemble label selector on train_loocv, choose conservative threshold on val, evaluate test once.",
        "method_specs": METHOD_SPECS,
        "ablation_specs": ABLATION_SPECS,
        "best_val": best,
        "final": final_rows,
        "model": {
            "feature_names": best_model["feature_names"],
            "weights": best_model["weights"],
            "bias": best_model["bias"],
            "means": best_model["means"],
            "stds": best_model["stds"],
            "l2": best_model["l2"],
            "source_policy": best_model.get("source_policy", "all"),
        },
    }
    with (args.output_dir / "aco_v44_method_ensemble_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--feature-sets", default="consensus,no_label,all")
    parser.add_argument("--l2-grid", default="0.001,0.003,0.01,0.03,0.1,0.3")
    parser.add_argument("--source-policies", default="all,no_weak_only,weak_requires_top3_support,weak_requires_consensus")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "final": payload["final"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
