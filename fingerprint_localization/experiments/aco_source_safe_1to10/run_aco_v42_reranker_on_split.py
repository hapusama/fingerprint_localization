#!/usr/bin/env python3
"""ACO 4.2: supervised Top-3 reranker over ACO 4.0 candidates.

Protocol:
- Keep the original 1:10 augmented 6:2:2 split.
- Keep RSSI+ Top3 as the only candidate set.
- Train a packet-wise softmax ranker on train_loocv candidate rows.
- Select the model and conservative replacement threshold on val.
- Evaluate test once with the selected configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "aco_v42_reranker"
DEFAULT_ACO_V4_DIR = DEFAULT_RESULTS_DIR / "aco_v4_best"
DEFAULT_ACO_V41_DIR = DEFAULT_RESULTS_DIR / "aco_v41_softgate_best"
DEFAULT_ACO_V2_DIR = DEFAULT_RESULTS_DIR / "aco_v2"
EPS = 1e-12


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


def parse_label(label: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)_(\d+)$", label)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def natural_label_key(label: str) -> tuple[int, int]:
    return parse_label(label)


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


def rank_inv(values: dict[str, float], high_better: bool) -> dict[str, float]:
    labels = list(values)
    if high_better:
        ordered = sorted(labels, key=lambda label: (-values[label], natural_label_key(label)))
    else:
        ordered = sorted(labels, key=lambda label: (values[label], natural_label_key(label)))
    denom = max(1, len(ordered) - 1)
    return {label: 1.0 - rank / denom for rank, label in enumerate(ordered)}


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


def numeric(row: dict, key: str) -> float:
    return parse_float(row.get(key), 0.0)


def load_split(aco_v4_dir: Path, split: str) -> list[dict]:
    preds = {row["sample_index"]: row for row in read_csv(aco_v4_dir / f"{split}_predictions.csv")}
    cands = defaultdict(dict)
    for row in read_csv(aco_v4_dir / f"{split}_candidate_scores.csv"):
        cands[row["sample_index"]][row["candidate_label"]] = row
    return [{"prediction": preds[idx], "candidates": cands[idx]} for idx in sorted(preds, key=lambda x: int(x))]


def load_v41_candidates(aco_v41_dir: Path, split: str) -> dict[str, dict[str, dict]]:
    path = aco_v41_dir / f"{split}_candidate_scores.csv"
    if not path.exists():
        return {}
    out = defaultdict(dict)
    for row in read_csv(path):
        out[row["sample_index"]][row["candidate_label"]] = row
    return out


def load_aux_candidates(aco_dir: Path, split: str) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    pred_path = aco_dir / f"{split}_predictions.csv"
    cand_path = aco_dir / f"{split}_candidate_scores.csv"
    preds = {row["sample_index"]: row for row in read_csv(pred_path)} if pred_path.exists() else {}
    cands = defaultdict(dict)
    if cand_path.exists():
        for row in read_csv(cand_path):
            cands[row["sample_index"]][row["candidate_label"]] = row
    return preds, cands


def best_path_counts(path_labels: str, candidates: Sequence[str]) -> dict[str, float]:
    labels = [part for part in path_labels.split(";") if part]
    total = len(labels) or 1
    counts = Counter(labels)
    return {label: counts.get(label, 0) / total for label in candidates}


def build_groups(aco_v4_dir: Path, aco_v41_dir: Path, split: str, aco_v2_dir: Path | None = None) -> list[dict]:
    base_rows = load_split(aco_v4_dir, split)
    v41 = load_v41_candidates(aco_v41_dir, split)
    v2_preds, v2_cands = load_aux_candidates(aco_v2_dir, split) if aco_v2_dir else ({}, {})
    groups = []
    for item in base_rows:
        pred = item["prediction"]
        candidates = list(item["candidates"])
        true_label = pred["true_label"]
        base_label = pred["aco_score4_label"]
        rssi_label = pred["rssi_top1_label"]
        raw_label = pred.get("raw_winner_label", "")
        v2_pred = v2_preds.get(pred["sample_index"], {})
        v2_vote_label = v2_pred.get("aco_vote_label", "")
        v2_pheromone_label = v2_pred.get("aco_pheromone_label", "")
        v2_path_label = v2_pred.get("aco_path_mode_label", "")
        path_rate = best_path_counts(pred.get("best_path_labels", ""), candidates)
        c0, l0 = parse_label(rssi_label)
        base_c, base_l = parse_label(base_label)
        metric_maps = {
            "self_pheromone": {label: numeric(item["candidates"][label], "self_pheromone") for label in candidates},
            "elite_vote": {label: numeric(item["candidates"][label], "elite_vote") for label in candidates},
            "score4": {label: numeric(item["candidates"][label], "score4") for label in candidates},
            "candidate_mean_obs": {label: numeric(item["candidates"][label], "candidate_mean_obs") for label in candidates},
            "candidate_cost_norm": {label: numeric(item["candidates"][label], "candidate_cost_norm") for label in candidates},
            "template_reliability": {label: numeric(item["candidates"][label], "template_reliability") for label in candidates},
            "cost_veto": {label: numeric(item["candidates"][label], "cost_veto") for label in candidates},
        }
        high_metrics = {"self_pheromone", "elite_vote", "score4", "template_reliability", "cost_veto"}
        norm_high = {name: normalize_high(values) if name not in high_metrics else normalize_low(values) for name, values in metric_maps.items()}
        ranks = {name: rank_inv(values, name in high_metrics) for name, values in metric_maps.items()}
        rows = []
        for rank, label in enumerate(candidates):
            row = item["candidates"][label]
            v41row = v41.get(pred["sample_index"], {}).get(label, {})
            v2row = v2_cands.get(pred["sample_index"], {}).get(label, {})
            cand_c, cand_l = parse_label(label)
            features = {
                "bias_feature": 1.0,
                "rssi_rank_inv": 1.0 - rank / max(1, len(candidates) - 1),
                "is_rssi_top1": float(label == rssi_label),
                "is_aco_base": float(label == base_label),
                "is_raw_winner": numeric(row, "is_raw_winner"),
                "raw_margin": numeric(row, "raw_margin"),
                "same_corridor_as_rssi": float(cand_c == c0),
                "same_corridor_as_base": float(cand_c == base_c),
                "abs_loc_delta_rssi": abs(cand_l - l0) / 54.0,
                "abs_loc_delta_base": abs(cand_l - base_l) / 54.0,
                "candidate_corridor": cand_c,
                "candidate_location_scaled": cand_l / 54.0,
                "best_path_candidate_rate": path_rate[label],
                "best_path_garbage_count": numeric(pred, "best_path_garbage_count"),
                "Q_seg": numeric(row, "Q_seg"),
                "segment_cost_std": numeric(row, "segment_cost_std"),
                "alpha_shrink": numeric(row, "alpha_shrink"),
                "R_ws": numeric(v41row, "R_ws"),
                "Sep_gate": numeric(v41row, "Sep_gate"),
                "q_s_mean": numeric(v41row, "q_s_mean"),
                "cost_veto41": numeric(v41row, "cost_veto"),
                "score41": numeric(v41row, "score41"),
                "v2_self_pheromone": numeric(v2row, "self_pheromone"),
                "v2_elite_vote": numeric(v2row, "elite_vote"),
                "v2_template_reliability": numeric(v2row, "template_reliability"),
                "is_v2_vote": float(label == v2_vote_label),
                "is_v2_pheromone": float(label == v2_pheromone_label),
                "is_v2_path_mode": float(label == v2_path_label),
                "v2_agrees_v4": float(v2_vote_label == base_label and label == base_label),
                "v2_disagrees_v4_supports_candidate": float(v2_vote_label != base_label and label == v2_vote_label),
            }
            for name in metric_maps:
                features[name] = metric_maps[name][label]
                features[f"{name}_norm"] = norm_high[name][label]
                features[f"{name}_rank_inv"] = ranks[name][label]
            features["score4_margin"] = margin_high(metric_maps["score4"])
            features["vote_margin"] = margin_high(metric_maps["elite_vote"])
            features["pheromone_margin"] = margin_high(metric_maps["self_pheromone"])
            features["cost_margin"] = margin_low(metric_maps["candidate_mean_obs"])
            features["raw_margin_x_is_raw_winner"] = features["raw_margin"] * features["is_raw_winner"]
            features["rssi_x_score4_norm"] = features["is_rssi_top1"] * features["score4_norm"]
            features["rssi_x_low_cost"] = features["is_rssi_top1"] * features["candidate_mean_obs_norm"]
            features["base_x_score_margin"] = features["is_aco_base"] * features["score4_margin"]
            features["raw_x_low_cost"] = features["is_raw_winner"] * features["candidate_mean_obs_norm"]
            features["rel_sep_product"] = features["R_ws"] * features["Sep_gate"]
            features["v41_score_x_rel"] = features["score41"] * features["R_ws"]
            features["v2_vote_x_score4_norm"] = features["is_v2_vote"] * features["score4_norm"]
            features["v2_vote_x_low_cost"] = features["is_v2_vote"] * features["candidate_mean_obs_norm"]
            features["v2_vote_x_v4_disagree"] = features["is_v2_vote"] * float(v2_vote_label != base_label)
            features["v2_vote_x_raw_winner"] = features["is_v2_vote"] * features["is_raw_winner"]
            features["v2_vote_x_rssi_top1"] = features["is_v2_vote"] * features["is_rssi_top1"]
            features["v2_vote_x_template_rel"] = features["is_v2_vote"] * features["template_reliability_norm"]
            rows.append(
                {
                    "split": split,
                    "sample_index": int(pred["sample_index"]),
                    "file_name": pred["file_name"],
                    "packet_index": pred["packet_index"],
                    "true_label": true_label,
                    "candidate_label": label,
                    "target": int(label == true_label),
                    "true_in_top3": int(pred["true_in_rssi_topk"]),
                    "base_label": base_label,
                    "base_correct": int(base_label == true_label),
                    "rssi_top1_label": rssi_label,
                    "features": features,
                }
            )
        groups.append(
            {
                "split": split,
                "sample_index": int(pred["sample_index"]),
                "file_name": pred["file_name"],
                "packet_index": pred["packet_index"],
                "true_label": true_label,
                "true_in_top3": int(pred["true_in_rssi_topk"]),
                "base_label": base_label,
                "base_correct": int(base_label == true_label),
                "rows": rows,
            }
        )
    return groups


def feature_names_for(groups: Sequence[dict], feature_set: str) -> list[str]:
    all_names = sorted({name for group in groups for row in group["rows"] for name in row["features"]})
    if feature_set == "core":
        keep = {
            "rssi_rank_inv", "is_rssi_top1", "is_aco_base", "is_raw_winner", "raw_margin",
            "self_pheromone_norm", "elite_vote_norm", "score4_norm", "candidate_mean_obs_norm",
            "candidate_cost_norm_norm", "template_reliability_norm", "cost_veto_norm",
            "self_pheromone_rank_inv", "elite_vote_rank_inv", "score4_rank_inv", "candidate_mean_obs_rank_inv",
            "score4_margin", "vote_margin", "pheromone_margin", "cost_margin",
            "best_path_candidate_rate", "Q_seg", "segment_cost_std", "alpha_shrink",
        }
        return [name for name in all_names if name in keep]
    if feature_set == "with_v41":
        return [name for name in all_names if name not in {"candidate_corridor", "candidate_location_scaled"} and not name.startswith("v2_") and not name.startswith("is_v2")]
    if feature_set == "with_v2":
        return [
            name for name in all_names
            if name not in {"candidate_corridor", "candidate_location_scaled"}
            and (not name.startswith("R_ws"))
        ]
    return all_names


def compute_scaler(groups: Sequence[dict], feature_names: Sequence[str]) -> tuple[list[float], list[float]]:
    values = []
    for group in groups:
        for row in group["rows"]:
            values.append([row["features"].get(name, 0.0) for name in feature_names])
    means = []
    stds = []
    for j in range(len(feature_names)):
        col = [row[j] for row in values]
        mean = sum(col) / len(col)
        var = sum((value - mean) ** 2 for value in col) / len(col)
        means.append(mean)
        stds.append(math.sqrt(var) if var > EPS else 1.0)
    return means, stds


def vectorize(row: dict, feature_names: Sequence[str], means: Sequence[float], stds: Sequence[float]) -> list[float]:
    return [(row["features"].get(name, 0.0) - means[j]) / stds[j] for j, name in enumerate(feature_names)]


def train_softmax_ranker(
    groups: Sequence[dict],
    feature_names: Sequence[str],
    l2: float,
    epochs: int,
    lr: float,
    seed: int,
) -> dict:
    train_groups = [group for group in groups if group["true_in_top3"]]
    means, stds = compute_scaler(train_groups, feature_names)
    rng = random.Random(seed)
    weights = [0.0 for _ in feature_names]
    bias = 0.0
    train_vectors = [
        (
            [vectorize(row, feature_names, means, stds) for row in group["rows"]],
            [row["target"] for row in group["rows"]],
        )
        for group in train_groups
    ]
    for _epoch in range(epochs):
        shuffled = list(train_vectors)
        rng.shuffle(shuffled)
        grad = [0.0 for _ in feature_names]
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
    return {
        "feature_names": list(feature_names),
        "means": means,
        "stds": stds,
        "weights": weights,
        "bias": bias,
        "l2": l2,
    }


def score_rows(group: dict, model: dict) -> list[dict]:
    names = model["feature_names"]
    rows = []
    for row in group["rows"]:
        x = vectorize(row, names, model["means"], model["stds"])
        score = model["bias"] + sum(model["weights"][j] * x[j] for j in range(len(names)))
        rows.append({"row": row, "score": score})
    rows.sort(key=lambda item: (-item["score"], natural_label_key(item["row"]["candidate_label"])))
    return rows


def evaluate(groups: Sequence[dict], model: dict | None, method: str, theta: float | None = None) -> tuple[dict, list[dict]]:
    n = len(groups)
    base_correct = sum(group["base_correct"] for group in groups)
    final_correct = 0
    trigger = 0
    w2r = 0
    r2w = 0
    rows = []
    for group in groups:
        true_label = group["true_label"]
        base_label = group["base_label"]
        if model is None:
            final_label = base_label
            ml_label = ""
            ml_margin = 0.0
            base_margin = 0.0
            top_score = ""
            base_score = ""
        else:
            scored = score_rows(group, model)
            ml_label = scored[0]["row"]["candidate_label"]
            score_by_label = {item["row"]["candidate_label"]: item["score"] for item in scored}
            top_score = scored[0]["score"]
            base_score = score_by_label.get(base_label, scored[-1]["score"])
            ml_margin = scored[0]["score"] - (scored[1]["score"] if len(scored) > 1 else scored[0]["score"])
            base_margin = top_score - base_score
            if theta is not None and (ml_label == base_label or base_margin < theta):
                final_label = base_label
            else:
                final_label = ml_label
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
                "file_name": group["file_name"],
                "packet_index": group["packet_index"],
                "true_label": true_label,
                "true_in_top3": group["true_in_top3"],
                "base_label": base_label,
                "base_correct": base_ok,
                "ml_label": ml_label,
                "final_label": final_label,
                "final_correct": final_ok,
                "triggered": int(final_label != base_label),
                "ml_margin": ml_margin,
                "base_margin": base_margin,
                "ml_top_score": top_score,
                "base_score": base_score,
            }
        )
    metrics = {
        "method": method,
        "split": groups[0]["split"] if groups else "",
        "packet_count": n,
        "top3_recall": sum(group["true_in_top3"] for group in groups) / n if n else 0.0,
        "base_correct": base_correct,
        "base_accuracy": base_correct / n if n else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / n if n else 0.0,
        "trigger_count": trigger,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "theta": "" if theta is None else theta,
    }
    return metrics, rows


def choose_best(rows: Sequence[dict]) -> dict:
    return max(rows, key=lambda row: (row["final_accuracy"], row["net_gain"], -row["R2W"], -row["trigger_count"]))


def theta_grid(groups: Sequence[dict], model: dict) -> list[float]:
    margins = []
    for group in groups:
        scored = score_rows(group, model)
        ml_label = scored[0]["row"]["candidate_label"]
        if ml_label == group["base_label"]:
            continue
        score_by_label = {item["row"]["candidate_label"]: item["score"] for item in scored}
        margins.append(scored[0]["score"] - score_by_label.get(group["base_label"], scored[-1]["score"]))
    if not margins:
        return [-1e9, 1e9]
    ordered = sorted(margins)
    qs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0]
    values = sorted(set(round(ordered[round((len(ordered) - 1) * q)], 8) for q in qs))
    return [-1e9] + values + [value + 1e-6 for value in values] + [1e9]


def feature_importance(model: dict) -> list[dict]:
    rows = [
        {"feature": name, "weight": weight, "abs_weight": abs(weight)}
        for name, weight in zip(model["feature_names"], model["weights"])
    ]
    rows.sort(key=lambda row: (-row["abs_weight"], row["feature"]))
    return rows


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        split: build_groups(args.aco_v4_dir, args.aco_v41_dir, split, args.aco_v2_dir)
        for split in ["train_loocv", "val", "test"]
    }
    all_candidate_rows = []
    for split_groups in groups.values():
        for group in split_groups:
            for row in group["rows"]:
                flat = {key: value for key, value in row.items() if key != "features"}
                flat.update(row["features"])
                all_candidate_rows.append(flat)
    feature_names_all = sorted({key for row in all_candidate_rows for key in row if key not in {
        "split", "sample_index", "file_name", "packet_index", "true_label", "candidate_label", "target",
        "true_in_top3", "base_label", "base_correct", "rssi_top1_label",
    }})
    write_csv(
        args.output_dir / "aco_v42_candidate_features.csv",
        all_candidate_rows,
        [
            "split", "sample_index", "file_name", "packet_index", "true_label", "candidate_label", "target",
            "true_in_top3", "base_label", "base_correct", "rssi_top1_label",
        ] + feature_names_all,
    )

    summary_rows = []
    prediction_rows = []
    for split, split_groups in groups.items():
        metrics, preds = evaluate(split_groups, None, "aco_v4_base")
        summary_rows.append(metrics)
        prediction_rows.extend(preds)

    best = None
    best_model = None
    for feature_set in ["core", "with_v41", "with_v2", "all"]:
        names = feature_names_for(groups["train_loocv"], feature_set)
        for l2 in [float(part) for part in args.l2_grid.split(",") if part.strip()]:
            print(f"training feature_set={feature_set} l2={l2}", flush=True)
            model = train_softmax_ranker(groups["train_loocv"], names, l2, args.epochs, args.learning_rate, args.seed)
            raw_metrics, _ = evaluate(groups["val"], model, f"aco_v42_{feature_set}_raw", None)
            raw_metrics["feature_set"] = feature_set
            raw_metrics["l2"] = l2
            raw_metrics["selection"] = "raw_ml"
            summary_rows.append(raw_metrics)
            for theta in theta_grid(groups["val"], model):
                metrics, _ = evaluate(groups["val"], model, f"aco_v42_{feature_set}_gate", theta)
                metrics["feature_set"] = feature_set
                metrics["l2"] = l2
                metrics["selection"] = "theta_on_val"
                summary_rows.append(metrics)
                if best is None or (
                    metrics["final_accuracy"],
                    metrics["net_gain"],
                    -metrics["R2W"],
                    -metrics["trigger_count"],
                ) > (
                    best["final_accuracy"],
                    best["net_gain"],
                    -best["R2W"],
                    -best["trigger_count"],
                ):
                    best = metrics
                    best_model = model
    assert best is not None and best_model is not None

    final_rows = []
    for split, split_groups in groups.items():
        metrics, preds = evaluate(split_groups, best_model, "aco_v42_selected", parse_float(best["theta"], 0.0))
        metrics["feature_set"] = best["feature_set"]
        metrics["l2"] = best["l2"]
        final_rows.append(metrics)
        prediction_rows.extend(preds)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))

    write_csv(args.output_dir / "aco_v42_selection_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "aco_v42_final_summary.csv", final_rows, list(final_rows[0].keys()))
    write_csv(args.output_dir / "aco_v42_feature_importance.csv", feature_importance(best_model), ["feature", "weight", "abs_weight"])
    payload = {
        "protocol": "Train Top3 softmax reranker on train_loocv, choose conservative replacement threshold on val, evaluate test once.",
        "best_val": best,
        "final": final_rows,
        "model": {
            "feature_names": best_model["feature_names"],
            "weights": best_model["weights"],
            "bias": best_model["bias"],
            "means": best_model["means"],
            "stds": best_model["stds"],
            "l2": best_model["l2"],
        },
    }
    with (args.output_dir / "aco_v42_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aco-v4-dir", type=Path, default=DEFAULT_ACO_V4_DIR)
    parser.add_argument("--aco-v41-dir", type=Path, default=DEFAULT_ACO_V41_DIR)
    parser.add_argument("--aco-v2-dir", type=Path, default=DEFAULT_ACO_V2_DIR)
    parser.add_argument("--l2-grid", default="0.003,0.01,0.03,0.1,0.3")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "final": payload["final"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
