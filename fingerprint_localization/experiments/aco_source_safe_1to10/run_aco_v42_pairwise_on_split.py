#!/usr/bin/env python3
"""ACO 4.2 pairwise challenger correction over RSSI+ Top3."""

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
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_aco_v42_reranker_on_split as base42  # noqa: E402


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "aco_v42_pairwise"
EPS = 1e-12


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def feature_names_for(groups: Sequence[dict], feature_set: str) -> list[str]:
    names = base42.feature_names_for(groups, feature_set)
    # Pairwise diffs of exact label IDs are brittle; keep relational features.
    drop = {"candidate_corridor", "candidate_location_scaled", "bias_feature"}
    return [name for name in names if name not in drop]


def make_pairs(groups: Sequence[dict], feature_names: Sequence[str]) -> list[dict]:
    pairs = []
    for group in groups:
        incumbent = None
        for row in group["rows"]:
            if row["candidate_label"] == group["base_label"]:
                incumbent = row
                break
        if incumbent is None:
            continue
        for challenger in group["rows"]:
            if challenger["candidate_label"] == incumbent["candidate_label"]:
                continue
            diff = {
                name: challenger["features"].get(name, 0.0) - incumbent["features"].get(name, 0.0)
                for name in feature_names
            }
            pairs.append(
                {
                    "split": group["split"],
                    "sample_index": group["sample_index"],
                    "file_name": group["file_name"],
                    "packet_index": group["packet_index"],
                    "true_label": group["true_label"],
                    "true_in_top3": group["true_in_top3"],
                    "base_label": group["base_label"],
                    "base_correct": group["base_correct"],
                    "challenger_label": challenger["candidate_label"],
                    "target": int(challenger["candidate_label"] == group["true_label"]),
                    "diff": diff,
                }
            )
    return pairs


def standardizer(pairs: Sequence[dict], feature_names: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    means = {}
    stds = {}
    for name in feature_names:
        values = [pair["diff"].get(name, 0.0) for pair in pairs]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        means[name] = mean
        stds[name] = math.sqrt(var) if var > EPS else 1.0
    return means, stds


def vectorize(pair: dict, names: Sequence[str], means: dict[str, float], stds: dict[str, float]) -> list[float]:
    return [(pair["diff"].get(name, 0.0) - means[name]) / stds[name] for name in names]


def train_model(pairs: Sequence[dict], names: Sequence[str], l2: float, pos_weight: float, epochs: int, lr: float) -> dict:
    means, stds = standardizer(pairs, names)
    xs = [vectorize(pair, names, means, stds) for pair in pairs]
    ys = [pair["target"] for pair in pairs]
    weights = [0.0 for _ in names]
    bias = 0.0
    total_weight = sum(pos_weight if y else 1.0 for y in ys) or 1.0
    for _ in range(epochs):
        grad = [0.0 for _ in names]
        grad_b = 0.0
        for x, y in zip(xs, ys):
            p = sigmoid(bias + sum(weights[j] * x[j] for j in range(len(weights))))
            sw = pos_weight if y else 1.0
            err = (p - y) * sw
            grad_b += err
            for j in range(len(weights)):
                grad[j] += err * x[j]
        for j in range(len(weights)):
            grad[j] = grad[j] / total_weight + l2 * weights[j]
            weights[j] -= lr * grad[j]
        bias -= lr * grad_b / total_weight
    return {"feature_names": list(names), "means": means, "stds": stds, "weights": weights, "bias": bias, "l2": l2, "pos_weight": pos_weight}


def pair_prob(pair: dict, model: dict) -> float:
    x = vectorize(pair, model["feature_names"], model["means"], model["stds"])
    return sigmoid(model["bias"] + sum(model["weights"][j] * x[j] for j in range(len(x))))


def evaluate(groups: Sequence[dict], pairs: Sequence[dict], model: dict, threshold: float, method: str) -> tuple[dict, list[dict]]:
    by_sample = defaultdict(list)
    for pair in pairs:
        row = dict(pair)
        row["probability"] = pair_prob(pair, model)
        by_sample[pair["sample_index"]].append(row)
    for rows in by_sample.values():
        rows.sort(key=lambda row: (-row["probability"], base42.natural_label_key(row["challenger_label"])))
    pred_rows = []
    final_correct = triggers = w2r = r2w = 0
    for group in groups:
        candidates = by_sample.get(group["sample_index"], [])
        best = candidates[0] if candidates else None
        if best and best["probability"] >= threshold:
            final_label = best["challenger_label"]
            prob = best["probability"]
        else:
            final_label = group["base_label"]
            prob = best["probability"] if best else 0.0
        base_ok = int(group["base_label"] == group["true_label"])
        final_ok = int(final_label == group["true_label"])
        final_correct += final_ok
        triggers += int(final_label != group["base_label"])
        w2r += int((not base_ok) and final_ok)
        r2w += int(base_ok and not final_ok)
        pred_rows.append(
            {
                "method": method,
                "split": group["split"],
                "sample_index": group["sample_index"],
                "file_name": group["file_name"],
                "packet_index": group["packet_index"],
                "true_label": group["true_label"],
                "true_in_top3": group["true_in_top3"],
                "base_label": group["base_label"],
                "base_correct": base_ok,
                "final_label": final_label,
                "final_correct": final_ok,
                "triggered": int(final_label != group["base_label"]),
                "best_challenger_label": best["challenger_label"] if best else "",
                "best_challenger_probability": prob,
                "threshold": threshold,
            }
        )
    n = len(groups)
    base_correct = sum(group["base_correct"] for group in groups)
    return {
        "method": method,
        "split": groups[0]["split"] if groups else "",
        "packet_count": n,
        "top3_recall": sum(group["true_in_top3"] for group in groups) / n if n else 0.0,
        "base_correct": base_correct,
        "base_accuracy": base_correct / n if n else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / n if n else 0.0,
        "trigger_count": triggers,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "threshold": threshold,
    }, pred_rows


def choose_best(rows: Sequence[dict]) -> dict:
    return max(rows, key=lambda row: (row["final_accuracy"], row["net_gain"], -row["R2W"], -row["trigger_count"]))


def threshold_grid(pairs: Sequence[dict], model: dict) -> list[float]:
    probs = sorted(pair_prob(pair, model) for pair in pairs)
    qs = [0.0, 0.50, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0]
    values = sorted(set(round(probs[round((len(probs) - 1) * q)], 8) for q in qs))
    return [0.0] + values + [value + 1e-6 for value in values] + [1.0]


def weight_rows(model: dict) -> list[dict]:
    rows = [
        {"feature": name, "weight": weight, "abs_weight": abs(weight)}
        for name, weight in zip(model["feature_names"], model["weights"])
    ]
    rows.sort(key=lambda row: (-row["abs_weight"], row["feature"]))
    return rows


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = {split: base42.build_groups(args.aco_v4_dir, args.aco_v41_dir, split) for split in ["train_loocv", "val", "test"]}
    summary_rows = []
    best = None
    best_model = None
    best_pairs = None
    for feature_set in ["core", "with_v41", "all"]:
        names = feature_names_for(groups["train_loocv"], feature_set)
        for l2 in [float(part) for part in args.l2_grid.split(",") if part.strip()]:
            train_pairs = make_pairs(groups["train_loocv"], names)
            pos = sum(pair["target"] for pair in train_pairs)
            neg = len(train_pairs) - pos
            auto_pos_weight = neg / max(1, pos)
            for pw in [float(part) if float(part) > 0 else auto_pos_weight for part in args.positive_weight_grid.split(",") if part.strip()]:
                print(f"training pairwise feature_set={feature_set} l2={l2} pos_weight={pw:.3g}", flush=True)
                model = train_model(train_pairs, names, l2, pw, args.epochs, args.learning_rate)
                val_pairs = make_pairs(groups["val"], names)
                for threshold in threshold_grid(val_pairs, model):
                    metrics, _ = evaluate(groups["val"], val_pairs, model, threshold, f"aco_v42_pairwise_{feature_set}")
                    metrics["feature_set"] = feature_set
                    metrics["l2"] = l2
                    metrics["positive_weight"] = pw
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
                        best_pairs = {split: make_pairs(groups[split], names) for split in ["train_loocv", "val", "test"]}
    assert best is not None and best_model is not None and best_pairs is not None
    final_rows = []
    for split in ["train_loocv", "val", "test"]:
        metrics, preds = evaluate(groups[split], best_pairs[split], best_model, float(best["threshold"]), "aco_v42_pairwise_selected")
        metrics["feature_set"] = best["feature_set"]
        metrics["l2"] = best["l2"]
        metrics["positive_weight"] = best["positive_weight"]
        final_rows.append(metrics)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))
    write_csv(args.output_dir / "aco_v42_pairwise_selection_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "aco_v42_pairwise_final_summary.csv", final_rows, list(final_rows[0].keys()))
    write_csv(args.output_dir / "aco_v42_pairwise_feature_importance.csv", weight_rows(best_model), ["feature", "weight", "abs_weight"])
    payload = {
        "protocol": "Pairwise challenger model trained on train_loocv, threshold selected on val, test evaluated once.",
        "best_val": best,
        "final": final_rows,
        "model": {
            "feature_names": best_model["feature_names"],
            "weights": best_model["weights"],
            "bias": best_model["bias"],
            "means": best_model["means"],
            "stds": best_model["stds"],
            "l2": best_model["l2"],
            "positive_weight": best_model["pos_weight"],
        },
    }
    with (args.output_dir / "aco_v42_pairwise_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aco-v4-dir", type=Path, default=base42.DEFAULT_ACO_V4_DIR)
    parser.add_argument("--aco-v41-dir", type=Path, default=base42.DEFAULT_ACO_V41_DIR)
    parser.add_argument("--l2-grid", default="0.003,0.01,0.03,0.1,0.3")
    parser.add_argument("--positive-weight-grid", default="0,1,2,4")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "final": payload["final"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
