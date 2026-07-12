#!/usr/bin/env python3
"""Pairwise logistic replacement model for ML-ACO 2.0.

For each packet, the ACO 2.0 vote winner is treated as the incumbent. Every
other RSSI+ Top-3 candidate becomes a challenger, represented by
`challenger_features - incumbent_features`. The model predicts whether to
replace the ACO winner with the challenger.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


PAIRWISE_DIR = Path(__file__).resolve().parent
MODEL_V3_DIR = PAIRWISE_DIR.parent
if str(MODEL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_V3_DIR))

import aco_packet_path_v2 as aco2  # noqa: E402
import ml_aco_v2_ranker as mlbase  # noqa: E402


DEFAULT_BASE_DIR = PAIRWISE_DIR / "output_gaussian_noise_1to10_group_safe"
DEFAULT_FEATURE_CSV = DEFAULT_BASE_DIR / "ml_candidate_features.csv"
DEFAULT_SPLIT_CSV = DEFAULT_BASE_DIR / "group_safe_split_assignments.csv"
DEFAULT_METRICS_JSON = DEFAULT_BASE_DIR / "ml_aco_v2_ranker_metrics.json"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "pairwise_correction"
EPS = 1e-12

META_COLUMNS = {
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
}


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


def sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def load_feature_rows(path: Path) -> tuple[list[dict], list[str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"No candidate feature rows found in {path}")
    feature_names = [name for name in rows[0].keys() if name not in META_COLUMNS]
    for row in rows:
        row["sample_index"] = int(row["sample_index"])
        row["packet_index"] = int(float(row["packet_index"]))
        row["target"] = int(float(row["target"]))
        row["true_in_top3"] = int(float(row["true_in_top3"]))
        row["features"] = {name: parse_float(row.get(name)) for name in feature_names}
    return rows, feature_names


def group_rows(rows: Sequence[dict]) -> dict[str, list[dict]]:
    by_split: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {"rows": []}))
    for row in rows:
        group = by_split[row["split"]][row["sample_index"]]
        group["rows"].append(row)
        for key in [
            "split",
            "sample_index",
            "base_packet_key",
            "file_name",
            "packet_index",
            "true_label",
            "true_in_top3",
            "aco_vote_label",
        ]:
            group[key] = row[key]
    out = {}
    for split, groups in by_split.items():
        out[split] = []
        for group in groups.values():
            group["rows"].sort(key=lambda item: aco2.natural_label_key(item["candidate_label"]))
            group["aco_vote_correct"] = int(group["aco_vote_label"] == group["true_label"])
            out[split].append(group)
        out[split].sort(key=lambda group: group["sample_index"])
    return out


def make_pairwise_dataset(groups_by_split: dict[str, list[dict]], feature_names: Sequence[str]) -> dict[str, list[dict]]:
    pairs_by_split = {}
    for split, groups in groups_by_split.items():
        pairs = []
        for group in groups:
            incumbent = None
            for row in group["rows"]:
                if row["candidate_label"] == group["aco_vote_label"]:
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
                label = int(challenger["candidate_label"] == group["true_label"])
                pairs.append(
                    {
                        "split": split,
                        "sample_index": group["sample_index"],
                        "base_packet_key": group["base_packet_key"],
                        "file_name": group["file_name"],
                        "packet_index": group["packet_index"],
                        "true_label": group["true_label"],
                        "true_in_top3": group["true_in_top3"],
                        "aco_vote_label": group["aco_vote_label"],
                        "aco_vote_correct": group["aco_vote_correct"],
                        "challenger_label": challenger["candidate_label"],
                        "target": label,
                        "diff": diff,
                    }
                )
        pairs_by_split[split] = pairs
    return pairs_by_split


def flatten_pair_rows(pairs_by_split: dict[str, list[dict]], feature_names: Sequence[str]) -> list[dict]:
    rows = []
    for pairs in pairs_by_split.values():
        for pair in pairs:
            row = {key: value for key, value in pair.items() if key != "diff"}
            row.update({f"diff_{name}": pair["diff"].get(name, 0.0) for name in feature_names})
            rows.append(row)
    return rows


def standardizer(pairs: Sequence[dict], feature_names: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    means = {}
    stds = {}
    for name in feature_names:
        values = [pair["diff"].get(name, 0.0) for pair in pairs]
        if not values:
            values = [0.0]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        means[name] = mean
        stds[name] = std if std > EPS else 1.0
    return means, stds


def vectorize(pair: dict, feature_names: Sequence[str], means: dict[str, float], stds: dict[str, float]) -> list[float]:
    return [(pair["diff"].get(name, 0.0) - means[name]) / stds[name] for name in feature_names]


def train_pairwise_logistic(
    pairs: Sequence[dict],
    feature_names: Sequence[str],
    l2: float,
    positive_weight: float,
    epochs: int,
    learning_rate: float,
) -> dict:
    means, stds = standardizer(pairs, feature_names)
    xs = [vectorize(pair, feature_names, means, stds) for pair in pairs]
    ys = [int(pair["target"]) for pair in pairs]
    dim = len(feature_names)
    weights = [0.0] * dim
    bias = 0.0
    total_weight = sum(positive_weight if y else 1.0 for y in ys) or 1.0
    for _epoch in range(epochs):
        grad = [0.0] * dim
        grad_b = 0.0
        for x, y in zip(xs, ys):
            p = sigmoid(bias + sum(weights[j] * x[j] for j in range(dim)))
            sample_weight = positive_weight if y else 1.0
            err = (p - y) * sample_weight
            grad_b += err
            for j in range(dim):
                grad[j] += err * x[j]
        for j in range(dim):
            grad[j] = grad[j] / total_weight + l2 * weights[j]
            weights[j] -= learning_rate * grad[j]
        bias -= learning_rate * grad_b / total_weight
    return {
        "feature_names": list(feature_names),
        "means": means,
        "stds": stds,
        "weights": weights,
        "bias": bias,
        "l2": l2,
        "positive_weight": positive_weight,
    }


def pair_probability(pair: dict, model: dict) -> float:
    x = vectorize(pair, model["feature_names"], model["means"], model["stds"])
    return sigmoid(model["bias"] + sum(model["weights"][j] * x[j] for j in range(len(x))))


def build_pair_lookup(pairs: Sequence[dict], model: dict) -> dict[int, list[dict]]:
    lookup = defaultdict(list)
    for pair in pairs:
        row = dict(pair)
        row["probability"] = pair_probability(pair, model)
        lookup[pair["sample_index"]].append(row)
    for sample_pairs in lookup.values():
        sample_pairs.sort(
            key=lambda item: (-item["probability"], aco2.natural_label_key(item["challenger_label"]))
        )
    return lookup


def evaluate_replacement(
    groups: Sequence[dict],
    pairs: Sequence[dict],
    model: dict,
    threshold: float,
    method: str,
) -> tuple[dict, list[dict]]:
    pair_lookup = build_pair_lookup(pairs, model)
    n = len(groups)
    base_correct = sum(group["aco_vote_correct"] for group in groups)
    final_correct = 0
    trigger_count = 0
    w2r = 0
    r2w = 0
    prediction_rows = []
    for group in groups:
        best_pair = pair_lookup.get(group["sample_index"], [])
        if best_pair and best_pair[0]["probability"] >= threshold:
            final_label = best_pair[0]["challenger_label"]
            challenger_prob = best_pair[0]["probability"]
        else:
            final_label = group["aco_vote_label"]
            challenger_prob = best_pair[0]["probability"] if best_pair else 0.0
        base_ok = int(group["aco_vote_label"] == group["true_label"])
        final_ok = int(final_label == group["true_label"])
        final_correct += final_ok
        trigger = int(final_label != group["aco_vote_label"])
        trigger_count += trigger
        w2r += int((not base_ok) and final_ok)
        r2w += int(base_ok and not final_ok)
        prediction_rows.append(
            {
                "method": method,
                "split": group["split"],
                "sample_index": group["sample_index"],
                "base_packet_key": group["base_packet_key"],
                "file_name": group["file_name"],
                "packet_index": group["packet_index"],
                "true_label": group["true_label"],
                "true_in_top3": group["true_in_top3"],
                "aco_vote_label": group["aco_vote_label"],
                "final_label": final_label,
                "base_correct": base_ok,
                "final_correct": final_ok,
                "triggered": trigger,
                "best_challenger_label": best_pair[0]["challenger_label"] if best_pair else "",
                "best_challenger_probability": challenger_prob,
                "threshold": threshold,
            }
        )
    metrics = {
        "method": method,
        "split": groups[0]["split"] if groups else "",
        "packet_count": n,
        "baseline_correct": base_correct,
        "baseline_accuracy": base_correct / n if n else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / n if n else 0.0,
        "trigger_count": trigger_count,
        "W2R": w2r,
        "R2W": r2w,
        "net": w2r - r2w,
        "threshold": threshold,
    }
    return metrics, prediction_rows


def choose_best(rows: Sequence[dict]) -> dict:
    return max(rows, key=lambda row: (row["final_accuracy"], row["net"], -row["R2W"], -row["trigger_count"]))


def weight_rows(model: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    for name, weight in zip(model["feature_names"], model["weights"]):
        rows.append({"feature": name, "weight": weight, "abs_weight": abs(weight)})
    positive = sorted([row for row in rows if row["weight"] >= 0], key=lambda row: (-row["weight"], row["feature"]))
    negative = sorted([row for row in rows if row["weight"] < 0], key=lambda row: (row["weight"], row["feature"]))
    return positive, negative


def split_audit(split_csv: Path, metrics_json: Path | None) -> dict:
    rows = read_csv(split_csv)
    audit = mlbase.leakage_audit(rows)
    source_audit = {}
    if metrics_json and metrics_json.exists():
        with metrics_json.open(encoding="utf-8") as f:
            source_audit = json.load(f).get("split_info", {}).get("source_split_audit", {})
    return {"group_safe_split_audit": audit, "source_split_audit": source_audit}


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows, feature_names = load_feature_rows(args.feature_csv)
    groups_by_split = group_rows(candidate_rows)
    pairs_by_split = make_pairwise_dataset(groups_by_split, feature_names)
    flat_pairs = flatten_pair_rows(pairs_by_split, feature_names)
    pair_fields = [
        "split",
        "sample_index",
        "base_packet_key",
        "file_name",
        "packet_index",
        "true_label",
        "true_in_top3",
        "aco_vote_label",
        "aco_vote_correct",
        "challenger_label",
        "target",
    ] + [f"diff_{name}" for name in feature_names]
    write_csv(args.output_dir / "pairwise_correction_dataset.csv", flat_pairs, pair_fields)

    train_pairs = pairs_by_split["train_loocv"]
    pos_count = sum(pair["target"] for pair in train_pairs)
    neg_count = len(train_pairs) - pos_count
    positive_weight = args.positive_weight
    if positive_weight <= 0:
        positive_weight = neg_count / max(1, pos_count)

    model = train_pairwise_logistic(
        train_pairs,
        feature_names,
        args.l2,
        positive_weight,
        args.epochs,
        args.learning_rate,
    )

    sweep_rows = []
    for threshold in [float(item) for item in args.threshold_grid.split(",") if item.strip()]:
        metrics, _preds = evaluate_replacement(
            groups_by_split["val"],
            pairs_by_split["val"],
            model,
            threshold,
            "pairwise_logistic_replacement",
        )
        sweep_rows.append(metrics)
    best_val = choose_best(sweep_rows)
    threshold = float(best_val["threshold"])

    summary_rows = []
    prediction_rows = []
    for split in ["train_loocv", "val", "test"]:
        metrics, preds = evaluate_replacement(
            groups_by_split[split],
            pairs_by_split[split],
            model,
            threshold,
            "pairwise_logistic_replacement",
        )
        summary_rows.append(metrics)
        prediction_rows.extend(preds)

    positive_rows, negative_rows = weight_rows(model)
    write_csv(args.output_dir / "pairwise_threshold_sweep.csv", sweep_rows, list(sweep_rows[0].keys()))
    write_csv(args.output_dir / "pairwise_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(args.output_dir / "pairwise_predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    write_csv(args.output_dir / "pairwise_top_positive_features.csv", positive_rows, ["feature", "weight", "abs_weight"])
    write_csv(args.output_dir / "pairwise_top_negative_features.csv", negative_rows, ["feature", "weight", "abs_weight"])

    audit = split_audit(args.split_csv, args.metrics_json)
    payload = {
        "method": "group-safe pairwise logistic replacement model",
        "data_policy": "Uses existing group-safe ML-ACO split and ACO 2.0 candidate features.",
        "leakage_audit": audit,
        "pair_counts": {
            split: {
                "pairs": len(pairs),
                "positive_pairs": sum(pair["target"] for pair in pairs),
                "negative_pairs": len(pairs) - sum(pair["target"] for pair in pairs),
            }
            for split, pairs in pairs_by_split.items()
        },
        "training": {
            "l2": args.l2,
            "positive_weight": positive_weight,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
        "selected_threshold": threshold,
        "val_selection": best_val,
        "summary": summary_rows,
        "top_positive_features": positive_rows[: args.top_n],
        "top_negative_features": negative_rows[: args.top_n],
    }
    with (args.output_dir / "pairwise_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-grid", default="0.9,0.92,0.94,0.95,0.96,0.97,0.98,0.985,0.99,0.992,0.995")
    parser.add_argument("--positive-weight", type=float, default=0.0, help="<=0 uses neg/pos automatic weighting")
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.12)
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["leakage_audit"], indent=2, ensure_ascii=False))
    for row in payload["summary"]:
        print(json.dumps(row, ensure_ascii=False))
    print("top_positive_features")
    for row in payload["top_positive_features"][:10]:
        print(json.dumps(row, ensure_ascii=False))
    print("top_negative_features")
    for row in payload["top_negative_features"][:10]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
