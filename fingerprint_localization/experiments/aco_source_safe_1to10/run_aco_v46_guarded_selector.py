#!/usr/bin/env python3
"""ACO 4.6 guarded reliability selector.

This freezes the useful V4.5 finding: keep the reliability-aware selector, but
veto two unstable challenger families before thresholding:

* weak-source candidates that are not in the ACO Top3;
* ACO2+ablation candidates without weak-source support.

The model is still trained by ACO4.5.  This script only applies guards, chooses
the replacement margin on validation, and evaluates test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_aco_v42_reranker_on_split as base42  # noqa: E402


RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_V45_DIR = RESULTS_DIR / "aco_v45_reliability_selector"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "aco_v46_guarded_selector"


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def load_model(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    return payload["model"]


def score_row(row: dict, model: dict) -> float:
    score = model["bias"]
    for name, mean, std, weight in zip(model["feature_names"], model["means"], model["stds"], model["weights"]):
        score += weight * ((parse_float(row.get(name)) - mean) / std)
    return score


def candidate_is_guarded(row: dict) -> bool:
    if row["candidate_label"] == row["base_label"]:
        return False
    weak_not_top3 = parse_float(row.get("weak_hit_count")) > 0.0 and parse_float(row.get("in_aco_top3")) <= 0.0
    v2_ablation_without_weak = (
        parse_float(row.get("aco2_hit_count")) > 0.0
        and parse_float(row.get("ablation_hit_count")) > 0.0
        and parse_float(row.get("weak_hit_count")) <= 0.0
    )
    return weak_not_top3 or v2_ablation_without_weak


def group_samples(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["sample_index"]), []).append(row)
    samples = []
    for sample_index, sample_rows in sorted(grouped.items()):
        sample_rows.sort(key=lambda row: base42.natural_label_key(row["candidate_label"]))
        samples.append(
            {
                "split": sample_rows[0]["split"],
                "sample_index": sample_index,
                "true_label": sample_rows[0]["true_label"],
                "base_label": sample_rows[0]["base_label"],
                "base_correct": int(sample_rows[0]["base_label"] == sample_rows[0]["true_label"]),
                "true_in_candidates": int(any(int(row["target"]) for row in sample_rows)),
                "rows": sample_rows,
            }
        )
    return samples


def score_groups(groups: Sequence[dict], model: dict) -> None:
    for group in groups:
        for row in group["rows"]:
            row["_selector_score"] = score_row(row, model)


def best_allowed(group: dict) -> tuple[dict, float, float]:
    allowed = [row for row in group["rows"] if not candidate_is_guarded(row)]
    if not allowed:
        allowed = [row for row in group["rows"] if row["candidate_label"] == group["base_label"]] or group["rows"][:1]
    allowed.sort(key=lambda row: (-parse_float(row.get("_selector_score")), base42.natural_label_key(row["candidate_label"])))
    top = allowed[0]
    base_row = next((row for row in group["rows"] if row["candidate_label"] == group["base_label"]), allowed[-1])
    top_score = parse_float(top.get("_selector_score"))
    base_score = parse_float(base_row.get("_selector_score"))
    return top, top_score, base_score


def theta_grid(groups: Sequence[dict]) -> list[float]:
    values = []
    for group in groups:
        top, top_score, base_score = best_allowed(group)
        if top["candidate_label"] != group["base_label"]:
            values.append(top_score - base_score)
    if not values:
        return [math.inf]
    ordered = sorted(values)
    out = [-1e9, 1e9]
    for value in ordered:
        out.append(value)
        out.append(value + 1e-9)
    return sorted(set(out))


def evaluate(groups: Sequence[dict], theta: float, method: str) -> tuple[dict, list[dict]]:
    n = len(groups)
    base_correct = final_correct = trigger = w2r = r2w = oracle = guarded_top = 0
    pred_rows = []
    for group in groups:
        true_label = group["true_label"]
        base_label = group["base_label"]
        oracle += int(any(int(row["target"]) for row in group["rows"]))
        top, top_score, base_score = best_allowed(group)
        selector_label = top["candidate_label"]
        selector_margin = top_score - base_score
        if selector_label == base_label or selector_margin < theta:
            final_label = base_label
        else:
            final_label = selector_label
        guarded_top += int(any(candidate_is_guarded(row) for row in group["rows"]))
        base_ok = int(base_label == true_label)
        final_ok = int(final_label == true_label)
        base_correct += base_ok
        final_correct += final_ok
        trigger += int(final_label != base_label)
        w2r += int((not base_ok) and final_ok)
        r2w += int(base_ok and not final_ok)
        pred_rows.append(
            {
                "method": method,
                "split": group["split"],
                "sample_index": group["sample_index"],
                "true_label": true_label,
                "true_in_candidates": group["true_in_candidates"],
                "base_label": base_label,
                "base_correct": base_ok,
                "selector_label": selector_label,
                "selector_source_names": top.get("source_names", ""),
                "final_label": final_label,
                "final_correct": final_ok,
                "triggered": int(final_label != base_label),
                "selector_margin": selector_margin,
                "selector_top_score": top_score,
                "base_score": base_score,
                "top_candidate_guarded": int(candidate_is_guarded(top)),
                "W2R": int((not base_ok) and final_ok),
                "R2W": int(base_ok and not final_ok),
            }
        )
    metrics = {
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
        "theta": theta,
        "guard_policy": "weak_not_top3;v2ab_requires_weak",
    }
    return metrics, pred_rows


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.v45_dir / "aco_v45_metrics.json")
    rows = read_csv(args.v45_dir / "method_candidate_features_v45.csv")
    groups = {}
    for split in ["train_loocv", "val", "test"]:
        split_groups = group_samples([row for row in rows if row["split"] == split])
        score_groups(split_groups, model)
        groups[split] = split_groups

    summary_rows = []
    best = None
    for theta in theta_grid(groups["val"]):
        metrics, _ = evaluate(groups["val"], theta, "aco_v46_guarded_gate")
        summary_rows.append(metrics)
        key = (
            metrics["final_accuracy"],
            metrics["net_gain"],
            -metrics["R2W"],
            metrics["W2R"],
            -metrics["trigger_count"],
        )
        if best is None or key > best[0]:
            best = (key, theta, metrics)
    assert best is not None
    theta = best[1]

    final_rows = []
    for split, split_groups in groups.items():
        metrics, preds = evaluate(split_groups, theta, "aco_v46_selected")
        final_rows.append(metrics)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))
    write_csv(args.output_dir / "aco_v46_selection_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(args.output_dir / "aco_v46_final_summary.csv", final_rows, list(final_rows[0].keys()))

    payload = {
        "protocol": "Reuse ACO4.5 model; apply weak/V2 guards; choose replacement threshold on val; evaluate test.",
        "best_val": best[2],
        "final": final_rows,
        "guard_policy": ["weak_not_top3", "v2ab_requires_weak"],
    }
    with (args.output_dir / "aco_v46_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v45-dir", type=Path, default=DEFAULT_V45_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "final": payload["final"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
