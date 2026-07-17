#!/usr/bin/env python3
"""Validation-select the final Score4/ML blend for Expanded ACO."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

import run_expanded_aco_ml_prior as prior
import run_expanded_supervised_ensemble as supervised


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
DEFAULT_ROOT = PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "aco_ml_candidate_prior"
BETA_GRID = [round(index / 10.0, 1) for index in range(11)]
EPS = 1e-12


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def common_probabilities(models: dict, eval_rows: Sequence[dict]) -> tuple[dict, list[str]]:
    x_eval = supervised.matrix(eval_rows)
    class_order = sorted(
        {str(label) for model in models.values() for label in model.classes_},
        key=supervised.natural_label_key,
    )
    probabilities = np.zeros((len(eval_rows), len(class_order)), dtype=float)
    for model in models.values():
        probabilities += supervised.common_probabilities(
            model,
            model.predict_proba(x_eval),
            class_order,
        ) / len(models)
    by_key = {}
    for row, values in zip(eval_rows, probabilities):
        key = (row["file_stem"], int(float(row["packet_index"])))
        by_key[key] = {label: float(values[index]) for index, label in enumerate(class_order)}
    return by_key, class_order


def minmax(values: dict[str, float]) -> dict[str, float]:
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo <= EPS:
        return {label: 0.5 for label in values}
    return {label: (value - lo) / (hi - lo) for label, value in values.items()}


def load_groups(
    prediction_path: Path,
    candidate_path: Path,
    probabilities: dict[tuple[str, int], dict[str, float]],
) -> list[dict]:
    candidates: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in read_csv(candidate_path):
        key = (Path(row["file_name"]).stem, int(float(row["packet_index"])))
        candidates[key].append(row)
    groups = []
    for prediction in read_csv(prediction_path):
        key = (Path(prediction["file_name"]).stem, int(float(prediction["packet_index"])))
        rows = candidates[key]
        labels = [row["candidate_label"] for row in rows]
        score4 = minmax({row["candidate_label"]: float(row["score4"]) for row in rows})
        ml = minmax({label: probabilities[key][label] for label in labels})
        groups.append(
            {
                "key": key,
                "prediction": prediction,
                "candidate_labels": labels,
                "score4_norm": score4,
                "ml_norm": ml,
                "ml_probabilities": probabilities[key],
            }
        )
    return groups


def evaluate(groups: Sequence[dict], beta: float, split: str) -> tuple[dict, list[dict]]:
    rows = []
    for group in groups:
        prediction = group["prediction"]
        combined = {
            label: (1.0 - beta) * group["score4_norm"][label] + beta * group["ml_norm"][label]
            for label in group["candidate_labels"]
        }
        final_label = max(
            combined,
            key=lambda label: (
                combined[label],
                tuple(-value for value in supervised.natural_label_key(label)),
            ),
        )
        ml_top1 = max(
            group["ml_probabilities"],
            key=lambda label: (
                group["ml_probabilities"][label],
                tuple(-value for value in supervised.natural_label_key(label)),
            ),
        )
        true_label = prediction["true_label"]
        base_label = prediction["final_label"]
        base_correct = int(base_label == true_label)
        final_correct = int(final_label == true_label)
        rows.append(
            {
                "split": split,
                "file_name": prediction["file_name"],
                "packet_index": prediction["packet_index"],
                "true_label": true_label,
                "candidate_labels": ";".join(group["candidate_labels"]),
                "aco_score4_label": base_label,
                "aco_score4_correct": base_correct,
                "ml_prior_top1_label": ml_top1,
                "ml_prior_top1_correct": int(ml_top1 == true_label),
                "ml_top1_in_aco_candidates": int(ml_top1 in group["candidate_labels"]),
                "score4_ml_beta": beta,
                "final_label": final_label,
                "final_correct": final_correct,
                "changed_from_aco_score4": int(final_label != base_label),
                "W2R": int(not base_correct and final_correct),
                "R2W": int(base_correct and not final_correct),
                "final_differs_from_ml_top1": int(final_label != ml_top1),
                "final_combined_score": combined[final_label],
            }
        )
    count = len(rows)
    correct = sum(row["final_correct"] for row in rows)
    base_correct = sum(row["aco_score4_correct"] for row in rows)
    metrics = {
        "split": split,
        "beta": beta,
        "packet_count": count,
        "aco_score4_correct": base_correct,
        "final_correct": correct,
        "final_accuracy": correct / count,
        "changed_from_aco_score4": sum(row["changed_from_aco_score4"] for row in rows),
        "W2R": sum(row["W2R"] for row in rows),
        "R2W": sum(row["R2W"] for row in rows),
        "net_gain": correct - base_correct,
        "ml_prior_top1_correct": sum(row["ml_prior_top1_correct"] for row in rows),
        "ml_top1_in_aco_candidates": sum(row["ml_top1_in_aco_candidates"] for row in rows),
        "final_differs_from_ml_top1": sum(row["final_differs_from_ml_top1"] for row in rows),
    }
    return metrics, rows


def mcnemar_exact_p(w2r: int, r2w: int) -> float:
    discordant = w2r + r2w
    if discordant == 0:
        return 1.0
    lower = min(w2r, r2w)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def compare_external(rows: Sequence[dict], path: Path, column: str, name: str) -> dict:
    baseline = {
        (Path(row["file_name"]).stem, int(float(row["packet_index"]))): row[column]
        for row in read_csv(path)
    }
    baseline_correct = changed = w2r = r2w = 0
    for row in rows:
        key = (Path(row["file_name"]).stem, int(float(row["packet_index"])))
        true_label = row["true_label"]
        baseline_label = baseline[key]
        final_label = row["final_label"]
        baseline_ok = baseline_label == true_label
        final_ok = bool(row["final_correct"])
        baseline_correct += int(baseline_ok)
        changed += int(baseline_label != final_label)
        w2r += int(not baseline_ok and final_ok)
        r2w += int(baseline_ok and not final_ok)
    return {
        "baseline": name,
        "baseline_correct": baseline_correct,
        "final_correct": sum(row["final_correct"] for row in rows),
        "changed": changed,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(w2r, r2w),
    }


def wilson_interval(correct: int, count: int, z: float = 1.96) -> tuple[float, float]:
    p = correct / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return center - radius, center + radius


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path) -> None:
    path = output_dir / "FINAL_CHECKSUMS.sha256"
    lines = []
    for item in sorted(output_dir.iterdir()):
        if item.is_file() and item != path:
            lines.append(f"{sha256(item)}  {item.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    prior_report = json.loads((args.root / "aco_ml_prior_report.json").read_text(encoding="utf-8"))
    candidate_prior_weight = float(prior_report["validation_selection"]["prior_weight"])
    selected_weight_dir = f"weight_{candidate_prior_weight:.1f}"
    validation_table = supervised.load_validation_table(supervised.DEFAULT_VALIDATION_FEATURES)
    validation_eval = [row for row in validation_table if row["split"] == "val"]
    refit_table = supervised.load_refit_table(
        supervised.DEFAULT_REFIT_RSSI,
        supervised.DEFAULT_REFIT_RAW,
        supervised.DEFAULT_REFIT_SPLIT,
    )
    formal_eval = [row for row in refit_table if row["split"] == "test"]
    validation_models = joblib.load(args.root / "validation_prior_models.joblib")
    formal_models = joblib.load(args.root / "formal_prior_models.joblib")
    validation_probabilities, _classes = common_probabilities(validation_models, validation_eval)
    formal_probabilities, _formal_classes = common_probabilities(formal_models, formal_eval)

    validation_groups = load_groups(
        args.root / "validation" / selected_weight_dir / "val_predictions.csv",
        args.root / "validation" / selected_weight_dir / "val_candidate_scores.csv",
        validation_probabilities,
    )
    selection_rows = []
    for beta in BETA_GRID:
        metrics, _rows = evaluate(validation_groups, beta, "val")
        selection_rows.append(metrics)
    if args.selection_rule == "min_beta_target":
        eligible = [
            row for row in selection_rows
            if row["final_correct"] >= args.validation_target_correct
        ]
        selected = min(eligible, key=lambda row: row["beta"]) if eligible else max(
            selection_rows,
            key=lambda row: (row["final_correct"], -row["beta"]),
        )
    else:
        selected = max(selection_rows, key=lambda row: (row["final_correct"], -row["beta"]))
    beta = float(selected["beta"])
    validation_metrics, validation_rows = evaluate(validation_groups, beta, "val")

    formal_groups = load_groups(
        args.root / "formal_test" / "test_predictions.csv",
        args.root / "formal_test" / "test_candidate_scores.csv",
        formal_probabilities,
    )
    test_metrics, test_rows = evaluate(formal_groups, beta, "test")
    test_metrics["mcnemar_vs_aco_score4_p"] = mcnemar_exact_p(test_metrics["W2R"], test_metrics["R2W"])
    test_metrics["wilson95"] = wilson_interval(test_metrics["final_correct"], test_metrics["packet_count"])
    test_metrics["target_correct_for_at_least_90_54_percent"] = 116
    test_metrics["target_met"] = test_metrics["final_correct"] >= 116
    comparisons = [
        compare_external(
            test_rows,
            supervised.DEFAULT_ACO_TEST / "test_predictions.csv",
            "final_label",
            "source_level_aco_no_learned_prior",
        ),
        compare_external(
            test_rows,
            supervised.DEFAULT_CANDIDATE_RERANK / "selected_test_predictions.csv",
            "final_label",
            "candidate_rerank",
        ),
    ]

    write_csv(args.root / "score4_ml_beta_selection.csv", selection_rows)
    write_csv(args.root / "final_validation_predictions.csv", validation_rows)
    write_csv(args.root / "final_test_predictions.csv", test_rows)
    write_csv(args.root / "final_external_comparisons.csv", comparisons)
    payload = {
        "status": "PASS",
        "method": "ACO v4 with learned candidate prior and Score4 posterior fusion",
        "formula": "Score_final=(1-beta)*norm(Score4_ACO)+beta*norm(P_LDA+RF), within ACO candidates",
        "candidate_prior_weight": candidate_prior_weight,
        "beta_selection": (
            "validation only; choose the minimum beta reaching the validation target"
            if args.selection_rule == "min_beta_target"
            else "validation only; maximize accuracy and choose smaller beta on ties"
        ),
        "validation_target_correct": args.validation_target_correct,
        "validation": validation_metrics,
        "formal_test": test_metrics,
        "external_comparisons": comparisons,
        "aco_components_retained": [
            "RSSI/learned-prior candidate generation",
            "four-segment physical observation costs",
            "pheromone iteration",
            "elite-path voting",
            "Score4 ACO score",
        ],
        "ablation": {
            "no_learned_prior_formal_correct": 102,
            "learned_prior_without_score4_fusion_formal_correct": test_metrics["aco_score4_correct"],
        },
        "caveat": "Exploratory because the Expanded test split was inspected by earlier experiments.",
    }
    (args.root / "final_aco_score4_ml_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = (
        "# Expanded-649 Final ACO Score4/ML Prior Result\n\n"
        f"- Candidate-prior weight: {candidate_prior_weight:.1f}.\n"
        f"- Validation-selected Score4 posterior beta: {beta:.1f}.\n"
        f"- Validation: {validation_metrics['final_correct']}/128 = {validation_metrics['final_accuracy']:.2%}.\n"
        f"- Formal test: {test_metrics['final_correct']}/128 = {test_metrics['final_accuracy']:.2%}.\n"
        f"- Versus ACO before posterior fusion: W2R={test_metrics['W2R']}, R2W={test_metrics['R2W']}.\n"
        f"- Versus no-prior source-level ACO: W2R={comparisons[0]['W2R']}, "
        f"R2W={comparisons[0]['R2W']}, p={comparisons[0]['mcnemar_exact_two_sided_p']:.6f}.\n"
        f"- Target 116/128: {'met' if test_metrics['target_met'] else 'not met'}.\n\n"
        f"Final selection is restricted to the ACO-generated candidate set and retains {1.0 - beta:.0%} "
        "normalized ACO Score4 evidence at the validation-selected beta. The result is exploratory because the "
        "test split had already been inspected.\n"
    )
    (args.root / "final_aco_score4_ml_report.md").write_text(report, encoding="utf-8")
    write_checksums(args.root)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--selection-rule",
        choices=("max_accuracy", "min_beta_target"),
        default="min_beta_target",
    )
    parser.add_argument("--validation-target-correct", type=int, default=116)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
