#!/usr/bin/env python3
"""ACO 4.8 exploratory guarded V4.7 rescue rules.

V4.7 is frozen.  This script keeps the same selected rescue order but adds two
conservative guards:

* B/D rescue candidates must not score more than 1.0 below the current stage-1
  label under the reliability selector;
* the cross-corridor KNN transition rule (A) keeps its transition evidence but
  must have selector score >= -0.75.

This is exploratory and should be reported separately from the frozen V4.7
result unless it passes stability checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_aco_v47_two_stage_rules as v47  # noqa: E402


RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_V45_DIR = RESULTS_DIR / "aco_v45_reliability_selector"
DEFAULT_V46_DIR = RESULTS_DIR / "aco_v46_guarded_selector"
DEFAULT_ACO2_DIR = RESULTS_DIR / "aco_v2"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "aco_v48_guarded_v47_rules"
RULE_ORDER = ["D_knnmfr_same_score06", "B_v43raw_score0", "A_knn_cross_trans05"]
RANKER = "score"
RELATIVE_SCORE_FLOOR = -1.0
A_CROSS_SCORE_FLOOR = -0.75
RANDOM_SEEDS = [20260701, 20260702, 20260703, 20260704, 20260705]
SPLITS = ["train_loocv", "val", "test"]


def parse_float(value: object, default: float = 0.0) -> float:
    return v47.parse_float(value, default)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        preferred = [
            "setting",
            "split",
            "sample_index",
            "true_label",
            "stage1_label",
            "final_label",
            "primary_rule",
            "W2R",
            "R2W",
        ]
        seen = set(preferred)
        fieldnames = [name for name in preferred if any(name in row for row in rows)]
        fieldnames += sorted({key for row in rows for key in row} - seen)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_candidate_groups(v45_dir: Path) -> dict[str, dict[int, list[dict]]]:
    model = v47.load_model(v45_dir / "aco_v45_metrics.json")
    rows = [v47.enrich_row(row, model) for row in read_csv(v45_dir / "method_candidate_features_v45.csv")]
    return v47.group_candidates(rows)


def load_stage1(v46_dir: Path) -> dict[str, dict[int, dict]]:
    return v47.load_stage1_predictions(v46_dir)


def load_aco2(aco2_dir: Path) -> dict[int, dict]:
    out = {}
    for split in SPLITS:
        for row in read_csv(aco2_dir / f"{split}_predictions.csv"):
            out[int(row["sample_index"])] = row
    return out


def label_corridor(label: str) -> str:
    return label.split("_", 1)[0] if "_" in label else ""


def choose_label(
    rows: Sequence[dict],
    stage1: dict,
    rules: dict,
    rankers: dict,
) -> dict:
    true_label = stage1["true_label"]
    stage1_label = stage1["final_label"]
    base_row = next((row for row in rows if row["candidate_label"] == stage1_label), None)
    base_score = parse_float(base_row.get("_score")) if base_row else -999.0
    candidates = []
    for row in rows:
        if row["candidate_label"] == stage1_label:
            continue
        hits = [name for name in RULE_ORDER if rules[name](row)]
        if not hits:
            continue
        primary = hits[0]
        score = parse_float(row.get("_score"))
        if primary == "A_knn_cross_trans05":
            if score < A_CROSS_SCORE_FLOOR:
                continue
        elif score - base_score < RELATIVE_SCORE_FLOOR:
            continue
        candidates.append((RULE_ORDER.index(primary), -rankers[RANKER](row), row["candidate_label"], row, hits))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        _rule_idx, _rank_score, _label, chosen, fired_rules = candidates[0]
        final_label = chosen["candidate_label"]
        primary_rule = fired_rules[0]
    else:
        chosen = None
        fired_rules = []
        final_label = stage1_label
        primary_rule = ""

    stage1_ok = int(stage1_label == true_label)
    final_ok = int(final_label == true_label)
    out = {
        "method": "aco_v48_guarded_v47_rules",
        "split": stage1["split"],
        "sample_index": stage1["sample_index"],
        "true_label": true_label,
        "true_corridor": label_corridor(true_label),
        "true_in_candidates": int(any(row["candidate_label"] == true_label for row in rows)),
        "stage1_label": stage1_label,
        "stage1_correct": stage1_ok,
        "final_label": final_label,
        "final_correct": final_ok,
        "triggered": int(final_label != stage1_label),
        "primary_rule": primary_rule,
        "fired_rules": ";".join(fired_rules),
        "W2R": int((not stage1_ok) and final_ok),
        "R2W": int(stage1_ok and not final_ok),
        "W2W": int((not stage1_ok) and (not final_ok) and final_label != stage1_label),
        "R2R": int(stage1_ok and final_ok and final_label != stage1_label),
        "base_score": base_score,
    }
    if chosen is not None:
        out.update(
            {
                "rule_label": chosen["candidate_label"],
                "rule_source_names": chosen.get("source_names", ""),
                "rule_score": chosen.get("_score", ""),
                "score_minus_base": parse_float(chosen.get("_score")) - base_score,
                "rule_transition_rel": chosen.get("transition_rel", ""),
                "rule_in_aco_top3": chosen.get("in_aco_top3", ""),
                "rule_weak_hit_count": chosen.get("weak_hit_count", ""),
                "rule_raw_chirp_source_hit": chosen.get("raw_chirp_source_hit", ""),
                "rule_same_corridor": chosen.get("_same_corridor", ""),
                "rule_cross_corridor": chosen.get("_cross_corridor", ""),
            }
        )
    return out


def evaluate_split(groups: dict[int, list[dict]], stage1_rows: dict[int, dict], split: str) -> tuple[dict, list[dict]]:
    rules = v47.build_rules()
    rankers = v47.build_rankers()
    pred_rows = []
    for sample_index, rows in sorted(groups.items()):
        pred_rows.append(choose_label(rows, stage1_rows[sample_index], rules, rankers))
    n = len(pred_rows)
    stage1_correct = sum(int(row["stage1_correct"]) for row in pred_rows)
    final_correct = sum(int(row["final_correct"]) for row in pred_rows)
    oracle = sum(int(row["true_in_candidates"]) for row in pred_rows)
    w2r = sum(int(row["W2R"]) for row in pred_rows)
    r2w = sum(int(row["R2W"]) for row in pred_rows)
    trigger = sum(int(row["triggered"]) for row in pred_rows)
    metrics = {
        "method": "aco_v48_guarded_v47_rules",
        "split": split,
        "packet_count": n,
        "candidate_oracle_correct": oracle,
        "candidate_oracle_accuracy": oracle / n if n else 0.0,
        "stage1_correct": stage1_correct,
        "stage1_accuracy": stage1_correct / n if n else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / n if n else 0.0,
        "trigger_count": trigger,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "rule_order": ";".join(RULE_ORDER),
        "ranker": RANKER,
        "relative_score_floor": RELATIVE_SCORE_FLOOR,
        "a_cross_score_floor": A_CROSS_SCORE_FLOOR,
    }
    return metrics, pred_rows


def metric_for_setting(setting: str, rows: Sequence[dict], aco2: dict[int, dict]) -> dict:
    n = len(rows)
    aco2_correct = sum(int(aco2[int(row["sample_index"])]["aco_vote_label"] == row["true_label"]) for row in rows)
    stage1_correct = sum(int(row["stage1_correct"]) for row in rows)
    final_correct = sum(int(row["final_correct"]) for row in rows)
    oracle = sum(int(row["true_in_candidates"]) for row in rows)
    w2r = sum(int(row["W2R"]) for row in rows)
    r2w = sum(int(row["R2W"]) for row in rows)
    trigger = sum(int(row["triggered"]) for row in rows)
    errors = [row for row in rows if not int(row["final_correct"])]
    errors_in_pool = sum(int(row["true_in_candidates"]) for row in errors)
    return {
        "setting": setting,
        "packet_count": n,
        "ACO2_correct": aco2_correct,
        "ACO2": aco2_correct / n if n else 0.0,
        "V4.6_correct": stage1_correct,
        "V4.6": stage1_correct / n if n else 0.0,
        "V4.8_correct": final_correct,
        "V4.8": final_correct / n if n else 0.0,
        "oracle_correct": oracle,
        "oracle": oracle / n if n else 0.0,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "trigger_count": trigger,
        "remaining_errors": len(errors),
        "remaining_errors_true_in_candidate_pool": errors_in_pool,
        "remaining_errors_true_not_in_candidate_pool": len(errors) - errors_in_pool,
    }


def stratified_random_partitions(rows: Sequence[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: int(item["sample_index"])):
        by_label[row["true_label"]].append(row)
    parts = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * 0.60)
        n_val = round(n * 0.20)
        parts["train"].extend(items[:n_train])
        parts["val"].extend(items[n_train : n_train + n_val])
        parts["test"].extend(items[n_train + n_val :])
    return parts


def rule_summary(setting: str, rows: Sequence[dict]) -> list[dict]:
    by_rule: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if int(row["triggered"]):
            by_rule[row["primary_rule"]].append(row)
    out = []
    for rule, items in sorted(by_rule.items()):
        out.append(
            {
                "setting": setting,
                "rule": rule,
                "trigger_count": len(items),
                "W2R": sum(int(row["W2R"]) for row in items),
                "R2W": sum(int(row["R2W"]) for row in items),
                "net_gain": sum(int(row["W2R"]) - int(row["R2W"]) for row in items),
                "sample_indices": ";".join(str(row["sample_index"]) for row in items),
            }
        )
    return out


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_candidate_groups(args.v45_dir)
    stage1 = load_stage1(args.v46_dir)
    aco2 = load_aco2(args.aco2_dir)

    final_summary = []
    all_pred_rows = []
    for split in SPLITS:
        metrics, preds = evaluate_split(groups[split], stage1[split], split)
        final_summary.append(metrics)
        all_pred_rows.extend(preds)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds)
    write_csv(args.output_dir / "aco_v48_final_summary.csv", final_summary, list(final_summary[0].keys()))

    settings: dict[str, list[dict]] = {
        "current_train_loocv": [row for row in all_pred_rows if row["split"] == "train_loocv"],
        "current_val": [row for row in all_pred_rows if row["split"] == "val"],
        "current_test": [row for row in all_pred_rows if row["split"] == "test"],
    }
    for idx, seed in enumerate(RANDOM_SEEDS, start=1):
        parts = stratified_random_partitions(all_pred_rows, seed)
        for part, part_rows in parts.items():
            settings[f"random_{idx:02d}_seed_{seed}_{part}"] = part_rows
    for corridor in ["0", "1", "2"]:
        settings[f"corridor_{corridor}_test"] = [
            row for row in all_pred_rows if label_corridor(row["true_label"]) == corridor
        ]

    summary_rows = [metric_for_setting(setting, rows, aco2) for setting, rows in settings.items()]
    rule_rows = [rule_row for setting, rows in settings.items() for rule_row in rule_summary(setting, rows)]
    modified_rows = [
        {"setting": setting, **row}
        for setting, rows in settings.items()
        for row in rows
        if int(row["triggered"])
    ]
    write_csv(args.output_dir / "stability_summary.csv", summary_rows)
    write_csv(args.output_dir / "rule_summary_by_setting.csv", rule_rows)
    write_csv(args.output_dir / "modified_samples_by_setting.csv", modified_rows)

    payload = {
    "protocol": "Exploratory V4.8: frozen V4.7 rule order plus B/D relative score guard and A cross-corridor score floor.",
        "warning": "This is not the frozen V4.7 result. Treat as exploratory until independently validated.",
        "rule_order": RULE_ORDER,
        "ranker": RANKER,
        "relative_score_floor": RELATIVE_SCORE_FLOOR,
        "a_cross_score_floor": A_CROSS_SCORE_FLOOR,
        "final": final_summary,
        "outputs": {
            "summary": str(args.output_dir / "stability_summary.csv"),
            "final_summary": str(args.output_dir / "aco_v48_final_summary.csv"),
            "rule_summary": str(args.output_dir / "rule_summary_by_setting.csv"),
            "modified_samples": str(args.output_dir / "modified_samples_by_setting.csv"),
        },
    }
    with (args.output_dir / "aco_v48_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v45-dir", type=Path, default=DEFAULT_V45_DIR)
    parser.add_argument("--v46-dir", type=Path, default=DEFAULT_V46_DIR)
    parser.add_argument("--aco2-dir", type=Path, default=DEFAULT_ACO2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
