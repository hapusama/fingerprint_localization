#!/usr/bin/env python3
"""Stability validation for the frozen ACO 4.7 rescue-rule layer.

This script intentionally does not tune or edit ACO4.7.  It loads the current
ACO2, ACO4.6 and V4.5 candidate-pool outputs, applies the already selected
ACO4.7 rule order/ranker, and reports stability across current, random, and
leave-corridor-out evaluation partitions.

The random and corridor-out checks are evaluation-only repartitions over the
existing full prediction/candidate pool.  They are meant to validate the frozen
rescue rules' behavior, not to rerun the full lower-level ACO pipeline.
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
DEFAULT_ACO2_DIR = RESULTS_DIR / "aco_v2"
DEFAULT_V45_DIR = RESULTS_DIR / "aco_v45_reliability_selector"
DEFAULT_V46_DIR = RESULTS_DIR / "aco_v46_guarded_selector"
DEFAULT_V47_DIR = RESULTS_DIR / "aco_v47_two_stage_rules"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "aco_v47_stability_validation"
DEFAULT_RANDOM_SEEDS = "20260701,20260702,20260703,20260704,20260705"
DEFAULT_FROZEN_ORDER = ["D_knnmfr_same_score06", "B_v43raw_score0", "A_knn_cross_trans05"]
DEFAULT_FROZEN_RANKER = "score"
EVAL_SPLITS = ["train_loocv", "val", "test"]


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
            "partition",
            "rule",
            "sample_index",
            "true_label",
            "aco2_label",
            "v46_label",
            "v47_label",
            "primary_rule",
        ]
        seen = {name for name in preferred}
        rest = sorted({key for row in rows for key in row} - seen)
        fieldnames = [name for name in preferred if any(name in row for row in rows)] + rest
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: object, default: float = 0.0) -> float:
    return v47.parse_float(value, default)


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def accuracy(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def label_corridor(label: str) -> str:
    return label.split("_", 1)[0] if "_" in label else ""


def load_prediction_pool(results_dir: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for split in EVAL_SPLITS:
        for row in read_csv(results_dir / f"{split}_predictions.csv"):
            sample_index = parse_int(row["sample_index"])
            row = dict(row)
            row["original_split"] = split
            out[sample_index] = row
    return out


def load_candidate_groups(v45_dir: Path) -> dict[int, list[dict]]:
    model = v47.load_model(v45_dir / "aco_v45_metrics.json")
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in read_csv(v45_dir / "method_candidate_features_v45.csv"):
        enriched = v47.enrich_row(row, model)
        groups[parse_int(enriched["sample_index"])].append(enriched)
    for rows in groups.values():
        rows.sort(key=lambda row: row["candidate_label"])
    return dict(groups)


def load_frozen_selection(v47_dir: Path) -> tuple[list[str], str]:
    metrics_path = v47_dir / "aco_v47_metrics.json"
    if not metrics_path.exists():
        return list(DEFAULT_FROZEN_ORDER), DEFAULT_FROZEN_RANKER
    with metrics_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    order = payload.get("selected_rule_order") or DEFAULT_FROZEN_ORDER
    ranker = payload.get("selected_ranker") or DEFAULT_FROZEN_RANKER
    return list(order), str(ranker)


def choose_v47_label(
    sample_index: int,
    candidate_rows: Sequence[dict],
    stage1_row: dict,
    rule_order: Sequence[str],
    ranker_name: str,
    rules: dict,
    rankers: dict,
) -> dict:
    true_label = stage1_row["true_label"]
    stage1_label = stage1_row["final_label"]
    final_label = stage1_label
    chosen = None
    fired_rules: list[str] = []
    candidates = []
    for row in candidate_rows:
        if row["candidate_label"] == stage1_label:
            continue
        hits = [name for name in rule_order if rules[name](row)]
        if not hits:
            continue
        first_rule = min(rule_order.index(name) for name in hits)
        candidates.append((first_rule, -rankers[ranker_name](row), row["candidate_label"], row, hits))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        first_rule_idx, _neg_score, _label, chosen, fired_rules = candidates[0]
        final_label = chosen["candidate_label"]
        primary_rule = rule_order[first_rule_idx]
    else:
        primary_rule = ""

    stage1_ok = int(stage1_label == true_label)
    final_ok = int(final_label == true_label)
    true_in_candidates = int(any(row["candidate_label"] == true_label for row in candidate_rows))
    out = {
        "sample_index": sample_index,
        "original_split": stage1_row.get("original_split", stage1_row.get("split", "")),
        "true_label": true_label,
        "true_corridor": label_corridor(true_label),
        "true_in_candidates": true_in_candidates,
        "v46_label": stage1_label,
        "v46_correct": stage1_ok,
        "v47_label": final_label,
        "v47_correct": final_ok,
        "triggered": int(final_label != stage1_label),
        "primary_rule": primary_rule,
        "fired_rules": ";".join(fired_rules),
        "W2R": int((not stage1_ok) and final_ok),
        "R2W": int(stage1_ok and not final_ok),
        "W2W": int((not stage1_ok) and (not final_ok) and final_label != stage1_label),
        "R2R": int(stage1_ok and final_ok and final_label != stage1_label),
    }
    if chosen is not None:
        out.update(
            {
                "rule_label": chosen["candidate_label"],
                "rule_source_names": chosen.get("source_names", ""),
                "rule_score": chosen.get("_score", ""),
                "rule_transition_rel": chosen.get("transition_rel", ""),
                "rule_in_aco_top3": chosen.get("in_aco_top3", ""),
                "rule_weak_hit_count": chosen.get("weak_hit_count", ""),
                "rule_raw_chirp_source_hit": chosen.get("raw_chirp_source_hit", ""),
                "rule_knn": chosen.get("_knn", ""),
                "rule_mfr": chosen.get("_mfr", ""),
                "rule_v43_challenger": chosen.get("_v43_challenger", ""),
                "rule_same_corridor": chosen.get("_same_corridor", ""),
                "rule_cross_corridor": chosen.get("_cross_corridor", ""),
                "cand_vs_base_same_corridor": chosen.get("cand_vs_base_same_corridor", ""),
                "cand_vs_base_loc_delta": chosen.get("cand_vs_base_loc_delta", ""),
            }
        )
    return out


def build_full_records(
    aco2: dict[int, dict],
    v46_rows: dict[int, dict],
    candidate_groups: dict[int, list[dict]],
    rule_order: Sequence[str],
    ranker_name: str,
) -> dict[int, dict]:
    rules = v47.build_rules()
    rankers = v47.build_rankers()
    sample_ids = sorted(set(aco2) & set(v46_rows) & set(candidate_groups))
    records = {}
    for sample_index in sample_ids:
        rec = choose_v47_label(
            sample_index,
            candidate_groups[sample_index],
            v46_rows[sample_index],
            rule_order,
            ranker_name,
            rules,
            rankers,
        )
        aco2_row = aco2[sample_index]
        aco2_label = aco2_row["aco_vote_label"]
        rec.update(
            {
                "aco2_label": aco2_label,
                "aco2_correct": int(aco2_label == rec["true_label"]),
                "file_name": aco2_row.get("file_name", ""),
                "packet_index": aco2_row.get("packet_index", ""),
            }
        )
        records[sample_index] = rec
    return records


def metric_row(setting: str, sample_ids: Sequence[int], records: dict[int, dict]) -> dict:
    n = len(sample_ids)
    aco2_correct = sum(records[idx]["aco2_correct"] for idx in sample_ids)
    v46_correct = sum(records[idx]["v46_correct"] for idx in sample_ids)
    v47_correct = sum(records[idx]["v47_correct"] for idx in sample_ids)
    oracle_correct = sum(records[idx]["true_in_candidates"] for idx in sample_ids)
    w2r = sum(records[idx]["W2R"] for idx in sample_ids)
    r2w = sum(records[idx]["R2W"] for idx in sample_ids)
    trigger_count = sum(records[idx]["triggered"] for idx in sample_ids)
    v47_errors = [idx for idx in sample_ids if not records[idx]["v47_correct"]]
    errors_in_pool = sum(records[idx]["true_in_candidates"] for idx in v47_errors)
    errors_not_in_pool = len(v47_errors) - errors_in_pool
    aco2_acc = accuracy(aco2_correct, n)
    v46_acc = accuracy(v46_correct, n)
    v47_acc = accuracy(v47_correct, n)
    oracle_acc = accuracy(oracle_correct, n)
    return {
        "setting": setting,
        "packet_count": n,
        "ACO2_correct": aco2_correct,
        "ACO2": aco2_acc,
        "V4.6_correct": v46_correct,
        "V4.6": v46_acc,
        "V4.7_correct": v47_correct,
        "V4.7": v47_acc,
        "oracle_correct": oracle_correct,
        "oracle": oracle_acc,
        "ACO2_oracle_gap": oracle_acc - aco2_acc,
        "V4.7_oracle_gap": oracle_acc - v47_acc,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "trigger_count": trigger_count,
        "v47_remaining_errors": len(v47_errors),
        "v47_errors_true_in_candidate_pool": errors_in_pool,
        "v47_errors_true_not_in_candidate_pool": errors_not_in_pool,
        "v47_errors_in_pool_rate": accuracy(errors_in_pool, len(v47_errors)),
    }


def rule_summary_rows(setting: str, sample_ids: Sequence[int], records: dict[int, dict]) -> list[dict]:
    rows = []
    by_rule: dict[str, list[int]] = defaultdict(list)
    for idx in sample_ids:
        rec = records[idx]
        if rec["triggered"]:
            by_rule[rec["primary_rule"]].append(idx)
    for rule in sorted(by_rule):
        ids = by_rule[rule]
        rows.append(
            {
                "setting": setting,
                "rule": rule,
                "trigger_count": len(ids),
                "W2R": sum(records[idx]["W2R"] for idx in ids),
                "R2W": sum(records[idx]["R2W"] for idx in ids),
                "net_gain": sum(records[idx]["W2R"] - records[idx]["R2W"] for idx in ids),
                "W2W": sum(records[idx]["W2W"] for idx in ids),
                "R2R": sum(records[idx]["R2R"] for idx in ids),
                "sample_indices": ";".join(str(idx) for idx in ids),
            }
        )
    return rows


def modified_sample_rows(setting: str, sample_ids: Sequence[int], records: dict[int, dict]) -> list[dict]:
    out = []
    for idx in sample_ids:
        rec = records[idx]
        if not rec["triggered"]:
            continue
        row = {"setting": setting}
        row.update(rec)
        out.append(row)
    return out


def current_split_members(records: dict[int, dict]) -> dict[str, list[int]]:
    out = {"current_train_loocv": [], "current_val": [], "current_test": []}
    for idx, rec in records.items():
        split = rec["original_split"]
        if split == "train_loocv":
            out["current_train_loocv"].append(idx)
        elif split == "val":
            out["current_val"].append(idx)
        elif split == "test":
            out["current_test"].append(idx)
    return {key: sorted(value) for key, value in out.items()}


def stratified_random_split(records: dict[int, dict], seed: int) -> dict[str, list[int]]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, rec in records.items():
        by_label[rec["true_label"]].append(idx)
    splits = {"train": [], "val": [], "test": []}
    for label in sorted(by_label):
        ids = list(by_label[label])
        rng.shuffle(ids)
        n = len(ids)
        n_train = round(n * 0.60)
        n_val = round(n * 0.20)
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])
    return {key: sorted(value) for key, value in splits.items()}


def corridor_test_splits(records: dict[int, dict]) -> dict[str, list[int]]:
    out = {}
    for corridor in ["0", "1", "2"]:
        out[f"corridor_{corridor}_test"] = sorted(
            idx for idx, rec in records.items() if rec["true_corridor"] == corridor
        )
    return out


def load_existing_v47(v47_dir: Path) -> dict[int, dict]:
    return load_prediction_pool(v47_dir)


def recompute_check(records: dict[int, dict], existing_v47: dict[int, dict]) -> list[dict]:
    rows = []
    for idx, rec in sorted(records.items()):
        existing = existing_v47.get(idx)
        if not existing:
            rows.append({"sample_index": idx, "issue": "missing_existing_v47"})
            continue
        if existing.get("final_label") != rec["v47_label"]:
            rows.append(
                {
                    "sample_index": idx,
                    "issue": "final_label_mismatch",
                    "existing_final_label": existing.get("final_label", ""),
                    "recomputed_final_label": rec["v47_label"],
                    "existing_fired_rules": existing.get("fired_rules", ""),
                    "recomputed_fired_rules": rec["fired_rules"],
                }
            )
    return rows


def parse_seed_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rule_order, ranker_name = load_frozen_selection(args.v47_dir)
    aco2 = load_prediction_pool(args.aco2_dir)
    v46_rows = load_prediction_pool(args.v46_dir)
    candidate_groups = load_candidate_groups(args.v45_dir)
    records = build_full_records(aco2, v46_rows, candidate_groups, rule_order, ranker_name)

    current_members = current_split_members(records)
    random_seeds = parse_seed_list(args.random_seeds)
    random_assignments_rows = []
    random_settings: dict[str, list[int]] = {}
    for split_idx, seed in enumerate(random_seeds, start=1):
        parts = stratified_random_split(records, seed)
        for partition, ids in parts.items():
            setting = f"random_{split_idx:02d}_seed_{seed}_{partition}"
            random_settings[setting] = ids
            for sample_index in ids:
                random_assignments_rows.append(
                    {
                        "random_split": f"random_{split_idx:02d}",
                        "seed": seed,
                        "partition": partition,
                        "sample_index": sample_index,
                        "true_label": records[sample_index]["true_label"],
                    }
                )
    corridor_settings = corridor_test_splits(records)

    all_settings: dict[str, list[int]] = {}
    all_settings.update(current_members)
    all_settings.update(random_settings)
    all_settings.update(corridor_settings)

    summary_rows = [metric_row(setting, ids, records) for setting, ids in all_settings.items()]
    rule_rows = []
    modified_rows = []
    for setting, ids in all_settings.items():
        rule_rows.extend(rule_summary_rows(setting, ids, records))
        modified_rows.extend(modified_sample_rows(setting, ids, records))

    current_settings = set(current_members)
    random_setting_names = set(random_settings)
    corridor_setting_names = set(corridor_settings)

    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "current_split_metrics.csv", [row for row in summary_rows if row["setting"] in current_settings])
    write_csv(args.output_dir / "random_split_metrics.csv", [row for row in summary_rows if row["setting"] in random_setting_names])
    write_csv(args.output_dir / "leave_corridor_out_metrics.csv", [row for row in summary_rows if row["setting"] in corridor_setting_names])
    write_csv(args.output_dir / "rule_summary_by_setting.csv", rule_rows)
    write_csv(args.output_dir / "current_split_rule_summary.csv", [row for row in rule_rows if row["setting"] in current_settings])
    write_csv(args.output_dir / "random_split_rule_summary.csv", [row for row in rule_rows if row["setting"] in random_setting_names])
    write_csv(args.output_dir / "leave_corridor_out_rule_summary.csv", [row for row in rule_rows if row["setting"] in corridor_setting_names])
    write_csv(args.output_dir / "modified_samples_all_settings.csv", modified_rows)
    write_csv(args.output_dir / "current_split_modified_samples.csv", [row for row in modified_rows if row["setting"] in current_settings])
    write_csv(args.output_dir / "random_split_assignments.csv", random_assignments_rows)
    write_csv(args.output_dir / "full_pool_frozen_v47_records.csv", records.values())

    check_rows = recompute_check(records, load_existing_v47(args.v47_dir))
    write_csv(args.output_dir / "v47_recomputed_check.csv", check_rows, ["sample_index", "issue", "existing_final_label", "recomputed_final_label", "existing_fired_rules", "recomputed_fired_rules"] if check_rows else ["sample_index", "issue"])

    metadata = {
        "protocol": (
            "Freeze current ACO4.7 rescue rule definitions, selected rule order, and ranker. "
            "Evaluate current split, stratified random repartitions, and leave-corridor-out "
            "partitions over the existing full prediction/candidate pool. No thresholds or "
            "rule logic are modified."
        ),
        "important_limitation": (
            "Random and corridor-out rows are evaluation-only repartitions of existing ACO2/V4.6/V4.5 "
            "outputs; they do not rerun the full lower-level ACO training pipeline for each split."
        ),
        "frozen_rule_order": rule_order,
        "frozen_ranker": ranker_name,
        "random_seeds": random_seeds,
        "sample_count": len(records),
        "recompute_mismatch_count": len(check_rows),
        "outputs": {
            "summary": str(args.output_dir / "summary.csv"),
            "current_split_metrics": str(args.output_dir / "current_split_metrics.csv"),
            "random_split_metrics": str(args.output_dir / "random_split_metrics.csv"),
            "leave_corridor_out_metrics": str(args.output_dir / "leave_corridor_out_metrics.csv"),
            "rule_summary_by_setting": str(args.output_dir / "rule_summary_by_setting.csv"),
            "random_split_rule_summary": str(args.output_dir / "random_split_rule_summary.csv"),
            "leave_corridor_out_rule_summary": str(args.output_dir / "leave_corridor_out_rule_summary.csv"),
            "current_split_modified_samples": str(args.output_dir / "current_split_modified_samples.csv"),
        },
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return {"metadata": metadata, "summary": summary_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aco2-dir", type=Path, default=DEFAULT_ACO2_DIR)
    parser.add_argument("--v45-dir", type=Path, default=DEFAULT_V45_DIR)
    parser.add_argument("--v46-dir", type=Path, default=DEFAULT_V46_DIR)
    parser.add_argument("--v47-dir", type=Path, default=DEFAULT_V47_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seeds", default=DEFAULT_RANDOM_SEEDS)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    for row in payload["summary"]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
