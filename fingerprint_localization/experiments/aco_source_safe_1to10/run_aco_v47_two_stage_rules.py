#!/usr/bin/env python3
"""ACO 4.7 guarded selector plus train+val-selected rescue rules.

Stage 1 is ACO4.6: ACO4.5 reliability scores with weak/V2 guards.
Stage 2 enumerates a small fixed library of rescue rules and selects the rule
order/ranker using train+val only.  Test is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import permutations
from pathlib import Path
from typing import Callable, Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))


RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_V45_DIR = RESULTS_DIR / "aco_v45_reliability_selector"
DEFAULT_V46_DIR = RESULTS_DIR / "aco_v46_guarded_selector"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "aco_v47_two_stage_rules"


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
        return json.load(f)["model"]


def score_row(row: dict, model: dict) -> float:
    score = model["bias"]
    for name, mean, std, weight in zip(model["feature_names"], model["means"], model["stds"], model["weights"]):
        score += weight * ((parse_float(row.get(name)) - mean) / std)
    return score


def enrich_row(row: dict, model: dict) -> dict:
    row = dict(row)
    sources = {source for source in row.get("source_names", "").split(";") if source}
    row["_score"] = score_row(row, model)
    row["_knn"] = int("knn" in sources)
    row["_mfr"] = int("mfr_prev" in sources)
    row["_v43_challenger"] = int("v43_challenger" in sources)
    row["_raw_chirp"] = int(parse_float(row.get("raw_chirp_source_hit")) > 0.0)
    row["_top3"] = int(parse_float(row.get("in_aco_top3")) > 0.0)
    row["_cross_corridor"] = int(parse_float(row.get("cand_vs_base_same_corridor")) < 0.5)
    row["_same_corridor"] = int(not row["_cross_corridor"])
    row["_weak_pair"] = int(parse_float(row.get("weak_hit_count")) >= 2.0)
    row["_base_strong"] = int(
        parse_float(row.get("base_top3_is_rssi_top1")) > 0.0
        and parse_float(row.get("base_top3_is_v2_vote")) > 0.0
        and parse_float(row.get("base_top3_score4_norm")) >= 0.99
        and parse_float(row.get("base_top3_elite_vote_norm")) >= 0.99
    )
    return row


Rule = Callable[[dict], bool]
Ranker = Callable[[dict], float]


def build_rules() -> dict[str, Rule]:
    return {
        "A_knn_cross_trans05": lambda row: bool(
            row["_knn"]
            and row["_top3"]
            and row["_cross_corridor"]
            and parse_float(row.get("transition_rel")) >= 0.5
        ),
        "B_v43raw_score0": lambda row: bool(
            row["_v43_challenger"]
            and row["_raw_chirp"]
            and row["_top3"]
            and parse_float(row.get("_score")) >= 0.0
        ),
        "C_weak2_same_score06": lambda row: bool(
            row["_weak_pair"]
            and row["_top3"]
            and row["_same_corridor"]
            and parse_float(row.get("_score")) >= 0.6
        ),
        "D_knnmfr_same_score06": lambda row: bool(
            row["_knn"]
            and row["_mfr"]
            and row["_top3"]
            and row["_same_corridor"]
            and parse_float(row.get("_score")) >= 0.6
        ),
        "F_knnmfr_trans05": lambda row: bool(
            row["_knn"]
            and row["_mfr"]
            and row["_top3"]
            and parse_float(row.get("transition_rel")) >= 0.5
        ),
    }


def build_rankers() -> dict[str, Ranker]:
    return {
        "score": lambda row: parse_float(row.get("_score")),
        "transition_rel": lambda row: parse_float(row.get("transition_rel")),
        "top3_obs": lambda row: parse_float(row.get("top3_candidate_mean_obs_norm")),
        "top3_cost41": lambda row: parse_float(row.get("top3_cost_veto41")),
    }


def load_stage1_predictions(v46_dir: Path) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for split in ["train_loocv", "val", "test"]:
        out[split] = {
            int(row["sample_index"]): row
            for row in read_csv(v46_dir / f"{split}_predictions.csv")
        }
    return out


def group_candidates(rows: Sequence[dict]) -> dict[str, dict[int, list[dict]]]:
    grouped: dict[str, dict[int, list[dict]]] = {}
    for row in rows:
        grouped.setdefault(row["split"], {}).setdefault(int(row["sample_index"]), []).append(row)
    return grouped


def evaluate_combo(
    groups: dict[int, list[dict]],
    stage1_rows: dict[int, dict],
    rule_order: Sequence[str],
    ranker_name: str,
    rules: dict[str, Rule],
    rankers: dict[str, Ranker],
    method: str,
) -> tuple[dict, list[dict]]:
    final_correct = base_correct = trigger = w2r = r2w = oracle = 0
    pred_rows = []
    for sample_index, rows in sorted(groups.items()):
        stage1 = stage1_rows[sample_index]
        true_label = stage1["true_label"]
        stage1_label = stage1["final_label"]
        final_label = stage1_label
        fired_rules: list[str] = []
        chosen = None
        candidates = []
        for row in rows:
            if row["candidate_label"] == stage1_label:
                continue
            hits = [name for name in rule_order if rules[name](row)]
            if hits:
                first_rule = min(rule_order.index(name) for name in hits)
                candidates.append((first_rule, -rankers[ranker_name](row), row, hits))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]["candidate_label"]))
            chosen = candidates[0][2]
            fired_rules = candidates[0][3]
            final_label = chosen["candidate_label"]
        oracle += int(any(row["candidate_label"] == true_label for row in rows))
        stage1_ok = int(stage1_label == true_label)
        final_ok = int(final_label == true_label)
        base_correct += stage1_ok
        final_correct += final_ok
        trigger += int(final_label != stage1_label)
        w2r += int((not stage1_ok) and final_ok)
        r2w += int(stage1_ok and not final_ok)
        pred_rows.append(
            {
                "method": method,
                "split": stage1["split"],
                "sample_index": sample_index,
                "true_label": true_label,
                "true_in_candidates": int(any(row["candidate_label"] == true_label for row in rows)),
                "stage1_label": stage1_label,
                "stage1_correct": stage1_ok,
                "final_label": final_label,
                "final_correct": final_ok,
                "triggered": int(final_label != stage1_label),
                "rule_label": "" if chosen is None else chosen["candidate_label"],
                "rule_source_names": "" if chosen is None else chosen.get("source_names", ""),
                "fired_rules": ";".join(fired_rules),
                "rule_score": "" if chosen is None else chosen.get("_score", ""),
                "W2R": int((not stage1_ok) and final_ok),
                "R2W": int(stage1_ok and not final_ok),
            }
        )
    n = len(groups)
    metrics = {
        "method": method,
        "split": next(iter(stage1_rows.values()))["split"] if stage1_rows else "",
        "packet_count": n,
        "candidate_oracle_correct": oracle,
        "candidate_oracle_accuracy": oracle / n if n else 0.0,
        "stage1_correct": base_correct,
        "stage1_accuracy": base_correct / n if n else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / n if n else 0.0,
        "trigger_count": trigger,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "rule_order": ";".join(rule_order),
        "ranker": ranker_name,
    }
    return metrics, pred_rows


def combo_key(train_metrics: dict, val_metrics: dict, rule_order: Sequence[str]) -> tuple:
    """Train+val-only model selection objective."""
    return (
        train_metrics["net_gain"] + val_metrics["net_gain"],
        val_metrics["final_correct"],
        val_metrics["net_gain"],
        -val_metrics["R2W"],
        train_metrics["final_correct"],
        -train_metrics["R2W"],
        -len(rule_order),
        ";".join(rule_order),
    )


def candidate_rule_orders(rule_names: Sequence[str], max_rules: int) -> Iterable[tuple[str, ...]]:
    yield ()
    for size in range(1, max_rules + 1):
        yield from permutations(rule_names, size)


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.v45_dir / "aco_v45_metrics.json")
    rows = [enrich_row(row, model) for row in read_csv(args.v45_dir / "method_candidate_features_v45.csv")]
    groups = group_candidates(rows)
    stage1 = load_stage1_predictions(args.v46_dir)
    rules = build_rules()
    rankers = build_rankers()

    selection_rows = []
    best = None
    best_order: tuple[str, ...] = ()
    best_ranker = "score"
    for order in candidate_rule_orders(list(rules), args.max_rules):
        ranker_names = ["score"] if not order else list(rankers)
        for ranker_name in ranker_names:
            train_metrics, _ = evaluate_combo(
                groups["train_loocv"], stage1["train_loocv"], order, ranker_name, rules, rankers, "aco_v47_candidate"
            )
            val_metrics, _ = evaluate_combo(
                groups["val"], stage1["val"], order, ranker_name, rules, rankers, "aco_v47_candidate"
            )
            row = {
                "rule_order": ";".join(order),
                "ranker": ranker_name,
                "train_final_correct": train_metrics["final_correct"],
                "train_net_gain": train_metrics["net_gain"],
                "train_W2R": train_metrics["W2R"],
                "train_R2W": train_metrics["R2W"],
                "val_final_correct": val_metrics["final_correct"],
                "val_net_gain": val_metrics["net_gain"],
                "val_W2R": val_metrics["W2R"],
                "val_R2W": val_metrics["R2W"],
                "train_val_net_gain": train_metrics["net_gain"] + val_metrics["net_gain"],
            }
            selection_rows.append(row)
            key = combo_key(train_metrics, val_metrics, order)
            if best is None or key > best:
                best = key
                best_order = order
                best_ranker = ranker_name
    assert best is not None

    final_rows = []
    for split in ["train_loocv", "val", "test"]:
        metrics, preds = evaluate_combo(
            groups[split], stage1[split], best_order, best_ranker, rules, rankers, "aco_v47_selected"
        )
        final_rows.append(metrics)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))

    write_csv(args.output_dir / "aco_v47_selection_summary.csv", selection_rows, list(selection_rows[0].keys()))
    write_csv(args.output_dir / "aco_v47_final_summary.csv", final_rows, list(final_rows[0].keys()))

    payload = {
        "protocol": "Stage1=A4.6. Stage2 enumerates fixed rescue rules and selects order/ranker using train+val only.",
        "selected_rule_order": list(best_order),
        "selected_ranker": best_ranker,
        "selection_key": best,
        "final": final_rows,
    }
    with (args.output_dir / "aco_v47_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v45-dir", type=Path, default=DEFAULT_V45_DIR)
    parser.add_argument("--v46-dir", type=Path, default=DEFAULT_V46_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rules", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"selected_rule_order": payload["selected_rule_order"], "final": payload["final"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
