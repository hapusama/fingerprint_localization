#!/usr/bin/env python3
"""ACO 4.3 pairwise correction with raw/chirp expert features.

This variant treats the ACO 4.2 interactions final prediction as the incumbent
and trains a pairwise challenger model to decide when another RSSI+ Top-3
candidate should replace it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_aco_v42_pairwise_on_split as pair42  # noqa: E402
import run_aco_v42_reranker_on_split as base42  # noqa: E402
import run_aco_v43_reranker_raw_chirp_features as v43rank  # noqa: E402


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "aco_v43_pairwise_raw_chirp_features"
DEFAULT_INCUMBENT_DIR = EXPERIMENT_DIR / "results" / "aco_v42_reranker_interactions"
RAW_CHIRP_PREFIXES = ("v21_", "v22_", "raw_chirp_")


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


def load_incumbents(path: Path, split: str) -> dict[int, dict]:
    return {int(row["sample_index"]): row for row in read_csv(path / f"{split}_predictions.csv")}


def parse_label(label: str) -> tuple[int, int]:
    return base42.parse_label(label)


def add_incumbent_features(groups: Sequence[dict], incumbent_rows: dict[int, dict]) -> None:
    for group in groups:
        original_v4_base = group["base_label"]
        incumbent = incumbent_rows[group["sample_index"]]
        incumbent_label = incumbent["final_label"]
        inc_c, inc_l = parse_label(incumbent_label)
        group["v4_base_label"] = original_v4_base
        group["incumbent_method"] = incumbent.get("method", "")
        group["base_label"] = incumbent_label
        group["base_correct"] = int(incumbent_label == group["true_label"])
        for row in group["rows"]:
            label = row["candidate_label"]
            cand_c, cand_l = parse_label(label)
            features = row["features"]
            is_incumbent = float(label == incumbent_label)
            features["is_incumbent"] = is_incumbent
            features["is_v4_base"] = float(label == original_v4_base)
            features["same_corridor_as_incumbent"] = float(cand_c == inc_c)
            features["abs_loc_delta_incumbent"] = abs(cand_l - inc_l) / 54.0
            features["incumbent_x_score4_norm"] = is_incumbent * features.get("score4_norm", 0.0)
            features["incumbent_x_low_cost"] = is_incumbent * features.get("candidate_mean_obs_norm", 0.0)
            features["incumbent_x_v2_vote"] = is_incumbent * features.get("is_v2_vote", 0.0)
            features["incumbent_x_raw_chirp_agree"] = is_incumbent * features.get("raw_chirp_agree_winner", 0.0)


def feature_names_for(groups: Sequence[dict], feature_set: str) -> list[str]:
    if feature_set in {"core_raw_chirp", "with_v41_raw_chirp", "with_v2_raw_chirp", "all_raw_chirp"}:
        names = v43rank.feature_names_for(groups, feature_set)
    else:
        names = base42.feature_names_for(groups, feature_set)
    extra = {
        "is_incumbent",
        "is_v4_base",
        "same_corridor_as_incumbent",
        "abs_loc_delta_incumbent",
        "incumbent_x_score4_norm",
        "incumbent_x_low_cost",
        "incumbent_x_v2_vote",
        "incumbent_x_raw_chirp_agree",
    }
    names = sorted(set(names) | extra)
    drop = {"candidate_corridor", "candidate_location_scaled", "bias_feature"}
    return [name for name in names if name not in drop]


def flatten_pair_rows(pairs: Sequence[dict], names: Sequence[str]) -> list[dict]:
    rows = []
    for pair in pairs:
        row = {key: value for key, value in pair.items() if key != "diff"}
        for name in names:
            row[f"diff_{name}"] = pair["diff"].get(name, 0.0)
        rows.append(row)
    return rows


def build_augmented_groups(args: argparse.Namespace) -> dict[str, list[dict]]:
    packet_rows = v43rank.load_raw_chirp_packets(args.raw_chirp_features_csv)
    groups = {
        split: base42.build_groups(args.aco_v4_dir, args.aco_v41_dir, split, args.aco_v2_dir)
        for split in ["train_loocv", "val", "test"]
    }
    for split, split_groups in groups.items():
        v43rank.augment_groups_with_raw_chirp(split_groups, packet_rows)
        add_incumbent_features(split_groups, load_incumbents(args.incumbent_dir, split))
    return groups


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = build_augmented_groups(args)
    summary_rows = []
    best = None
    best_model = None
    best_pairs = None
    best_names = None

    for feature_set in [part.strip() for part in args.feature_sets.split(",") if part.strip()]:
        names = feature_names_for(groups["train_loocv"], feature_set)
        train_pairs = pair42.make_pairs(groups["train_loocv"], names)
        pos = sum(pair["target"] for pair in train_pairs)
        neg = len(train_pairs) - pos
        auto_pos_weight = neg / max(1, pos)
        for l2 in [float(part) for part in args.l2_grid.split(",") if part.strip()]:
            for pw_text in [part for part in args.positive_weight_grid.split(",") if part.strip()]:
                pw_value = float(pw_text)
                pos_weight = auto_pos_weight if pw_value <= 0 else pw_value
                print(
                    f"training pairwise feature_set={feature_set} l2={l2} pos_weight={pos_weight:.3g}",
                    flush=True,
                )
                model = pair42.train_model(train_pairs, names, l2, pos_weight, args.epochs, args.learning_rate)
                val_pairs = pair42.make_pairs(groups["val"], names)
                for threshold in pair42.threshold_grid(val_pairs, model):
                    metrics, _ = pair42.evaluate(
                        groups["val"], val_pairs, model, threshold, f"aco_v43_pairwise_{feature_set}"
                    )
                    metrics["feature_set"] = feature_set
                    metrics["l2"] = l2
                    metrics["positive_weight"] = pos_weight
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
                        best_names = names
                        best_pairs = {
                            split: pair42.make_pairs(groups[split], names)
                            for split in ["train_loocv", "val", "test"]
                        }

    assert best is not None and best_model is not None and best_pairs is not None and best_names is not None
    final_rows = []
    for split in ["train_loocv", "val", "test"]:
        metrics, preds = pair42.evaluate(
            groups[split], best_pairs[split], best_model, float(best["threshold"]), "aco_v43_pairwise_selected"
        )
        metrics["feature_set"] = best["feature_set"]
        metrics["l2"] = best["l2"]
        metrics["positive_weight"] = best["positive_weight"]
        final_rows.append(metrics)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))

    train_pair_rows = flatten_pair_rows(pair42.make_pairs(groups["train_loocv"], best_names), best_names)
    write_csv(args.output_dir / "train_pair_features.csv", train_pair_rows, list(train_pair_rows[0].keys()))
    write_csv(args.output_dir / "aco_v43_pairwise_selection_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "aco_v43_pairwise_final_summary.csv", final_rows, list(final_rows[0].keys()))
    write_csv(args.output_dir / "aco_v43_pairwise_feature_importance.csv", pair42.weight_rows(best_model), ["feature", "weight", "abs_weight"])
    payload = {
        "protocol": (
            "Pairwise challenger model over ACO4.2 interactions incumbent, with V2.1 raw "
            "Gaussian and V2.2 chirp-shrink expert features. Select on validation; evaluate test once."
        ),
        "incumbent_dir": str(args.incumbent_dir),
        "raw_chirp_features_csv": str(args.raw_chirp_features_csv),
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
    with (args.output_dir / "aco_v43_pairwise_raw_chirp_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--incumbent-dir", type=Path, default=DEFAULT_INCUMBENT_DIR)
    parser.add_argument("--raw-chirp-features-csv", type=Path, default=v43rank.DEFAULT_RAW_CHIRP_FEATURES_CSV)
    parser.add_argument("--aco-v4-dir", type=Path, default=base42.DEFAULT_ACO_V4_DIR)
    parser.add_argument("--aco-v41-dir", type=Path, default=base42.DEFAULT_ACO_V41_DIR)
    parser.add_argument("--aco-v2-dir", type=Path, default=base42.DEFAULT_ACO_V2_DIR)
    parser.add_argument("--feature-sets", default="with_v41_raw_chirp,with_v2_raw_chirp,all_raw_chirp")
    parser.add_argument("--l2-grid", default="0.003,0.01,0.03,0.1")
    parser.add_argument("--positive-weight-grid", default="0,1,2,4,8")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "final": payload["final"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
