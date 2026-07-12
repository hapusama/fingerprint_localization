#!/usr/bin/env python3
"""ACO 4.3 expert-stack reranker.

Adds explicit ACO2 ablation expert labels (V1.0..V2.7 vote/pheromone/path)
on top of the raw/chirp expert features and trains the same conservative
ACO4.2-style Top-3 softmax reranker.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_aco_v42_reranker_on_split as base42  # noqa: E402
import run_aco_v43_reranker_raw_chirp_features as v43rank  # noqa: E402


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "aco_v43_reranker_expert_stack"
DEFAULT_ABLATION_DIR = EXPERIMENT_DIR / "results" / "aco_v2_ablation"
DEFAULT_VERSIONS = "v1_0,v2_1,v2_2,v2_3,v2_4,v2_5,v2_6,v2_7"
EXPERT_PREFIXES = ("expert_", "v21_", "v22_", "raw_chirp_")


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


def load_ablation_predictions(ablation_dir: Path, versions: Sequence[str], split: str) -> dict[str, dict[int, dict]]:
    out = {}
    for version in versions:
        path = ablation_dir / version / f"{split}_predictions.csv"
        out[version] = {int(row["sample_index"]): row for row in read_csv(path)}
    return out


def augment_groups_with_experts(
    groups: Sequence[dict],
    ablation_rows: dict[str, dict[int, dict]],
    versions: Sequence[str],
) -> None:
    for group in groups:
        sample_index = group["sample_index"]
        labels_by_expert = {}
        for version in versions:
            pred = ablation_rows[version][sample_index]
            for suffix, column in [
                ("vote", "aco_vote_label"),
                ("pheromone", "aco_pheromone_label"),
                ("path", "aco_path_mode_label"),
            ]:
                labels_by_expert[f"{version}_{suffix}"] = pred[column]
        vote_counter = Counter(label for name, label in labels_by_expert.items() if name.endswith("_vote"))
        all_counter = Counter(labels_by_expert.values())
        top_vote_label, top_vote_count = ("", 0)
        if vote_counter:
            top_vote_label, top_vote_count = max(
                vote_counter.items(), key=lambda item: (item[1], -base42.natural_label_key(item[0])[0], -base42.natural_label_key(item[0])[1])
            )
        top_any_label, top_any_count = ("", 0)
        if all_counter:
            top_any_label, top_any_count = max(
                all_counter.items(), key=lambda item: (item[1], -base42.natural_label_key(item[0])[0], -base42.natural_label_key(item[0])[1])
            )
        for row in group["rows"]:
            label = row["candidate_label"]
            features = row["features"]
            low_cost = features.get("candidate_mean_obs_norm", 0.0)
            score4 = features.get("score4_norm", 0.0)
            raw_agree = features.get("raw_chirp_agree_winner", 0.0)
            v2_support = max(features.get("is_v2_vote", 0.0), features.get("v2_disagrees_v4_supports_candidate", 0.0))
            vote_hits = sum(1 for name, expert_label in labels_by_expert.items() if name.endswith("_vote") and expert_label == label)
            pheromone_hits = sum(1 for name, expert_label in labels_by_expert.items() if name.endswith("_pheromone") and expert_label == label)
            path_hits = sum(1 for name, expert_label in labels_by_expert.items() if name.endswith("_path") and expert_label == label)
            all_hits = vote_hits + pheromone_hits + path_hits
            features.update(
                {
                    "expert_vote_hit_count": float(vote_hits),
                    "expert_pheromone_hit_count": float(pheromone_hits),
                    "expert_path_hit_count": float(path_hits),
                    "expert_all_hit_count": float(all_hits),
                    "expert_vote_hit_frac": vote_hits / max(1, len(versions)),
                    "expert_all_hit_frac": all_hits / max(1, len(versions) * 3),
                    "expert_is_top_vote_consensus": float(label == top_vote_label),
                    "expert_top_vote_count": float(top_vote_count if label == top_vote_label else 0),
                    "expert_is_top_any_consensus": float(label == top_any_label),
                    "expert_top_any_count": float(top_any_count if label == top_any_label else 0),
                    "expert_vote_x_low_cost": vote_hits * low_cost,
                    "expert_all_x_low_cost": all_hits * low_cost,
                    "expert_vote_x_score4_norm": vote_hits * score4,
                    "expert_all_x_score4_norm": all_hits * score4,
                    "expert_vote_x_raw_chirp_agree": vote_hits * raw_agree,
                    "expert_all_x_raw_chirp_agree": all_hits * raw_agree,
                    "expert_vote_x_v2_support": vote_hits * v2_support,
                    "expert_all_x_v2_support": all_hits * v2_support,
                }
            )
            for version in versions:
                for suffix, column in [
                    ("vote", "aco_vote_label"),
                    ("pheromone", "aco_pheromone_label"),
                    ("path", "aco_path_mode_label"),
                ]:
                    hit = float(ablation_rows[version][sample_index][column] == label)
                    features[f"expert_{version}_{suffix}"] = hit
                    features[f"expert_{version}_{suffix}_x_low_cost"] = hit * low_cost
                    features[f"expert_{version}_{suffix}_x_raw_chirp_agree"] = hit * raw_agree


def expert_feature_names(groups: Sequence[dict]) -> list[str]:
    return sorted(
        {
            name
            for group in groups
            for row in group["rows"]
            for name in row["features"]
            if name.startswith(EXPERT_PREFIXES)
        }
    )


def feature_names_for(groups: Sequence[dict], feature_set: str) -> list[str]:
    expert_names = set(expert_feature_names(groups))
    if feature_set == "with_v41_expert_stack":
        return sorted(set(base42.feature_names_for(groups, "with_v41")) | expert_names)
    if feature_set == "with_v2_expert_stack":
        return sorted(set(base42.feature_names_for(groups, "with_v2")) | expert_names)
    if feature_set == "all_expert_stack":
        return sorted(
            {
                name
                for group in groups
                for row in group["rows"]
                for name in row["features"]
                if name not in {"candidate_corridor", "candidate_location_scaled"}
            }
        )
    return v43rank.feature_names_for(groups, feature_set)


def flatten_candidate_rows(groups: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for split_groups in groups.values():
        for group in split_groups:
            for row in group["rows"]:
                flat = {key: value for key, value in row.items() if key != "features"}
                flat.update(row["features"])
                rows.append(flat)
    return rows


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    versions = [part.strip() for part in args.versions.split(",") if part.strip()]
    packet_rows = v43rank.load_raw_chirp_packets(args.raw_chirp_features_csv)
    groups = {
        split: base42.build_groups(args.aco_v4_dir, args.aco_v41_dir, split, args.aco_v2_dir)
        for split in ["train_loocv", "val", "test"]
    }
    for split, split_groups in groups.items():
        v43rank.augment_groups_with_raw_chirp(split_groups, packet_rows)
        augment_groups_with_experts(
            split_groups,
            load_ablation_predictions(args.ablation_dir, versions, split),
            versions,
        )

    candidate_rows = flatten_candidate_rows(groups)
    static_columns = {
        "split", "sample_index", "file_name", "packet_index", "true_label", "candidate_label",
        "target", "true_in_top3", "base_label", "base_correct", "rssi_top1_label",
    }
    feature_names_all = sorted({key for row in candidate_rows for key in row if key not in static_columns})
    write_csv(
        args.output_dir / "aco_v43_expert_candidate_features.csv",
        candidate_rows,
        [
            "split", "sample_index", "file_name", "packet_index", "true_label", "candidate_label",
            "target", "true_in_top3", "base_label", "base_correct", "rssi_top1_label",
        ] + feature_names_all,
    )

    summary_rows = []
    prediction_rows = []
    for split, split_groups in groups.items():
        metrics, preds = base42.evaluate(split_groups, None, "aco_v4_base")
        summary_rows.append(metrics)
        prediction_rows.extend(preds)

    best = None
    best_model = None
    for feature_set in [part.strip() for part in args.feature_sets.split(",") if part.strip()]:
        names = feature_names_for(groups["train_loocv"], feature_set)
        for l2 in [float(part) for part in args.l2_grid.split(",") if part.strip()]:
            print(f"training feature_set={feature_set} l2={l2}", flush=True)
            model = base42.train_softmax_ranker(
                groups["train_loocv"], names, l2, args.epochs, args.learning_rate, args.seed
            )
            raw_metrics, _ = base42.evaluate(groups["val"], model, f"aco_v43_{feature_set}_raw", None)
            raw_metrics["feature_set"] = feature_set
            raw_metrics["l2"] = l2
            raw_metrics["selection"] = "raw_ml"
            summary_rows.append(raw_metrics)
            for theta in base42.theta_grid(groups["val"], model):
                metrics, _ = base42.evaluate(groups["val"], model, f"aco_v43_{feature_set}_gate", theta)
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
        metrics, preds = base42.evaluate(split_groups, best_model, "aco_v43_expert_selected", base42.parse_float(best["theta"], 0.0))
        metrics["feature_set"] = best["feature_set"]
        metrics["l2"] = best["l2"]
        final_rows.append(metrics)
        prediction_rows.extend(preds)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))

    write_csv(args.output_dir / "aco_v43_expert_selection_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "aco_v43_expert_final_summary.csv", final_rows, list(final_rows[0].keys()))
    write_csv(args.output_dir / "aco_v43_expert_feature_importance.csv", base42.feature_importance(best_model), ["feature", "weight", "abs_weight"])
    payload = {
        "protocol": (
            "Train ACO4.2-style Top3 softmax reranker with raw/chirp and ACO2 ablation "
            "expert-label stack. Select on validation; evaluate test once."
        ),
        "versions": versions,
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
    with (args.output_dir / "aco_v43_expert_stack_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--versions", default=DEFAULT_VERSIONS)
    parser.add_argument("--raw-chirp-features-csv", type=Path, default=v43rank.DEFAULT_RAW_CHIRP_FEATURES_CSV)
    parser.add_argument("--aco-v4-dir", type=Path, default=base42.DEFAULT_ACO_V4_DIR)
    parser.add_argument("--aco-v41-dir", type=Path, default=base42.DEFAULT_ACO_V41_DIR)
    parser.add_argument("--aco-v2-dir", type=Path, default=base42.DEFAULT_ACO_V2_DIR)
    parser.add_argument("--feature-sets", default="with_v41_expert_stack,with_v2_expert_stack,all_expert_stack")
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
