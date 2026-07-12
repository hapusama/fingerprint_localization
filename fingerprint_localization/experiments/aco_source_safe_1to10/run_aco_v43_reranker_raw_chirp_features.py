#!/usr/bin/env python3
"""ACO 4.3: add V2.1 raw Gaussian and V2.2 chirp-shrink expert features.

Protocol:
- Keep the same RSSI+ Top-3 candidate set and 6:2:2 split as ACO 4.2.
- Reuse the ACO 4.2 softmax reranker training/validation protocol.
- Add per-candidate features derived from the ACO 2.0 raw/chirp experts.
- Select feature set, L2, and replacement threshold on validation only.
- Evaluate test once with the selected configuration.
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

import run_aco_v42_reranker_on_split as base42  # noqa: E402


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "aco_v43_reranker_raw_chirp_features"
DEFAULT_RAW_CHIRP_FEATURES_CSV = (
    EXPERIMENT_DIR / "results" / "aco_v43_raw_chirp_challenger" / "packet_challenger_features.csv"
)
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


def parse_rank_costs(text: str) -> dict[str, float]:
    out = {}
    for part in (text or "").split(";"):
        if not part or ":" not in part:
            continue
        label, value = part.rsplit(":", 1)
        out[label] = base42.parse_float(value, 0.0)
    return out


def load_raw_chirp_packets(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for row in read_csv(path):
        rows[(row["split"], int(row["sample_index"]))] = row
    return rows


def raw_feature_names(groups: Sequence[dict]) -> list[str]:
    return sorted(
        {
            name
            for group in groups
            for row in group["rows"]
            for name in row["features"]
            if name.startswith(RAW_CHIRP_PREFIXES)
        }
    )


def augment_groups_with_raw_chirp(groups: Sequence[dict], packet_rows: dict[tuple[str, int], dict]) -> None:
    for group in groups:
        packet = packet_rows.get((group["split"], group["sample_index"]), {})
        raw_scores = parse_rank_costs(packet.get("raw_rank", ""))
        chirp_scores = parse_rank_costs(packet.get("chirp_rank", ""))
        raw_quality = base42.normalize_high(raw_scores)
        chirp_quality = base42.normalize_high(chirp_scores)
        raw_rank_inv = base42.rank_inv(raw_scores, high_better=False) if raw_scores else {}
        chirp_rank_inv = base42.rank_inv(chirp_scores, high_better=False) if chirp_scores else {}
        raw_winner = packet.get("raw_winner", "")
        chirp_winner = packet.get("chirp_winner", "")
        agreed_winner = packet.get("agreed_winner", "")
        raw_margin = base42.parse_float(packet.get("raw_margin_v21"), 0.0)
        chirp_margin = base42.parse_float(packet.get("chirp_margin_v22"), 0.0)
        min_margin = min(raw_margin, chirp_margin)
        base_label = group["base_label"]
        base_raw_cost = raw_scores.get(base_label, 0.0)
        base_chirp_cost = chirp_scores.get(base_label, 0.0)
        raw_chirp_agree = float(bool(agreed_winner))

        for row in group["rows"]:
            label = row["candidate_label"]
            features = row["features"]
            low_cost = features.get("candidate_mean_obs_norm", 0.0)
            score4 = features.get("score4_norm", 0.0)
            template_rel = features.get("template_reliability_norm", 0.0)
            v2_vote = features.get("is_v2_vote", 0.0)
            v2_support = max(v2_vote, features.get("v2_disagrees_v4_supports_candidate", 0.0))
            is_base = features.get("is_aco_base", 0.0)
            is_rssi = features.get("is_rssi_top1", 0.0)
            is_raw_winner = float(label == raw_winner)
            is_chirp_winner = float(label == chirp_winner)
            is_agreed_winner = float(bool(agreed_winner) and label == agreed_winner)
            winner_count = is_raw_winner + is_chirp_winner
            any_winner = float(winner_count > 0.0)
            raw_cost = raw_scores.get(label, base_raw_cost)
            chirp_cost = chirp_scores.get(label, base_chirp_cost)

            features.update(
                {
                    "v21_raw_gaussian_winner": is_raw_winner,
                    "v22_chirp_shrink_winner": is_chirp_winner,
                    "raw_chirp_agree_packet": raw_chirp_agree,
                    "raw_chirp_agree_winner": is_agreed_winner,
                    "raw_chirp_any_winner": any_winner,
                    "raw_chirp_winner_count": winner_count,
                    "v21_raw_margin": raw_margin,
                    "v22_chirp_margin": chirp_margin,
                    "raw_chirp_min_margin": min_margin,
                    "v21_raw_cost": raw_cost,
                    "v22_chirp_cost": chirp_cost,
                    "v21_raw_cost_quality": raw_quality.get(label, 0.0),
                    "v22_chirp_cost_quality": chirp_quality.get(label, 0.0),
                    "v21_raw_rank_inv": raw_rank_inv.get(label, 0.0),
                    "v22_chirp_rank_inv": chirp_rank_inv.get(label, 0.0),
                    "v21_raw_cost_delta_vs_base": base_raw_cost - raw_cost,
                    "v22_chirp_cost_delta_vs_base": base_chirp_cost - chirp_cost,
                    "v21_raw_margin_x_winner": raw_margin * is_raw_winner,
                    "v22_chirp_margin_x_winner": chirp_margin * is_chirp_winner,
                    "raw_chirp_min_margin_x_agree_winner": min_margin * is_agreed_winner,
                    "raw_chirp_agree_x_low_cost": is_agreed_winner * low_cost,
                    "raw_chirp_agree_x_score4_norm": is_agreed_winner * score4,
                    "raw_chirp_agree_x_template_rel": is_agreed_winner * template_rel,
                    "raw_chirp_agree_x_v2_support": is_agreed_winner * v2_support,
                    "raw_chirp_agree_x_v2_vote": is_agreed_winner * v2_vote,
                    "raw_chirp_agree_x_not_base": is_agreed_winner * (1.0 - is_base),
                    "raw_chirp_agree_x_not_rssi_top1": is_agreed_winner * (1.0 - is_rssi),
                    "raw_chirp_any_x_low_cost": any_winner * low_cost,
                    "raw_chirp_any_x_score4_norm": any_winner * score4,
                    "raw_chirp_any_x_v2_support": any_winner * v2_support,
                    "v21_raw_winner_x_low_cost": is_raw_winner * low_cost,
                    "v22_chirp_winner_x_low_cost": is_chirp_winner * low_cost,
                    "v21_raw_winner_x_v2_support": is_raw_winner * v2_support,
                    "v22_chirp_winner_x_v2_support": is_chirp_winner * v2_support,
                }
            )


def feature_names_for(groups: Sequence[dict], feature_set: str) -> list[str]:
    raw_names = set(raw_feature_names(groups))
    if feature_set == "core_raw_chirp":
        return sorted(set(base42.feature_names_for(groups, "core")) | raw_names)
    if feature_set == "with_v41_raw_chirp":
        return sorted(set(base42.feature_names_for(groups, "with_v41")) | raw_names)
    if feature_set == "with_v2_raw_chirp":
        return sorted(set(base42.feature_names_for(groups, "with_v2")) | raw_names)
    if feature_set == "all_raw_chirp":
        return sorted(
            {
                name
                for group in groups
                for row in group["rows"]
                for name in row["features"]
                if name not in {"candidate_corridor", "candidate_location_scaled"}
            }
        )
    return base42.feature_names_for(groups, feature_set)


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
    packet_rows = load_raw_chirp_packets(args.raw_chirp_features_csv)
    groups = {
        split: base42.build_groups(args.aco_v4_dir, args.aco_v41_dir, split, args.aco_v2_dir)
        for split in ["train_loocv", "val", "test"]
    }
    for split_groups in groups.values():
        augment_groups_with_raw_chirp(split_groups, packet_rows)

    candidate_rows = flatten_candidate_rows(groups)
    static_columns = {
        "split", "sample_index", "file_name", "packet_index", "true_label", "candidate_label",
        "target", "true_in_top3", "base_label", "base_correct", "rssi_top1_label",
    }
    feature_names_all = sorted({key for row in candidate_rows for key in row if key not in static_columns})
    write_csv(
        args.output_dir / "aco_v43_candidate_features.csv",
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
        metrics, preds = base42.evaluate(split_groups, best_model, "aco_v43_selected", base42.parse_float(best["theta"], 0.0))
        metrics["feature_set"] = best["feature_set"]
        metrics["l2"] = best["l2"]
        final_rows.append(metrics)
        prediction_rows.extend(preds)
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, list(preds[0].keys()))

    write_csv(args.output_dir / "aco_v43_selection_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))
    write_csv(args.output_dir / "aco_v43_final_summary.csv", final_rows, list(final_rows[0].keys()))
    write_csv(args.output_dir / "aco_v43_feature_importance.csv", base42.feature_importance(best_model), ["feature", "weight", "abs_weight"])
    payload = {
        "protocol": (
            "Train ACO4.2-style Top3 softmax reranker with V2.1 raw Gaussian and V2.2 "
            "chirp-shrink expert features. Select on validation; evaluate test once."
        ),
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
        },
    }
    with (args.output_dir / "aco_v43_reranker_raw_chirp_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-chirp-features-csv", type=Path, default=DEFAULT_RAW_CHIRP_FEATURES_CSV)
    parser.add_argument("--aco-v4-dir", type=Path, default=base42.DEFAULT_ACO_V4_DIR)
    parser.add_argument("--aco-v41-dir", type=Path, default=base42.DEFAULT_ACO_V41_DIR)
    parser.add_argument("--aco-v2-dir", type=Path, default=base42.DEFAULT_ACO_V2_DIR)
    parser.add_argument(
        "--feature-sets",
        default="core_raw_chirp,with_v41_raw_chirp,with_v2_raw_chirp,all_raw_chirp",
    )
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
