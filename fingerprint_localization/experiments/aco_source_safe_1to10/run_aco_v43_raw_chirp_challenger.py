#!/usr/bin/env python3
"""ACO 4.3 raw/chirp challenger over the ACO 4.2 interactions output.

Protocol:
- Keep ACO 4.2 interactions as the base prediction.
- Recompute V2.1 raw-bin Gaussian and V2.2 chirp-shrink bin winners inside
  each RSSI+ Top-3 candidate set.
- Select a conservative challenger gate on validation only.
- Evaluate the selected rule on test once.
"""

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
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
MODEL_V3_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(MODEL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_V3_DIR))

import aco_packet_path_v2 as aco2  # noqa: E402
import run_aco_v2_ablation_on_split as ablation  # noqa: E402
import run_aco_v2_on_split as split_runner  # noqa: E402


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "aco_v43_raw_chirp_challenger"
DEFAULT_BASE_DIR = EXPERIMENT_DIR / "results" / "aco_v42_reranker_interactions"
EPS = 1e-12


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


def build_aco_args(args: argparse.Namespace) -> argparse.Namespace:
    return split_runner.build_args(args)


def candidate_rank(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda label: (scores[label], aco2.natural_label_key(label)))


def relative_margin(scores: dict[str, float], winner: str) -> float:
    ordered = sorted(scores.values())
    if len(ordered) < 2:
        return 0.0
    best = scores[winner]
    second = ordered[1]
    return max(0.0, (second - best) / (abs(second) + EPS))


def score_candidates(
    sample: aco2.SegmentPacket,
    candidates: Sequence[str],
    templates: dict[str, aco2.BinTemplate],
    aco_args: argparse.Namespace,
    config: ablation.AblationConfig,
) -> dict[str, float]:
    return {
        label: sum(
            ablation.bin_cost(shape, templates[label], aco_args, config)
            for shape in sample.segment_shapes
        )
        for label in candidates
        if label in templates
    }


def load_candidate_features(base_dir: Path) -> dict[str, dict[int, dict[str, dict]]]:
    rows = read_csv(base_dir / "aco_v42_candidate_features.csv")
    grouped: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[row["split"]][int(row["sample_index"])][row["candidate_label"]] = row
    return grouped


def load_base_predictions(base_dir: Path, split: str) -> dict[int, dict]:
    return {int(row["sample_index"]): row for row in read_csv(base_dir / f"{split}_predictions.csv")}


def build_samples_and_templates(args: argparse.Namespace):
    aco_args = build_aco_args(args)
    rssi_packets = aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, _q4_offsets, _thresholds = aco2.base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = split_runner.load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = aco2.prepare_chirp_fields(aco_args, labels)
    configs = {cfg.version: cfg for cfg in ablation.ABLATIONS}
    raw_config = configs["V2.1"]
    chirp_config = configs["V2.2"]
    templates = {
        "raw": ablation.build_templates_for_ablation(
            samples, labels, split_indices["train"], chirp_shapes, chirp_struct, aco_args, raw_config
        ),
        "chirp": ablation.build_templates_for_ablation(
            samples, labels, split_indices["train"], chirp_shapes, chirp_struct, aco_args, chirp_config
        ),
    }
    return aco_args, samples, split_indices, templates, raw_config, chirp_config, chirp_metadata


def build_packet_rows(args: argparse.Namespace) -> dict[str, list[dict]]:
    aco_args, samples, split_indices, templates, raw_config, chirp_config, _metadata = build_samples_and_templates(args)
    candidate_features = load_candidate_features(args.base_dir)
    rows_by_split = {}
    for split in ["train_loocv", "val", "test"]:
        base_preds = load_base_predictions(args.base_dir, split)
        rows = []
        for sample_index in split_indices["train" if split == "train_loocv" else split]:
            pred = base_preds[sample_index]
            sample = samples[sample_index]
            candidates = list(candidate_features[split][sample_index])
            raw_scores = score_candidates(sample, candidates, templates["raw"], aco_args, raw_config)
            chirp_scores = score_candidates(sample, candidates, templates["chirp"], aco_args, chirp_config)
            raw_order = candidate_rank(raw_scores)
            chirp_order = candidate_rank(chirp_scores)
            raw_winner = raw_order[0] if raw_order else ""
            chirp_winner = chirp_order[0] if chirp_order else ""
            agreed_winner = raw_winner if raw_winner and raw_winner == chirp_winner else ""
            features = candidate_features[split][sample_index]
            base_label = pred["final_label"]
            row = {
                "split": split,
                "sample_index": sample_index,
                "file_name": pred["file_name"],
                "packet_index": pred["packet_index"],
                "true_label": pred["true_label"],
                "base_label": base_label,
                "base_correct": int(pred["final_correct"]),
                "true_in_top3": int(pred["true_in_top3"]),
                "raw_winner": raw_winner,
                "chirp_winner": chirp_winner,
                "raw_chirp_agree": int(bool(agreed_winner)),
                "agreed_winner": agreed_winner,
                "raw_margin_v21": relative_margin(raw_scores, raw_winner) if raw_winner else 0.0,
                "chirp_margin_v22": relative_margin(chirp_scores, chirp_winner) if chirp_winner else 0.0,
                "raw_rank": ";".join(f"{label}:{raw_scores[label]:.6g}" for label in raw_order),
                "chirp_rank": ";".join(f"{label}:{chirp_scores[label]:.6g}" for label in chirp_order),
            }
            for prefix, label in [("raw", raw_winner), ("chirp", chirp_winner), ("agree", agreed_winner)]:
                feat = features.get(label, {}) if label else {}
                row[f"{prefix}_candidate_mean_obs_norm"] = parse_float(feat.get("candidate_mean_obs_norm"), 0.0)
                row[f"{prefix}_candidate_cost_norm"] = parse_float(feat.get("candidate_cost_norm"), 1.0)
                row[f"{prefix}_raw_x_low_cost"] = parse_float(feat.get("raw_x_low_cost"), 0.0)
                row[f"{prefix}_is_rssi_top1"] = parse_float(feat.get("is_rssi_top1"), 0.0)
                row[f"{prefix}_is_raw_winner"] = parse_float(feat.get("is_raw_winner"), 0.0)
                row[f"{prefix}_is_v2_vote"] = parse_float(feat.get("is_v2_vote"), 0.0)
                row[f"{prefix}_v2_disagrees_v4_supports_candidate"] = parse_float(
                    feat.get("v2_disagrees_v4_supports_candidate"), 0.0
                )
                row[f"{prefix}_v2_vote_x_raw_winner"] = parse_float(feat.get("v2_vote_x_raw_winner"), 0.0)
                row[f"{prefix}_v2_vote_x_low_cost"] = parse_float(feat.get("v2_vote_x_low_cost"), 0.0)
                row[f"{prefix}_v2_vote_x_score4_norm"] = parse_float(feat.get("v2_vote_x_score4_norm"), 0.0)
                row[f"{prefix}_v2_vote_x_template_rel"] = parse_float(feat.get("v2_vote_x_template_rel"), 0.0)
                row[f"{prefix}_v2_agrees_v4"] = parse_float(feat.get("v2_agrees_v4"), 0.0)
                row[f"{prefix}_rssi_rank_inv"] = parse_float(feat.get("rssi_rank_inv"), 0.0)
                row[f"{prefix}_score4_norm"] = parse_float(feat.get("score4_norm"), 0.0)
                row[f"{prefix}_score41"] = parse_float(feat.get("score41"), 0.0)
                row[f"{prefix}_template_reliability_norm"] = parse_float(feat.get("template_reliability_norm"), 0.0)
                row[f"{prefix}_alpha_shrink"] = parse_float(feat.get("alpha_shrink"), 0.0)
            base_feat = features.get(base_label, {})
            for feature_name, default in [
                ("candidate_mean_obs_norm", 0.0),
                ("candidate_cost_norm", 1.0),
                ("is_rssi_top1", 0.0),
                ("is_raw_winner", 0.0),
                ("is_v2_vote", 0.0),
                ("v2_disagrees_v4_supports_candidate", 0.0),
                ("v2_vote_x_raw_winner", 0.0),
                ("v2_vote_x_low_cost", 0.0),
                ("v2_vote_x_score4_norm", 0.0),
                ("v2_agrees_v4", 0.0),
                ("rssi_rank_inv", 0.0),
                ("score4_norm", 0.0),
                ("score41", 0.0),
                ("template_reliability_norm", 0.0),
                ("alpha_shrink", 0.0),
            ]:
                row[f"base_{feature_name}"] = parse_float(base_feat.get(feature_name), default)
            rows.append(row)
        rows_by_split[split] = rows
    return rows_by_split


def challenger_label(row: dict, mode: str) -> str:
    if mode == "agree":
        return row["agreed_winner"]
    if mode == "raw":
        return row["raw_winner"]
    if mode == "chirp":
        return row["chirp_winner"]
    if mode == "raw_then_chirp":
        return row["raw_winner"] or row["chirp_winner"]
    raise ValueError(f"Unknown challenger mode: {mode}")


def passes_gate(row: dict, label: str, rule: dict) -> bool:
    if not label or label == row["base_label"]:
        return False
    if rule["mode"] == "agree" and not row["raw_chirp_agree"]:
        return False
    prefix = "agree" if rule["mode"] == "agree" else ("raw" if label == row["raw_winner"] else "chirp")
    v2_support = (
        row[f"{prefix}_is_v2_vote"] >= 0.5
        or row[f"{prefix}_v2_disagrees_v4_supports_candidate"] >= 0.5
        or row[f"{prefix}_v2_vote_x_raw_winner"] > 0.0
    )
    if rule["require_v2_support"] and not v2_support:
        return False
    if rule["require_v2_vote"] and row[f"{prefix}_is_v2_vote"] < 0.5:
        return False
    if rule["require_v2_disagree_support"] and row[f"{prefix}_v2_disagrees_v4_supports_candidate"] < 0.5:
        return False
    if (
        rule["protect_base_rssi_v2"]
        and row["base_is_rssi_top1"] >= 0.5
        and row["base_is_v2_vote"] >= 0.5
        and not v2_support
    ):
        return False
    if (
        rule["protect_base_strong"]
        and row["base_candidate_mean_obs_norm"] >= rule["base_strong_min_obs"]
        and row["base_score4_norm"] >= rule["base_strong_min_score4"]
        and row["base_is_v2_vote"] >= 0.5
        and not v2_support
    ):
        return False
    if row["raw_margin_v21"] < rule["min_raw_margin"]:
        return False
    if row["chirp_margin_v22"] < rule["min_chirp_margin"]:
        return False
    if row[f"{prefix}_candidate_mean_obs_norm"] < rule["min_low_cost"]:
        return False
    if row[f"{prefix}_candidate_cost_norm"] > rule["max_cost_norm"]:
        return False
    if row[f"{prefix}_rssi_rank_inv"] < rule["min_rssi_rank_inv"]:
        return False
    if rule["require_not_rssi_top1"] and row[f"{prefix}_is_rssi_top1"] >= 0.5:
        return False
    return True


def apply_rule(rows: Sequence[dict], rule: dict, method: str) -> tuple[dict, list[dict]]:
    out_rows = []
    base_correct = sum(int(row["base_correct"]) for row in rows)
    final_correct = 0
    trigger = 0
    w2r = 0
    r2w = 0
    for row in rows:
        label = challenger_label(row, rule["mode"])
        use_challenger = passes_gate(row, label, rule)
        final_label = label if use_challenger else row["base_label"]
        base_ok = int(row["base_correct"])
        final_ok = int(final_label == row["true_label"])
        final_correct += final_ok
        trigger += int(final_label != row["base_label"])
        w2r += int((not base_ok) and final_ok)
        r2w += int(base_ok and not final_ok)
        out = dict(row)
        out.update(
            {
                "method": method,
                "final_label": final_label,
                "final_correct": final_ok,
                "triggered": int(final_label != row["base_label"]),
                "W2R": int((not base_ok) and final_ok),
                "R2W": int(base_ok and not final_ok),
            }
        )
        out_rows.append(out)
    n = len(rows)
    metrics = {
        "method": method,
        "split": rows[0]["split"] if rows else "",
        "packet_count": n,
        "base_correct": base_correct,
        "base_accuracy": base_correct / n if n else 0.0,
        "final_correct": final_correct,
        "final_accuracy": final_correct / n if n else 0.0,
        "trigger_count": trigger,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
    }
    metrics.update(rule)
    return metrics, out_rows


def rule_grid() -> list[dict]:
    modes = ["agree", "raw", "chirp"]
    raw_margins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
    chirp_margins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
    low_costs = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80]
    max_costs = [1.0, 0.80, 0.60, 0.40]
    rank_mins = [0.0, 0.5]
    gate_profiles = [
        {
            "gate_profile": "base_raw_chirp",
            "require_v2_support": 0,
            "require_v2_vote": 0,
            "require_v2_disagree_support": 0,
            "protect_base_rssi_v2": 0,
            "protect_base_strong": 0,
            "base_strong_min_obs": 0.0,
            "base_strong_min_score4": 0.0,
        },
        {
            "gate_profile": "require_v2_support",
            "require_v2_support": 1,
            "require_v2_vote": 0,
            "require_v2_disagree_support": 0,
            "protect_base_rssi_v2": 0,
            "protect_base_strong": 0,
            "base_strong_min_obs": 0.0,
            "base_strong_min_score4": 0.0,
        },
        {
            "gate_profile": "require_v2_vote",
            "require_v2_support": 1,
            "require_v2_vote": 1,
            "require_v2_disagree_support": 0,
            "protect_base_rssi_v2": 0,
            "protect_base_strong": 0,
            "base_strong_min_obs": 0.0,
            "base_strong_min_score4": 0.0,
        },
        {
            "gate_profile": "require_v2_disagree_support",
            "require_v2_support": 1,
            "require_v2_vote": 0,
            "require_v2_disagree_support": 1,
            "protect_base_rssi_v2": 0,
            "protect_base_strong": 0,
            "base_strong_min_obs": 0.0,
            "base_strong_min_score4": 0.0,
        },
        {
            "gate_profile": "protect_base_rssi_v2",
            "require_v2_support": 0,
            "require_v2_vote": 0,
            "require_v2_disagree_support": 0,
            "protect_base_rssi_v2": 1,
            "protect_base_strong": 0,
            "base_strong_min_obs": 0.0,
            "base_strong_min_score4": 0.0,
        },
        {
            "gate_profile": "protect_base_strong",
            "require_v2_support": 0,
            "require_v2_vote": 0,
            "require_v2_disagree_support": 0,
            "protect_base_rssi_v2": 0,
            "protect_base_strong": 1,
            "base_strong_min_obs": 0.90,
            "base_strong_min_score4": 0.90,
        },
        {
            "gate_profile": "protect_base_rssi_v2_and_strong",
            "require_v2_support": 0,
            "require_v2_vote": 0,
            "require_v2_disagree_support": 0,
            "protect_base_rssi_v2": 1,
            "protect_base_strong": 1,
            "base_strong_min_obs": 0.90,
            "base_strong_min_score4": 0.90,
        },
    ]
    rules = []
    for mode in modes:
        for min_raw in raw_margins:
            for min_chirp in chirp_margins:
                if mode == "raw" and min_chirp > 0.0:
                    continue
                if mode == "chirp" and min_raw > 0.0:
                    continue
                for min_low in low_costs:
                    for max_cost in max_costs:
                        for min_rank in rank_mins:
                            for profile in gate_profiles:
                                rule = {
                                    "mode": mode,
                                    "min_raw_margin": min_raw,
                                    "min_chirp_margin": min_chirp,
                                    "min_low_cost": min_low,
                                    "max_cost_norm": max_cost,
                                    "min_rssi_rank_inv": min_rank,
                                    "require_not_rssi_top1": 0,
                                }
                                rule.update(profile)
                                rules.append(rule)
    return rules


def choose_best(rows: Sequence[dict]) -> dict:
    return max(
        rows,
        key=lambda row: (
            row["final_accuracy"],
            row["net_gain"],
            -row["R2W"],
            row["W2R"],
            -row["trigger_count"],
        ),
    )


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = build_packet_rows(args)

    write_csv(
        args.output_dir / "packet_challenger_features.csv",
        [row for rows in rows_by_split.values() for row in rows],
        sorted({key for rows in rows_by_split.values() for row in rows for key in row}),
    )

    all_rules = rule_grid()
    selection_rows = []
    for rule in all_rules:
        metrics, _ = apply_rule(rows_by_split["val"], rule, f"aco_v43_{rule['mode']}")
        selection_rows.append(metrics)
    best = choose_best(selection_rows)
    selected_rule = {key: best[key] for key in all_rules[0]}

    final_rows = []
    prediction_rows = {}
    for split in ["train_loocv", "val", "test"]:
        metrics, preds = apply_rule(rows_by_split[split], selected_rule, "aco_v43_selected")
        final_rows.append(metrics)
        prediction_rows[split] = preds
        write_csv(args.output_dir / f"{split}_predictions.csv", preds, sorted({key for row in preds for key in row}))

    write_csv(args.output_dir / "selection_summary.csv", selection_rows, sorted({key for row in selection_rows for key in row}))
    write_csv(args.output_dir / "final_summary.csv", final_rows, list(final_rows[0].keys()))
    payload = {
        "protocol": (
            "Raw/chirp challenger over ACO4.2 interactions. Rule selected on validation only; "
            "test evaluated once with selected rule."
        ),
        "best_val": best,
        "selected_rule": selected_rule,
        "final": final_rows,
    }
    with (args.output_dir / "aco_v43_raw_chirp_challenger_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--rssi-csv", type=Path, default=split_runner.DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=split_runner.DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=split_runner.DEFAULT_SPLIT_CSV)
    parser.add_argument("--chirp-template-csv", type=Path, default=aco2.DEFAULT_CHIRP_TEMPLATE_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=aco2.DEFAULT_CHIRP_STRUCT_CSV)
    parser.add_argument("--location-csv", type=Path, default=aco2.DEFAULT_LOCATION_CSV)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--segment-count", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--rssi-weight", type=float, default=0.45)
    parser.add_argument("--bin-weight", type=float, default=0.02)
    parser.add_argument("--energy-weight", type=float, default=0.20)
    parser.add_argument("--raw-weight", type=float, default=0.55)
    parser.add_argument("--q4-weight", type=float, default=0.0)
    parser.add_argument("--shrinkage-lambda", type=float, default=8.0)
    parser.add_argument("--phy-var-c0", type=float, default=0.05)
    parser.add_argument("--phy-var-c1", type=float, default=0.50)
    parser.add_argument("--phy-var-c2", type=float, default=1.0)
    parser.add_argument("--sigma0-sq", type=float, default=0.02)
    parser.add_argument("--min-variance", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--logdet-weight", type=float, default=0.05)
    parser.add_argument("--normalize-bin-cost", action="store_true", default=True)
    parser.add_argument("--garbage-cost", type=float, default=1.0)
    parser.add_argument("--lambda0-switch", type=float, default=0.70)
    parser.add_argument("--switch-eta", type=float, default=0.20)
    parser.add_argument("--lambda-div", type=float, default=0.20)
    parser.add_argument("--lambda-g", type=float, default=0.50)
    parser.add_argument("--max-garbage", type=int, default=2)
    parser.add_argument("--garbage-overuse-penalty", type=float, default=4.0)
    parser.add_argument("--lambda-c", type=float, default=0.15)
    parser.add_argument("--tau-stay", type=float, default=1.4)
    parser.add_argument("--tau-switch", type=float, default=0.35)
    parser.add_argument("--pheromone-power", type=float, default=1.0)
    parser.add_argument("--heuristic-power", type=float, default=1.4)
    parser.add_argument("--evaporation", type=float, default=0.25)
    parser.add_argument("--min-pheromone", type=float, default=1e-4)
    parser.add_argument("--aco-temperature", type=float, default=None)
    parser.add_argument("--q4-shift-grid", default="-0.25,0,0.25")
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "final": payload["final"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
