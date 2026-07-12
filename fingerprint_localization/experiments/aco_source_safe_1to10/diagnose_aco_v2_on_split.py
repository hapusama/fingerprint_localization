#!/usr/bin/env python3
"""Diagnostics for ACO 2.0 on the fixed 1:10 Gaussian-noise test split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
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


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "aco_v2_diagnostics"
DEFAULT_ABLATION_DIR = EXPERIMENT_DIR / "results" / "aco_v2_ablation"
DEFAULT_ACO_V2_DIR = EXPERIMENT_DIR / "results" / "aco_v2"
EPS = 1e-12


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
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


def median(values: Sequence[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def quantile(values: Sequence[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def load_predictions(path: Path) -> dict[int, dict]:
    return {int(row["sample_index"]): row for row in read_csv(path)}


def rank_string(items: Sequence[tuple[str, float]], reverse: bool = False) -> str:
    ranked = sorted(items, key=lambda item: (-item[1], aco2.natural_label_key(item[0])) if reverse else (item[1], aco2.natural_label_key(item[0])))
    return ";".join(f"{label}:{score:.6g}" for label, score in ranked)


def rank_labels(items: Sequence[tuple[str, float]], reverse: bool = False) -> list[str]:
    ranked = sorted(items, key=lambda item: (-item[1], aco2.natural_label_key(item[0])) if reverse else (item[1], aco2.natural_label_key(item[0])))
    return [label for label, _score in ranked]


def label_rank(label: str, labels: Sequence[str]) -> int:
    try:
        return list(labels).index(label) + 1
    except ValueError:
        return 0


def change_matrix(args: argparse.Namespace) -> tuple[list[dict], dict[str, dict[int, dict]]]:
    by_version = {}
    for version in ["v1_0", "v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v2_6", "v2_7"]:
        by_version[version] = load_predictions(args.ablation_dir / version / "test_predictions.csv")
    v1 = by_version["v1_0"]
    rows = []
    version_meta = {config.version.lower().replace(".", "_"): config for config in ablation.ABLATIONS}
    for version in ["v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v2_6", "v2_7"]:
        plus = minus = both_correct = both_wrong = 0
        for sample_index, base_row in v1.items():
            other = by_version[version][sample_index]
            base_ok = int(base_row["aco_vote_correct"])
            other_ok = int(other["aco_vote_correct"])
            if not base_ok and other_ok:
                plus += 1
            elif base_ok and not other_ok:
                minus += 1
            elif base_ok and other_ok:
                both_correct += 1
            else:
                both_wrong += 1
        config = version_meta[version]
        n = len(v1)
        rows.append(
            {
                "compare_to": "V1.0",
                "version": config.version,
                "change": config.change,
                "purpose": config.purpose,
                "v1_wrong_vx_correct": plus,
                "v1_correct_vx_wrong": minus,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "net_gain": plus - minus,
                "v1_accuracy": (plus * 0 + both_correct + minus) / n if n else 0.0,
                "vx_accuracy": (plus + both_correct) / n if n else 0.0,
                "packet_count": n,
            }
        )
    return rows, by_version


def build_bin_rankings(args: argparse.Namespace) -> tuple[dict[int, dict], dict]:
    aco_args = split_runner.build_args(args)
    rssi_packets = aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, _thresholds = aco2.base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = split_runner.load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    chirp_shapes, chirp_struct, _chirp_metadata = aco2.prepare_chirp_fields(aco_args, labels)
    train_indices = split_indices["train"]
    config_by_version = {config.version: config for config in ablation.ABLATIONS}
    configs = {
        "raw_gaussian_v2_1": config_by_version["V2.1"],
        "chirp_mean_v2_2": config_by_version["V2.2"],
        "huber_garbage_v2_5": config_by_version["V2.5"],
        "full_v2_7": config_by_version["V2.7"],
    }
    templates = {
        name: ablation.build_templates_for_ablation(samples, labels, train_indices, chirp_shapes, chirp_struct, aco_args, config)
        for name, config in configs.items()
    }
    rankings: dict[int, dict] = {}
    for test_index in split_indices["test"]:
        ranked = aco2.base.class_rank(rssi_rows, labels, train_indices, test_index, aco_args.rssi_class_k)
        candidates = [label for label, _score in ranked[: aco_args.top_k]]
        row = {
            "true_label": labels[test_index],
            "rssi_rank": rank_string([(label, score) for label, score in ranked[: aco_args.top_k]]),
            "rssi_rank_labels": candidates,
        }
        for name, config in configs.items():
            scores = []
            for label in candidates:
                total = sum(
                    ablation.bin_cost(shape, templates[name][label], aco_args, config)
                    for shape in samples[test_index].segment_shapes
                )
                scores.append((label, total))
            labels_ranked = rank_labels(scores)
            row[f"{name}_rank"] = rank_string(scores)
            row[f"{name}_winner"] = labels_ranked[0] if labels_ranked else ""
            row[f"{name}_true_rank"] = label_rank(labels[test_index], labels_ranked)
        rankings[test_index] = row
    metadata = {
        "sample_count": len(samples),
        "train_count": len(split_indices["train"]),
        "test_count": len(split_indices["test"]),
    }
    return rankings, metadata


def load_candidate_rankings(path: Path) -> dict[int, dict]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in read_csv(path):
        grouped[int(row["sample_index"])].append(row)
    out = {}
    for sample_index, rows in grouped.items():
        pheromone_scores = [(row["candidate_label"], parse_float(row["self_pheromone"])) for row in rows]
        vote_scores = [(row["candidate_label"], parse_float(row["elite_vote"])) for row in rows]
        pheromone_labels = rank_labels(pheromone_scores, reverse=True)
        vote_labels = rank_labels(vote_scores, reverse=True)
        vote_sorted = sorted(vote_scores, key=lambda item: (-item[1], aco2.natural_label_key(item[0])))
        margin = vote_sorted[0][1] - vote_sorted[1][1] if len(vote_sorted) >= 2 else 0.0
        out[sample_index] = {
            "self_pheromone_rank": rank_string(pheromone_scores, reverse=True),
            "self_pheromone_winner": pheromone_labels[0] if pheromone_labels else "",
            "vote_rank": rank_string(vote_scores, reverse=True),
            "vote_winner": vote_labels[0] if vote_labels else "",
            "vote_margin": margin,
        }
    return out


def top3_wrong_details(
    args: argparse.Namespace,
    by_version: dict[str, dict[int, dict]],
    bin_rankings: dict[int, dict],
) -> tuple[list[dict], list[dict]]:
    final_predictions = load_predictions(args.aco_v2_dir / "test_predictions.csv")
    final_candidate_ranks = load_candidate_rankings(args.aco_v2_dir / "test_candidate_scores.csv")
    rows = []
    for sample_index, pred in final_predictions.items():
        if int(pred["true_in_rssi_topk"]) != 1 or int(pred["aco_vote_correct"]) == 1:
            continue
        true_label = pred["true_label"]
        bin_row = bin_rankings[sample_index]
        cand_row = final_candidate_ranks[sample_index]
        self_labels = [part.split(":", 1)[0] for part in cand_row["self_pheromone_rank"].split(";") if part]
        rows.append(
            {
                "sample_index": sample_index,
                "file_name": pred["file_name"],
                "packet_index": pred["packet_index"],
                "true_label": true_label,
                "rssi_top3": pred["rssi_topk_candidates"],
                "rssi_top3_with_scores": bin_row["rssi_rank"],
                "rssi_top1_label": pred["rssi_top1_label"],
                "original_aco_v1_vote": by_version["v1_0"][sample_index]["aco_vote_label"],
                "aco_v2_vote": pred["aco_vote_label"],
                "raw_gaussian_rank": bin_row["raw_gaussian_v2_1_rank"],
                "raw_gaussian_winner": bin_row["raw_gaussian_v2_1_winner"],
                "raw_gaussian_true_rank": bin_row["raw_gaussian_v2_1_true_rank"],
                "chirp_shrink_rank": bin_row["chirp_mean_v2_2_rank"],
                "chirp_shrink_winner": bin_row["chirp_mean_v2_2_winner"],
                "chirp_shrink_true_rank": bin_row["chirp_mean_v2_2_true_rank"],
                "self_pheromone_rank": cand_row["self_pheromone_rank"],
                "self_pheromone_winner": cand_row["self_pheromone_winner"],
                "self_pheromone_true_rank": label_rank(true_label, self_labels),
                "aco_vote_rank": cand_row["vote_rank"],
                "aco_vote_margin": cand_row["vote_margin"],
                "raw_gaussian_true_first": int(bin_row["raw_gaussian_v2_1_winner"] == true_label),
                "chirp_shrink_true_first": int(bin_row["chirp_mean_v2_2_winner"] == true_label),
                "self_pheromone_true_first": int(cand_row["self_pheromone_winner"] == true_label),
                "rssi_top1_true": int(pred["rssi_top1_label"] == true_label),
                "v1_vote_true": int(by_version["v1_0"][sample_index]["aco_vote_label"] == true_label),
            }
        )
    n = len(rows)
    summary = [
        {
            "subset": "true_in_top3_and_aco_v2_vote_wrong",
            "packet_count": n,
            "raw_gaussian_true_first": sum(row["raw_gaussian_true_first"] for row in rows),
            "chirp_shrink_true_first": sum(row["chirp_shrink_true_first"] for row in rows),
            "self_pheromone_true_first": sum(row["self_pheromone_true_first"] for row in rows),
            "rssi_top1_true": sum(row["rssi_top1_true"] for row in rows),
            "v1_vote_true": sum(row["v1_vote_true"] for row in rows),
            "mean_vote_margin": sum(row["aco_vote_margin"] for row in rows) / n if n else 0.0,
            "median_vote_margin": median([row["aco_vote_margin"] for row in rows]) if n else 0.0,
        }
    ]
    return rows, summary


def oracle_diagnostics(
    args: argparse.Namespace,
    by_version: dict[str, dict[int, dict]],
    bin_rankings: dict[int, dict],
) -> tuple[list[dict], list[dict]]:
    final_predictions = load_predictions(args.aco_v2_dir / "test_predictions.csv")
    rows = []
    expert_correct = Counter()
    for sample_index, pred in final_predictions.items():
        true_label = pred["true_label"]
        experts = {
            "rssi_top1": pred["rssi_top1_label"],
            "original_aco_v1_vote": by_version["v1_0"][sample_index]["aco_vote_label"],
            "aco_v2_self_pheromone": pred["aco_pheromone_label"],
            "raw_gaussian_winner": bin_rankings[sample_index]["raw_gaussian_v2_1_winner"],
            "chirp_shrink_winner": bin_rankings[sample_index]["chirp_mean_v2_2_winner"],
            "huber_garbage_v2_5_vote": by_version["v2_5"][sample_index]["aco_vote_label"],
        }
        oracle_correct = int(any(label == true_label for label in experts.values()))
        row = {
            "sample_index": sample_index,
            "file_name": pred["file_name"],
            "packet_index": pred["packet_index"],
            "true_label": true_label,
            "true_in_rssi_top3": pred["true_in_rssi_topk"],
            "oracle_correct": oracle_correct,
        }
        for name, label in experts.items():
            ok = int(label == true_label)
            expert_correct[name] += ok
            row[f"{name}_label"] = label
            row[f"{name}_correct"] = ok
        rows.append(row)
    n = len(rows)
    summary = [
        {
            "expert": expert,
            "correct": expert_correct[expert],
            "packet_count": n,
            "accuracy": expert_correct[expert] / n if n else 0.0,
        }
        for expert in [
            "rssi_top1",
            "original_aco_v1_vote",
            "aco_v2_self_pheromone",
            "raw_gaussian_winner",
            "chirp_shrink_winner",
            "huber_garbage_v2_5_vote",
        ]
    ]
    oracle_correct = sum(row["oracle_correct"] for row in rows)
    summary.append(
        {
            "expert": "oracle_any_expert",
            "correct": oracle_correct,
            "packet_count": n,
            "accuracy": oracle_correct / n if n else 0.0,
        }
    )
    return rows, summary


def range_stats(values: Sequence[float]) -> dict:
    values = list(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "q75": quantile(values, 0.75),
        "q90": quantile(values, 0.90),
        "max": max(values),
        "zero_rate": sum(1 for value in values if abs(value) <= EPS) / len(values),
    }


def cost_scale_diagnostics(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rows = read_csv(args.aco_v2_dir / "test_segment_costs.csv")
    non_g = [row for row in rows if row["candidate_label"] != aco2.GARBAGE_LABEL]
    weights = {"C_R": 0.45, "C_E": 0.20, "C_W": 0.55, "C_bin": 0.02, "C_Q": 0.0, "C_bin_raw": 0.02}
    detailed = []
    grouped_segment: dict[tuple[int, int], list[dict]] = defaultdict(list)
    grouped_packet: dict[int, list[dict]] = defaultdict(list)
    for row in non_g:
        sample_index = int(row["sample_index"])
        segment_index = int(row["segment_index"])
        grouped_segment[(sample_index, segment_index)].append(row)
        grouped_packet[sample_index].append(row)

    for (sample_index, segment_index), group in grouped_segment.items():
        for cost_name in ["C_R", "C_E", "C_W", "C_bin", "C_bin_raw", "C_Q"]:
            vals = [parse_float(row[cost_name]) for row in group if row.get(cost_name, "") != ""]
            if not vals:
                continue
            diff = max(vals) - min(vals)
            detailed.append(
                {
                    "scope": "segment_top3",
                    "sample_index": sample_index,
                    "segment_index": segment_index,
                    "cost_name": cost_name,
                    "range": diff,
                    "weight": weights.get(cost_name, 1.0),
                    "weighted_range": diff * weights.get(cost_name, 1.0),
                }
            )

    for sample_index, group in grouped_packet.items():
        by_candidate: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for row in group:
            label = row["candidate_label"]
            for cost_name in ["C_R", "C_E", "C_W", "C_bin", "C_bin_raw", "C_Q"]:
                if row.get(cost_name, "") != "":
                    by_candidate[label][cost_name] += parse_float(row[cost_name])
        for cost_name in ["C_R", "C_E", "C_W", "C_bin", "C_bin_raw", "C_Q"]:
            vals = [costs[cost_name] for costs in by_candidate.values() if cost_name in costs]
            if not vals:
                continue
            diff = max(vals) - min(vals)
            detailed.append(
                {
                    "scope": "packet_sum_top3",
                    "sample_index": sample_index,
                    "segment_index": "",
                    "cost_name": cost_name,
                    "range": diff,
                    "weight": weights.get(cost_name, 1.0),
                    "weighted_range": diff * weights.get(cost_name, 1.0),
                }
            )

    summary = []
    for scope in ["segment_top3", "packet_sum_top3"]:
        for cost_name in ["C_R", "C_E", "C_W", "C_bin", "C_bin_raw", "C_Q"]:
            items = [row for row in detailed if row["scope"] == scope and row["cost_name"] == cost_name]
            if not items:
                continue
            stats = range_stats([row["range"] for row in items])
            weighted = range_stats([row["weighted_range"] for row in items])
            summary.append(
                {
                    "scope": scope,
                    "cost_name": cost_name,
                    "weight": items[0]["weight"],
                    "range_mean": stats["mean"],
                    "range_median": stats["median"],
                    "range_q90": stats["q90"],
                    "range_max": stats["max"],
                    "range_zero_rate": stats["zero_rate"],
                    "weighted_range_mean": weighted["mean"],
                    "weighted_range_median": weighted["median"],
                    "weighted_range_q90": weighted["q90"],
                    "weighted_range_max": weighted["max"],
                }
            )
    return detailed, summary


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows, by_version = change_matrix(args)
    bin_rankings, ranking_metadata = build_bin_rankings(args)
    wrong_rows, wrong_summary = top3_wrong_details(args, by_version, bin_rankings)
    oracle_rows, oracle_summary = oracle_diagnostics(args, by_version, bin_rankings)
    cost_rows, cost_summary = cost_scale_diagnostics(args)

    write_csv(args.output_dir / "diagnosis1_change_matrix.csv", matrix_rows, list(matrix_rows[0].keys()))
    write_csv(args.output_dir / "diagnosis2_top3_wrong_packets.csv", wrong_rows, list(wrong_rows[0].keys()) if wrong_rows else ["sample_index"])
    write_csv(args.output_dir / "diagnosis2_top3_wrong_summary.csv", wrong_summary, list(wrong_summary[0].keys()))
    write_csv(args.output_dir / "diagnosis3_oracle_packets.csv", oracle_rows, list(oracle_rows[0].keys()))
    write_csv(args.output_dir / "diagnosis3_oracle_summary.csv", oracle_summary, list(oracle_summary[0].keys()))
    write_csv(args.output_dir / "diagnosis4_cost_scale_by_segment.csv", cost_rows, list(cost_rows[0].keys()))
    write_csv(args.output_dir / "diagnosis4_cost_scale_summary.csv", cost_summary, list(cost_summary[0].keys()))
    payload = {
        "method": "ACO 2.0 diagnostics on gaussian_noise_1to10_split test set",
        "source": "external_design_notes/蚁群算法2.0诊断.md",
        "ranking_metadata": ranking_metadata,
        "diagnosis1_change_matrix": matrix_rows,
        "diagnosis2_top3_wrong_summary": wrong_summary,
        "diagnosis3_oracle_summary": oracle_summary,
        "diagnosis4_cost_scale_summary": cost_summary,
    }
    with (args.output_dir / "aco_v2_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--aco-v2-dir", type=Path, default=DEFAULT_ACO_V2_DIR)
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
    print(json.dumps(payload["diagnosis2_top3_wrong_summary"], indent=2, ensure_ascii=False))
    print(json.dumps(payload["diagnosis3_oracle_summary"], indent=2, ensure_ascii=False))
    print(f"Wrote {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
