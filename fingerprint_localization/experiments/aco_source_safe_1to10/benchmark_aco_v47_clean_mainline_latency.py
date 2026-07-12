#!/usr/bin/env python3
"""Benchmark the fixed clean-mainline ACO V4.7 online latency.

This script intentionally does not train selectors, sweep thresholds, or
enumerate rescue-rule orders.  It measures the fixed inference path used by
the reported V4.7 result:

1. build the V4.4 method-candidate pool from cached expert predictions;
2. add V4.5 reliability features with train-only reliability tables;
3. apply the fixed V4.6 guarded selector;
4. apply the frozen V4.7 rescue-rule order.

The source expert predictions are treated as cached inputs here.  Their own
offline generation is not part of this latency number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import run_aco_v44_method_ensemble_selector as v44  # noqa: E402
import run_aco_v45_reliability_selector as v45  # noqa: E402
import run_aco_v46_guarded_selector as v46  # noqa: E402
import run_aco_v47_two_stage_rules as v47  # noqa: E402


RESULTS_DIR = EXPERIMENT_DIR / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "aco_v47_clean_mainline_latency"
SPLITS = ["train_loocv", "val", "test"]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def now() -> float:
    return time.perf_counter()


def flatten_v44_groups(groups_by_split: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for split_groups in groups_by_split.values():
        for group in split_groups:
            for row in group["rows"]:
                flat = {key: value for key, value in row.items() if key != "features"}
                flat.update(row["features"])
                rows.append(flat)
    return rows


def metric_accuracy(final_rows: Sequence[dict]) -> dict[str, dict]:
    out = {}
    for row in final_rows:
        n = row["packet_count"]
        out[row["split"]] = {
            "packet_count": n,
            "correct": row["final_correct"],
            "accuracy": row["final_accuracy"],
            "W2R": row["W2R"],
            "R2W": row["R2W"],
            "net_gain": row["net_gain"],
            "trigger_count": row["trigger_count"],
            "oracle": row["candidate_oracle_accuracy"],
        }
    return out


def stage_summary(seconds: Sequence[float]) -> dict:
    values = sorted(seconds)
    if not values:
        return {}

    def q(p: float) -> float:
        idx = round((len(values) - 1) * p)
        return values[idx]

    return {
        "count": len(values),
        "mean_us": statistics.fmean(values) * 1e6,
        "median_us": statistics.median(values) * 1e6,
        "p90_us": q(0.90) * 1e6,
        "p95_us": q(0.95) * 1e6,
        "p99_us": q(0.99) * 1e6,
        "min_us": values[0] * 1e6,
        "max_us": values[-1] * 1e6,
    }


def batch_once(args: argparse.Namespace) -> dict:
    t0 = now()
    model = v47.load_model(args.v45_dir / "aco_v45_metrics.json")
    v46_metrics = load_json(args.v46_dir / "aco_v46_metrics.json")
    v47_metrics = load_json(args.v47_dir / "aco_v47_metrics.json")
    theta = v46.parse_float(v46_metrics["best_val"]["theta"])
    rule_order = v47_metrics["selected_rule_order"]
    ranker_name = v47_metrics["selected_ranker"]
    rules = v47.build_rules()
    rankers = v47.build_rankers()
    t_models = now()

    v44_args = argparse.Namespace(results_dir=args.results_dir)
    top3_features = v44.load_top3_features(args.results_dir)
    raw_packets = v44.load_raw_chirp_packets(args.results_dir)
    v44_groups = {
        split: v44.build_groups(v44_args, split, top3_features, raw_packets)
        for split in SPLITS
    }
    raw_rows = flatten_v44_groups(v44_groups)
    t_candidate_pool = now()

    by_split = v45.group_rows(raw_rows)
    rel = v45.build_reliability_tables(by_split["train_loocv"])
    engineered_rows = v45.add_engineered_features(raw_rows, rel, v45.numeric_columns(raw_rows))
    t_v45_features = now()

    v45_groups = {
        split: v45.group_samples([row for row in engineered_rows if row["split"] == split])
        for split in SPLITS
    }
    stage1_rows = {}
    v46_final = []
    for split in SPLITS:
        v46.score_groups(v45_groups[split], model)
        metrics, preds = v46.evaluate(v45_groups[split], theta, "aco_v46_fixed")
        v46_final.append(metrics)
        stage1_rows[split] = {int(row["sample_index"]): row for row in preds}
    t_v46_stage1 = now()

    enriched_rows = [v47.enrich_row(row, model) for row in engineered_rows]
    candidate_groups = v47.group_candidates(enriched_rows)
    v47_final = []
    for split in SPLITS:
        metrics, _preds = v47.evaluate_combo(
            candidate_groups[split],
            stage1_rows[split],
            rule_order,
            ranker_name,
            rules,
            rankers,
            "aco_v47_fixed",
        )
        v47_final.append(metrics)
    t_v47_rescue = now()

    return {
        "timing_s": {
            "load_fixed_models": t_models - t0,
            "candidate_pool_from_cached_experts": t_candidate_pool - t_models,
            "v45_reliability_feature_engineering": t_v45_features - t_candidate_pool,
            "v46_guarded_selector": t_v46_stage1 - t_v45_features,
            "v47_frozen_rescue": t_v47_rescue - t_v46_stage1,
            "total_cached_expert_pipeline": t_v47_rescue - t0,
        },
        "row_counts": {
            "candidate_rows": len(raw_rows),
            "engineered_rows": len(engineered_rows),
            "samples": sum(len(v44_groups[split]) for split in SPLITS),
            "test_samples": len(v44_groups["test"]),
        },
        "v46_metrics": metric_accuracy(v46_final),
        "v47_metrics": metric_accuracy(v47_final),
    }


def load_source_cache(results_dir: Path) -> dict:
    method_preds = {split: v44.load_method_predictions(results_dir, split) for split in SPLITS}
    ablation_preds = {split: v44.load_ablation_predictions(results_dir, split) for split in SPLITS}
    top3_features = v44.load_top3_features(results_dir)
    raw_packets = v44.load_raw_chirp_packets(results_dir)
    sample_ids = {
        split: sorted(
            set.intersection(
                *[
                    set(rows)
                    for rows in {**method_preds[split], **ablation_preds[split]}.values()
                ]
            )
        )
        for split in SPLITS
    }
    return {
        "method_preds": method_preds,
        "ablation_preds": ablation_preds,
        "top3_features": top3_features,
        "raw_packets": raw_packets,
        "sample_ids": sample_ids,
    }


def build_group_from_cached_sources(cache: dict, split: str, sample_index: int) -> dict:
    sources = {**cache["method_preds"][split], **cache["ablation_preds"][split]}
    true_label = next(iter(sources.values()))[sample_index]["true_label"]
    labels_by_source = {name: rows[sample_index]["label"] for name, rows in sources.items()}
    family_by_source = {name: rows[sample_index]["family"] for name, rows in sources.items()}
    packet = cache["raw_packets"].get(split, {}).get(sample_index, {})
    raw_winner = packet.get("raw_winner", "")
    chirp_winner = packet.get("chirp_winner", "")
    agreed_winner = packet.get("agreed_winner", "")
    for extra_label in [raw_winner, chirp_winner, agreed_winner]:
        if extra_label:
            labels_by_source.setdefault(f"expert_extra_{extra_label}", extra_label)
            family_by_source.setdefault(f"expert_extra_{extra_label}", "raw_chirp")

    label_counts = v44.Counter(labels_by_source.values())
    top_count = max(label_counts.values()) if label_counts else 0
    ordered_counts = sorted(label_counts.values(), reverse=True)
    second_count = ordered_counts[1] if len(ordered_counts) > 1 else 0
    v43_label = labels_by_source.get("v43_raw_chirp", "")
    v42_label = labels_by_source.get("v42_interactions", "")
    v2_label = labels_by_source.get("aco_v2", "")
    knn_label = labels_by_source.get("knn", "")
    mfr_label = labels_by_source.get("mfr_prev", "")
    top3_by_label = cache["top3_features"].get(split, {}).get(sample_index, {})
    rssi_label = ""
    if top3_by_label:
        rssi_label = next(iter(top3_by_label.values())).get("rssi_top1_label", "")

    rows = []
    for label in sorted(label_counts, key=v44.base42.natural_label_key):
        hit_sources = [name for name, pred_label in labels_by_source.items() if pred_label == label]
        family_counts = v44.Counter(family_by_source[name] for name in hit_sources)
        same_v43, delta_v43 = v44.label_distance(label, v43_label)
        same_v42, delta_v42 = v44.label_distance(label, v42_label)
        same_v2, delta_v2 = v44.label_distance(label, v2_label)
        same_rssi, delta_rssi = v44.label_distance(label, rssi_label)
        same_knn, delta_knn = v44.label_distance(label, knn_label)
        same_mfr, delta_mfr = v44.label_distance(label, mfr_label)
        candidate_features = top3_by_label.get(label, {})
        features = {
            "bias_feature": 1.0,
            "source_hit_count": float(len(hit_sources)),
            "source_hit_frac": len(hit_sources) / max(1, len(labels_by_source)),
            "is_consensus_top": float(label_counts[label] == top_count),
            "consensus_margin": float(label_counts[label] - second_count),
            "aco4_hit_count": float(family_counts["aco4"]),
            "aco2_hit_count": float(family_counts["aco2"]),
            "ablation_hit_count": float(family_counts["ablation"]),
            "weak_hit_count": float(family_counts["weak"]),
            "raw_chirp_source_hit": float(family_counts["raw_chirp"]),
            "same_corridor_as_v43": same_v43,
            "abs_loc_delta_v43": delta_v43,
            "same_corridor_as_v42": same_v42,
            "abs_loc_delta_v42": delta_v42,
            "same_corridor_as_v2": same_v2,
            "abs_loc_delta_v2": delta_v2,
            "same_corridor_as_rssi": same_rssi,
            "abs_loc_delta_rssi": delta_rssi,
            "same_corridor_as_knn": same_knn,
            "abs_loc_delta_knn": delta_knn,
            "same_corridor_as_mfr": same_mfr,
            "abs_loc_delta_mfr": delta_mfr,
            "is_raw_winner_packet": float(label == raw_winner),
            "is_chirp_winner_packet": float(label == chirp_winner),
            "is_raw_chirp_agreed_packet": float(bool(agreed_winner) and label == agreed_winner),
            "raw_margin_v21": v44.parse_float(packet.get("raw_margin_v21"), 0.0),
            "chirp_margin_v22": v44.parse_float(packet.get("chirp_margin_v22"), 0.0),
            "raw_chirp_min_margin_packet": min(
                v44.parse_float(packet.get("raw_margin_v21"), 0.0),
                v44.parse_float(packet.get("chirp_margin_v22"), 0.0),
            ),
            "in_aco_top3": float(label in top3_by_label),
        }
        for name in v44.METHOD_SPECS:
            method_name = name[0]
            features[f"src_{method_name}"] = float(labels_by_source.get(method_name) == label)
        for name in v44.ABLATION_SPECS:
            method_name = name[0]
            features[f"src_{method_name}"] = float(labels_by_source.get(method_name) == label)
        for key in v44.TOP3_FEATURE_KEYS:
            features[f"top3_{key}"] = v44.parse_float(candidate_features.get(key), 0.0)
        ca, la = v44.base42.parse_label(label)
        features["candidate_corridor"] = float(ca)
        features["candidate_location_scaled"] = la / 54.0
        rows.append(
            {
                "split": split,
                "sample_index": sample_index,
                "true_label": true_label,
                "candidate_label": label,
                "target": int(label == true_label),
                "base_label": v43_label,
                "base_correct": int(v43_label == true_label),
                "features": features,
                "source_names": ";".join(hit_sources),
            }
        )
    return {
        "split": split,
        "sample_index": sample_index,
        "true_label": true_label,
        "base_label": v43_label,
        "base_correct": int(v43_label == true_label),
        "true_in_candidates": int(true_label in label_counts),
        "rows": rows,
    }


def flatten_single_group(group: dict) -> list[dict]:
    rows = []
    for row in group["rows"]:
        flat = {key: value for key, value in row.items() if key != "features"}
        flat.update(row["features"])
        rows.append(flat)
    return rows


def run_online_once(
    split: str,
    sample_index: int,
    cache: dict,
    rel: dict,
    model: dict,
    theta: float,
    rule_order: Sequence[str],
    ranker_name: str,
    rules: dict,
    rankers: dict,
) -> tuple[dict, dict[str, float]]:
    t0 = now()
    raw_group = build_group_from_cached_sources(cache, split, sample_index)
    raw_rows = flatten_single_group(raw_group)
    t_candidate = now()

    engineered_rows = v45.add_engineered_features(raw_rows, rel, [])
    v45_group = v45.group_samples(engineered_rows)[0]
    t_features = now()

    v46.score_groups([v45_group], model)
    _stage1_metrics, stage1_preds = v46.evaluate([v45_group], theta, "aco_v46_fixed")
    stage1 = {sample_index: stage1_preds[0]}
    t_v46 = now()

    v47_rows = [v47.enrich_row(row, model) for row in engineered_rows]
    _metrics, final_preds = v47.evaluate_combo(
        {sample_index: v47_rows},
        stage1,
        rule_order,
        ranker_name,
        rules,
        rankers,
        "aco_v47_fixed",
    )
    t_v47 = now()
    return final_preds[0], {
        "candidate_pool_from_cached_experts": t_candidate - t0,
        "v45_reliability_feature_engineering": t_features - t_candidate,
        "v46_guarded_selector": t_v46 - t_features,
        "v47_frozen_rescue": t_v47 - t_v46,
        "online_total_cached_expert_pipeline": t_v47 - t0,
    }


def online_benchmark(args: argparse.Namespace, rel: dict) -> dict:
    t0 = now()
    cache = load_source_cache(args.results_dir)
    model = v47.load_model(args.v45_dir / "aco_v45_metrics.json")
    v46_metrics = load_json(args.v46_dir / "aco_v46_metrics.json")
    v47_metrics = load_json(args.v47_dir / "aco_v47_metrics.json")
    theta = v46.parse_float(v46_metrics["best_val"]["theta"])
    rule_order = v47_metrics["selected_rule_order"]
    ranker_name = v47_metrics["selected_ranker"]
    rules = v47.build_rules()
    rankers = v47.build_rankers()
    preload_s = now() - t0

    sample_ids = cache["sample_ids"][args.online_split]
    timings: dict[str, list[float]] = {
        "candidate_pool_from_cached_experts": [],
        "v45_reliability_feature_engineering": [],
        "v46_guarded_selector": [],
        "v47_frozen_rescue": [],
        "online_total_cached_expert_pipeline": [],
    }
    first_pass_preds = []
    for repeat in range(args.online_repeats):
        for sample_index in sample_ids:
            pred, timing = run_online_once(
                args.online_split,
                sample_index,
                cache,
                rel,
                model,
                theta,
                rule_order,
                ranker_name,
                rules,
                rankers,
            )
            for key, value in timing.items():
                timings[key].append(value)
            if repeat == 0:
                first_pass_preds.append(pred)

    n = len(first_pass_preds)
    correct = sum(int(row["final_correct"]) for row in first_pass_preds)
    trigger = sum(int(row["triggered"]) for row in first_pass_preds)
    w2r = sum(int(row["W2R"]) for row in first_pass_preds)
    r2w = sum(int(row["R2W"]) for row in first_pass_preds)
    return {
        "preload_s": preload_s,
        "split": args.online_split,
        "sample_count": n,
        "repeats": args.online_repeats,
        "total_timed_samples": n * args.online_repeats,
        "accuracy_check": {
            "correct": correct,
            "accuracy": correct / n if n else 0.0,
            "trigger_count": trigger,
            "W2R": w2r,
            "R2W": r2w,
            "net_gain": w2r - r2w,
        },
        "per_packet_latency": {key: stage_summary(values) for key, values in timings.items()},
    }


def aggregate_batch_runs(batch_runs: Sequence[dict]) -> dict:
    timings = {key: [] for key in batch_runs[0]["timing_s"]}
    for run in batch_runs:
        for key, value in run["timing_s"].items():
            timings[key].append(value)
    samples = batch_runs[0]["row_counts"]["samples"]
    return {
        "repeats": len(batch_runs),
        "samples_per_run": samples,
        "test_samples": batch_runs[0]["row_counts"]["test_samples"],
        "row_counts": batch_runs[0]["row_counts"],
        "timing_s": {
            key: {
                "mean_s": statistics.fmean(values),
                "median_s": statistics.median(values),
                "runs_s": values,
                "mean_ms_per_sample": statistics.fmean(values) * 1000.0 / samples,
            }
            for key, values in timings.items()
        },
        "v46_metrics": batch_runs[0]["v46_metrics"],
        "v47_metrics": batch_runs[0]["v47_metrics"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--v45-dir", type=Path, default=RESULTS_DIR / "aco_v45_reliability_selector")
    parser.add_argument("--v46-dir", type=Path, default=RESULTS_DIR / "aco_v46_guarded_selector")
    parser.add_argument("--v47-dir", type=Path, default=RESULTS_DIR / "aco_v47_two_stage_rules")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-repeats", type=int, default=5)
    parser.add_argument("--online-repeats", type=int, default=200)
    parser.add_argument("--online-split", choices=SPLITS, default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch_runs = [batch_once(args) for _ in range(args.batch_repeats)]
    batch = aggregate_batch_runs(batch_runs)

    # Reuse the batch-built raw candidate rows only to define the fixed train
    # reliability table for online timing.
    v44_args = argparse.Namespace(results_dir=args.results_dir)
    top3_features = v44.load_top3_features(args.results_dir)
    raw_packets = v44.load_raw_chirp_packets(args.results_dir)
    v44_groups = {
        split: v44.build_groups(v44_args, split, top3_features, raw_packets)
        for split in SPLITS
    }
    raw_rows = flatten_v44_groups(v44_groups)
    rel = v45.build_reliability_tables(v45.group_rows(raw_rows)["train_loocv"])
    online = online_benchmark(args, rel)

    payload = {
        "latency_scope": {
            "included": [
                "V4.4 candidate pool fusion from cached expert predictions",
                "V4.5 reliability feature engineering using fixed train reliability tables",
                "V4.6 fixed guarded selector",
                "V4.7 frozen rescue rules",
            ],
            "excluded": [
                "selector training",
                "threshold or rule-order search",
                "stability-validation random split generation",
                "offline generation of source expert prediction CSVs",
                "raw IQ/RSSI acquisition from radio hardware",
            ],
        },
        "batch_cached_expert_pipeline": batch,
        "online_cached_expert_pipeline": online,
    }
    out_path = args.output_dir / "clean_mainline_latency_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
