#!/usr/bin/env python3
"""Sweep lambda parameters for segment-4 physical-pathcost ACO."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
PKG_ROOT = PROJECT_ROOT / "fingerprint_localization"
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

import run_aco_segment4_physical_pathcost as phys  # noqa: E402
import run_aco_segmented as segmented  # noqa: E402
import run_experiment as re  # noqa: E402
from model.v3 import aco_packet_path as aco  # noqa: E402


EPS = 1e-12


def parse_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def transition_components(candidates: Sequence[str], location_meta: dict[str, dict], d0: float) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    sw = []
    dist = []
    vis = []
    for left in candidates:
        left_meta = location_meta.get(left, {})
        sw_row = []
        dist_row = []
        vis_row = []
        for right in candidates:
            right_meta = location_meta.get(right, {})
            sw_row.append(1.0 if left != right else 0.0)
            left_distance = float(left_meta.get("distance_m", 0.0))
            right_distance = float(right_meta.get("distance_m", left_distance))
            dist_row.append((abs(left_distance - right_distance) / max(d0, EPS)) ** 2)
            left_state = left_meta.get("visibility_state")
            right_state = right_meta.get("visibility_state", left_state)
            vis_row.append(1.0 if left_state != right_state else 0.0)
        sw.append(sw_row)
        dist.append(dist_row)
        vis.append(vis_row)
    return sw, dist, vis


def precompute_records(
    samples: Sequence[aco.PacketSample],
    q4_offsets: Sequence[float],
    chirp_priors: dict,
    location_meta: dict[str, dict],
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    args,
) -> tuple[list[dict], dict]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    prototypes = aco.build_symbol_prototypes(samples, labels, train_indices)
    pathloss_model = phys.fit_pathloss_model(samples, train_indices, location_meta)
    d0 = phys.nearest_spacing(location_meta, sorted({labels[idx] for idx in train_indices}, key=aco.natural_label_key))
    records = []
    topk_contains = 0
    rssi_correct = 0
    for test_index in eval_indices:
        sample = samples[test_index]
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        rssi_ranked = aco.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        rssi_pred = candidates[0]
        rssi_costs = {label: score for label, score in rssi_ranked if label in candidates}
        chirp_weight = aco.chirp_separability(candidates, chirp_priors)
        obs_costs, _rows = aco.build_observation_costs(
            sample,
            candidates,
            rssi_costs,
            prototypes,
            q4_offsets,
            chirp_weight,
            args,
        )
        pathloss_costs = phys.build_pathloss_costs(sample, candidates, len(obs_costs), location_meta, pathloss_model)
        sw, dist, vis = transition_components(candidates, location_meta, d0)
        records.append(
            {
                "sample_index": test_index,
                "true_label": sample.label,
                "candidates": candidates,
                "obs_costs": obs_costs,
                "pathloss_costs": pathloss_costs,
                "sw": sw,
                "dist": dist,
                "vis": vis,
                "stable_scores": [chirp_priors.get(label, {}).get("stable", 0.0) for label in candidates],
            }
        )
        rssi_correct += int(rssi_pred == sample.label)
        topk_contains += int(sample.label in candidates)
    metadata = {
        "d0_m": d0,
        "pathloss_model_coef": ";".join(f"{value:.8g}" for value in pathloss_model["coef"]),
        "pathloss_model_sigma": pathloss_model["sigma"],
        "rssi_top1_accuracy": rssi_correct / len(records) if records else 0.0,
        "rssi_topk_recall": topk_contains / len(records) if records else 0.0,
    }
    return records, metadata


def matrix_for_combo(record: dict, combo: dict) -> list[list[float]]:
    k = len(record["candidates"])
    return [
        [
            combo["lambda_sw"] * record["sw"][i][j]
            + combo["lambda_d"] * record["dist"][i][j]
            + combo["lambda_v"] * record["vis"][i][j]
            for j in range(k)
        ]
        for i in range(k)
    ]


def run_aco_record(record: dict, combo: dict, args, rng: random.Random) -> dict:
    candidates = record["candidates"]
    obs_costs = record["obs_costs"]
    pathloss_costs = record["pathloss_costs"]
    transition_costs = matrix_for_combo(record, combo)
    k = len(candidates)
    pheromone = []
    for i in range(k):
        row = []
        for j in range(k):
            base = args.tau_stay if i == j else args.tau_switch
            if i == j:
                base *= 1.0 + args.chirp_self_loop_boost * record["stable_scores"][j]
            row.append(base)
        pheromone.append(row)

    best_path = []
    best_cost = float("inf")
    elite_vote = [0.0] * k
    for _ in range(args.iterations):
        paths = []
        for _ant in range(args.ants):
            first_weights = [
                math.exp(-args.heuristic_power * (obs_costs[0][j] + combo["lambda_pl"] * pathloss_costs[0][j]))
                for j in range(k)
            ]
            path = [aco.weighted_choice(first_weights, rng)]
            for s in range(1, len(obs_costs)):
                prev = path[-1]
                weights = []
                for j in range(k):
                    local_cost = obs_costs[s][j] + transition_costs[prev][j] + combo["lambda_pl"] * pathloss_costs[s][j]
                    weights.append((pheromone[prev][j] ** args.pheromone_power) * math.exp(-args.heuristic_power * local_cost))
                path.append(aco.weighted_choice(weights, rng))
            cost = phys.physical_path_cost(
                path,
                obs_costs,
                transition_costs,
                pathloss_costs,
                combo["lambda_div"],
                combo["lambda_pl"],
            )
            paths.append((cost, path))
            if cost < best_cost:
                best_cost = cost
                best_path = list(path)
        paths.sort(key=lambda item: item[0])
        elite = paths[: max(1, min(args.elite_ants, len(paths)))]
        for i in range(k):
            for j in range(k):
                pheromone[i][j] *= 1.0 - args.evaporation
                pheromone[i][j] = max(args.min_pheromone, pheromone[i][j])
        for cost, path in elite:
            deposit = 1.0 / (cost + EPS)
            for s in range(1, len(path)):
                pheromone[path[s - 1]][path[s]] += deposit
            for choice in path:
                elite_vote[choice] += deposit

    path_counts = Counter(best_path)
    path_mode_idx = min(range(k), key=lambda idx: (-path_counts.get(idx, 0), candidates[idx]))
    pheromone_idx = max(range(k), key=lambda idx: (pheromone[idx][idx], -idx))
    vote_idx = max(range(k), key=lambda idx: (elite_vote[idx], -idx))
    return {
        "path_mode_label": candidates[path_mode_idx],
        "pheromone_label": candidates[pheromone_idx],
        "vote_label": candidates[vote_idx],
        "best_cost": best_cost,
        "best_path_labels": ";".join(candidates[idx] for idx in best_path),
    }


def evaluate_records(records: Sequence[dict], combo: dict, args) -> dict:
    rng = random.Random(args.seed)
    correct = Counter()
    predictions = []
    for record in records:
        result = run_aco_record(record, combo, args, rng)
        true_label = record["true_label"]
        correct["path_mode"] += int(result["path_mode_label"] == true_label)
        correct["pheromone"] += int(result["pheromone_label"] == true_label)
        correct["vote"] += int(result["vote_label"] == true_label)
        predictions.append({**record, **result})
    n = len(records)
    return {
        "packet_count": n,
        "aco_path_mode_correct": correct["path_mode"],
        "aco_path_mode_accuracy": correct["path_mode"] / n if n else 0.0,
        "aco_pheromone_correct": correct["pheromone"],
        "aco_pheromone_accuracy": correct["pheromone"] / n if n else 0.0,
        "aco_vote_correct": correct["vote"],
        "aco_vote_accuracy": correct["vote"] / n if n else 0.0,
    }


def evaluate_records_with_predictions(records: Sequence[dict], combo: dict, args) -> tuple[dict, list[dict]]:
    rng = random.Random(args.seed)
    correct = Counter()
    predictions = []
    for record in records:
        result = run_aco_record(record, combo, args, rng)
        true_label = record["true_label"]
        correct["path_mode"] += int(result["path_mode_label"] == true_label)
        correct["pheromone"] += int(result["pheromone_label"] == true_label)
        correct["vote"] += int(result["vote_label"] == true_label)
        predictions.append(
            {
                "sample_index": record["sample_index"],
                "true_label": true_label,
                "candidates": ";".join(record["candidates"]),
                "aco_path_mode_label": result["path_mode_label"],
                "aco_path_mode_correct": int(result["path_mode_label"] == true_label),
                "aco_pheromone_label": result["pheromone_label"],
                "aco_pheromone_correct": int(result["pheromone_label"] == true_label),
                "aco_vote_label": result["vote_label"],
                "aco_vote_correct": int(result["vote_label"] == true_label),
                "best_cost": result["best_cost"],
                "best_path_labels": result["best_path_labels"],
            }
        )
    n = len(records)
    return (
        {
            "packet_count": n,
            "aco_path_mode_correct": correct["path_mode"],
            "aco_path_mode_accuracy": correct["path_mode"] / n if n else 0.0,
            "aco_pheromone_correct": correct["pheromone"],
            "aco_pheromone_accuracy": correct["pheromone"] / n if n else 0.0,
            "aco_vote_correct": correct["vote"],
            "aco_vote_accuracy": correct["vote"] / n if n else 0.0,
        },
        predictions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-sw-grid", default="0.3,0.5,0.7,0.9")
    parser.add_argument("--lambda-d-grid", default="0,0.03,0.1")
    parser.add_argument("--lambda-v-grid", default="0,0.1,0.2")
    parser.add_argument("--lambda-div-grid", default="0,0.1,0.2")
    parser.add_argument("--lambda-pl-grid", default="0,0.05,0.2")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    base_args = SimpleNamespace(
        top_k=args.top_k,
        rssi_class_k=args.rssi_class_k,
        ants=args.ants,
        aco_iterations=args.iterations,
        elite_ants=args.elite_ants,
        seed=args.seed,
    )
    aco_args = re.aco_args_from_cli(base_args)
    aco_args.iterations = args.iterations

    paths = {
        "rssi": EXPERIMENT_DIR / "data" / "noisy_rssi_plus_packet_level_54points.csv",
        "raw": EXPERIMENT_DIR / "data" / "noisy_lora_frequency_s17_54points.csv",
        "spectrum": EXPERIMENT_DIR / "data" / "noisy_subbin_spectrum_long.csv",
    }
    base_samples, q4_offsets, _base_thresholds = re.build_aco_samples(paths, base_args)
    samples, thresholds = segmented.rebuild_segmented_samples(
        base_samples,
        q4_offsets,
        segment_count=4,
        auto_peak_quantile=0.10,
        auto_q4_dev_quantile=0.75,
        q4_peak_offset_max=0.50,
        q4_peak_to_side_threshold=6.0,
    )
    split = segmented.read_split(EXPERIMENT_DIR / "data" / "split_assignments.csv")
    indices = re.indices_for_split(samples, split)
    chirp_priors = aco.read_chirp_priors(re.DEFAULT_INPUTS["chirp"])
    location_meta = phys.read_location_physics(PKG_ROOT / "docs" / "location_distance_54points.csv")

    train_idx = indices["train"]
    records_by_split = {}
    metadata_by_split = {}
    for split_name, eval_idx in {
        "train_loocv": indices["train"],
        "val": indices["val"],
        "test": indices["test"],
    }.items():
        records, metadata = precompute_records(
            samples,
            q4_offsets,
            chirp_priors,
            location_meta,
            train_idx,
            eval_idx,
            aco_args,
        )
        records_by_split[split_name] = records
        metadata_by_split[split_name] = metadata

    combos = [
        {
            "lambda_sw": lambda_sw,
            "lambda_d": lambda_d,
            "lambda_v": lambda_v,
            "lambda_div": lambda_div,
            "lambda_pl": lambda_pl,
        }
        for lambda_sw, lambda_d, lambda_v, lambda_div, lambda_pl in itertools.product(
            parse_grid(args.lambda_sw_grid),
            parse_grid(args.lambda_d_grid),
            parse_grid(args.lambda_v_grid),
            parse_grid(args.lambda_div_grid),
            parse_grid(args.lambda_pl_grid),
        )
    ]

    out_dir = EXPERIMENT_DIR / "results" / "aco_segments4_physical_pathcost_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    for idx, combo in enumerate(combos, start=1):
        metrics = evaluate_records(records_by_split["val"], combo, aco_args)
        row = {
            "combo_index": idx,
            **combo,
            **metadata_by_split["val"],
            **metrics,
        }
        sweep_rows.append(row)
        if idx % 25 == 0 or idx == len(combos):
            best_so_far = max(
                sweep_rows,
                key=lambda item: (
                    item["aco_vote_accuracy"],
                    item["aco_path_mode_accuracy"],
                    item["aco_pheromone_accuracy"],
                ),
            )
            print(
                f"{idx}/{len(combos)} best_val_vote={best_so_far['aco_vote_accuracy']:.6f} "
                f"lambda_sw={best_so_far['lambda_sw']} lambda_d={best_so_far['lambda_d']} "
                f"lambda_v={best_so_far['lambda_v']} lambda_div={best_so_far['lambda_div']} "
                f"lambda_pl={best_so_far['lambda_pl']}",
                flush=True,
            )

    fields = list(sweep_rows[0].keys())
    write_csv(out_dir / "validation_sweep_results.csv", sweep_rows, fields)
    best = max(
        sweep_rows,
        key=lambda item: (
            item["aco_vote_accuracy"],
            item["aco_path_mode_accuracy"],
            item["aco_pheromone_accuracy"],
            -item["lambda_d"],
            -item["lambda_pl"],
        ),
    )
    best_combo = {key: best[key] for key in ["lambda_sw", "lambda_d", "lambda_v", "lambda_div", "lambda_pl"]}

    final_rows = []
    for split_name in ["train_loocv", "val", "test"]:
        metrics, predictions = evaluate_records_with_predictions(records_by_split[split_name], best_combo, aco_args)
        row = {
            "method": "aco_segments4_physical_pathcost_sweep_best",
            "split": split_name,
            "segment_count": 4,
            "source_symbol_count": 16,
            "symbols_per_segment": 4,
            "ants": args.ants,
            "iterations": args.iterations,
            **best_combo,
            **metadata_by_split[split_name],
            **thresholds,
            **metrics,
        }
        final_rows.append(row)
        with (out_dir / f"best_{split_name}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)
        write_csv(
            out_dir / f"best_{split_name}_predictions.csv",
            predictions,
            list(predictions[0].keys()) if predictions else ["sample_index"],
        )

    final_fields = sorted({key for row in final_rows for key in row})
    preferred = [
        "method",
        "split",
        "segment_count",
        "ants",
        "iterations",
        "lambda_sw",
        "lambda_d",
        "lambda_v",
        "lambda_div",
        "lambda_pl",
        "packet_count",
        "rssi_top1_accuracy",
        "rssi_topk_recall",
        "aco_path_mode_accuracy",
        "aco_pheromone_accuracy",
        "aco_vote_accuracy",
        "d0_m",
        "pathloss_model_sigma",
    ]
    final_fields = preferred + [key for key in final_fields if key not in preferred]
    write_csv(out_dir / "best_summary.csv", final_rows, final_fields)
    with (out_dir / "sweep_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selection_metric": "validation aco_vote_accuracy; ties by path_mode then pheromone",
                "grid_size": len(combos),
                "best_combo": best_combo,
                "best_validation_row": best,
                "thresholds": thresholds,
                "args": vars(args),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"best_combo": best_combo, "final": final_rows}, indent=2, ensure_ascii=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
