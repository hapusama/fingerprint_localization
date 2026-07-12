#!/usr/bin/env python3
"""Segment-4 ACO with the physical path-cost formula from the screenshot."""

from __future__ import annotations

import argparse
import csv
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

import run_aco_segmented as segmented  # noqa: E402
import run_experiment as re  # noqa: E402
from model.v3 import aco_packet_path as aco  # noqa: E402


EPS = 1e-12
RSSI_MEDIAN_INDEX = 2


def read_location_physics(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["position_key"]
            state = row.get("c_i（NLOS-2，LOS-1，OLOS-0）", row.get("c_i", "0"))
            rows[label] = {
                "distance_m": float(row["distance_m"]),
                "visibility_state": int(float(state)),
                "observed_flag": int(float(row.get("o_i(1=有数据，0-无数据)", 0) or 0)),
            }
    return rows


def nearest_spacing(location_meta: dict[str, dict], labels: Sequence[str]) -> float:
    distances = sorted(
        location_meta[label]["distance_m"]
        for label in labels
        if label in location_meta
    )
    gaps = [b - a for a, b in zip(distances, distances[1:]) if b - a > EPS]
    return aco.median(gaps) if gaps else 1.0


def solve_linear_system(a: list[list[float]], b: list[float]) -> list[float]:
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        if abs(aug[col][col]) <= EPS:
            continue
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [value - factor * aug[col][idx] for idx, value in enumerate(aug[row])]
    return [aug[row][-1] for row in range(n)]


def fit_pathloss_model(samples: Sequence[aco.PacketSample], train_indices: Sequence[int], location_meta: dict[str, dict]) -> dict:
    x_rows = []
    y_values = []
    for idx in train_indices:
        sample = samples[idx]
        meta = location_meta.get(sample.label)
        if not meta:
            continue
        distance = max(meta["distance_m"], 1.0)
        state = float(meta["visibility_state"])
        x_rows.append([1.0, math.log10(distance), state])
        y_values.append(float(sample.rssi_plus[RSSI_MEDIAN_INDEX]))
    if not x_rows:
        return {"coef": [-70.0, -20.0, 0.0], "sigma": 1.0}

    dim = len(x_rows[0])
    xtx = [[0.0 for _ in range(dim)] for _ in range(dim)]
    xty = [0.0 for _ in range(dim)]
    ridge = 1e-6
    for row, y in zip(x_rows, y_values):
        for i in range(dim):
            xty[i] += row[i] * y
            for j in range(dim):
                xtx[i][j] += row[i] * row[j]
    for i in range(dim):
        xtx[i][i] += ridge
    coef = solve_linear_system(xtx, xty)
    residuals = [
        y - sum(coef[j] * row[j] for j in range(dim))
        for row, y in zip(x_rows, y_values)
    ]
    sigma = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return {"coef": coef, "sigma": sigma if sigma > EPS else 1.0}


def predict_pathloss_rssi(label: str, location_meta: dict[str, dict], model: dict) -> float:
    meta = location_meta.get(label)
    if not meta:
        return 0.0
    row = [1.0, math.log10(max(meta["distance_m"], 1.0)), float(meta["visibility_state"])]
    return sum(model["coef"][idx] * row[idx] for idx in range(len(row)))


def normalize_costs(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo <= EPS:
        return {key: 0.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def build_pathloss_costs(
    sample: aco.PacketSample,
    candidates: Sequence[str],
    segment_count: int,
    location_meta: dict[str, dict],
    pathloss_model: dict,
) -> list[list[float]]:
    observed = float(sample.rssi_plus[RSSI_MEDIAN_INDEX])
    raw = {}
    for label in candidates:
        predicted = predict_pathloss_rssi(label, location_meta, pathloss_model)
        raw[label] = ((observed - predicted) / (pathloss_model["sigma"] + EPS)) ** 2
    normalized = normalize_costs(raw)
    return [[normalized.get(label, 0.0) for label in candidates] for _ in range(segment_count)]


def build_transition_costs(
    candidates: Sequence[str],
    location_meta: dict[str, dict],
    d0: float,
    lambda_sw: float,
    lambda_d: float,
    lambda_v: float,
) -> list[list[float]]:
    matrix = []
    for left in candidates:
        left_meta = location_meta.get(left, {})
        row = []
        for right in candidates:
            right_meta = location_meta.get(right, {})
            switched = 1.0 if left != right else 0.0
            left_distance = float(left_meta.get("distance_m", 0.0))
            right_distance = float(right_meta.get("distance_m", left_distance))
            jump = abs(left_distance - right_distance)
            left_state = left_meta.get("visibility_state")
            right_state = right_meta.get("visibility_state", left_state)
            state_changed = 1.0 if left_state != right_state else 0.0
            row.append(
                lambda_sw * switched
                + lambda_d * ((jump / max(d0, EPS)) ** 2)
                + lambda_v * state_changed
            )
        matrix.append(row)
    return matrix


def physical_path_cost(
    path: Sequence[int],
    obs_costs: Sequence[Sequence[float]],
    transition_costs: Sequence[Sequence[float]],
    pathloss_costs: Sequence[Sequence[float]],
    lambda_div: float,
    lambda_pl: float,
) -> float:
    obs = sum(obs_costs[s][choice] for s, choice in enumerate(path))
    transition = sum(transition_costs[path[s - 1]][path[s]] for s in range(1, len(path)))
    pathloss = sum(pathloss_costs[s][choice] for s, choice in enumerate(path))
    return obs + transition + lambda_div * len(set(path)) + lambda_pl * pathloss


def run_physical_aco_for_packet(
    obs_costs: Sequence[Sequence[float]],
    pathloss_costs: Sequence[Sequence[float]],
    candidates: Sequence[str],
    chirp_priors: dict,
    location_meta: dict[str, dict],
    d0: float,
    args: argparse.Namespace,
    rng: random.Random,
) -> dict:
    k = len(candidates)
    transition_costs = build_transition_costs(
        candidates,
        location_meta,
        d0,
        args.lambda_sw,
        args.lambda_d,
        args.lambda_v,
    )
    stable_scores = [chirp_priors.get(label, {}).get("stable", 0.0) for label in candidates]
    pheromone = []
    for i in range(k):
        row = []
        for j in range(k):
            base = args.tau_stay if i == j else args.tau_switch
            if i == j:
                base *= 1.0 + args.chirp_self_loop_boost * stable_scores[j]
            row.append(base)
        pheromone.append(row)

    best_path = []
    best_cost = float("inf")
    elite_vote = [0.0] * k

    for _ in range(args.iterations):
        paths = []
        for _ant in range(args.ants):
            first_weights = [
                math.exp(-args.heuristic_power * (obs_costs[0][j] + args.lambda_pl * pathloss_costs[0][j]))
                for j in range(k)
            ]
            path = [aco.weighted_choice(first_weights, rng)]
            for s in range(1, len(obs_costs)):
                prev = path[-1]
                weights = []
                for j in range(k):
                    local_cost = (
                        obs_costs[s][j]
                        + transition_costs[prev][j]
                        + args.lambda_pl * pathloss_costs[s][j]
                    )
                    weights.append((pheromone[prev][j] ** args.pheromone_power) * math.exp(-args.heuristic_power * local_cost))
                path.append(aco.weighted_choice(weights, rng))
            cost = physical_path_cost(
                path,
                obs_costs,
                transition_costs,
                pathloss_costs,
                args.lambda_div,
                args.lambda_pl,
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
        "best_path": best_path,
        "best_cost": best_cost,
        "path_mode_label": candidates[path_mode_idx],
        "pheromone_label": candidates[pheromone_idx],
        "vote_label": candidates[vote_idx],
        "self_pheromone": {candidates[idx]: pheromone[idx][idx] for idx in range(k)},
        "elite_vote": {candidates[idx]: elite_vote[idx] for idx in range(k)},
    }


def evaluate_physical_pathcost(
    samples: Sequence[aco.PacketSample],
    q4_offsets: Sequence[float],
    chirp_priors: dict,
    thresholds: dict,
    location_meta: dict[str, dict],
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    rng = random.Random(args.seed)
    predictions = []
    correct = Counter()
    topk_contains = 0
    q4_reliable_total = 0
    symbol_total = 0
    prototype_cache = {}
    pathloss_model = fit_pathloss_model(samples, train_indices, location_meta)
    d0 = nearest_spacing(location_meta, sorted({labels[idx] for idx in train_indices}, key=aco.natural_label_key))

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
        prototype_key = tuple(train_indices)
        if prototype_key not in prototype_cache:
            prototype_cache[prototype_key] = aco.build_symbol_prototypes(samples, labels, train_indices)
        prototypes = prototype_cache[prototype_key]
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
        pathloss_costs = build_pathloss_costs(sample, candidates, len(obs_costs), location_meta, pathloss_model)
        result = run_physical_aco_for_packet(
            obs_costs,
            pathloss_costs,
            candidates,
            chirp_priors,
            location_meta,
            d0,
            args,
            rng,
        )
        path_labels = [candidates[idx] for idx in result["best_path"]]
        topk_contains += int(sample.label in candidates)
        q4_reliable_total += sum(1 for symbol in sample.symbols if symbol.q4_reliable)
        symbol_total += len(sample.symbols)
        correct["rssi"] += int(rssi_pred == sample.label)
        correct["path_mode"] += int(result["path_mode_label"] == sample.label)
        correct["pheromone"] += int(result["pheromone_label"] == sample.label)
        correct["vote"] += int(result["vote_label"] == sample.label)
        predictions.append(
            {
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": int(rssi_pred == sample.label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "aco_path_mode_label": result["path_mode_label"],
                "aco_path_mode_correct": int(result["path_mode_label"] == sample.label),
                "aco_pheromone_label": result["pheromone_label"],
                "aco_pheromone_correct": int(result["pheromone_label"] == sample.label),
                "aco_vote_label": result["vote_label"],
                "aco_vote_correct": int(result["vote_label"] == sample.label),
                "best_path_cost": result["best_cost"],
                "best_path_labels": ";".join(path_labels),
                "q4_reliable_segments": sum(1 for symbol in sample.symbols if symbol.q4_reliable),
                "segment_count": len(sample.symbols),
            }
        )

    n = len(predictions)
    metrics = {
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "rssi_top1_correct": correct["rssi"],
        "rssi_top1_accuracy": correct["rssi"] / n if n else 0.0,
        "rssi_topk_contains_true": topk_contains,
        "rssi_topk_recall": topk_contains / n if n else 0.0,
        "aco_path_mode_correct": correct["path_mode"],
        "aco_path_mode_accuracy": correct["path_mode"] / n if n else 0.0,
        "aco_pheromone_correct": correct["pheromone"],
        "aco_pheromone_accuracy": correct["pheromone"] / n if n else 0.0,
        "aco_vote_correct": correct["vote"],
        "aco_vote_accuracy": correct["vote"] / n if n else 0.0,
        "q4_reliable_segment_count": q4_reliable_total,
        "segment_observation_count": symbol_total,
        "q4_reliable_segment_rate": q4_reliable_total / symbol_total if symbol_total else 0.0,
        "d0_m": d0,
        "pathloss_model_coef": ";".join(f"{value:.8g}" for value in pathloss_model["coef"]),
        "pathloss_model_sigma": pathloss_model["sigma"],
        **thresholds,
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--lambda-sw", type=float, default=0.70)
    parser.add_argument("--lambda-d", type=float, default=0.35)
    parser.add_argument("--lambda-v", type=float, default=0.20)
    parser.add_argument("--lambda-div", type=float, default=0.20)
    parser.add_argument("--lambda-pl", type=float, default=0.20)
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
    for name in ["lambda_sw", "lambda_d", "lambda_v", "lambda_div", "lambda_pl"]:
        setattr(aco_args, name, getattr(args, name))
    aco_args.iterations = args.iterations

    paths = {
        "rssi": EXPERIMENT_DIR / "data" / "noisy_rssi_plus_packet_level_54points.csv",
        "raw": EXPERIMENT_DIR / "data" / "noisy_lora_frequency_s17_54points.csv",
        "spectrum": EXPERIMENT_DIR / "data" / "noisy_subbin_spectrum_long.csv",
    }
    base_samples, q4_offsets, _base_thresholds = re.build_aco_samples(paths, base_args)
    segmented_samples, thresholds = segmented.rebuild_segmented_samples(
        base_samples,
        q4_offsets,
        segment_count=4,
        auto_peak_quantile=0.10,
        auto_q4_dev_quantile=0.75,
        q4_peak_offset_max=0.50,
        q4_peak_to_side_threshold=6.0,
    )
    split = segmented.read_split(EXPERIMENT_DIR / "data" / "split_assignments.csv")
    indices = re.indices_for_split(segmented_samples, split)
    chirp_priors = aco.read_chirp_priors(re.DEFAULT_INPUTS["chirp"])
    location_meta = read_location_physics(PKG_ROOT / "docs" / "location_distance_54points.csv")

    out_dir = EXPERIMENT_DIR / "results" / "aco_segments4_physical_pathcost"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for split_name in ["train_loocv", "val", "test"]:
        train_idx = indices["train"]
        eval_idx = indices["train"] if split_name == "train_loocv" else indices[split_name]
        metrics, predictions = evaluate_physical_pathcost(
            segmented_samples,
            q4_offsets,
            chirp_priors,
            thresholds,
            location_meta,
            train_idx,
            eval_idx,
            aco_args,
        )
        metrics = {
            "method": "aco_segments4_physical_pathcost",
            "split": split_name,
            "segment_count": 4,
            "source_symbol_count": 16,
            "symbols_per_segment": 4,
            "ants": args.ants,
            "iterations": args.iterations,
            "lambda_sw": args.lambda_sw,
            "lambda_d": args.lambda_d,
            "lambda_v": args.lambda_v,
            "lambda_div": args.lambda_div,
            "lambda_pl": args.lambda_pl,
            **metrics,
        }
        summary.append(metrics)
        with (out_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        if predictions:
            re.write_csv(out_dir / f"{split_name}_predictions.csv", predictions, list(predictions[0].keys()))

    fields = segmented.metric_fields(summary)
    for extra in [
        "iterations",
        "lambda_sw",
        "lambda_d",
        "lambda_v",
        "lambda_div",
        "lambda_pl",
        "d0_m",
        "pathloss_model_coef",
        "pathloss_model_sigma",
    ]:
        if extra not in fields:
            fields.append(extra)
    re.write_csv(out_dir / "aco_segments4_physical_pathcost_summary.csv", summary, fields)
    with (out_dir / "pathcost_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "formula": "sum C_obs + sum C_trans + lambda_div*N_unique + lambda_pl*sum C_pl",
                "C_trans": "lambda_sw*1[i!=j] + lambda_d*(dist(i,j)^2/d0^2) + lambda_v*1[v_i!=v_j]",
                "C_pl": "per-packet median RSSI mismatch against train-fitted pathloss model, normalized within candidates",
                "config": summary[0] if summary else {},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
