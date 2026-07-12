#!/usr/bin/env python3
"""ACO 2.0 ablation on the fixed 1:10 Gaussian-noise split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
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
import run_aco_v2_on_split as split_runner  # noqa: E402


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "aco_v2_ablation"
EPS = 1e-12


@dataclass(frozen=True)
class AblationConfig:
    version: str
    change: str
    purpose: str
    use_bin: bool
    chirp_mean_shrinkage: bool
    physical_var_shrinkage: bool
    huber: bool
    garbage_state: bool
    dynamic_switch: bool
    reliability_pheromone: bool


ABLATIONS = [
    AblationConfig(
        version="V1.0",
        change="first-version 4-segment ACO baseline",
        purpose="baseline",
        use_bin=False,
        chirp_mean_shrinkage=False,
        physical_var_shrinkage=False,
        huber=False,
        garbage_state=False,
        dynamic_switch=False,
        reliability_pheromone=False,
    ),
    AblationConfig(
        version="V2.1",
        change="add empirical raw-bin Gaussian likelihood",
        purpose="test whether Gaussian likelihood helps beyond raw distance",
        use_bin=True,
        chirp_mean_shrinkage=False,
        physical_var_shrinkage=False,
        huber=False,
        garbage_state=False,
        dynamic_switch=False,
        reliability_pheromone=False,
    ),
    AblationConfig(
        version="V2.2",
        change="add chirp-generated mean template shrinkage",
        purpose="test whether chirp shape prior helps",
        use_bin=True,
        chirp_mean_shrinkage=True,
        physical_var_shrinkage=False,
        huber=False,
        garbage_state=False,
        dynamic_switch=False,
        reliability_pheromone=False,
    ),
    AblationConfig(
        version="V2.3",
        change="add physical covariance shrinkage",
        purpose="test whether multipath reliability helps",
        use_bin=True,
        chirp_mean_shrinkage=True,
        physical_var_shrinkage=True,
        huber=False,
        garbage_state=False,
        dynamic_switch=False,
        reliability_pheromone=False,
    ),
    AblationConfig(
        version="V2.4",
        change="add Huber bin likelihood",
        purpose="test whether bad segment suppression helps",
        use_bin=True,
        chirp_mean_shrinkage=True,
        physical_var_shrinkage=True,
        huber=True,
        garbage_state=False,
        dynamic_switch=False,
        reliability_pheromone=False,
    ),
    AblationConfig(
        version="V2.5",
        change="add garbage state",
        purpose="test whether abnormal segment rejection helps",
        use_bin=True,
        chirp_mean_shrinkage=True,
        physical_var_shrinkage=True,
        huber=True,
        garbage_state=True,
        dynamic_switch=False,
        reliability_pheromone=False,
    ),
    AblationConfig(
        version="V2.6",
        change="add evidence-driven dynamic switch penalty",
        purpose="test whether evidence-driven switching helps",
        use_bin=True,
        chirp_mean_shrinkage=True,
        physical_var_shrinkage=True,
        huber=True,
        garbage_state=True,
        dynamic_switch=True,
        reliability_pheromone=False,
    ),
    AblationConfig(
        version="V2.7",
        change="add physical self-loop pheromone initialization/update",
        purpose="test whether reliable templates speed convergence",
        use_bin=True,
        chirp_mean_shrinkage=True,
        physical_var_shrinkage=True,
        huber=True,
        garbage_state=True,
        dynamic_switch=True,
        reliability_pheromone=True,
    ),
]


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def template_variance(values: Sequence[float], center: float) -> float:
    if not values:
        return 1.0
    value = sum((v - center) ** 2 for v in values) / len(values)
    if value > EPS:
        return value
    fallback = aco2.safe_iqr(values) ** 2
    return fallback if fallback > EPS else 1.0


def empirical_templates(
    packets: Sequence[aco2.SegmentPacket],
    labels: Sequence[str],
    train_indices: Sequence[int],
) -> dict[str, dict]:
    by_label_packets: dict[str, list[list[float]]] = {}
    by_label_segments: dict[str, list[list[float]]] = {}
    for idx in train_indices:
        label = labels[idx]
        by_label_packets.setdefault(label, []).append(aco2.packet_median_shape(packets[idx]))
        by_label_segments.setdefault(label, []).extend(packets[idx].segment_shapes)
    out = {}
    for label, rows in by_label_packets.items():
        dim = len(aco2.RAW_OFFSETS)
        mu = [aco2.median([row[j] for row in rows]) for j in range(dim)]
        segments = by_label_segments[label]
        var = [
            template_variance([row[j] for row in segments], mu[j])
            for j in range(dim)
        ]
        out[label] = {"mu": mu, "var": var, "n": len(rows)}
    return out


def build_templates_for_ablation(
    packets: Sequence[aco2.SegmentPacket],
    labels: Sequence[str],
    train_indices: Sequence[int],
    chirp_shapes: dict[str, tuple[list[float], str]],
    chirp_struct: dict[str, tuple[dict, str]],
    args: argparse.Namespace,
    config: AblationConfig,
) -> dict[str, aco2.BinTemplate]:
    emp = empirical_templates(packets, labels, train_indices)
    if not emp:
        return {}
    global_mu = [aco2.median([item["mu"][j] for item in emp.values()]) for j in range(len(aco2.RAW_OFFSETS))]
    global_var = [aco2.median([item["var"][j] for item in emp.values()]) for j in range(len(aco2.RAW_OFFSETS))]
    out = {}
    for label, item in emp.items():
        mu_phy, phy_source = chirp_shapes.get(label, (global_mu, "empirical_global_fallback"))
        struct, struct_source = chirp_struct.get(label, ({"k_ratio": 1.0, "tau_rms": 0.0}, "structure_default"))
        n = item["n"]
        alpha = n / (n + args.shrinkage_lambda)
        if config.chirp_mean_shrinkage:
            mu = [alpha * item["mu"][j] + (1.0 - alpha) * mu_phy[j] for j in range(len(aco2.RAW_OFFSETS))]
        else:
            mu = list(item["mu"])

        if config.physical_var_shrinkage:
            sigma_phy = (
                args.phy_var_c0
                + args.phy_var_c1 / (aco2.parse_float(struct.get("k_ratio"), 1.0) + 1.0)
                + args.phy_var_c2 * aco2.parse_float(struct.get("tau_rms"), 0.0)
            )
            sigma_phy = max(sigma_phy, args.min_variance)
            var_phy = [sigma_phy for _ in aco2.RAW_OFFSETS]
            var = [
                alpha * item["var"][j] + (1.0 - alpha) * var_phy[j] + args.sigma0_sq
                for j in range(len(aco2.RAW_OFFSETS))
            ]
        else:
            var_phy = list(global_var)
            var = [item["var"][j] + args.sigma0_sq for j in range(len(aco2.RAW_OFFSETS))]
        var = [max(v, args.min_variance) for v in var]
        reliability = 1.0 / (sum(var) + EPS)
        out[label] = aco2.BinTemplate(
            label=label,
            mu=mu,
            var=var,
            mu_emp=list(item["mu"]),
            var_emp=list(item["var"]),
            mu_phy=mu_phy,
            var_phy=var_phy,
            n_packets=n,
            alpha_shrink=alpha,
            reliability=reliability,
            chirp_source=f"{phy_source};{struct_source}",
        )
    return out


def bin_cost(shape: Sequence[float], template: aco2.BinTemplate, args: argparse.Namespace, config: AblationConfig) -> float:
    mahal = sum(
        ((shape[j] - template.mu[j]) ** 2) / (template.var[j] + EPS)
        for j in range(len(aco2.RAW_OFFSETS))
    )
    if config.huber:
        mahal = aco2.huber_mahalanobis(mahal, args.huber_delta)
    logdet = sum(math.log(template.var[j] + EPS) for j in range(len(aco2.RAW_OFFSETS)))
    return mahal + args.logdet_weight * logdet


def build_observation_costs(
    packet: aco2.SegmentPacket,
    candidates: Sequence[str],
    rssi_costs: dict[str, float],
    templates: dict[str, aco2.BinTemplate],
    segment_prototypes: dict,
    q4_offsets: Sequence[float],
    args: argparse.Namespace,
    config: AblationConfig,
) -> tuple[list[list[float]], list[dict]]:
    shift_grid = [float(part.strip()) for part in args.q4_shift_grid.split(",") if part.strip()]
    rssi_norm = aco2.normalize_scores({label: rssi_costs[label] for label in candidates})
    obs_costs = []
    rows = []
    for segment_idx, shape in enumerate(packet.segment_shapes):
        if config.use_bin:
            raw_bin_costs = {label: bin_cost(shape, templates[label], args, config) for label in candidates}
            bin_norm = aco2.normalize_scores(raw_bin_costs) if args.normalize_bin_cost else raw_bin_costs
        else:
            raw_bin_costs = {label: 0.0 for label in candidates}
            bin_norm = {label: 0.0 for label in candidates}
        e_cost = {
            label: aco2.base.robust_scalar_cost(packet.segment_zw[segment_idx][0], segment_prototypes["energy"][label])
            for label in candidates
        }
        w_cost = {
            label: aco2.base.robust_vector_cost(packet.segment_zw[segment_idx], segment_prototypes["zw"][label])
            for label in candidates
        }
        if args.q4_weight > 0 and packet.segment_q4_reliable[segment_idx]:
            q_cost = {
                label: aco2.base.q4_shape_cost(
                    packet.segment_q4_curves[segment_idx],
                    segment_prototypes["q4"][label],
                    q4_offsets,
                    shift_grid,
                )
                for label in candidates
            }
        else:
            q_cost = {label: 0.0 for label in candidates}
        e_norm = aco2.normalize_scores(e_cost)
        w_norm = aco2.normalize_scores(w_cost)
        q_norm = aco2.normalize_scores(q_cost)
        costs = []
        for label in candidates:
            q_weight = args.q4_weight if packet.segment_q4_reliable[segment_idx] else 0.0
            c_obs = (
                args.rssi_weight * rssi_norm.get(label, 0.0)
                + (args.bin_weight if config.use_bin else 0.0) * bin_norm[label]
                + args.energy_weight * e_norm[label]
                + args.raw_weight * w_norm[label]
                + q_weight * q_norm[label]
            )
            costs.append(c_obs)
            rows.append(
                {
                    "segment_index": segment_idx,
                    "candidate_label": label,
                    "C_obs": c_obs,
                    "C_R": rssi_norm.get(label, 0.0),
                    "C_bin": bin_norm[label],
                    "C_bin_raw": raw_bin_costs[label],
                    "C_E": e_norm[label],
                    "C_W": w_norm[label],
                    "C_Q": q_norm[label],
                    "q4_reliable": int(packet.segment_q4_reliable[segment_idx]),
                }
            )
        if config.garbage_state:
            costs.append(args.garbage_cost)
            rows.append(
                {
                    "segment_index": segment_idx,
                    "candidate_label": aco2.GARBAGE_LABEL,
                    "C_obs": args.garbage_cost,
                    "C_R": "",
                    "C_bin": "",
                    "C_bin_raw": "",
                    "C_E": "",
                    "C_W": "",
                    "C_Q": "",
                    "q4_reliable": int(packet.segment_q4_reliable[segment_idx]),
                }
            )
        obs_costs.append(costs)
    return obs_costs, rows


def template_distance(left: aco2.BinTemplate, right: aco2.BinTemplate) -> float:
    return sum(
        ((left.mu[j] - right.mu[j]) ** 2)
        / (0.5 * (left.var[j] + right.var[j]) + EPS)
        for j in range(len(aco2.RAW_OFFSETS))
    )


def switch_penalty(
    prev_idx: int,
    next_idx: int,
    segment_idx: int,
    obs_costs: Sequence[Sequence[float]],
    template_distances: Sequence[Sequence[float]],
    args: argparse.Namespace,
    config: AblationConfig,
) -> float:
    if prev_idx == next_idx:
        return 0.0
    garbage_idx = len(obs_costs[segment_idx]) - 1 if config.garbage_state else -1
    if config.garbage_state and (prev_idx == garbage_idx or next_idx == garbage_idx):
        return 0.0
    if not config.dynamic_switch:
        return args.lambda0_switch
    advantage = max(0.0, obs_costs[segment_idx][prev_idx] - obs_costs[segment_idx][next_idx])
    return args.lambda0_switch / (1.0 + args.switch_eta * template_distances[prev_idx][next_idx] * advantage)


def path_cost(
    path: Sequence[int],
    obs_costs: Sequence[Sequence[float]],
    template_distances: Sequence[Sequence[float]],
    args: argparse.Namespace,
    config: AblationConfig,
) -> float:
    total = sum(obs_costs[s][choice] for s, choice in enumerate(path))
    total += sum(switch_penalty(path[s - 1], path[s], s, obs_costs, template_distances, args, config) for s in range(1, len(path)))
    if config.garbage_state:
        garbage_idx = len(obs_costs[0]) - 1
        unique = len({idx for idx in path if idx != garbage_idx})
        n_g = sum(1 for idx in path if idx == garbage_idx)
        total += args.lambda_div * unique + args.lambda_g * n_g
        if n_g > args.max_garbage:
            total += args.garbage_overuse_penalty * (n_g - args.max_garbage)
    else:
        unique = len(set(path))
        total += args.lambda_div * unique
    return total


def run_aco(
    obs_costs: Sequence[Sequence[float]],
    candidates: Sequence[str],
    templates: dict[str, aco2.BinTemplate],
    args: argparse.Namespace,
    config: AblationConfig,
    rng: random.Random,
) -> dict:
    k = len(candidates)
    node_count = k + (1 if config.garbage_state else 0)
    garbage_idx = node_count - 1 if config.garbage_state else -1
    reliabilities = [templates[label].reliability for label in candidates]
    med_rel = aco2.median(reliabilities) if reliabilities else 1.0
    rel_norm = [rel / (med_rel + EPS) for rel in reliabilities]
    template_dist = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i != j:
                template_dist[i][j] = template_distance(templates[left], templates[right])

    pheromone = []
    for i in range(node_count):
        row = []
        for j in range(node_count):
            if i == j and i < k:
                boost = (1.0 + args.lambda_c * rel_norm[i]) if config.reliability_pheromone else 1.0
                row.append(args.tau_stay * boost)
            elif i == j:
                row.append(args.tau_stay)
            else:
                row.append(args.tau_switch)
        pheromone.append(row)

    best_path: list[int] = []
    best_cost = float("inf")
    elite_vote = [0.0 for _ in range(node_count)]

    for _iter in range(args.iterations):
        paths = []
        for _ant in range(args.ants):
            first_weights = [math.exp(-args.heuristic_power * cost) for cost in obs_costs[0]]
            path = [aco2.weighted_choice(first_weights, rng)]
            for s in range(1, len(obs_costs)):
                prev = path[-1]
                weights = []
                for j in range(node_count):
                    penalty = switch_penalty(prev, j, s, obs_costs, template_dist, args, config)
                    eta = math.exp(-obs_costs[s][j])
                    weights.append((pheromone[prev][j] ** args.pheromone_power) * (eta ** args.heuristic_power) * math.exp(-penalty))
                path.append(aco2.weighted_choice(weights, rng))
            cost = path_cost(path, obs_costs, template_dist, args, config)
            paths.append((cost, path))
            if cost < best_cost:
                best_cost = cost
                best_path = list(path)

        paths.sort(key=lambda item: item[0])
        elite = paths[: max(1, min(args.elite_ants, len(paths)))]
        temp = args.aco_temperature
        if temp is None or temp <= EPS:
            temp = aco2.median([cost for cost, _path in elite])
            temp = temp if temp > EPS else 1.0
        weights = [math.exp(-cost / (temp + EPS)) for cost, _path in elite]
        total = sum(weights) or 1.0
        weights = [value / total for value in weights]
        for i in range(node_count):
            for j in range(node_count):
                pheromone[i][j] *= 1.0 - args.evaporation
                pheromone[i][j] = max(args.min_pheromone, pheromone[i][j])
        for (_cost, path), weight in zip(elite, weights):
            for choice in path:
                if not config.garbage_state or choice != garbage_idx:
                    elite_vote[choice] += weight
            for prev, cur in zip(path, path[1:]):
                if config.reliability_pheromone and prev == cur and cur < k:
                    pheromone[prev][cur] += weight * (1.0 + args.lambda_c * rel_norm[cur])
                else:
                    pheromone[prev][cur] += weight

    non_g_path = [idx for idx in best_path if not config.garbage_state or idx != garbage_idx]
    if non_g_path:
        path_counts = Counter(non_g_path)
        path_mode_idx = min(range(k), key=lambda idx: (-path_counts.get(idx, 0), candidates[idx]))
    else:
        path_mode_idx = min(range(k), key=lambda idx: obs_costs[0][idx])
    pheromone_idx = max(range(k), key=lambda idx: (pheromone[idx][idx], -idx))
    vote_idx = max(range(k), key=lambda idx: (elite_vote[idx], -idx))
    return {
        "best_path": best_path,
        "best_path_labels": [aco2.GARBAGE_LABEL if config.garbage_state and idx == garbage_idx else candidates[idx] for idx in best_path],
        "best_cost": best_cost,
        "path_mode_label": candidates[path_mode_idx],
        "pheromone_label": candidates[pheromone_idx],
        "vote_label": candidates[vote_idx],
        "garbage_count": sum(1 for idx in best_path if config.garbage_state and idx == garbage_idx),
    }


def evaluate_split(
    samples: Sequence[aco2.SegmentPacket],
    q4_offsets: Sequence[float],
    chirp_shapes,
    chirp_struct,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    split_name: str,
    args: argparse.Namespace,
    config: AblationConfig,
) -> tuple[dict, list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    rng = random.Random(args.seed)
    correct = Counter()
    topk_contains = 0
    predictions = []

    templates = build_templates_for_ablation(samples, labels, train_indices, chirp_shapes, chirp_struct, args, config)
    prototypes = aco2.build_segment_prototypes(samples, labels, train_indices)
    for test_index in eval_indices:
        sample = samples[test_index]
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        rssi_ranked = aco2.base.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        rssi_pred = candidates[0]
        rssi_costs = {label: score for label, score in rssi_ranked if label in candidates}
        obs_costs, _rows = build_observation_costs(
            sample,
            candidates,
            rssi_costs,
            templates,
            prototypes,
            q4_offsets,
            args,
            config,
        )
        result = run_aco(obs_costs, candidates, templates, args, config, rng)
        topk_contains += int(sample.label in candidates)
        correct["rssi"] += int(rssi_pred == sample.label)
        for key in ["path_mode", "pheromone", "vote"]:
            correct[key] += int(result[f"{key}_label"] == sample.label)
        predictions.append(
            {
                "version": config.version,
                "split": split_name,
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
                "best_path_labels": ";".join(result["best_path_labels"]),
                "best_path_garbage_count": result["garbage_count"],
            }
        )

    n = len(predictions)
    metrics = {
        "version": config.version,
        "split": split_name,
        "change": config.change,
        "purpose": config.purpose,
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "segment_count": args.segment_count,
        "rssi_class_k": args.rssi_class_k,
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
        "garbage_state_usage_mean": sum(int(row["best_path_garbage_count"]) for row in predictions) / n if n else 0.0,
        "use_bin": int(config.use_bin),
        "chirp_mean_shrinkage": int(config.chirp_mean_shrinkage),
        "physical_var_shrinkage": int(config.physical_var_shrinkage),
        "huber": int(config.huber),
        "garbage_state": int(config.garbage_state),
        "dynamic_switch": int(config.dynamic_switch),
        "reliability_pheromone": int(config.reliability_pheromone),
    }
    return metrics, predictions


def metric_fields(rows: Sequence[dict]) -> list[str]:
    preferred = [
        "version",
        "split",
        "change",
        "purpose",
        "packet_count",
        "location_count",
        "top_k",
        "segment_count",
        "rssi_class_k",
        "rssi_top1_accuracy",
        "rssi_topk_recall",
        "aco_path_mode_accuracy",
        "aco_pheromone_accuracy",
        "aco_vote_accuracy",
        "garbage_state_usage_mean",
        "use_bin",
        "chirp_mean_shrinkage",
        "physical_var_shrinkage",
        "huber",
        "garbage_state",
        "dynamic_switch",
        "reliability_pheromone",
    ]
    return preferred + sorted({key for row in rows for key in row} - set(preferred))


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aco_args = split_runner.build_args(args)
    rssi_packets = aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, thresholds = aco2.base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = split_runner.load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = aco2.prepare_chirp_fields(aco_args, labels)

    eval_plan = [
        ("train_loocv", split_indices["train"], split_indices["train"]),
        ("val", split_indices["train"], split_indices["val"]),
        ("test", split_indices["train"], split_indices["test"]),
    ]
    summary_rows = []
    for config in ABLATIONS:
        version_dir = args.output_dir / config.version.lower().replace(".", "_")
        version_dir.mkdir(parents=True, exist_ok=True)
        version_rows = []
        for split_name, train_indices, eval_indices in eval_plan:
            metrics, predictions = evaluate_split(
                samples,
                q4_offsets,
                chirp_shapes,
                chirp_struct,
                train_indices,
                eval_indices,
                split_name,
                aco_args,
                config,
            )
            metrics.update(thresholds)
            version_rows.append(metrics)
            summary_rows.append(metrics)
            with (version_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            if predictions:
                write_csv(version_dir / f"{split_name}_predictions.csv", predictions, list(predictions[0].keys()))
        write_csv(version_dir / f"{config.version.lower().replace('.', '_')}_summary.csv", version_rows, metric_fields(version_rows))

    write_csv(args.output_dir / "aco_v2_ablation_summary.csv", summary_rows, metric_fields(summary_rows))
    metadata = {
        "method": "ACO 2.0 ablation on gaussian_noise_1to10_split",
        "data_policy": "Consumes existing 1:10 Gaussian-noise augmented CSVs and split_assignments.csv; does not regenerate noise or split.",
        "sample_counts": {
            "aligned": len(samples),
            "train": len(split_indices["train"]),
            "val": len(split_indices["val"]),
            "test": len(split_indices["test"]),
            "locations": len(set(labels)),
        },
        "base_args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(aco_args).items()
        },
        "symbol_thresholds": thresholds,
        "chirp_template_field": chirp_metadata,
        "ablation_versions": [config.__dict__ for config in ABLATIONS],
        "summary": summary_rows,
    }
    with (args.output_dir / "aco_v2_ablation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    metadata = run(parse_args())
    print(json.dumps(metadata["sample_counts"], indent=2, ensure_ascii=False))
    for row in metadata["summary"]:
        if row["split"] == "test":
            print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
