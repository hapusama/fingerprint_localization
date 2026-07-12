#!/usr/bin/env python3
"""ACO 4.0 evidence-reliability aware packet path search.

This module builds on ``aco_packet_path_v2`` and keeps its data preparation,
chirp-shrinkage templates, garbage state, and dynamic switch penalty.  The
packet-level search adds RSSI Top-1 and raw-bin weak priors, segment-stability
gating, cost veto, stability-aware switching/garbage costs, and a Score4 final
selector.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Sequence

import aco_packet_path_v2 as aco2


EPS = aco2.EPS
GARBAGE_LABEL = aco2.GARBAGE_LABEL


def z_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(var)
    if std <= EPS:
        return [0.0 for _ in values]
    return [(value - mean) / std for value in values]


def build_observation_costs_v4(
    packet: aco2.SegmentPacket,
    candidates: Sequence[str],
    rssi_costs: dict[str, float],
    templates: dict[str, aco2.BinTemplate],
    segment_prototypes: dict,
    q4_offsets: Sequence[float],
    args,
) -> tuple[list[list[float]], list[dict], dict]:
    shift_grid = [float(part.strip()) for part in args.q4_shift_grid.split(",") if part.strip()]
    rssi_norm = aco2.normalize_scores({label: rssi_costs[label] for label in candidates})

    per_segment_components = []
    rows = []
    for segment_idx, shape in enumerate(packet.segment_shapes):
        raw_bin_costs = {label: aco2.bin_cost(shape, templates[label], args) for label in candidates}
        bin_norm = aco2.normalize_scores(raw_bin_costs) if args.normalize_bin_cost else raw_bin_costs
        e_cost = {
            label: aco2.base.robust_scalar_cost(packet.segment_zw[segment_idx][0], segment_prototypes["energy"][label])
            for label in candidates
        }
        w_cost = {
            label: aco2.base.robust_vector_cost(packet.segment_zw[segment_idx], segment_prototypes["zw"][label])
            for label in candidates
        }
        if packet.segment_q4_reliable[segment_idx]:
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
        per_segment_components.append(
            {
                "bin_raw": raw_bin_costs,
                "bin": bin_norm,
                "energy": aco2.normalize_scores(e_cost),
                "zw": aco2.normalize_scores(w_cost),
                "q4": aco2.normalize_scores(q_cost),
            }
        )

    segment_base_mins = []
    for components in per_segment_components:
        costs = []
        for label in candidates:
            q_weight = args.q4_weight if packet.segment_q4_reliable[len(segment_base_mins)] else 0.0
            costs.append(
                args.energy_weight * components["energy"][label]
                + args.raw_weight * components["zw"][label]
                + q_weight * components["q4"][label]
                + args.bin_weight * components["bin"][label]
            )
        segment_base_mins.append(min(costs) if costs else 0.0)

    mean_min = sum(segment_base_mins) / len(segment_base_mins) if segment_base_mins else 0.0
    segment_cost_std = math.sqrt(
        sum((value - mean_min) ** 2 for value in segment_base_mins) / len(segment_base_mins)
    ) if segment_base_mins else 0.0
    t_seg = getattr(args, "t_seg_resolved", None) or getattr(args, "t_seg", None) or 1.0
    q_seg = 1.0 / (1.0 + segment_cost_std / max(t_seg, EPS))

    rssi_top1 = candidates[0] if candidates else ""
    packet_bin_raw = {
        label: sum(components["bin_raw"][label] for components in per_segment_components) / max(1, len(per_segment_components))
        for label in candidates
    }
    raw_ranked = sorted(packet_bin_raw.items(), key=lambda item: (item[1], item[0]))
    raw_winner = raw_ranked[0][0] if raw_ranked else ""
    raw_best = raw_ranked[0][1] if raw_ranked else 0.0
    raw_second = raw_ranked[1][1] if len(raw_ranked) > 1 else raw_best
    raw_margin = max(0.0, (raw_second - raw_best) / (raw_second + EPS)) if raw_ranked else 0.0

    obs_costs = []
    candidate_obs_sum = {label: 0.0 for label in candidates}
    candidate_base_sum = {label: 0.0 for label in candidates}
    for segment_idx, components in enumerate(per_segment_components):
        costs = []
        for label in candidates:
            c_r = rssi_norm.get(label, 0.0)
            c_bin = components["bin"][label]
            c_e = components["energy"][label]
            c_w = components["zw"][label]
            c_q = components["q4"][label]
            q_weight = args.q4_weight if packet.segment_q4_reliable[segment_idx] else 0.0
            c_seg = args.energy_weight * c_e + args.raw_weight * c_w + q_weight * c_q + args.bin_weight * c_bin
            c_obs = args.rssi_weight * c_r + q_seg * c_seg
            candidate_obs_sum[label] += c_obs
            candidate_base_sum[label] += c_seg
            costs.append(c_obs)
            rows.append(
                {
                    "segment_index": segment_idx,
                    "candidate_label": label,
                    "C_obs": c_obs,
                    "C_seg_base": c_seg,
                    "C_R": c_r,
                    "C_bin": c_bin,
                    "C_bin_raw": components["bin_raw"][label],
                    "C_E": c_e,
                    "C_W": c_w,
                    "C_Q": c_q,
                    "q4_reliable": int(packet.segment_q4_reliable[segment_idx]),
                    "q4_peak_offset": packet.segment_q4_peak_offsets[segment_idx],
                    "q4_peak_to_side_db": packet.segment_q4_peak_to_side_db[segment_idx],
                    "q4_dev_from_packet": packet.segment_q4_dev_from_packet[segment_idx],
                    "template_reliability": templates[label].reliability,
                    "Q_seg": q_seg,
                    "segment_cost_std": segment_cost_std,
                    "T_seg": t_seg,
                    "rssi_top1_label": rssi_top1,
                    "raw_winner_label": raw_winner,
                    "raw_margin": raw_margin,
                }
            )
        garbage_cost = max(args.garbage_cost_min, args.garbage_cost - args.lambda_garbage_stability * (1.0 - q_seg))
        costs.append(garbage_cost)
        rows.append(
            {
                "segment_index": segment_idx,
                "candidate_label": GARBAGE_LABEL,
                "C_obs": garbage_cost,
                "C_seg_base": "",
                "C_R": "",
                "C_bin": "",
                "C_bin_raw": "",
                "C_E": "",
                "C_W": "",
                "C_Q": "",
                "q4_reliable": int(packet.segment_q4_reliable[segment_idx]),
                "q4_peak_offset": packet.segment_q4_peak_offsets[segment_idx],
                "q4_peak_to_side_db": packet.segment_q4_peak_to_side_db[segment_idx],
                "q4_dev_from_packet": packet.segment_q4_dev_from_packet[segment_idx],
                "template_reliability": "",
                "Q_seg": q_seg,
                "segment_cost_std": segment_cost_std,
                "T_seg": t_seg,
                "rssi_top1_label": rssi_top1,
                "raw_winner_label": raw_winner,
                "raw_margin": raw_margin,
            }
        )
        obs_costs.append(costs)

    candidate_mean_obs = {
        label: candidate_obs_sum[label] / max(1, len(per_segment_components))
        for label in candidates
    }
    norm_mean_obs = aco2.normalize_scores(candidate_mean_obs)
    veto = {label: math.exp(-args.lambda_veto * norm_mean_obs[label]) for label in candidates}
    meta = {
        "q_seg": q_seg,
        "segment_cost_std": segment_cost_std,
        "t_seg": t_seg,
        "rssi_top1": rssi_top1,
        "raw_winner": raw_winner,
        "raw_margin": raw_margin,
        "candidate_mean_obs": candidate_mean_obs,
        "candidate_mean_base": {label: candidate_base_sum[label] / max(1, len(per_segment_components)) for label in candidates},
        "candidate_cost_norm": norm_mean_obs,
        "veto": veto,
        "garbage_cost": obs_costs[0][-1] if obs_costs else args.garbage_cost,
    }
    return obs_costs, rows, meta


def dynamic_switch_penalty_v4(
    prev_idx: int,
    next_idx: int,
    segment_idx: int,
    obs_costs: Sequence[Sequence[float]],
    template_distances: Sequence[Sequence[float]],
    q_seg: float,
    args,
) -> float:
    base_penalty = aco2.dynamic_switch_penalty(prev_idx, next_idx, segment_idx, obs_costs, template_distances, args)
    return base_penalty * (1.0 + args.lambda_q_switch * (1.0 - q_seg))


def path_cost_v4(
    path: Sequence[int],
    obs_costs: Sequence[Sequence[float]],
    template_distances: Sequence[Sequence[float]],
    priors: Sequence[float],
    veto: Sequence[float],
    q_seg: float,
    args,
) -> float:
    garbage_idx = len(obs_costs[0]) - 1
    total = 0.0
    for s, choice in enumerate(path):
        if choice == garbage_idx:
            total += obs_costs[s][choice]
        else:
            total += obs_costs[s][choice] - priors[choice] / max(1, len(obs_costs)) - math.log(max(veto[choice], EPS))
    total += sum(
        dynamic_switch_penalty_v4(path[s - 1], path[s], s, obs_costs, template_distances, q_seg, args)
        for s in range(1, len(path))
    )
    unique = len({idx for idx in path if idx != garbage_idx})
    n_g = sum(1 for idx in path if idx == garbage_idx)
    total += args.lambda_div * unique + args.lambda_g * n_g
    if n_g > args.max_garbage:
        total += args.garbage_overuse_penalty * (n_g - args.max_garbage)
    return total


def run_aco_v4_for_packet(
    obs_costs: Sequence[Sequence[float]],
    candidates: Sequence[str],
    templates: dict[str, aco2.BinTemplate],
    meta: dict,
    args,
    rng: random.Random,
) -> dict:
    k = len(candidates)
    garbage_idx = k
    node_count = k + 1
    reliabilities = [templates[label].reliability for label in candidates]
    med_rel = aco2.median(reliabilities) if reliabilities else 1.0
    rel_norm = [rel / (med_rel + EPS) for rel in reliabilities]
    template_dist = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i != j:
                template_dist[i][j] = aco2.template_distance(templates[left], templates[right])

    rssi_top1 = meta["rssi_top1"]
    raw_winner = meta["raw_winner"]
    raw_margin = meta["raw_margin"]
    q_seg = meta["q_seg"]
    priors = [
        args.lambda_rssi_prior * int(label == rssi_top1)
        + args.lambda_raw_prior * raw_margin * int(label == raw_winner)
        for label in candidates
    ]
    veto = [meta["veto"][label] for label in candidates]

    pheromone = []
    for i in range(node_count):
        row = []
        for j in range(node_count):
            if i == j and i != garbage_idx:
                rssi_boost = args.lambda_rssi_prior if candidates[i] == rssi_top1 else 0.0
                row.append(args.tau_stay * (1.0 + rssi_boost + args.lambda_c * rel_norm[i]))
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
            first_weights = []
            for j, cost in enumerate(obs_costs[0]):
                eta = math.exp(-cost) if j == garbage_idx else math.exp(-cost + priors[j] / max(1, len(obs_costs))) * veto[j]
                first_weights.append(eta ** args.heuristic_power)
            path = [aco2.weighted_choice(first_weights, rng)]
            for s in range(1, len(obs_costs)):
                prev = path[-1]
                weights = []
                for j in range(node_count):
                    penalty = dynamic_switch_penalty_v4(prev, j, s, obs_costs, template_dist, q_seg, args)
                    eta = math.exp(-obs_costs[s][j]) if j == garbage_idx else math.exp(-obs_costs[s][j] + priors[j] / max(1, len(obs_costs))) * veto[j]
                    weights.append((pheromone[prev][j] ** args.pheromone_power) * (eta ** args.heuristic_power) * math.exp(-penalty))
                path.append(aco2.weighted_choice(weights, rng))
            cost = path_cost_v4(path, obs_costs, template_dist, priors, veto, q_seg, args)
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
        weight_total = sum(weights) or 1.0
        weights = [value / weight_total for value in weights]

        for i in range(node_count):
            for j in range(node_count):
                pheromone[i][j] *= 1.0 - args.evaporation
                pheromone[i][j] = max(args.min_pheromone, pheromone[i][j])
        for (_cost, path), weight in zip(elite, weights):
            for choice in path:
                if choice != garbage_idx:
                    elite_vote[choice] += weight
            for prev, cur in zip(path, path[1:]):
                if prev == cur and cur != garbage_idx:
                    raw_boost = args.lambda_raw_prior * raw_margin if candidates[cur] == raw_winner else 0.0
                    rssi_boost = args.lambda_rssi_prior if candidates[cur] == rssi_top1 else 0.0
                    pheromone[prev][cur] += weight * (1.0 + rssi_boost + raw_boost + args.lambda_c * rel_norm[cur])
                else:
                    pheromone[prev][cur] += weight

    non_g_path = [idx for idx in best_path if idx != garbage_idx]
    if non_g_path:
        path_counts = Counter(non_g_path)
        path_mode_idx = min(range(k), key=lambda idx: (-path_counts.get(idx, 0), candidates[idx]))
    else:
        path_mode_idx = min(range(k), key=lambda idx: obs_costs[0][idx])
    pheromone_values = [pheromone[idx][idx] for idx in range(k)]
    vote_values = [elite_vote[idx] for idx in range(k)]
    cost_values = [meta["candidate_mean_obs"][label] for label in candidates]
    z_pheromone = z_scores(pheromone_values)
    z_vote = z_scores(vote_values)
    z_cost = z_scores(cost_values)
    score4_values = []
    for idx, label in enumerate(candidates):
        score4_values.append(
            z_pheromone[idx]
            + args.lambda_score_vote * z_vote[idx]
            - args.lambda_score_cost * z_cost[idx]
            + args.lambda_rssi_prior * int(label == rssi_top1)
            + args.lambda_raw_prior * raw_margin * int(label == raw_winner)
        )
    score4_idx = max(range(k), key=lambda idx: (score4_values[idx], -idx))
    pheromone_idx = max(range(k), key=lambda idx: (pheromone_values[idx], -idx))
    vote_idx = max(range(k), key=lambda idx: (vote_values[idx], -idx))
    path_labels = [GARBAGE_LABEL if idx == garbage_idx else candidates[idx] for idx in best_path]
    return {
        "best_path": best_path,
        "best_path_labels": path_labels,
        "best_cost": best_cost,
        "path_mode_label": candidates[path_mode_idx],
        "pheromone_label": candidates[pheromone_idx],
        "vote_label": candidates[vote_idx],
        "score4_label": candidates[score4_idx],
        "self_pheromone": {label: pheromone[idx][idx] for idx, label in enumerate(candidates)},
        "elite_vote": {label: elite_vote[idx] for idx, label in enumerate(candidates)},
        "score4": {label: score4_values[idx] for idx, label in enumerate(candidates)},
        "garbage_count": sum(1 for idx in best_path if idx == garbage_idx),
    }
