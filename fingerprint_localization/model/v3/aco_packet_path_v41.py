#!/usr/bin/env python3
"""ACO 4.1 with weakly supervised multipath reliability gates."""

from __future__ import annotations

import math
import random
from collections import Counter
from statistics import median
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


def sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo <= EPS:
        return {key: 0.5 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def source_confidence(source: str) -> float:
    parts = source.split(";")
    score = 0.0
    for part in parts:
        if "measured_chirp" in part:
            score = max(score, 1.0)
        elif "interpolated" in part or "nearest" in part:
            score = max(score, 0.5)
    return score


def shape_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= EPS or right_norm <= EPS:
        return 0.5
    return 0.5 * (dot / (left_norm * right_norm) + 1.0)


def candidate_weak_supervision(
    candidates: Sequence[str],
    templates: dict[str, aco2.BinTemplate],
    chirp_struct,
    args,
) -> dict:
    k_values = {}
    tau_values = {}
    pdiff_values = {}
    tr_values = {}
    sim_values = {}
    conf_values = {}
    for label in candidates:
        struct, _source = chirp_struct.get(label, ({"k_ratio": 1.0, "tau_rms": 0.0}, "structure_default"))
        k_ratio = aco2.parse_float(struct.get("k_ratio"), 1.0)
        tau_rms = aco2.parse_float(struct.get("tau_rms"), 0.0)
        k_values[label] = math.log1p(max(0.0, k_ratio))
        tau_values[label] = max(0.0, tau_rms)
        pdiff_values[label] = 1.0 / (1.0 + max(0.0, k_ratio))
        tr_values[label] = sum(templates[label].var)
        sim_values[label] = shape_similarity(templates[label].mu_emp, templates[label].mu_phy)
        conf_values[label] = source_confidence(templates[label].chirp_source)

    k_norm = normalize(k_values)
    tau_norm = normalize(tau_values)
    pdiff_norm = normalize(pdiff_values)
    tr_norm = normalize(tr_values)
    reliability = {}
    struct_features = {}
    for label in candidates:
        score = (
            args.ws_bias
            + args.ws_a1 * k_norm[label]
            - args.ws_a2 * tau_norm[label]
            - args.ws_a3 * tr_norm[label]
            + args.ws_a4 * sim_values[label]
            + args.ws_a5 * conf_values[label]
        )
        reliability[label] = sigmoid(score)
        struct_features[label] = [k_norm[label], pdiff_norm[label], tau_norm[label]]
    return {
        "reliability": reliability,
        "k_norm": k_norm,
        "tau_norm": tau_norm,
        "pdiff_norm": pdiff_norm,
        "tr_norm": tr_norm,
        "sim": sim_values,
        "conf": conf_values,
        "struct_features": struct_features,
    }


def weak_template_distances(
    candidates: Sequence[str],
    templates: dict[str, aco2.BinTemplate],
    ws: dict,
    args,
) -> tuple[list[list[float]], dict[str, float], dict[str, float]]:
    n = len(candidates)
    distances = [[0.0 for _ in range(n + 1)] for _ in range(n + 1)]
    for i, left in enumerate(candidates):
        for j, right in enumerate(candidates):
            if i == j:
                continue
            struct_dist = sum(
                (ws["struct_features"][left][dim] - ws["struct_features"][right][dim]) ** 2
                for dim in range(3)
            )
            raw_dist = aco2.template_distance(templates[left], templates[right])
            distances[i][j] = (
                args.ws_omega_template * raw_dist
                + args.ws_omega_struct * struct_dist
            ) * 0.5 * (ws["reliability"][left] + ws["reliability"][right])
    sep_raw = {}
    sep_gate = {}
    for i, label in enumerate(candidates):
        others = [distances[i][j] for j in range(n) if j != i]
        sep = min(others) if others else 0.0
        sep_raw[label] = sep
        sep_gate[label] = sep / (sep + args.ws_sep_scale + EPS)
    return distances, sep_raw, sep_gate


def packet_median_zw(packet: aco2.SegmentPacket) -> list[float]:
    dim = len(packet.segment_zw[0])
    return [median([zw[j] for zw in packet.segment_zw]) for j in range(dim)]


def build_observation_costs_v41(
    packet: aco2.SegmentPacket,
    candidates: Sequence[str],
    rssi_costs: dict[str, float],
    templates: dict[str, aco2.BinTemplate],
    chirp_struct,
    segment_prototypes: dict,
    q4_offsets: Sequence[float],
    args,
) -> tuple[list[list[float]], list[dict], dict]:
    shift_grid = [float(part.strip()) for part in args.q4_shift_grid.split(",") if part.strip()]
    rssi_norm = aco2.normalize_scores({label: rssi_costs[label] for label in candidates})
    ws = candidate_weak_supervision(candidates, templates, chirp_struct, args)

    per_segment_components = []
    segment_base_mins = []
    segment_obs_mins = []
    for segment_idx, shape in enumerate(packet.segment_shapes):
        raw_bin_costs = {label: aco2.bin_cost(shape, templates[label], args) for label in candidates}
        bin_norm = aco2.normalize_scores(raw_bin_costs) if args.normalize_bin_cost else raw_bin_costs
        e_norm = aco2.normalize_scores(
            {
                label: aco2.base.robust_scalar_cost(packet.segment_zw[segment_idx][0], segment_prototypes["energy"][label])
                for label in candidates
            }
        )
        w_norm = aco2.normalize_scores(
            {
                label: aco2.base.robust_vector_cost(packet.segment_zw[segment_idx], segment_prototypes["zw"][label])
                for label in candidates
            }
        )
        if packet.segment_q4_reliable[segment_idx]:
            q_norm = aco2.normalize_scores(
                {
                    label: aco2.base.q4_shape_cost(
                        packet.segment_q4_curves[segment_idx],
                        segment_prototypes["q4"][label],
                        q4_offsets,
                        shift_grid,
                    )
                    for label in candidates
                }
            )
        else:
            q_norm = {label: 0.0 for label in candidates}

        q_weight = args.q4_weight if packet.segment_q4_reliable[segment_idx] else 0.0
        components = {}
        base_costs = []
        obs_costs = []
        for label in candidates:
            c_seg = (
                args.energy_weight * e_norm[label]
                + args.raw_weight * w_norm[label]
                + q_weight * q_norm[label]
                + args.bin_weight * bin_norm[label]
            )
            components[label] = {
                "C_R": rssi_norm.get(label, 0.0),
                "C_bin": bin_norm[label],
                "C_bin_raw": raw_bin_costs[label],
                "C_E": e_norm[label],
                "C_W": w_norm[label],
                "C_Q": q_norm[label],
                "C_seg_base": c_seg,
            }
            base_costs.append(c_seg)
            obs_costs.append(args.rssi_weight * rssi_norm.get(label, 0.0) + c_seg)
        per_segment_components.append(components)
        segment_base_mins.append(min(base_costs) if base_costs else 0.0)
        segment_obs_mins.append(min(obs_costs) if obs_costs else 0.0)

    mean_min = sum(segment_base_mins) / len(segment_base_mins) if segment_base_mins else 0.0
    segment_cost_std = math.sqrt(
        sum((value - mean_min) ** 2 for value in segment_base_mins) / len(segment_base_mins)
    ) if segment_base_mins else 0.0
    t_seg = getattr(args, "t_seg_resolved", None) or getattr(args, "t_seg", None) or 1.0
    q_pkt = 1.0 / (1.0 + segment_cost_std / max(t_seg, EPS))

    median_zw = packet_median_zw(packet)
    segment_deviation = [
        sum((packet.segment_zw[segment_idx][j] - median_zw[j]) ** 2 for j in range(len(median_zw)))
        for segment_idx in range(len(packet.segment_zw))
    ]
    t_d = getattr(args, "t_d_resolved", None) or getattr(args, "t_d", None) or 1.0
    t_c = getattr(args, "t_c_resolved", None) or getattr(args, "t_c", None) or 1.0
    q_s = [
        max(
            args.min_segment_reliability,
            math.exp(-segment_deviation[idx] / max(t_d, EPS))
            * math.exp(-segment_obs_mins[idx] / max(t_c, EPS)),
        )
        for idx in range(len(segment_deviation))
    ]

    rssi_top1 = candidates[0] if candidates else ""
    packet_bin_raw = {
        label: sum(components[label]["C_bin_raw"] for components in per_segment_components) / max(1, len(per_segment_components))
        for label in candidates
    }
    raw_ranked = sorted(packet_bin_raw.items(), key=lambda item: (item[1], item[0]))
    raw_winner = raw_ranked[0][0] if raw_ranked else ""
    raw_best = raw_ranked[0][1] if raw_ranked else 0.0
    raw_second = raw_ranked[1][1] if len(raw_ranked) > 1 else raw_best
    raw_margin = max(0.0, (raw_second - raw_best) / (raw_second + EPS)) if raw_ranked else 0.0
    template_distances, sep_raw, sep_gate = weak_template_distances(candidates, templates, ws, args)

    obs_costs = []
    rows = []
    candidate_obs_sum = {label: 0.0 for label in candidates}
    candidate_base_sum = {label: 0.0 for label in candidates}
    for segment_idx, components in enumerate(per_segment_components):
        costs = []
        for label in candidates:
            c_seg = components[label]["C_seg_base"]
            c_obs = args.rssi_weight * components[label]["C_R"] + q_pkt * q_s[segment_idx] * c_seg
            candidate_obs_sum[label] += c_obs
            candidate_base_sum[label] += c_seg
            costs.append(c_obs)
            rows.append(
                {
                    "segment_index": segment_idx,
                    "candidate_label": label,
                    "C_obs": c_obs,
                    "C_seg_base": c_seg,
                    "C_R": components[label]["C_R"],
                    "C_bin": components[label]["C_bin"],
                    "C_bin_raw": components[label]["C_bin_raw"],
                    "C_E": components[label]["C_E"],
                    "C_W": components[label]["C_W"],
                    "C_Q": components[label]["C_Q"],
                    "q4_reliable": int(packet.segment_q4_reliable[segment_idx]),
                    "q4_peak_offset": packet.segment_q4_peak_offsets[segment_idx],
                    "q4_peak_to_side_db": packet.segment_q4_peak_to_side_db[segment_idx],
                    "q4_dev_from_packet": packet.segment_q4_dev_from_packet[segment_idx],
                    "template_reliability": templates[label].reliability,
                    "R_ws": ws["reliability"][label],
                    "Sep_ws": sep_raw[label],
                    "Sep_gate": sep_gate[label],
                    "Q_pkt": q_pkt,
                    "q_s": q_s[segment_idx],
                    "segment_deviation": segment_deviation[segment_idx],
                    "segment_cost_min": segment_obs_mins[segment_idx],
                    "segment_cost_std": segment_cost_std,
                    "T_seg": t_seg,
                    "T_d": t_d,
                    "T_c": t_c,
                    "rssi_top1_label": rssi_top1,
                    "raw_winner_label": raw_winner,
                    "raw_margin": raw_margin,
                }
            )
        garbage_cost = max(args.garbage_cost_min, args.garbage_cost - args.lambda_garbage_stability * (1.0 - q_s[segment_idx]))
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
                "R_ws": "",
                "Sep_ws": "",
                "Sep_gate": "",
                "Q_pkt": q_pkt,
                "q_s": q_s[segment_idx],
                "segment_deviation": segment_deviation[segment_idx],
                "segment_cost_min": segment_obs_mins[segment_idx],
                "segment_cost_std": segment_cost_std,
                "T_seg": t_seg,
                "T_d": t_d,
                "T_c": t_c,
                "rssi_top1_label": rssi_top1,
                "raw_winner_label": raw_winner,
                "raw_margin": raw_margin,
            }
        )
        obs_costs.append(costs)

    candidate_mean_obs = {label: candidate_obs_sum[label] / max(1, len(per_segment_components)) for label in candidates}
    norm_mean_obs = aco2.normalize_scores(candidate_mean_obs)
    veto = {
        label: math.exp(-args.lambda_veto * norm_mean_obs[label]) * math.exp(args.lambda_vr * ws["reliability"][label])
        for label in candidates
    }
    meta = {
        "q_seg": q_pkt,
        "q_pkt": q_pkt,
        "q_s": q_s,
        "segment_deviation": segment_deviation,
        "segment_cost_min": segment_obs_mins,
        "segment_cost_std": segment_cost_std,
        "t_seg": t_seg,
        "t_d": t_d,
        "t_c": t_c,
        "rssi_top1": rssi_top1,
        "raw_winner": raw_winner,
        "raw_margin": raw_margin,
        "candidate_mean_obs": candidate_mean_obs,
        "candidate_mean_base": {label: candidate_base_sum[label] / max(1, len(per_segment_components)) for label in candidates},
        "candidate_cost_norm": norm_mean_obs,
        "veto": veto,
        "R_ws": ws["reliability"],
        "Sep_ws": sep_raw,
        "Sep_gate": sep_gate,
        "template_distances": template_distances,
        "garbage_costs": [row[-1] for row in obs_costs],
    }
    return obs_costs, rows, meta


def dynamic_switch_penalty_v41(
    prev_idx: int,
    next_idx: int,
    segment_idx: int,
    obs_costs: Sequence[Sequence[float]],
    template_distances: Sequence[Sequence[float]],
    q_pkt: float,
    q_s: Sequence[float],
    args,
) -> float:
    if prev_idx == next_idx:
        return 0.0
    garbage_idx = len(obs_costs[segment_idx]) - 1
    if prev_idx == garbage_idx or next_idx == garbage_idx:
        return 0.0
    advantage = max(0.0, obs_costs[segment_idx][prev_idx] - obs_costs[segment_idx][next_idx])
    numerator = args.lambda0_switch * (1.0 + args.lambda_q_switch * (1.0 - q_pkt))
    denominator = 1.0 + args.switch_eta * q_s[segment_idx] * template_distances[prev_idx][next_idx] * advantage
    return numerator / denominator


def path_cost_v41(
    path: Sequence[int],
    obs_costs: Sequence[Sequence[float]],
    template_distances: Sequence[Sequence[float]],
    priors: Sequence[float],
    veto: Sequence[float],
    q_pkt: float,
    q_s: Sequence[float],
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
        dynamic_switch_penalty_v41(path[s - 1], path[s], s, obs_costs, template_distances, q_pkt, q_s, args)
        for s in range(1, len(path))
    )
    unique = len({idx for idx in path if idx != garbage_idx})
    n_g = sum(1 for idx in path if idx == garbage_idx)
    total += args.lambda_div * unique + args.lambda_g * n_g
    if n_g > args.max_garbage:
        total += args.garbage_overuse_penalty * (n_g - args.max_garbage)
    return total


def run_aco_v41_for_packet(
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
    template_dist = meta["template_distances"]
    rssi_top1 = meta["rssi_top1"]
    raw_winner = meta["raw_winner"]
    raw_margin = meta["raw_margin"]
    q_pkt = meta["q_pkt"]
    q_s = meta["q_s"]
    r_ws = [meta["R_ws"][label] for label in candidates]
    sep_gate = [meta["Sep_gate"][label] for label in candidates]
    priors = [
        args.lambda_rssi_prior * int(label == rssi_top1)
        + args.lambda_raw_prior * raw_margin * r_ws[idx] * sep_gate[idx] * int(label == raw_winner)
        for idx, label in enumerate(candidates)
    ]
    veto = [meta["veto"][label] for label in candidates]

    pheromone = []
    for i in range(node_count):
        row = []
        for j in range(node_count):
            if i == j and i != garbage_idx:
                rssi_boost = args.lambda_rssi_prior if candidates[i] == rssi_top1 else 0.0
                row.append(args.tau_stay * (1.0 + rssi_boost + args.lambda_c * r_ws[i]))
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
                    penalty = dynamic_switch_penalty_v41(prev, j, s, obs_costs, template_dist, q_pkt, q_s, args)
                    eta = math.exp(-obs_costs[s][j]) if j == garbage_idx else math.exp(-obs_costs[s][j] + priors[j] / max(1, len(obs_costs))) * veto[j]
                    weights.append((pheromone[prev][j] ** args.pheromone_power) * (eta ** args.heuristic_power) * math.exp(-penalty))
                path.append(aco2.weighted_choice(weights, rng))
            cost = path_cost_v41(path, obs_costs, template_dist, priors, veto, q_pkt, q_s, args)
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
                    rssi_boost = args.lambda_rssi_prior if candidates[cur] == rssi_top1 else 0.0
                    pheromone[prev][cur] += weight * (1.0 + rssi_boost + args.lambda_c * r_ws[cur])
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
    score_values = []
    for idx, label in enumerate(candidates):
        score_values.append(
            z_pheromone[idx]
            + args.lambda_score_vote * z_vote[idx]
            - args.lambda_score_cost * z_cost[idx]
            + args.lambda_rssi_prior * int(label == rssi_top1)
            + args.lambda_raw_prior * raw_margin * r_ws[idx] * sep_gate[idx] * int(label == raw_winner)
            + args.lambda_rel * r_ws[idx]
        )
    score_idx = max(range(k), key=lambda idx: (score_values[idx], -idx))
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
        "score4_label": candidates[score_idx],
        "self_pheromone": {label: pheromone[idx][idx] for idx, label in enumerate(candidates)},
        "elite_vote": {label: elite_vote[idx] for idx, label in enumerate(candidates)},
        "score4": {label: score_values[idx] for idx, label in enumerate(candidates)},
        "garbage_count": sum(1 for idx in best_path if idx == garbage_idx),
    }
