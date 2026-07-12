#!/usr/bin/env python3
"""ACO 2.0 with chirp-guided LoRa bin templates.

This is an experimental implementation of `external_design_notes/蚁群算法2.0.md`.
It keeps the original RSSI+ Top-K candidate generation and raw observation
terms, adds a shrinkage LoRa-bin Gaussian likelihood, adds a garbage state, and
uses evidence-driven switching penalties.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import aco_packet_path as base


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "fingerprint_localization"
DATA_ROOT = PACKAGE_ROOT / "data" / "mainline_202607"
DEFAULT_RSSI_CSV = DATA_ROOT / "inputs" / "rssi_plus_packet_level_54points.csv"
DEFAULT_SPECTRUM_CSV = DATA_ROOT / "external" / "subbin_spectrum_long.csv"
DEFAULT_CHIRP_TEMPLATE_CSV = DATA_ROOT / "features" / "02_chirp_synth_point_bins.csv"
DEFAULT_CHIRP_STRUCT_CSV = DATA_ROOT / "features" / "chirp_point_multipath_structure_features.csv"
DEFAULT_LOCATION_CSV = PACKAGE_ROOT / "docs" / "location_distance_54points.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3" / "output_aco_v2"

EPS = 1e-12
RAW_OFFSETS = base.RAW_OFFSETS
GARBAGE_LABEL = "__G__"


@dataclass
class SegmentPacket:
    key: base.PacketKey
    file_name: str
    packet_index: int
    label: str
    rssi_plus: list[float]
    segment_shapes: list[list[float]]
    segment_zw: list[list[float]]
    segment_q4_curves: list[list[float]]
    segment_q4_reliable: list[bool]
    segment_q4_peak_offsets: list[float]
    segment_q4_peak_to_side_db: list[float]
    segment_q4_dev_from_packet: list[float]


@dataclass
class BinTemplate:
    label: str
    mu: list[float]
    var: list[float]
    mu_emp: list[float]
    var_emp: list[float]
    mu_phy: list[float]
    var_phy: list[float]
    n_packets: int
    alpha_shrink: float
    reliability: float
    chirp_source: str


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_dict(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def natural_label_key(label: str) -> tuple[int, int]:
    return base.natural_label_key(label)


def point_display(label: str) -> str:
    return base.point_display(label)


def median(values: Sequence[float]) -> float:
    return base.median(list(values))


def safe_iqr(values: Sequence[float]) -> float:
    return base.safe_iqr(list(values))


def raw_shape(raw_bins: Sequence[float]) -> list[float]:
    logs = [math.log(max(float(value), EPS)) for value in raw_bins]
    center = sum(logs) / len(logs)
    return [value - center for value in logs]


def segment_bounds(n_symbols: int, segment_count: int) -> list[tuple[int, int]]:
    if n_symbols < segment_count:
        raise ValueError(f"Cannot split {n_symbols} symbols into {segment_count} nonempty segments")
    bounds = []
    for idx in range(segment_count):
        start = round(idx * n_symbols / segment_count)
        end = round((idx + 1) * n_symbols / segment_count)
        bounds.append((start, max(start + 1, min(end, n_symbols))))
    return bounds


def packet_to_segments(sample: base.PacketSample, segment_count: int) -> SegmentPacket:
    symbols = sorted(sample.symbols, key=lambda item: item.symbol_index)
    segment_shapes = []
    segment_zw = []
    segment_q4_curves = []
    segment_q4_reliable = []
    segment_q4_peak_offsets = []
    segment_q4_peak_to_side_db = []
    segment_q4_dev_from_packet = []
    for start, end in segment_bounds(len(symbols), segment_count):
        chunk = symbols[start:end]
        raw_bins = [
            median([symbol.raw_bins[j] for symbol in chunk])
            for j in range(len(RAW_OFFSETS))
        ]
        segment_shapes.append(raw_shape(raw_bins))
        segment_zw.append(base.raw_structure(raw_bins))
        segment_q4_curves.append(
            [
                median([symbol.q4_curve[j] for symbol in chunk])
                for j in range(len(chunk[0].q4_curve))
            ]
        )
        segment_q4_reliable.append(
            sum(1 for symbol in chunk if symbol.q4_reliable) >= max(1, math.ceil(len(chunk) / 2))
        )
        segment_q4_peak_offsets.append(median([symbol.q4_peak_offset for symbol in chunk]))
        segment_q4_peak_to_side_db.append(median([symbol.q4_peak_to_side_db for symbol in chunk]))
        segment_q4_dev_from_packet.append(median([symbol.q4_dev_from_packet for symbol in chunk]))
    return SegmentPacket(
        key=sample.key,
        file_name=sample.file_name,
        packet_index=sample.packet_index,
        label=sample.label,
        rssi_plus=sample.rssi_plus,
        segment_shapes=segment_shapes,
        segment_zw=segment_zw,
        segment_q4_curves=segment_q4_curves,
        segment_q4_reliable=segment_q4_reliable,
        segment_q4_peak_offsets=segment_q4_peak_offsets,
        segment_q4_peak_to_side_db=segment_q4_peak_to_side_db,
        segment_q4_dev_from_packet=segment_q4_dev_from_packet,
    )


def build_segment_packets(samples: Sequence[base.PacketSample], segment_count: int) -> list[SegmentPacket]:
    return [packet_to_segments(sample, segment_count) for sample in samples]


def packet_median_shape(packet: SegmentPacket) -> list[float]:
    return [
        median([shape[j] for shape in packet.segment_shapes])
        for j in range(len(RAW_OFFSETS))
    ]


def read_location_meta(path: Path) -> dict[str, dict]:
    rows = {}
    for row in read_csv_dict(path):
        label = row["position_key"]
        rows[label] = {
            "label": label,
            "corridor_id": parse_int(row["corridor_id"]),
            "location_id": parse_int(row["location_id"]),
            "distance_m": parse_float(row.get("distance_m")),
            "visibility_state": parse_int(row.get("c_i（NLOS-2，LOS-1，OLOS-0）", 0)),
        }
    return rows


def read_chirp_shapes(path: Path) -> dict[str, list[float]]:
    shapes = {}
    for row in read_csv_dict(path):
        label = f"{parse_int(row['corridor_id'])}_{parse_int(row['location_id'])}"
        mags = [parse_float(row[f"synth_mag_bin_{int(offset):+d}_mean"]) for offset in RAW_OFFSETS]
        shapes[label] = raw_shape(mags)
    return shapes


def read_chirp_structure(path: Path) -> dict[str, dict]:
    out = {}
    for row in read_csv_dict(path):
        label = row.get("position_key") or f"{parse_int(row['corridor_id'])}_{parse_int(row['location_id'])}"
        main = parse_float(row.get("main_effective_power_fraction"), 1.0)
        secondary = parse_float(row.get("secondary_effective_power_sum"), 0.0)
        tau = parse_float(row.get("equivalent_rms_delay_us"), 0.0)
        out[label] = {
            "k_ratio": main / (secondary + EPS),
            "tau_rms": tau,
            "source": "measured_chirp",
        }
    return out


def interpolate_field(
    labels: Sequence[str],
    measured: dict[str, object],
    locations: dict[str, dict],
    default: object,
) -> dict[str, tuple[object, str]]:
    filled = {}
    for label in labels:
        if label in measured:
            filled[label] = (measured[label], "measured_chirp")
            continue
        meta = locations.get(label)
        if meta is None:
            filled[label] = (default, "fallback_default")
            continue
        same = [
            (other_meta["distance_m"], other_label, measured[other_label])
            for other_label, other_meta in locations.items()
            if other_label in measured
            and other_meta["corridor_id"] == meta["corridor_id"]
            and other_meta["visibility_state"] == meta["visibility_state"]
        ]
        left = [(dist, lab, item) for dist, lab, item in same if dist < meta["distance_m"]]
        right = [(dist, lab, item) for dist, lab, item in same if dist > meta["distance_m"]]
        left_best = max(left, default=None, key=lambda item: item[0])
        right_best = min(right, default=None, key=lambda item: item[0])
        if left_best and right_best:
            left_value = left_best[2]
            right_value = right_best[2]
            if isinstance(left_value, list):
                value = [(left_value[j] + right_value[j]) / 2.0 for j in range(len(left_value))]
            else:
                value = {
                    key: (left_value[key] + right_value[key]) / 2.0
                    for key in left_value
                    if isinstance(left_value[key], (int, float))
                }
            filled[label] = (value, f"interpolated:{left_best[1]}|{right_best[1]}")
        elif left_best or right_best:
            near = left_best or right_best
            filled[label] = (near[2], f"nearest:{near[1]}")
        else:
            filled[label] = (default, "fallback_default")
    return filled


def empirical_templates(
    packets: Sequence[SegmentPacket],
    labels: Sequence[str],
    train_indices: Sequence[int],
) -> dict[str, dict]:
    by_label_packets: dict[str, list[list[float]]] = defaultdict(list)
    by_label_segments: dict[str, list[list[float]]] = defaultdict(list)
    for idx in train_indices:
        label = labels[idx]
        by_label_packets[label].append(packet_median_shape(packets[idx]))
        by_label_segments[label].extend(packets[idx].segment_shapes)
    out = {}
    for label, rows in by_label_packets.items():
        dim = len(RAW_OFFSETS)
        mu = [median([row[j] for row in rows]) for j in range(dim)]
        segment_rows = by_label_segments[label]
        var = []
        for j in range(dim):
            values = [row[j] for row in segment_rows]
            center = mu[j]
            value = sum((v - center) ** 2 for v in values) / len(values)
            var.append(value if value > EPS else safe_iqr(values) ** 2 + EPS)
        out[label] = {"mu": mu, "var": var, "n": len(rows)}
    return out


def build_segment_prototypes(
    packets: Sequence[SegmentPacket],
    labels: Sequence[str],
    train_indices: Sequence[int],
) -> dict:
    energy_by_label: dict[str, list[float]] = defaultdict(list)
    zw_by_label: dict[str, list[Sequence[float]]] = defaultdict(list)
    q4_by_label: dict[str, list[Sequence[float]]] = defaultdict(list)
    for idx in train_indices:
        label = labels[idx]
        packet = packets[idx]
        for segment_idx, zw in enumerate(packet.segment_zw):
            energy_by_label[label].append(zw[0])
            zw_by_label[label].append(zw)
            q4_by_label[label].append(packet.segment_q4_curves[segment_idx])
    return {
        "energy": base.robust_scalar_prototypes(energy_by_label),
        "zw": base.robust_vector_prototypes(zw_by_label),
        "q4": base.robust_vector_prototypes(q4_by_label),
    }


def build_templates(
    packets: Sequence[SegmentPacket],
    labels: Sequence[str],
    train_indices: Sequence[int],
    chirp_shapes: dict[str, tuple[list[float], str]],
    chirp_struct: dict[str, tuple[dict, str]],
    args: argparse.Namespace,
) -> dict[str, BinTemplate]:
    emp = empirical_templates(packets, labels, train_indices)
    out = {}
    global_mu = [median([item["mu"][j] for item in emp.values()]) for j in range(len(RAW_OFFSETS))]
    global_var = [median([item["var"][j] for item in emp.values()]) for j in range(len(RAW_OFFSETS))]
    for label, item in emp.items():
        mu_phy, phy_source = chirp_shapes.get(label, (global_mu, "empirical_global_fallback"))
        struct, struct_source = chirp_struct.get(label, ({"k_ratio": 1.0, "tau_rms": 0.0}, "structure_default"))
        sigma_phy = (
            args.phy_var_c0
            + args.phy_var_c1 / (parse_float(struct.get("k_ratio"), 1.0) + 1.0)
            + args.phy_var_c2 * parse_float(struct.get("tau_rms"), 0.0)
        )
        sigma_phy = max(sigma_phy, args.min_variance)
        var_phy = [sigma_phy for _ in RAW_OFFSETS]
        n = item["n"]
        alpha = n / (n + args.shrinkage_lambda)
        mu = [alpha * item["mu"][j] + (1.0 - alpha) * mu_phy[j] for j in range(len(RAW_OFFSETS))]
        var = [
            alpha * item["var"][j] + (1.0 - alpha) * var_phy[j] + args.sigma0_sq
            for j in range(len(RAW_OFFSETS))
        ]
        var = [max(v, args.min_variance) for v in var]
        reliability = 1.0 / (sum(var) + EPS)
        out[label] = BinTemplate(
            label=label,
            mu=mu,
            var=var,
            mu_emp=item["mu"],
            var_emp=item["var"],
            mu_phy=mu_phy,
            var_phy=var_phy,
            n_packets=n,
            alpha_shrink=alpha,
            reliability=reliability,
            chirp_source=f"{phy_source};{struct_source}",
        )
    return out


def huber_mahalanobis(value: float, delta: float) -> float:
    r = math.sqrt(max(value, 0.0))
    if r <= delta:
        return value
    return 2.0 * delta * r - delta * delta


def bin_cost(shape: Sequence[float], template: BinTemplate, args: argparse.Namespace) -> float:
    mahal = sum(
        ((shape[j] - template.mu[j]) ** 2) / (template.var[j] + EPS)
        for j in range(len(RAW_OFFSETS))
    )
    logdet = sum(math.log(template.var[j] + EPS) for j in range(len(RAW_OFFSETS)))
    return huber_mahalanobis(mahal, args.huber_delta) + args.logdet_weight * logdet


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo <= EPS:
        return {key: 0.0 for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def template_distance(left: BinTemplate, right: BinTemplate) -> float:
    return sum(
        ((left.mu[j] - right.mu[j]) ** 2)
        / (0.5 * (left.var[j] + right.var[j]) + EPS)
        for j in range(len(RAW_OFFSETS))
    )


def dynamic_switch_penalty(
    prev_idx: int,
    next_idx: int,
    segment_idx: int,
    obs_costs: Sequence[Sequence[float]],
    template_distances: Sequence[Sequence[float]],
    args: argparse.Namespace,
) -> float:
    if prev_idx == next_idx:
        return 0.0
    garbage_idx = len(obs_costs[segment_idx]) - 1
    if prev_idx == garbage_idx or next_idx == garbage_idx:
        return 0.0
    advantage = max(0.0, obs_costs[segment_idx][prev_idx] - obs_costs[segment_idx][next_idx])
    return args.lambda0_switch / (1.0 + args.switch_eta * template_distances[prev_idx][next_idx] * advantage)


def path_cost(
    path: Sequence[int],
    obs_costs: Sequence[Sequence[float]],
    template_distances: Sequence[Sequence[float]],
    args: argparse.Namespace,
) -> float:
    total = sum(obs_costs[s][choice] for s, choice in enumerate(path))
    total += sum(dynamic_switch_penalty(path[s - 1], path[s], s, obs_costs, template_distances, args) for s in range(1, len(path)))
    garbage_idx = len(obs_costs[0]) - 1
    unique = len({idx for idx in path if idx != garbage_idx})
    n_g = sum(1 for idx in path if idx == garbage_idx)
    total += args.lambda_div * unique + args.lambda_g * n_g
    if n_g > args.max_garbage:
        total += args.garbage_overuse_penalty * (n_g - args.max_garbage)
    return total


def weighted_choice(weights: Sequence[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= EPS:
        return rng.randrange(len(weights))
    pick = rng.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += weight
        if pick <= acc:
            return idx
    return len(weights) - 1


def run_aco_v2_for_packet(
    obs_costs: Sequence[Sequence[float]],
    candidates: Sequence[str],
    templates: dict[str, BinTemplate],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict:
    k = len(candidates)
    garbage_idx = k
    node_count = k + 1
    reliabilities = [templates[label].reliability for label in candidates]
    med_rel = median(reliabilities) if reliabilities else 1.0
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
            if i == j and i != garbage_idx:
                row.append(args.tau_stay * (1.0 + args.lambda_c * rel_norm[i]))
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
            path = [weighted_choice(first_weights, rng)]
            for s in range(1, len(obs_costs)):
                prev = path[-1]
                weights = []
                for j in range(node_count):
                    penalty = dynamic_switch_penalty(prev, j, s, obs_costs, template_dist, args)
                    eta = math.exp(-obs_costs[s][j])
                    weights.append((pheromone[prev][j] ** args.pheromone_power) * (eta ** args.heuristic_power) * math.exp(-penalty))
                path.append(weighted_choice(weights, rng))
            cost = path_cost(path, obs_costs, template_dist, args)
            paths.append((cost, path))
            if cost < best_cost:
                best_cost = cost
                best_path = list(path)

        paths.sort(key=lambda item: item[0])
        elite = paths[: max(1, min(args.elite_ants, len(paths)))]
        temp = args.aco_temperature
        if temp is None or temp <= EPS:
            temp = median([cost for cost, _path in elite])
            temp = temp if temp > EPS else 1.0
        weights = [math.exp(-cost / (temp + EPS)) for cost, _path in elite]
        weight_total = sum(weights) or 1.0
        weights = [value / weight_total for value in weights]

        for i in range(node_count):
            for j in range(node_count):
                pheromone[i][j] *= 1.0 - args.evaporation
                pheromone[i][j] = max(args.min_pheromone, pheromone[i][j])
        for (cost, path), weight in zip(elite, weights):
            for choice in path:
                if choice != garbage_idx:
                    elite_vote[choice] += weight
            for prev, cur in zip(path, path[1:]):
                if prev == cur and cur != garbage_idx:
                    pheromone[prev][cur] += weight * (1.0 + args.lambda_c * rel_norm[cur])
                else:
                    pheromone[prev][cur] += weight

    non_g_path = [idx for idx in best_path if idx != garbage_idx]
    if non_g_path:
        path_counts = Counter(non_g_path)
        path_mode_idx = min(range(k), key=lambda idx: (-path_counts.get(idx, 0), candidates[idx]))
    else:
        path_mode_idx = min(range(k), key=lambda idx: obs_costs[0][idx])
    pheromone_idx = max(range(k), key=lambda idx: (pheromone[idx][idx], -idx))
    vote_idx = max(range(k), key=lambda idx: (elite_vote[idx], -idx))
    path_labels = [GARBAGE_LABEL if idx == garbage_idx else candidates[idx] for idx in best_path]
    return {
        "best_path": best_path,
        "best_path_labels": path_labels,
        "best_cost": best_cost,
        "path_mode_label": candidates[path_mode_idx],
        "pheromone_label": candidates[pheromone_idx],
        "vote_label": candidates[vote_idx],
        "self_pheromone": {label: pheromone[idx][idx] for idx, label in enumerate(candidates)},
        "elite_vote": {label: elite_vote[idx] for idx, label in enumerate(candidates)},
        "garbage_count": sum(1 for idx in best_path if idx == garbage_idx),
    }


def build_observation_costs_v2(
    packet: SegmentPacket,
    candidates: Sequence[str],
    rssi_costs: dict[str, float],
    templates: dict[str, BinTemplate],
    segment_prototypes: dict,
    q4_offsets: Sequence[float],
    args: argparse.Namespace,
) -> tuple[list[list[float]], list[dict]]:
    shift_grid = [float(part.strip()) for part in args.q4_shift_grid.split(",") if part.strip()]
    rssi_norm = normalize_scores({label: rssi_costs[label] for label in candidates})
    obs_costs = []
    rows = []
    for segment_idx, shape in enumerate(packet.segment_shapes):
        raw_bin_costs = {label: bin_cost(shape, templates[label], args) for label in candidates}
        bin_norm = normalize_scores(raw_bin_costs) if args.normalize_bin_cost else raw_bin_costs
        e_cost = {
            label: base.robust_scalar_cost(packet.segment_zw[segment_idx][0], segment_prototypes["energy"][label])
            for label in candidates
        }
        w_cost = {
            label: base.robust_vector_cost(packet.segment_zw[segment_idx], segment_prototypes["zw"][label])
            for label in candidates
        }
        if packet.segment_q4_reliable[segment_idx]:
            q_cost = {
                label: base.q4_shape_cost(
                    packet.segment_q4_curves[segment_idx],
                    segment_prototypes["q4"][label],
                    q4_offsets,
                    shift_grid,
                )
                for label in candidates
            }
        else:
            q_cost = {label: 0.0 for label in candidates}
        e_norm = normalize_scores(e_cost)
        w_norm = normalize_scores(w_cost)
        q_norm = normalize_scores(q_cost)
        costs = []
        for label in candidates:
            c_r = rssi_norm.get(label, 0.0)
            c_bin = bin_norm[label]
            c_e = e_norm[label]
            c_w = w_norm[label]
            c_q = q_norm[label]
            q_weight = args.q4_weight if packet.segment_q4_reliable[segment_idx] else 0.0
            c_obs = (
                args.rssi_weight * c_r
                + args.bin_weight * c_bin
                + args.energy_weight * c_e
                + args.raw_weight * c_w
                + q_weight * c_q
            )
            costs.append(c_obs)
            rows.append(
                {
                    "segment_index": segment_idx,
                    "candidate_label": label,
                    "C_obs": c_obs,
                    "C_R": c_r,
                    "C_bin": c_bin,
                    "C_bin_raw": raw_bin_costs[label],
                    "C_E": c_e,
                    "C_W": c_w,
                    "C_Q": c_q,
                    "q4_reliable": int(packet.segment_q4_reliable[segment_idx]),
                    "q4_peak_offset": packet.segment_q4_peak_offsets[segment_idx],
                    "q4_peak_to_side_db": packet.segment_q4_peak_to_side_db[segment_idx],
                    "q4_dev_from_packet": packet.segment_q4_dev_from_packet[segment_idx],
                    "template_reliability": templates[label].reliability,
                }
            )
        costs.append(args.garbage_cost)
        rows.append(
            {
                "segment_index": segment_idx,
                "candidate_label": GARBAGE_LABEL,
                "C_obs": args.garbage_cost,
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
            }
        )
        obs_costs.append(costs)
    return obs_costs, rows


def evaluate(
    samples: Sequence[SegmentPacket],
    q4_offsets: Sequence[float],
    chirp_shapes,
    chirp_struct,
    args: argparse.Namespace,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    counts = Counter(labels)
    eval_indices = [idx for idx, label in enumerate(labels) if counts[label] >= 2]
    rng = random.Random(args.seed)
    correct = Counter()
    topk_contains = 0
    predictions = []
    candidate_rows = []
    segment_rows = []
    template_audit = []

    for test_index in eval_indices:
        sample = samples[test_index]
        train_indices = [idx for idx in eval_indices if idx != test_index]
        rssi_ranked = base.class_rank(rssi_rows, labels, train_indices, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        rssi_pred = candidates[0]
        topk_contains += int(sample.label in candidates)
        correct["rssi"] += int(rssi_pred == sample.label)
        templates = build_templates(samples, labels, train_indices, chirp_shapes, chirp_struct, args)
        if args.audit_templates and len(template_audit) < args.audit_template_limit:
            for label in candidates:
                tmpl = templates[label]
                template_audit.append(
                    {
                        "sample_index": test_index,
                        "candidate_label": label,
                        "n_packets": tmpl.n_packets,
                        "alpha_shrink": tmpl.alpha_shrink,
                        "reliability": tmpl.reliability,
                        "chirp_source": tmpl.chirp_source,
                        **{f"mu_{offset:+.0f}": tmpl.mu[j] for j, offset in enumerate(RAW_OFFSETS)},
                        **{f"var_{offset:+.0f}": tmpl.var[j] for j, offset in enumerate(RAW_OFFSETS)},
                    }
                )
        rssi_costs = {label: score for label, score in rssi_ranked if label in candidates}
        segment_prototypes = build_segment_prototypes(samples, labels, train_indices)
        obs_costs, rows = build_observation_costs_v2(
            sample,
            candidates,
            rssi_costs,
            templates,
            segment_prototypes,
            q4_offsets,
            args,
        )
        for row in rows:
            row.update({"sample_index": test_index, "file_name": sample.file_name, "packet_index": sample.packet_index, "true_label": sample.label})
            segment_rows.append(row)
        result = run_aco_v2_for_packet(obs_costs, candidates, templates, args, rng)
        for key in ["path_mode", "pheromone", "vote"]:
            correct[key] += int(result[f"{key}_label"] == sample.label)
        for label in candidates:
            candidate_rows.append(
                {
                    "sample_index": test_index,
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                    "candidate_label": label,
                    "self_pheromone": result["self_pheromone"].get(label, 0.0),
                    "elite_vote": result["elite_vote"].get(label, 0.0),
                    "template_reliability": templates[label].reliability,
                    "alpha_shrink": templates[label].alpha_shrink,
                    "chirp_source": templates[label].chirp_source,
                }
            )
        predictions.append(
            {
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "true_display": point_display(sample.label),
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
        "q4_reliable_segment_count": sum(
            sum(1 for flag in samples[idx].segment_q4_reliable if flag)
            for idx in eval_indices
        ),
        "segment_count_total": sum(len(samples[idx].segment_q4_reliable) for idx in eval_indices),
    }
    if metrics["segment_count_total"]:
        metrics["q4_reliable_segment_rate"] = metrics["q4_reliable_segment_count"] / metrics["segment_count_total"]
    return metrics, predictions, candidate_rows, segment_rows + template_audit


def prepare_chirp_fields(args: argparse.Namespace, labels: Sequence[str]) -> tuple[dict[str, tuple[list[float], str]], dict[str, tuple[dict, str]], dict]:
    locations = read_location_meta(args.location_csv)
    measured_shapes = read_chirp_shapes(args.chirp_template_csv)
    measured_struct = read_chirp_structure(args.chirp_structure_csv)
    default_shape = [0.0 for _ in RAW_OFFSETS]
    default_struct = {"k_ratio": 1.0, "tau_rms": 0.0}
    shape_field = interpolate_field(labels, measured_shapes, locations, default_shape)
    struct_field = interpolate_field(labels, measured_struct, locations, default_struct)
    metadata = {
        "measured_chirp_template_count": len(measured_shapes),
        "measured_chirp_structure_count": len(measured_struct),
        "labels_requiring_template": len(set(labels)),
        "template_sources": dict(Counter(source for _value, source in shape_field.values())),
        "structure_sources": dict(Counter(source for _value, source in struct_field.values())),
    }
    return shape_field, struct_field, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--chirp-template-csv", type=Path, default=DEFAULT_CHIRP_TEMPLATE_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=DEFAULT_CHIRP_STRUCT_CSV)
    parser.add_argument("--location-csv", type=Path, default=DEFAULT_LOCATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--segment-count", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
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
    parser.add_argument("--audit-templates", action="store_true", default=True)
    parser.add_argument("--audit-template-limit", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rssi_packets = base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, thresholds = base.read_symbol_packets(args.spectrum_csv, args)
    base_samples = base.align_samples(rssi_packets, symbol_packets)
    samples = build_segment_packets(base_samples, args.segment_count)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = prepare_chirp_fields(args, labels)
    metrics, predictions, candidate_rows, detail_rows = evaluate(samples, q4_offsets, chirp_shapes, chirp_struct, args)

    prediction_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "true_display",
        "rssi_top1_label",
        "rssi_top1_correct",
        "rssi_topk_candidates",
        "true_in_rssi_topk",
        "aco_path_mode_label",
        "aco_path_mode_correct",
        "aco_pheromone_label",
        "aco_pheromone_correct",
        "aco_vote_label",
        "aco_vote_correct",
        "best_path_cost",
        "best_path_labels",
        "best_path_garbage_count",
    ]
    write_csv(args.output_dir / "aco_v2_predictions.csv", predictions, prediction_fields)
    candidate_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "candidate_label",
        "self_pheromone",
        "elite_vote",
        "template_reliability",
        "alpha_shrink",
        "chirp_source",
    ]
    write_csv(args.output_dir / "aco_v2_candidate_scores.csv", candidate_rows, candidate_fields)
    if detail_rows:
        detail_fields = sorted({key for row in detail_rows for key in row})
        preferred = [
            "sample_index",
            "file_name",
            "packet_index",
            "true_label",
            "segment_index",
            "candidate_label",
            "C_obs",
            "C_R",
            "C_bin",
            "C_bin_raw",
            "C_E",
            "C_W",
            "C_Q",
            "q4_reliable",
        ]
        detail_fields = preferred + [key for key in detail_fields if key not in preferred]
        write_csv(args.output_dir / "aco_v2_segment_costs_and_templates.csv", detail_rows, detail_fields)
    write_csv(args.output_dir / "aco_v2_summary.csv", [metrics], list(metrics.keys()))
    payload = {
        "method": {
            "name": "ACO 2.0: original ACO observations plus chirp-shrinkage bin likelihood",
            "source": "external_design_notes/蚁群算法2.0.md",
            "notes": [
                "Uses chirp-projected LoRa bin templates from 20260626_chirp_lora_bin_projection/02_chirp_synth_point_bins.csv.",
                "Missing chirp templates are filled by same-corridor/same-visibility interpolation or nearest measured chirp template.",
                "Four preamble groups are noisy observations of one frozen packet channel.",
                "Keeps original ACO energy/raw/q4 observation terms, then adds the 2.0 bin Gaussian likelihood, garbage state, dynamic switching, and reliability-weighted self pheromone.",
            ],
        },
        "inputs": {
            "rssi_csv": str(args.rssi_csv),
            "spectrum_csv": str(args.spectrum_csv),
            "chirp_template_csv": str(args.chirp_template_csv),
            "chirp_structure_csv": str(args.chirp_structure_csv),
            "location_csv": str(args.location_csv),
        },
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "symbol_thresholds": thresholds,
        "chirp_template_field": chirp_metadata,
        "metrics": metrics,
    }
    with (args.output_dir / "aco_v2_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
