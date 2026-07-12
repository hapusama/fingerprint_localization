#!/usr/bin/env python3
"""Packet-internal ant colony optimization localization trial.

The ant path is not a time trajectory. For one packet, each preamble symbol is a
layer, and each layer chooses one RSSI+ Top-K location hypothesis. A good path
should mostly stay on one location while tolerating a few noisy symbols.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "fingerprint_localization"
DATA_ROOT = PACKAGE_ROOT / "data" / "mainline_202607"
DEFAULT_RSSI_CSV = DATA_ROOT / "inputs" / "rssi_plus_packet_level_54points.csv"
DEFAULT_SPECTRUM_CSV = DATA_ROOT / "external" / "subbin_spectrum_long.csv"
DEFAULT_CHIRP_CSV = DATA_ROOT / "features" / "chirp_point_multipath_structure_features.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3" / "output_aco"

EPS = 1e-12
RSSI_PLUS_COLUMNS = [
    "snr",
    "realtime_average_rssi",
    "median_rssi",
    "mode_rssi",
    "rssi_variance",
    "residual",
]
RAW_OFFSETS = [-2.0, -1.0, 0.0, 1.0, 2.0]
ZW_COLUMNS = ["E0", "C_peak", "R_side", "R_asym", "W_peak"]
PacketKey = Tuple[str, int]


@dataclass
class SymbolObservation:
    symbol_index: int
    raw_bins: List[float]
    zw: List[float]
    q4_curve: List[float]
    q4_peak_offset: float
    q4_peak_to_side_db: float
    q4_dev_from_packet: float
    q4_reliable: bool


@dataclass
class PacketSample:
    key: PacketKey
    file_name: str
    packet_index: int
    label: str
    rssi_plus: List[float]
    symbols: List[SymbolObservation]


def file_stem(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_int(value: object) -> int:
    return int(float(value))


def point_label(corridor_id: object, position_id: object) -> str:
    return f"{parse_int(corridor_id)}_{parse_int(position_id)}"


def natural_label_key(label: str) -> Tuple[int, int]:
    corridor, position = label.split("_", 1)
    return int(corridor), int(position)


def point_display(label: str) -> str:
    if not label:
        return ""
    corridor, position = label.split("_", 1)
    return f"c{corridor}p{position}"


def quantile(values: Sequence[float], q: float) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def iqr(values: Sequence[float]) -> float:
    return quantile(values, 0.75) - quantile(values, 0.25)


def safe_iqr(values: Sequence[float]) -> float:
    value = iqr(values)
    return value if value > EPS else 1.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def read_csv_dict(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_rssi_packets(path: Path) -> Dict[PacketKey, dict]:
    packets: Dict[PacketKey, dict] = {}
    for row in read_csv_dict(path):
        key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
        packets[key] = {
            "file_name": row["file_name"],
            "packet_index": key[1],
            "label": row["position_key"],
            "rssi_plus": [parse_float(row[col]) for col in RSSI_PLUS_COLUMNS],
        }
    return packets


def raw_structure(raw_bins: Sequence[float]) -> List[float]:
    a_m2, a_m1, a0, a_p1, a_p2 = [max(0.0, float(v)) for v in raw_bins]
    total = a_m2 + a_m1 + a0 + a_p1 + a_p2
    side = a_m2 + a_m1 + a_p1 + a_p2
    left = a_m2 + a_m1
    right = a_p1 + a_p2
    second_moment = sum((offset ** 2) * value for offset, value in zip(RAW_OFFSETS, raw_bins))
    return [
        math.log(a0 + EPS),
        a0 / (total + EPS),
        math.log((side + EPS) / (a0 + EPS)),
        math.log((right + EPS) / (left + EPS)),
        total / (second_moment + EPS),
    ]


def read_symbol_packets(path: Path, args: argparse.Namespace) -> Tuple[Dict[PacketKey, dict], List[float], dict]:
    raw_by_symbol: Dict[PacketKey, Dict[int, Dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    q4_by_symbol: Dict[PacketKey, Dict[int, Dict[float, dict]]] = defaultdict(lambda: defaultdict(dict))
    meta: Dict[PacketKey, dict] = {}
    q4_offsets_seen = set()

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = parse_int(row["q"])
            offset = round(parse_float(row["subbin_offset"]), 6)
            key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
            symbol_id = parse_int(row.get("local_symbol_index", row.get("preamble_symbol_index", 0)))
            meta[key] = {
                "file_name": row["file_name"],
                "packet_index": key[1],
                "label": point_label(row["corridor_id"], row["position_id"]),
            }
            if q == 1 and offset in RAW_OFFSETS:
                raw_by_symbol[key][symbol_id][offset] = parse_float(row["mag_raw"])
            elif q == 4:
                q4_offsets_seen.add(offset)
                q4_by_symbol[key][symbol_id][offset] = {
                    "db": parse_float(row["mag_db_rel_peak"]),
                    "mag": parse_float(row["mag_raw"]),
                }

    q4_offsets = sorted(q4_offsets_seen)
    preliminary: Dict[PacketKey, dict] = {}
    all_center = []
    all_q4_dev = []
    for key in sorted(set(raw_by_symbol) & set(q4_by_symbol)):
        complete_symbols = []
        for symbol_id in sorted(set(raw_by_symbol[key]) & set(q4_by_symbol[key])):
            raw_bins_map = raw_by_symbol[key][symbol_id]
            q4_bins_map = q4_by_symbol[key][symbol_id]
            if not all(offset in raw_bins_map for offset in RAW_OFFSETS):
                continue
            if not all(offset in q4_bins_map for offset in q4_offsets):
                continue
            raw_bins = [raw_bins_map[offset] for offset in RAW_OFFSETS]
            q4_curve = [q4_bins_map[offset]["db"] for offset in q4_offsets]
            q4_mags = [q4_bins_map[offset]["mag"] for offset in q4_offsets]
            peak_idx = max(range(len(q4_mags)), key=lambda idx: q4_mags[idx])
            side_db = [db for db, offset in zip(q4_curve, q4_offsets) if abs(offset) >= 1.0]
            peak_to_side = max(q4_curve) - mean(side_db) if side_db else 0.0
            complete_symbols.append(
                {
                    "symbol_index": symbol_id,
                    "raw_bins": raw_bins,
                    "zw": raw_structure(raw_bins),
                    "q4_curve": q4_curve,
                    "q4_peak_offset": q4_offsets[peak_idx],
                    "q4_peak_to_side_db": peak_to_side,
                }
            )
            all_center.append(raw_bins[2])
        if complete_symbols:
            packet_median = [
                median([symbol["q4_curve"][idx] for symbol in complete_symbols])
                for idx in range(len(q4_offsets))
            ]
            for symbol in complete_symbols:
                dev = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(symbol["q4_curve"], packet_median)) / len(q4_offsets)
                )
                symbol["q4_dev_from_packet"] = dev
                all_q4_dev.append(dev)
            preliminary[key] = {**meta[key], "symbols": complete_symbols}

    peak_threshold = args.peak_threshold
    if peak_threshold is None:
        peak_threshold = quantile(all_center, args.auto_peak_quantile)
    q4_dev_threshold = args.q4_dev_threshold
    if q4_dev_threshold is None:
        q4_dev_threshold = quantile(all_q4_dev, args.auto_q4_dev_quantile)

    packets: Dict[PacketKey, dict] = {}
    for key, packet in preliminary.items():
        symbols: List[SymbolObservation] = []
        for symbol in packet["symbols"]:
            a0 = symbol["raw_bins"][2]
            reliable = (
                abs(symbol["q4_peak_offset"]) < args.q4_peak_offset_max
                and a0 > peak_threshold
                and symbol["q4_peak_to_side_db"] > args.q4_peak_to_side_threshold
                and symbol["q4_dev_from_packet"] < q4_dev_threshold
            )
            symbols.append(
                SymbolObservation(
                    symbol_index=symbol["symbol_index"],
                    raw_bins=symbol["raw_bins"],
                    zw=symbol["zw"],
                    q4_curve=symbol["q4_curve"],
                    q4_peak_offset=symbol["q4_peak_offset"],
                    q4_peak_to_side_db=symbol["q4_peak_to_side_db"],
                    q4_dev_from_packet=symbol["q4_dev_from_packet"],
                    q4_reliable=reliable,
                )
            )
        packets[key] = {**packet, "symbols": symbols}

    thresholds = {
        "peak_threshold": peak_threshold,
        "q4_dev_threshold": q4_dev_threshold,
        "q4_peak_offset_max": args.q4_peak_offset_max,
        "q4_peak_to_side_threshold": args.q4_peak_to_side_threshold,
    }
    return packets, q4_offsets, thresholds


def align_samples(rssi_packets: Dict[PacketKey, dict], symbol_packets: Dict[PacketKey, dict]) -> List[PacketSample]:
    samples: List[PacketSample] = []
    common = sorted(
        set(rssi_packets) & set(symbol_packets),
        key=lambda key: (natural_label_key(rssi_packets[key]["label"]), key[1], key[0]),
    )
    for key in common:
        if rssi_packets[key]["label"] != symbol_packets[key]["label"]:
            raise ValueError(f"Label mismatch for {key}: {rssi_packets[key]['label']} vs {symbol_packets[key]['label']}")
        samples.append(
            PacketSample(
                key=key,
                file_name=symbol_packets[key]["file_name"],
                packet_index=key[1],
                label=rssi_packets[key]["label"],
                rssi_plus=rssi_packets[key]["rssi_plus"],
                symbols=symbol_packets[key]["symbols"],
            )
        )
    return samples


def zscore_stats(rows: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    dim = len(rows[0])
    means = [sum(row[j] for row in rows) / len(rows) for j in range(dim)]
    stds = []
    for j in range(dim):
        variance = sum((row[j] - means[j]) ** 2 for row in rows) / len(rows)
        std = math.sqrt(variance)
        stds.append(std if std > EPS else 1.0)
    return means, stds


def squared_distance(a: Sequence[float], b: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> float:
    return sum((((a[j] - means[j]) / stds[j]) - ((b[j] - means[j]) / stds[j])) ** 2 for j in range(len(a)))


def class_rank(
    rows: Sequence[Sequence[float]],
    labels: Sequence[str],
    train_indices: Sequence[int],
    test_index: int,
    class_neighbor_k: int,
) -> List[Tuple[str, float]]:
    train_rows = [rows[idx] for idx in train_indices]
    means, stds = zscore_stats(train_rows)
    ranked = []
    for label in sorted({labels[idx] for idx in train_indices}, key=natural_label_key):
        distances = [
            squared_distance(rows[idx], rows[test_index], means, stds)
            for idx in train_indices
            if labels[idx] == label
        ]
        if not distances:
            continue
        distances.sort()
        k_eff = min(class_neighbor_k, len(distances))
        ranked.append((label, sum(distances[:k_eff]) / k_eff))
    ranked.sort(key=lambda item: (item[1], natural_label_key(item[0])))
    return ranked


def robust_vector_prototypes(rows_by_label: Dict[str, List[Sequence[float]]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for label, rows in rows_by_label.items():
        dim = len(rows[0])
        out[label] = {
            "median": [median([row[j] for row in rows]) for j in range(dim)],
            "iqr": [safe_iqr([row[j] for row in rows]) for j in range(dim)],
            "count": len(rows),
        }
    return out


def robust_scalar_prototypes(rows_by_label: Dict[str, List[float]]) -> Dict[str, dict]:
    return {
        label: {"median": median(rows), "iqr": safe_iqr(rows), "count": len(rows)}
        for label, rows in rows_by_label.items()
    }


def build_symbol_prototypes(samples: Sequence[PacketSample], labels: Sequence[str], train_indices: Sequence[int]) -> dict:
    energy_by_label: Dict[str, List[float]] = defaultdict(list)
    zw_by_label: Dict[str, List[Sequence[float]]] = defaultdict(list)
    q4_by_label: Dict[str, List[Sequence[float]]] = defaultdict(list)
    for idx in train_indices:
        label = labels[idx]
        for symbol in samples[idx].symbols:
            energy_by_label[label].append(symbol.zw[0])
            zw_by_label[label].append(symbol.zw)
            q4_by_label[label].append(symbol.q4_curve)
    return {
        "energy": robust_scalar_prototypes(energy_by_label),
        "zw": robust_vector_prototypes(zw_by_label),
        "q4": robust_vector_prototypes(q4_by_label),
    }


def robust_vector_cost(value: Sequence[float], prototype: dict) -> float:
    return sum(((value[j] - prototype["median"][j]) / (prototype["iqr"][j] + EPS)) ** 2 for j in range(len(value))) / len(value)


def robust_scalar_cost(value: float, prototype: dict) -> float:
    return ((value - prototype["median"]) / (prototype["iqr"] + EPS)) ** 2


def interpolate_curve(curve: Sequence[float], offsets: Sequence[float], x: float) -> float:
    if x <= offsets[0]:
        return curve[0]
    if x >= offsets[-1]:
        return curve[-1]
    for idx in range(len(offsets) - 1):
        left = offsets[idx]
        right = offsets[idx + 1]
        if left <= x <= right:
            if abs(right - left) <= EPS:
                return curve[idx]
            frac = (x - left) / (right - left)
            return curve[idx] * (1.0 - frac) + curve[idx + 1] * frac
    return curve[-1]


def affine_mse(observed: Sequence[float], model: Sequence[float]) -> float:
    obs_mean = mean(observed)
    model_mean = mean(model)
    var_model = sum((v - model_mean) ** 2 for v in model)
    cov = sum((m - model_mean) * (o - obs_mean) for m, o in zip(model, observed))
    scale = cov / (var_model + EPS)
    scale = min(max(scale, 0.2), 2.5)
    offset = obs_mean - scale * model_mean
    return sum((o - (offset + scale * m)) ** 2 for o, m in zip(observed, model)) / len(observed)


def q4_shape_cost(observed: Sequence[float], prototype: dict, offsets: Sequence[float], shift_grid: Sequence[float]) -> float:
    proto_curve = prototype["median"]
    best = float("inf")
    for shift in shift_grid:
        shifted = [interpolate_curve(proto_curve, offsets, offset - shift) for offset in offsets]
        best = min(best, affine_mse(observed, shifted))
    return best


def normalize_costs(costs: Dict[str, float]) -> Dict[str, float]:
    if not costs:
        return {}
    lo = min(costs.values())
    hi = max(costs.values())
    if hi - lo <= EPS:
        return {label: 0.0 for label in costs}
    return {label: (value - lo) / (hi - lo) for label, value in costs.items()}


def read_chirp_priors(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    raw_rows = read_csv_dict(path)
    labels = []
    vectors: Dict[str, List[float]] = {}
    stable_scores: Dict[str, float] = {}
    fields = [
        "stable_secondary_path_count",
        "secondary_effective_power_sum",
        "equivalent_rms_delay_us",
        "effective_path_number",
        "unstable_secondary_peak_load",
        "post_to_precursor_power_ratio",
    ]
    for row in raw_rows:
        label = row.get("position_key") or point_label(row["corridor_id"], row["location_id"])
        labels.append(label)
        vectors[label] = [parse_float(row.get(field), 0.0) for field in fields]
        stable_scores[label] = parse_float(row.get("stable_detection_explained_fraction"), 0.0)
    if not vectors:
        return {}
    dim = len(fields)
    med = [median([vectors[label][j] for label in vectors]) for j in range(dim)]
    spread = [safe_iqr([vectors[label][j] for label in vectors]) for j in range(dim)]
    priors = {}
    for label, vector in vectors.items():
        priors[label] = {
            "vector": [(vector[j] - med[j]) / (spread[j] + EPS) for j in range(dim)],
            "stable": max(0.0, min(1.0, stable_scores.get(label, 0.0))),
        }
    return priors


def chirp_separability(candidates: Sequence[str], chirp_priors: Dict[str, dict]) -> float:
    distances = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            if left not in chirp_priors or right not in chirp_priors:
                continue
            lv = chirp_priors[left]["vector"]
            rv = chirp_priors[right]["vector"]
            distances.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(lv, rv)) / len(lv)))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


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


def path_cost(path: Sequence[int], obs_costs: Sequence[Sequence[float]], args: argparse.Namespace) -> float:
    obs = sum(obs_costs[s][choice] for s, choice in enumerate(path))
    switches = sum(1 for s in range(1, len(path)) if path[s] != path[s - 1])
    unique_count = len(set(path))
    return obs + args.switch_penalty * switches + args.diversity_penalty * unique_count


def run_aco_for_packet(
    obs_costs: Sequence[Sequence[float]],
    candidates: Sequence[str],
    chirp_priors: Dict[str, dict],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict:
    k = len(candidates)
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

    best_path: List[int] = []
    best_cost = float("inf")
    elite_vote = [0.0] * k

    for _ in range(args.iterations):
        paths = []
        for _ant in range(args.ants):
            first_weights = [math.exp(-args.heuristic_power * obs_costs[0][j]) for j in range(k)]
            path = [weighted_choice(first_weights, rng)]
            for s in range(1, len(obs_costs)):
                prev = path[-1]
                weights = []
                for j in range(k):
                    switch_factor = math.exp(-args.switch_penalty) if j != prev else 1.0
                    heuristic = math.exp(-obs_costs[s][j])
                    weights.append((pheromone[prev][j] ** args.pheromone_power) * (heuristic ** args.heuristic_power) * switch_factor)
                path.append(weighted_choice(weights, rng))
            cost = path_cost(path, obs_costs, args)
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
    path_mode_idx = min(
        range(k),
        key=lambda idx: (-path_counts.get(idx, 0), candidates[idx]),
    )
    pheromone_idx = max(range(k), key=lambda idx: (pheromone[idx][idx], -idx))
    vote_idx = max(range(k), key=lambda idx: (elite_vote[idx], -idx))
    return {
        "best_path": best_path,
        "best_cost": best_cost,
        "path_mode_label": candidates[path_mode_idx],
        "pheromone_label": candidates[pheromone_idx],
        "vote_label": candidates[vote_idx],
        "path_mode_counts": dict(path_counts),
        "self_pheromone": {candidates[idx]: pheromone[idx][idx] for idx in range(k)},
        "elite_vote": {candidates[idx]: elite_vote[idx] for idx in range(k)},
    }


def build_observation_costs(
    sample: PacketSample,
    candidates: Sequence[str],
    rssi_costs: Dict[str, float],
    prototypes: dict,
    q4_offsets: Sequence[float],
    chirp_weight: float,
    args: argparse.Namespace,
) -> Tuple[List[List[float]], List[dict]]:
    shift_grid = [float(part.strip()) for part in args.q4_shift_grid.split(",") if part.strip()]
    rssi_norm = normalize_costs({label: rssi_costs[label] for label in candidates})
    obs_costs: List[List[float]] = []
    rows = []
    for symbol in sample.symbols:
        e_cost = {
            label: robust_scalar_cost(symbol.zw[0], prototypes["energy"][label])
            for label in candidates
        }
        w_cost = {
            label: robust_vector_cost(symbol.zw, prototypes["zw"][label])
            for label in candidates
        }
        if symbol.q4_reliable:
            q_cost = {
                label: q4_shape_cost(symbol.q4_curve, prototypes["q4"][label], q4_offsets, shift_grid)
                for label in candidates
            }
        else:
            q_cost = {label: 0.0 for label in candidates}

        e_norm = normalize_costs(e_cost)
        w_norm = normalize_costs(w_cost)
        q_norm = normalize_costs(q_cost)
        costs_for_symbol = []
        for label in candidates:
            q_weight = args.q4_weight * (1.0 + args.chirp_q4_boost * chirp_weight) if symbol.q4_reliable else 0.0
            cost = (
                args.rssi_weight * rssi_norm.get(label, 0.0)
                + args.energy_weight * e_norm.get(label, 0.0)
                + args.raw_weight * w_norm.get(label, 0.0)
                + q_weight * q_norm.get(label, 0.0)
            )
            costs_for_symbol.append(cost)
            rows.append(
                {
                    "symbol_index": symbol.symbol_index,
                    "candidate_label": label,
                    "C_obs": cost,
                    "C_R": rssi_norm.get(label, 0.0),
                    "C_E": e_norm.get(label, 0.0),
                    "C_W": w_norm.get(label, 0.0),
                    "C_Q": q_norm.get(label, 0.0),
                    "q4_reliable": int(symbol.q4_reliable),
                    "q4_peak_offset": symbol.q4_peak_offset,
                    "q4_peak_to_side_db": symbol.q4_peak_to_side_db,
                    "q4_dev_from_packet": symbol.q4_dev_from_packet,
                }
            )
        obs_costs.append(costs_for_symbol)
    return obs_costs, rows


def evaluate(samples: Sequence[PacketSample], q4_offsets: Sequence[float], chirp_priors: Dict[str, dict], thresholds: dict, args: argparse.Namespace) -> Tuple[dict, List[dict], List[dict], List[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    counts = Counter(labels)
    eval_indices = [idx for idx, label in enumerate(labels) if counts[label] >= 2]
    rng = random.Random(args.seed)
    predictions = []
    score_rows = []
    symbol_rows = []
    correct = {"rssi": 0, "path_mode": 0, "pheromone": 0, "vote": 0}
    topk_contains = 0
    q4_reliable_total = 0
    symbol_total = 0

    for test_index in eval_indices:
        sample = samples[test_index]
        train_indices = [idx for idx in eval_indices if idx != test_index]
        rssi_ranked = class_rank(rssi_rows, labels, train_indices, test_index, args.rssi_class_k)
        candidates = [label for label, _ in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        rssi_pred = candidates[0]
        correct["rssi"] += int(rssi_pred == sample.label)
        topk_contains += int(sample.label in candidates)

        symbol_train_prototypes = build_symbol_prototypes(samples, labels, train_indices)
        rssi_costs = {label: score for label, score in rssi_ranked if label in candidates}
        chirp_weight = chirp_separability(candidates, chirp_priors)
        obs_costs, per_symbol_rows = build_observation_costs(
            sample,
            candidates,
            rssi_costs,
            symbol_train_prototypes,
            q4_offsets,
            chirp_weight,
            args,
        )
        for row in per_symbol_rows:
            row.update(
                {
                    "sample_index": test_index,
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                }
            )
            symbol_rows.append(row)
        q4_reliable_total += sum(1 for symbol in sample.symbols if symbol.q4_reliable)
        symbol_total += len(sample.symbols)

        aco = run_aco_for_packet(obs_costs, candidates, chirp_priors, args, rng)
        path_labels = [candidates[idx] for idx in aco["best_path"]]
        path_mode = aco["path_mode_label"]
        pheromone = aco["pheromone_label"]
        vote = aco["vote_label"]
        correct["path_mode"] += int(path_mode == sample.label)
        correct["pheromone"] += int(pheromone == sample.label)
        correct["vote"] += int(vote == sample.label)
        for label in candidates:
            score_rows.append(
                {
                    "sample_index": test_index,
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                    "candidate_label": label,
                    "candidate_display": point_display(label),
                    "self_pheromone": aco["self_pheromone"].get(label, 0.0),
                    "elite_vote": aco["elite_vote"].get(label, 0.0),
                    "chirp_candidate_separability": chirp_weight,
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
                "rssi_top1_display": point_display(rssi_pred),
                "rssi_top1_correct": int(rssi_pred == sample.label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "aco_path_mode_label": path_mode,
                "aco_path_mode_display": point_display(path_mode),
                "aco_path_mode_correct": int(path_mode == sample.label),
                "aco_pheromone_label": pheromone,
                "aco_pheromone_display": point_display(pheromone),
                "aco_pheromone_correct": int(pheromone == sample.label),
                "aco_vote_label": vote,
                "aco_vote_display": point_display(vote),
                "aco_vote_correct": int(vote == sample.label),
                "best_path_cost": aco["best_cost"],
                "best_path_labels": ";".join(path_labels),
                "best_path_unique_count": len(set(path_labels)),
                "q4_reliable_symbols": sum(1 for symbol in sample.symbols if symbol.q4_reliable),
                "symbol_count": len(sample.symbols),
                "chirp_candidate_separability": chirp_weight,
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
        "aco_vote_gain_vs_rssi_top1": correct["vote"] - correct["rssi"],
        "q4_reliable_symbol_count": q4_reliable_total,
        "symbol_count": symbol_total,
        "q4_reliable_symbol_rate": q4_reliable_total / symbol_total if symbol_total else 0.0,
        **thresholds,
    }
    return metrics, predictions, score_rows, symbol_rows


def build_symbol_feature_rows(samples: Sequence[PacketSample]) -> List[dict]:
    rows = []
    for sample in samples:
        for symbol in sample.symbols:
            row = {
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "position_key": sample.label,
                "symbol_index": symbol.symbol_index,
                "q4_reliable": int(symbol.q4_reliable),
                "q4_peak_offset": symbol.q4_peak_offset,
                "q4_peak_to_side_db": symbol.q4_peak_to_side_db,
                "q4_dev_from_packet": symbol.q4_dev_from_packet,
            }
            row.update({f"raw_bin_{int(offset):+d}": symbol.raw_bins[idx] for idx, offset in enumerate(RAW_OFFSETS)})
            row.update({name: symbol.zw[idx] for idx, name in enumerate(ZW_COLUMNS)})
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=DEFAULT_CHIRP_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rssi-weight", type=float, default=0.45)
    parser.add_argument("--energy-weight", type=float, default=0.20)
    parser.add_argument("--raw-weight", type=float, default=0.55)
    parser.add_argument("--q4-weight", type=float, default=0.15)
    parser.add_argument("--chirp-q4-boost", type=float, default=0.15)
    parser.add_argument("--chirp-self-loop-boost", type=float, default=0.20)
    parser.add_argument("--switch-penalty", type=float, default=0.70)
    parser.add_argument("--diversity-penalty", type=float, default=0.20)
    parser.add_argument("--pheromone-power", type=float, default=1.0)
    parser.add_argument("--heuristic-power", type=float, default=1.4)
    parser.add_argument("--evaporation", type=float, default=0.25)
    parser.add_argument("--tau-stay", type=float, default=1.4)
    parser.add_argument("--tau-switch", type=float, default=0.35)
    parser.add_argument("--min-pheromone", type=float, default=1e-4)
    parser.add_argument("--q4-shift-grid", default="-0.25,0,0.25")
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rssi_packets = read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, thresholds = read_symbol_packets(args.spectrum_csv, args)
    samples = align_samples(rssi_packets, symbol_packets)
    if not samples:
        raise RuntimeError("No aligned RSSI+/symbol-level packet samples found.")
    chirp_priors = read_chirp_priors(args.chirp_structure_csv)

    symbol_feature_rows = build_symbol_feature_rows(samples)
    symbol_feature_fields = [
        "file_name",
        "packet_index",
        "position_key",
        "symbol_index",
        "q4_reliable",
        "q4_peak_offset",
        "q4_peak_to_side_db",
        "q4_dev_from_packet",
    ]
    symbol_feature_fields += [f"raw_bin_{int(offset):+d}" for offset in RAW_OFFSETS]
    symbol_feature_fields += ZW_COLUMNS
    write_csv(args.output_dir / "aco_symbol_features.csv", symbol_feature_rows, symbol_feature_fields)

    metrics, predictions, score_rows, symbol_cost_rows = evaluate(samples, q4_offsets, chirp_priors, thresholds, args)
    summary_fields = [
        "packet_count",
        "location_count",
        "top_k",
        "rssi_top1_correct",
        "rssi_top1_accuracy",
        "rssi_topk_contains_true",
        "rssi_topk_recall",
        "aco_path_mode_correct",
        "aco_path_mode_accuracy",
        "aco_pheromone_correct",
        "aco_pheromone_accuracy",
        "aco_vote_correct",
        "aco_vote_accuracy",
        "aco_vote_gain_vs_rssi_top1",
        "q4_reliable_symbol_count",
        "symbol_count",
        "q4_reliable_symbol_rate",
        "peak_threshold",
        "q4_dev_threshold",
        "q4_peak_offset_max",
        "q4_peak_to_side_threshold",
    ]
    write_csv(args.output_dir / "aco_summary.csv", [metrics], summary_fields)

    prediction_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "true_display",
        "rssi_top1_label",
        "rssi_top1_display",
        "rssi_top1_correct",
        "rssi_topk_candidates",
        "true_in_rssi_topk",
        "aco_path_mode_label",
        "aco_path_mode_display",
        "aco_path_mode_correct",
        "aco_pheromone_label",
        "aco_pheromone_display",
        "aco_pheromone_correct",
        "aco_vote_label",
        "aco_vote_display",
        "aco_vote_correct",
        "best_path_cost",
        "best_path_labels",
        "best_path_unique_count",
        "q4_reliable_symbols",
        "symbol_count",
        "chirp_candidate_separability",
    ]
    write_csv(args.output_dir / "aco_predictions.csv", predictions, prediction_fields)

    score_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "candidate_label",
        "candidate_display",
        "self_pheromone",
        "elite_vote",
        "chirp_candidate_separability",
    ]
    write_csv(args.output_dir / "aco_candidate_scores.csv", score_rows, score_fields)

    symbol_cost_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "symbol_index",
        "candidate_label",
        "C_obs",
        "C_R",
        "C_E",
        "C_W",
        "C_Q",
        "q4_reliable",
        "q4_peak_offset",
        "q4_peak_to_side_db",
        "q4_dev_from_packet",
    ]
    write_csv(args.output_dir / "aco_symbol_candidate_costs.csv", symbol_cost_rows, symbol_cost_fields)

    payload = {
        "inputs": {
            "rssi_csv": str(args.rssi_csv),
            "spectrum_csv": str(args.spectrum_csv),
            "chirp_structure_csv": str(args.chirp_structure_csv),
            "aligned_packet_count": len(samples),
            "aligned_location_count": len(Counter(sample.label for sample in samples)),
            "q4_offsets": q4_offsets,
        },
        "data_policy": "uses trusted files under v2_output only; v2_output_wrong is not consumed",
        "method": {
            "name": "packet-internal ant colony optimization",
            "path": "one layer per preamble symbol; each node is one RSSI+ Top-K location hypothesis",
            "node_cost": "C_obs = alpha*C_R + beta*C_E + gamma*C_W + delta_s*C_Q",
            "q4_gate": "q4 segment cost is enabled only when center peak, energy, peak-to-side, and packet-median deviation gates pass",
            "outputs": "path-mode, self-pheromone, and elite-vote decisions are all reported",
        },
        "parameters": jsonable_args(args),
        "metrics": metrics,
    }
    with (args.output_dir / "aco_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
