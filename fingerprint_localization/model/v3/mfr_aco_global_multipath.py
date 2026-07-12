#!/usr/bin/env python3
"""Global multipath-field guided ACO for LoRa fingerprint localization.

MFR-ACO: Multipath-Field and Raw-bin guided Ant Colony Optimization.

This implements the first executable version from
`external_design_notes/全局多径蚁群.md`.
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
DEFAULT_CHIRP_CSV = DATA_ROOT / "features" / "chirp_point_multipath_structure_features.csv"
DEFAULT_LOCATION_CSV = PACKAGE_ROOT / "docs" / "location_distance_54points.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3" / "output_mfr_aco"

EPS = 1e-12
RAW_OFFSETS = base.RAW_OFFSETS
RSSI_COLUMNS = base.RSSI_PLUS_COLUMNS


@dataclass
class PacketEvidence:
    w_pkt: list[float]
    w_segments: list[list[float]]
    q_raw_sum: float


@dataclass
class MultipathPoint:
    label: str
    corridor_id: int
    location_id: int
    visibility_state: int
    distance_m: float
    z: list[float]
    z_norm: list[float]
    confidence: float
    source: str


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


def median(values: Sequence[float]) -> float:
    return base.median(list(values))


def safe_iqr(values: Sequence[float]) -> float:
    return base.safe_iqr(list(values))


def natural_label_key(label: str) -> tuple[int, int]:
    return base.natural_label_key(label)


def point_display(label: str) -> str:
    return base.point_display(label)


def raw_shape(raw_bins: Sequence[float]) -> list[float]:
    logs = [math.log(max(float(value), EPS)) for value in raw_bins]
    center = sum(logs) / len(logs)
    return [value - center for value in logs]


def segment_bounds(n_symbols: int, segment_count: int) -> list[tuple[int, int]]:
    if n_symbols < segment_count:
        raise ValueError(f"Cannot split {n_symbols} symbols into {segment_count} nonempty groups")
    bounds = []
    for idx in range(segment_count):
        start = round(idx * n_symbols / segment_count)
        end = round((idx + 1) * n_symbols / segment_count)
        bounds.append((start, max(start + 1, min(end, n_symbols))))
    return bounds


def packet_evidence(sample: base.PacketSample, segment_count: int) -> PacketEvidence:
    symbols = sorted(sample.symbols, key=lambda item: item.symbol_index)
    shapes = []
    for start, end in segment_bounds(len(symbols), segment_count):
        chunk = symbols[start:end]
        raw_bins = [
            median([symbol.raw_bins[j] for symbol in chunk])
            for j in range(len(RAW_OFFSETS))
        ]
        shapes.append(raw_shape(raw_bins))
    w_pkt = [
        median([shape[j] for shape in shapes])
        for j in range(len(RAW_OFFSETS))
    ]
    q_raw_sum = sum(
        sum((shape[j] - w_pkt[j]) ** 2 for j in range(len(RAW_OFFSETS)))
        for shape in shapes
    )
    return PacketEvidence(w_pkt=w_pkt, w_segments=shapes, q_raw_sum=q_raw_sum)


def build_all_packet_evidence(samples: Sequence[base.PacketSample], segment_count: int) -> list[PacketEvidence]:
    return [packet_evidence(sample, segment_count) for sample in samples]


def build_raw_prototypes(
    evidence: Sequence[PacketEvidence],
    labels: Sequence[str],
    train_indices: Sequence[int],
) -> dict[str, dict]:
    by_label: dict[str, list[list[float]]] = defaultdict(list)
    for idx in train_indices:
        by_label[labels[idx]].append(evidence[idx].w_pkt)
    prototypes = {}
    for label, rows in by_label.items():
        dim = len(rows[0])
        med = [median([row[j] for row in rows]) for j in range(dim)]
        var = []
        for j in range(dim):
            vals = [row[j] for row in rows]
            mu = sum(vals) / len(vals)
            value = sum((v - mu) ** 2 for v in vals) / len(vals)
            var.append(value if value > EPS else safe_iqr(vals) ** 2 + EPS)
        prototypes[label] = {"median": med, "var": var, "count": len(rows)}
    return prototypes


def raw_distance(w_pkt: Sequence[float], prototype: dict) -> float:
    return sum(
        ((w_pkt[j] - prototype["median"][j]) ** 2) / (prototype["var"][j] + EPS)
        for j in range(len(w_pkt))
    )


def compute_t_q(evidence: Sequence[PacketEvidence], indices: Sequence[int]) -> float:
    values = [evidence[idx].q_raw_sum for idx in indices]
    out = median(values)
    return out if out > EPS else 1.0


def compute_raw_temperature(
    evidence: Sequence[PacketEvidence],
    labels: Sequence[str],
    eval_indices: Sequence[int],
) -> float:
    values = []
    for test_index in eval_indices:
        train_indices = [idx for idx in eval_indices if idx != test_index]
        prototypes = build_raw_prototypes(evidence, labels, train_indices)
        if labels[test_index] in prototypes:
            values.append(raw_distance(evidence[test_index].w_pkt, prototypes[labels[test_index]]))
    out = median(values)
    return out if out > EPS else 1.0


def compute_rssi_temperature(
    rows: Sequence[Sequence[float]],
    labels: Sequence[str],
    eval_indices: Sequence[int],
    class_neighbor_k: int,
) -> float:
    values = []
    for test_index in eval_indices:
        train_indices = [idx for idx in eval_indices if idx != test_index]
        ranked = base.class_rank(rows, labels, train_indices, test_index, class_neighbor_k)
        scores = dict(ranked)
        if labels[test_index] in scores:
            values.append(scores[labels[test_index]])
    out = median(values)
    return out if out > EPS else 1.0


def read_location_rows(path: Path) -> dict[str, dict]:
    out = {}
    for row in read_csv_dict(path):
        label = row["position_key"]
        out[label] = {
            "label": label,
            "corridor_id": parse_int(row["corridor_id"]),
            "location_id": parse_int(row["location_id"]),
            "visibility_state": parse_int(row.get("c_i（NLOS-2，LOS-1，OLOS-0）", 0)),
            "distance_m": parse_float(row.get("distance_m"), 0.0),
        }
    return out


def chirp_z_from_row(row: dict) -> list[float]:
    main_fraction = parse_float(row.get("main_effective_power_fraction"), 1.0)
    diffuse_power = parse_float(row.get("secondary_effective_power_sum"), 0.0)
    tau_rms = parse_float(row.get("equivalent_rms_delay_us"), 0.0)
    k_ratio = main_fraction / (diffuse_power + EPS)
    return [
        math.log1p(max(k_ratio, 0.0)),
        math.log1p(max(diffuse_power, 0.0)),
        math.log1p(max(tau_rms, 0.0)),
    ]


def build_multipath_field(location_csv: Path, chirp_csv: Path) -> tuple[dict[str, MultipathPoint], dict]:
    locations = read_location_rows(location_csv)
    measured: dict[str, list[float]] = {}
    for row in read_csv_dict(chirp_csv):
        label = row.get("position_key")
        if not label:
            label = f"{parse_int(row['corridor_id'])}_{parse_int(row['location_id'])}"
        measured[label] = chirp_z_from_row(row)

    points_raw: dict[str, dict] = {}
    for label, meta in locations.items():
        if label in measured:
            points_raw[label] = {**meta, "z": measured[label], "confidence": 1.0, "source": "measured_chirp"}
            continue
        same_group = [
            (other_meta["distance_m"], other_label, measured[other_label])
            for other_label, other_meta in locations.items()
            if other_label in measured
            and other_meta["corridor_id"] == meta["corridor_id"]
            and other_meta["visibility_state"] == meta["visibility_state"]
        ]
        left = [(dist, lab, z) for dist, lab, z in same_group if dist < meta["distance_m"]]
        right = [(dist, lab, z) for dist, lab, z in same_group if dist > meta["distance_m"]]
        left_best = max(left, default=None, key=lambda item: item[0])
        right_best = min(right, default=None, key=lambda item: item[0])
        if left_best and right_best:
            z = [(left_best[2][j] + right_best[2][j]) / 2.0 for j in range(3)]
            conf = 0.5
            source = f"interpolated:{left_best[1]}|{right_best[1]}"
        elif left_best or right_best:
            near = left_best or right_best
            z = list(near[2])
            conf = 0.5
            source = f"nearest:{near[1]}"
        else:
            z = [0.0, 0.0, 0.0]
            conf = 0.0
            source = "unavailable"
        points_raw[label] = {**meta, "z": z, "confidence": conf, "source": source}

    valid = [item["z"] for item in points_raw.values() if item["confidence"] > 0.0]
    dim = 3
    means = [sum(row[j] for row in valid) / len(valid) for j in range(dim)]
    stds = []
    for j in range(dim):
        variance = sum((row[j] - means[j]) ** 2 for row in valid) / len(valid)
        std = math.sqrt(variance)
        stds.append(std if std > EPS else 1.0)

    field = {}
    for label, item in points_raw.items():
        z_norm = [(item["z"][j] - means[j]) / (stds[j] + EPS) for j in range(dim)]
        field[label] = MultipathPoint(
            label=label,
            corridor_id=item["corridor_id"],
            location_id=item["location_id"],
            visibility_state=item["visibility_state"],
            distance_m=item["distance_m"],
            z=item["z"],
            z_norm=z_norm,
            confidence=item["confidence"],
            source=item["source"],
        )
    metadata = {"z_mean": means, "z_std": stds, "measured_count": len(measured), "field_count": len(field)}
    return field, metadata


def multipath_distance(left: str, right: str, field: dict[str, MultipathPoint]) -> float:
    if left not in field or right not in field:
        return 0.0
    return sum((field[left].z_norm[j] - field[right].z_norm[j]) ** 2 for j in range(3))


def compute_multipath_temperature(field: dict[str, MultipathPoint], labels: Sequence[str]) -> float:
    values = []
    labels = sorted(set(labels), key=natural_label_key)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            if field.get(left, None) is None or field.get(right, None) is None:
                continue
            if field[left].confidence <= 0.0 or field[right].confidence <= 0.0:
                continue
            values.append(multipath_distance(left, right, field))
    out = median(values)
    return out if out > EPS else 1.0


def multipath_kernel(left: str, right: str, field: dict[str, MultipathPoint], t_m: float) -> float:
    return math.exp(-multipath_distance(left, right, field) / (t_m + EPS))


def multipath_reliability(label: str, field: dict[str, MultipathPoint]) -> float:
    item = field.get(label)
    if item is None or item.confidence <= 0.0:
        return 0.0
    raw = item.confidence * math.exp(-0.5 * (item.z_norm[1] + item.z_norm[2]))
    return max(0.0, min(item.confidence, raw))


def candidate_multipath_quantities(
    candidates: Sequence[str],
    field: dict[str, MultipathPoint],
    t_m: float,
) -> dict[str, dict]:
    quantities = {}
    for label in candidates:
        other_distances = [
            multipath_distance(label, other, field)
            for other in candidates
            if other != label
        ]
        min_distance = min(other_distances) if other_distances else 0.0
        sep = 1.0 - math.exp(-min_distance / (t_m + EPS))
        rel = multipath_reliability(label, field)
        quantities[label] = {"rel": rel, "sep": sep}
    return quantities


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi - lo <= EPS:
        return {label: 0.0 for label in scores}
    return {label: (value - lo) / (hi - lo) for label, value in scores.items()}


def path_mode(path: Sequence[int], candidates: Sequence[str]) -> str:
    counts = Counter(path)
    idx = min(range(len(candidates)), key=lambda item: (-counts.get(item, 0), candidates[item]))
    return candidates[idx]


def path_cost(
    path: Sequence[int],
    candidates: Sequence[str],
    eta_r: dict[str, float],
    eta_w: dict[str, float],
    raw_weight: dict[str, float],
    transition_boost: dict[tuple[str, str], float],
    k_m: dict[tuple[str, str], float],
    args: argparse.Namespace,
) -> float:
    labels = [candidates[idx] for idx in path]
    obs_r = sum(-math.log(eta_r[label] + EPS) for label in labels)
    mode_label = path_mode(path, candidates)
    c_w = -args.kappa_w * raw_weight[mode_label] * math.log(eta_w[mode_label] + EPS)
    transition = 0.0
    for prev, cur in zip(labels, labels[1:]):
        t_factor = transition_boost[(prev, cur)]
        transition += -math.log(t_factor + EPS)
    return obs_r + c_w + args.lambda_m * transition + args.lambda_div * len(set(labels))


def run_mfr_aco_for_packet(
    candidates: Sequence[str],
    eta_r: dict[str, float],
    eta_w: dict[str, float],
    raw_weight: dict[str, float],
    candidate_boost: dict[str, float],
    transition_boost: dict[tuple[str, str], float],
    k_m: dict[tuple[str, str], float],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict:
    h = args.search_depth
    k = len(candidates)
    pheromone = []
    for i, left in enumerate(candidates):
        row = []
        for j, right in enumerate(candidates):
            if i == j:
                base_tau = args.tau_stay * (candidate_boost[left] if args.ablation_stage >= 4 else 1.0)
            else:
                base_tau = args.tau_switch * (k_m[(left, right)] if args.ablation_stage >= 4 else 1.0)
            row.append(base_tau)
        pheromone.append(row)

    elite_vote = {label: 0.0 for label in candidates}
    best_path: list[int] = []
    best_cost = float("inf")

    for _iter in range(args.iterations):
        paths = []
        for _ant in range(args.ants):
            first_weights = [
                (eta_r[label] ** args.kappa_r)
                * (eta_w[label] ** ((args.kappa_w * raw_weight[label]) / h))
                * candidate_boost[label]
                for label in candidates
            ]
            path = [base.weighted_choice(first_weights, rng)]
            for _s in range(1, h):
                prev = path[-1]
                prev_label = candidates[prev]
                weights = []
                for j, label in enumerate(candidates):
                    t_factor = transition_boost[(prev_label, label)]
                    value = (
                        (pheromone[prev][j] ** args.pheromone_power)
                        * (eta_r[label] ** args.kappa_r)
                        * (eta_w[label] ** ((args.kappa_w * raw_weight[label]) / h))
                        * candidate_boost[label]
                        * (t_factor ** args.gamma)
                    )
                    weights.append(value)
                path.append(base.weighted_choice(weights, rng))
            cost = path_cost(path, candidates, eta_r, eta_w, raw_weight, transition_boost, k_m, args)
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
        unnormalized = [math.exp(-cost / (temp + EPS)) for cost, _path in elite]
        total = sum(unnormalized) or 1.0
        weights = [value / total for value in unnormalized]

        for i in range(k):
            for j in range(k):
                pheromone[i][j] *= 1.0 - args.evaporation
                pheromone[i][j] = max(args.min_pheromone, pheromone[i][j])

        for (cost, path), weight in zip(elite, weights):
            mode_label = path_mode(path, candidates)
            elite_vote[mode_label] += weight
            mode_idx = candidates.index(mode_label)
            pheromone[mode_idx][mode_idx] += weight * candidate_boost[mode_label] * eta_w[mode_label]
            for prev, cur in zip(path, path[1:]):
                if prev == cur:
                    continue
                left = candidates[prev]
                right = candidates[cur]
                pheromone[prev][cur] += weight * (k_m[(left, right)] if args.ablation_stage >= 4 else 1.0)

    best_labels = [candidates[idx] for idx in best_path]
    path_counts = Counter(best_path)
    path_mode_idx = min(range(k), key=lambda idx: (-path_counts.get(idx, 0), candidates[idx]))
    pheromone_idx = max(range(k), key=lambda idx: (pheromone[idx][idx], -idx))
    vote_label = max(candidates, key=lambda label: (elite_vote[label], -natural_label_key(label)[0], -natural_label_key(label)[1]))
    physical_score = {
        label: pheromone[idx][idx] * eta_r[label] * (eta_w[label] ** raw_weight[label])
        for idx, label in enumerate(candidates)
    }
    physical_label = max(candidates, key=lambda label: (physical_score[label], -natural_label_key(label)[0], -natural_label_key(label)[1]))
    return {
        "best_cost": best_cost,
        "best_path_labels": ";".join(best_labels),
        "path_mode_label": candidates[path_mode_idx],
        "pheromone_label": candidates[pheromone_idx],
        "vote_label": vote_label,
        "physical_label": physical_label,
        "elite_vote": elite_vote,
        "self_pheromone": {label: pheromone[idx][idx] for idx, label in enumerate(candidates)},
        "physical_score": physical_score,
    }


def evaluate_mfr_aco(
    samples: Sequence[base.PacketSample],
    evidence: Sequence[PacketEvidence],
    field: dict[str, MultipathPoint],
    args: argparse.Namespace,
) -> tuple[dict, list[dict], list[dict], dict]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    counts = Counter(labels)
    eval_indices = [idx for idx, label in enumerate(labels) if counts[label] >= 2]
    t_r = compute_rssi_temperature(rssi_rows, labels, eval_indices, args.rssi_class_k)
    t_w = compute_raw_temperature(evidence, labels, eval_indices)
    t_q = compute_t_q(evidence, eval_indices)
    t_m = args.multipath_temperature
    if t_m is None or t_m <= EPS:
        t_m = compute_multipath_temperature(field, labels)

    rng = random.Random(args.seed)
    predictions = []
    candidate_rows = []
    correct = Counter()
    topk_contains = 0

    for test_index in eval_indices:
        train_indices = [idx for idx in eval_indices if idx != test_index]
        rssi_ranked = base.class_rank(rssi_rows, labels, train_indices, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        raw_prototypes = build_raw_prototypes(evidence, labels, train_indices)
        rssi_scores = dict(rssi_ranked)
        eta_r = {label: math.exp(-rssi_scores[label] / (t_r + EPS)) for label in candidates}
        raw_distances = {
            label: raw_distance(evidence[test_index].w_pkt, raw_prototypes[label])
            for label in candidates
            if label in raw_prototypes
        }
        eta_w = {label: math.exp(-raw_distances.get(label, t_w) / (t_w + EPS)) for label in candidates}
        q_w = math.exp(-evidence[test_index].q_raw_sum / (t_q + EPS))
        mp = candidate_multipath_quantities(candidates, field, t_m)
        g_m = {label: q_w * mp[label]["rel"] * mp[label]["sep"] for label in candidates}
        k_m = {
            (left, right): multipath_kernel(left, right, field, t_m)
            for left in candidates
            for right in candidates
        }
        stage = args.ablation_stage
        if stage < 1:
            eta_w = {label: 1.0 for label in candidates}
            raw_weight = {label: 0.0 for label in candidates}
        elif stage == 1:
            raw_weight = {label: 1.0 for label in candidates}
        elif stage == 2:
            raw_weight = {label: q_w for label in candidates}
        else:
            raw_weight = {label: 1.0 + args.beta * g_m[label] for label in candidates}
        candidate_boost = {
            label: (1.0 + args.beta * g_m[label]) if stage >= 3 else 1.0
            for label in candidates
        }
        transition_boost = {}
        for left in candidates:
            for right in candidates:
                if stage >= 5:
                    transition_boost[(left, right)] = (1.0 + g_m[left]) if left == right else k_m[(left, right)]
                else:
                    transition_boost[(left, right)] = 1.0
        result = run_mfr_aco_for_packet(
            candidates,
            eta_r,
            eta_w,
            raw_weight,
            candidate_boost,
            transition_boost,
            k_m,
            args,
            rng,
        )
        true_label = labels[test_index]
        rssi_pred = candidates[0]
        topk_contains += int(true_label in candidates)
        outputs = {
            "rssi_top1": rssi_pred,
            "path_mode": result["path_mode_label"],
            "pheromone": result["pheromone_label"],
            "vote": result["vote_label"],
            "physical": result["physical_label"],
        }
        for name, pred in outputs.items():
            correct[name] += int(pred == true_label)
        predictions.append(
            {
                "sample_index": test_index,
                "file_name": samples[test_index].file_name,
                "packet_index": samples[test_index].packet_index,
                "true_label": true_label,
                "true_display": point_display(true_label),
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": int(rssi_pred == true_label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(true_label in candidates),
                "q_w": q_w,
                "aco_path_mode_label": result["path_mode_label"],
                "aco_path_mode_correct": int(result["path_mode_label"] == true_label),
                "aco_pheromone_label": result["pheromone_label"],
                "aco_pheromone_correct": int(result["pheromone_label"] == true_label),
                "aco_vote_label": result["vote_label"],
                "aco_vote_correct": int(result["vote_label"] == true_label),
                "mfr_physical_label": result["physical_label"],
                "mfr_physical_correct": int(result["physical_label"] == true_label),
                "best_cost": result["best_cost"],
                "best_path_labels": result["best_path_labels"],
            }
        )
        for label in candidates:
            item = field.get(label)
            candidate_rows.append(
                {
                    "sample_index": test_index,
                    "true_label": true_label,
                    "candidate_label": label,
                    "eta_r": eta_r[label],
                    "eta_w": eta_w[label],
                    "q_w": q_w,
                    "rel_m": mp[label]["rel"],
                    "sep_m": mp[label]["sep"],
                    "g_m": g_m[label],
                    "raw_weight": raw_weight[label],
                    "candidate_boost": candidate_boost[label],
                    "self_pheromone": result["self_pheromone"][label],
                    "elite_vote": result["elite_vote"][label],
                    "physical_score": result["physical_score"][label],
                    "multipath_confidence": item.confidence if item else 0.0,
                    "multipath_source": item.source if item else "",
                }
            )

    n = len(predictions)
    metrics = {
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "search_depth": args.search_depth,
        "ablation_stage": args.ablation_stage,
        "rssi_class_k": args.rssi_class_k,
        "rssi_top1_correct": correct["rssi_top1"],
        "rssi_top1_accuracy": correct["rssi_top1"] / n if n else 0.0,
        "rssi_topk_contains_true": topk_contains,
        "rssi_topk_recall": topk_contains / n if n else 0.0,
        "aco_path_mode_correct": correct["path_mode"],
        "aco_path_mode_accuracy": correct["path_mode"] / n if n else 0.0,
        "aco_pheromone_correct": correct["pheromone"],
        "aco_pheromone_accuracy": correct["pheromone"] / n if n else 0.0,
        "aco_vote_correct": correct["vote"],
        "aco_vote_accuracy": correct["vote"] / n if n else 0.0,
        "mfr_physical_correct": correct["physical"],
        "mfr_physical_accuracy": correct["physical"] / n if n else 0.0,
        "t_r": t_r,
        "t_w": t_w,
        "t_q": t_q,
        "t_m": t_m,
        "kappa_r": args.kappa_r,
        "kappa_w": args.kappa_w,
        "beta": args.beta,
        "gamma": args.gamma,
        "lambda_m": args.lambda_m,
        "lambda_div": args.lambda_div,
    }
    temperatures = {"t_r": t_r, "t_w": t_w, "t_q": t_q, "t_m": t_m}
    return metrics, predictions, candidate_rows, temperatures


def write_multipath_field(path: Path, field: dict[str, MultipathPoint]) -> None:
    rows = []
    for label in sorted(field, key=natural_label_key):
        item = field[label]
        rows.append(
            {
                "position_key": label,
                "corridor_id": item.corridor_id,
                "location_id": item.location_id,
                "visibility_state": item.visibility_state,
                "distance_m": item.distance_m,
                "z_k_main_ratio_log": item.z[0],
                "z_pdiff_log": item.z[1],
                "z_tau_rms_log": item.z[2],
                "z_norm_k": item.z_norm[0],
                "z_norm_pdiff": item.z_norm[1],
                "z_norm_tau": item.z_norm[2],
                "confidence": item.confidence,
                "source": item.source,
            }
        )
    write_csv(path, rows, list(rows[0].keys()) if rows else ["position_key"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--chirp-csv", type=Path, default=DEFAULT_CHIRP_CSV)
    parser.add_argument("--location-csv", type=Path, default=DEFAULT_LOCATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--search-depth", type=int, default=4)
    parser.add_argument("--raw-segments", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tau-stay", type=float, default=1.4)
    parser.add_argument("--tau-switch", type=float, default=0.35)
    parser.add_argument("--pheromone-power", type=float, default=1.0)
    parser.add_argument("--evaporation", type=float, default=0.25)
    parser.add_argument("--min-pheromone", type=float, default=1e-4)
    parser.add_argument("--aco-temperature", type=float, default=None)
    parser.add_argument("--multipath-temperature", type=float, default=None)
    parser.add_argument("--kappa-r", type=float, default=1.0)
    parser.add_argument("--kappa-w", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-m", type=float, default=1.0)
    parser.add_argument("--lambda-div", type=float, default=0.2)
    parser.add_argument("--ablation-stage", type=int, default=5, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rssi_packets = base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, symbol_thresholds = base.read_symbol_packets(args.spectrum_csv, args)
    samples = base.align_samples(rssi_packets, symbol_packets)
    if not samples:
        raise RuntimeError("No aligned RSSI+/symbol packet samples found.")
    evidence = build_all_packet_evidence(samples, args.raw_segments)
    field, field_metadata = build_multipath_field(args.location_csv, args.chirp_csv)
    metrics, predictions, candidate_rows, temperatures = evaluate_mfr_aco(samples, evidence, field, args)

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
        "q_w",
        "aco_path_mode_label",
        "aco_path_mode_correct",
        "aco_pheromone_label",
        "aco_pheromone_correct",
        "aco_vote_label",
        "aco_vote_correct",
        "mfr_physical_label",
        "mfr_physical_correct",
        "best_cost",
        "best_path_labels",
    ]
    write_csv(args.output_dir / "mfr_aco_predictions.csv", predictions, prediction_fields)
    candidate_fields = [
        "sample_index",
        "true_label",
        "candidate_label",
        "eta_r",
        "eta_w",
        "q_w",
        "rel_m",
        "sep_m",
        "g_m",
        "raw_weight",
        "candidate_boost",
        "self_pheromone",
        "elite_vote",
        "physical_score",
        "multipath_confidence",
        "multipath_source",
    ]
    write_csv(args.output_dir / "mfr_aco_candidate_scores.csv", candidate_rows, candidate_fields)
    summary_fields = list(metrics.keys())
    write_csv(args.output_dir / "mfr_aco_summary.csv", [metrics], summary_fields)
    write_multipath_field(args.output_dir / "global_multipath_field.csv", field)
    payload = {
        "inputs": {
            "rssi_csv": str(args.rssi_csv),
            "spectrum_csv": str(args.spectrum_csv),
            "chirp_csv": str(args.chirp_csv),
            "location_csv": str(args.location_csv),
        },
        "method": {
            "name": "MFR-ACO: Multipath-Field and Raw-bin guided Ant Colony Optimization",
            "evaluation": "packet-level leave-one-out over aligned packets with at least two samples per location",
            "candidate_generation": "RSSI+ class-KNN Top-K",
            "raw_evidence": "packet-level log-normalized q=1 raw bin[-2,+2], four preamble groups only estimate raw stability Q_W",
            "multipath_field": "chirp z=[log1p(K), log1p(Pdiff), log1p(tau_rms)] with same-corridor/same-visibility interpolation",
            "outputs": ["path_mode", "self_loop_pheromone", "elite_vote", "physical_score"],
        },
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "symbol_thresholds": symbol_thresholds,
        "temperatures": temperatures,
        "multipath_field": field_metadata,
        "metrics": metrics,
    }
    with (args.output_dir / "mfr_aco_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
