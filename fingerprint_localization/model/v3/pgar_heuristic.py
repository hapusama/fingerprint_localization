#!/usr/bin/env python3
"""V3 PGAR heuristic trial using the trusted 2026-06-23 processed data.

This version does not use v2 model results. It reads the 6.23 raw-data-derived
RSSI+ and LoRa frequency features, then evaluates the PGAR candidate reranker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = PROJECT_ROOT / "fingerprint_localization"
DATA_ROOT = PACKAGE_ROOT / "data" / "mainline_202607"
DEFAULT_Q4_SPECTRUM_CSV = DATA_ROOT / "external" / "subbin_spectrum_long.csv"

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
STRUCTURE_COLUMNS = ["E0", "C_peak", "R_side", "R_asym", "S_L", "S_R"]
DEFAULT_RSSI_CSV = DATA_ROOT / "inputs" / "rssi_plus_packet_level_54points.csv"
DEFAULT_RAW_FEATURE_CSV = DATA_ROOT / "inputs" / "lora_frequency_s17_54points.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3" / "output"

PacketKey = Tuple[str, int]


@dataclass
class Sample:
    key: PacketKey
    file_name: str
    packet_index: int
    label: str
    rssi_plus: list[float]
    raw_bins: list[float]
    structure: list[float]
    peak_mean: float
    peak_iqr: float
    q4_curve: list[float] | None
    q4_stability: float


def file_stem(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_int(value: object) -> int:
    return int(float(value))


def point_label(corridor_id: object, position_id: object) -> str:
    return f"{parse_int(corridor_id)}_{parse_int(position_id)}"


def natural_label_key(label: str) -> tuple[int, int]:
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


def rssi_rows_to_packets(rows: Sequence[dict]) -> dict[PacketKey, dict]:
    packets: dict[PacketKey, dict] = {}
    for row in rows:
        key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
        packets[key] = {
            "file_name": row["file_name"],
            "packet_index": key[1],
            "label": row["position_key"],
            "rssi_plus": [parse_float(row[col]) for col in RSSI_PLUS_COLUMNS],
        }
    return packets


def structure_features(raw_bins: Sequence[float]) -> list[float]:
    a_m2, a_m1, a0, a_p1, a_p2 = [max(0.0, float(v)) for v in raw_bins]
    total = a_m2 + a_m1 + a0 + a_p1 + a_p2
    side = a_m2 + a_m1 + a_p1 + a_p2
    left = a_m2 + a_m1
    right = a_p1 + a_p2
    return [
        math.log(a0 + EPS),
        a0 / (total + EPS),
        math.log((side + EPS) / (a0 + EPS)),
        math.log((right + EPS) / (left + EPS)),
        a0 - a_m1,
        a0 - a_p1,
    ]


def raw_rows_to_packets(rows: Sequence[dict]) -> dict[PacketKey, dict]:
    packets: dict[PacketKey, dict] = {}
    for row in rows:
        key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
        if "raw_bin_+0" in row:
            raw_bins = [parse_float(row[f"raw_bin_{int(offset):+d}"]) for offset in RAW_OFFSETS]
            structure = [parse_float(row[name]) for name in STRUCTURE_COLUMNS]
            peak_mean = parse_float(row["peak_mean"])
            peak_iqr = parse_float(row["peak_iqr"])
            label = row["position_key"]
        else:
            raw_bins = [parse_float(row[f"preamble_fft_mag_bin_{int(offset):+d}"]) for offset in RAW_OFFSETS]
            structure = structure_features(raw_bins)
            peak_mean = raw_bins[2]
            peak_iqr = parse_float(row.get("s17_j_s", row.get("preamble_peak_bin_std", 0.0)))
            label = row.get("position_key") or point_label(row["corridor_id"], row["position_id"])
        packets[key] = {
            "file_name": row["file_name"],
            "packet_index": key[1],
            "label": label,
            "raw_bins": raw_bins,
            "structure": structure,
            "peak_mean": peak_mean,
            "peak_iqr": peak_iqr,
        }
    return packets


def read_q4_curves(path: Path) -> tuple[dict[PacketKey, dict], list[float]]:
    symbols_by_packet: dict[PacketKey, dict[int, dict[float, float]]] = defaultdict(lambda: defaultdict(dict))
    meta: dict[PacketKey, dict] = {}
    offsets_seen: set[float] = set()
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if parse_int(row["q"]) != 4:
                continue
            offset = round(parse_float(row["subbin_offset"]), 6)
            key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
            symbol_id = parse_int(row.get("local_symbol_index", row.get("preamble_symbol_index", 0)))
            symbols_by_packet[key][symbol_id][offset] = parse_float(row["mag_db_rel_peak"])
            offsets_seen.add(offset)
            meta[key] = {
                "file_name": row["file_name"],
                "packet_index": key[1],
                "label": point_label(row["corridor_id"], row["position_id"]),
            }

    offsets = sorted(offsets_seen)
    packets: dict[PacketKey, dict] = {}
    for key, symbol_rows in symbols_by_packet.items():
        complete_symbols = [
            bins
            for bins in symbol_rows.values()
            if all(offset in bins for offset in offsets)
        ]
        if not complete_symbols:
            continue
        curve = [median([bins[offset] for bins in complete_symbols]) for offset in offsets]
        stability_by_offset = [iqr([bins[offset] for bins in complete_symbols]) for offset in offsets]
        packets[key] = {
            **meta[key],
            "q4_curve": curve,
            "q4_stability": median(stability_by_offset),
            "q4_symbol_count": len(complete_symbols),
        }
    return packets, offsets


def write_q4_curve_csv(path: Path, q4_packets: dict[PacketKey, dict], offsets: Sequence[float]) -> None:
    fields = ["file_name", "packet_index", "position_key", "q4_stability", "q4_symbol_count"]
    fields += [f"q4_db_{offset:+.2f}" for offset in offsets]
    rows = []
    for key in sorted(q4_packets, key=lambda item: (q4_packets[item]["label"], item[1], item[0])):
        item = q4_packets[key]
        row = {
            "file_name": item["file_name"],
            "packet_index": item["packet_index"],
            "position_key": item["label"],
            "q4_stability": item["q4_stability"],
            "q4_symbol_count": item["q4_symbol_count"],
        }
        row.update({f"q4_db_{offset:+.2f}": item["q4_curve"][idx] for idx, offset in enumerate(offsets)})
        rows.append(row)
    write_csv(path, rows, fields)


def align_samples(
    rssi: dict[PacketKey, dict],
    raw: dict[PacketKey, dict],
    q4: dict[PacketKey, dict] | None = None,
) -> list[Sample]:
    samples: list[Sample] = []
    common = sorted(
        set(rssi) & set(raw),
        key=lambda key: (natural_label_key(rssi[key]["label"]), rssi[key]["packet_index"], rssi[key]["file_name"]),
    )
    for key in common:
        if rssi[key]["label"] != raw[key]["label"]:
            raise ValueError(f"Label mismatch for {key}: {rssi[key]['label']} vs {raw[key]['label']}")
        q4_item = q4.get(key) if q4 is not None else None
        if q4_item is not None and q4_item["label"] != rssi[key]["label"]:
            raise ValueError(f"Q4 label mismatch for {key}: {q4_item['label']} vs {rssi[key]['label']}")
        samples.append(
            Sample(
                key=key,
                file_name=raw[key]["file_name"],
                packet_index=key[1],
                label=rssi[key]["label"],
                rssi_plus=rssi[key]["rssi_plus"],
                raw_bins=raw[key]["raw_bins"],
                structure=raw[key]["structure"],
                peak_mean=raw[key]["peak_mean"],
                peak_iqr=raw[key]["peak_iqr"],
                q4_curve=q4_item["q4_curve"] if q4_item is not None else None,
                q4_stability=q4_item["q4_stability"] if q4_item is not None else float("inf"),
            )
        )
    return samples


def zscore_stats(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    dim = len(rows[0])
    means = [sum(row[j] for row in rows) / len(rows) for j in range(dim)]
    stds: list[float] = []
    for j in range(dim):
        var = sum((row[j] - means[j]) ** 2 for row in rows) / len(rows)
        std = math.sqrt(var)
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
    candidate_labels: Sequence[str] | None = None,
) -> list[tuple[str, float]]:
    train_rows = [rows[idx] for idx in train_indices]
    means, stds = zscore_stats(train_rows)
    if candidate_labels is None:
        candidate_labels = sorted({labels[idx] for idx in train_indices}, key=natural_label_key)
    ranked = []
    for label in candidate_labels:
        label_distances = [
            squared_distance(rows[idx], rows[test_index], means, stds)
            for idx in train_indices
            if labels[idx] == label
        ]
        if not label_distances:
            continue
        label_distances.sort()
        k_eff = min(class_neighbor_k, len(label_distances))
        ranked.append((label, sum(label_distances[:k_eff]) / k_eff))
    ranked.sort(key=lambda item: (item[1], natural_label_key(item[0])))
    return ranked


def build_prototypes(structures: Sequence[Sequence[float]], labels: Sequence[str], train_indices: Sequence[int]) -> dict[str, dict[str, list[float]]]:
    by_label: dict[str, list[Sequence[float]]] = defaultdict(list)
    for idx in train_indices:
        by_label[labels[idx]].append(structures[idx])
    prototypes: dict[str, dict[str, list[float]]] = {}
    for label, rows in by_label.items():
        med = [median([row[j] for row in rows]) for j in range(len(STRUCTURE_COLUMNS))]
        spread = [safe_iqr([row[j] for row in rows]) for j in range(len(STRUCTURE_COLUMNS))]
        prototypes[label] = {"median": med, "iqr": spread}
    return prototypes


def candidate_weights(prototypes: dict[str, dict[str, list[float]]], candidates: Sequence[str]) -> list[float]:
    spreads = [iqr([prototypes[label]["median"][j] for label in candidates if label in prototypes]) for j in range(len(STRUCTURE_COLUMNS))]
    total = sum(spreads)
    if total <= EPS:
        return [1.0 / len(STRUCTURE_COLUMNS)] * len(STRUCTURE_COLUMNS)
    return [value / total for value in spreads]


def structure_distance(z: Sequence[float], prototype: dict[str, list[float]], weights: Sequence[float]) -> float:
    return sum(weights[j] * abs(z[j] - prototype["median"][j]) / (prototype["iqr"][j] + EPS) for j in range(len(STRUCTURE_COLUMNS)))


def build_q4_prototypes(
    q4_curves: Sequence[list[float] | None],
    labels: Sequence[str],
    train_indices: Sequence[int],
) -> dict[str, dict[str, list[float]]]:
    by_label: dict[str, list[list[float]]] = defaultdict(list)
    for idx in train_indices:
        curve = q4_curves[idx]
        if curve is not None:
            by_label[labels[idx]].append(curve)
    prototypes: dict[str, dict[str, list[float]]] = {}
    for label, rows in by_label.items():
        dim = len(rows[0])
        med = [median([row[j] for row in rows]) for j in range(dim)]
        spread = [safe_iqr([row[j] for row in rows]) for j in range(dim)]
        prototypes[label] = {"median": med, "iqr": spread}
    return prototypes


def q4_distance(curve: Sequence[float], prototype: dict[str, list[float]]) -> float:
    dim = len(curve)
    return sum(abs(curve[j] - prototype["median"][j]) / (prototype["iqr"][j] + EPS) for j in range(dim)) / dim


def q4_discriminability(left: str, right: str, prototypes: dict[str, dict[str, list[float]]]) -> float:
    if left not in prototypes or right not in prototypes:
        return 0.0
    left_proto = prototypes[left]
    right_proto = prototypes[right]
    dim = len(left_proto["median"])
    return sum(
        abs(left_proto["median"][j] - right_proto["median"][j])
        / (0.5 * (left_proto["iqr"][j] + right_proto["iqr"][j]) + EPS)
        for j in range(dim)
    ) / dim


def normalize_candidate_scores(ranked: Sequence[tuple[str, float]], candidates: Sequence[str]) -> dict[str, float]:
    raw = {label: score for label, score in ranked if label in candidates}
    if not raw:
        return {}
    lo = min(raw.values())
    hi = max(raw.values())
    if hi - lo <= EPS:
        return {label: 0.0 for label in raw}
    return {label: (score - lo) / (hi - lo) for label, score in raw.items()}


def resolve_auto_thresholds(samples: Sequence[Sample], args: argparse.Namespace) -> tuple[float, float]:
    peak_threshold = args.peak_threshold
    peak_iqr_threshold = args.peak_iqr_threshold
    if peak_threshold is None:
        peak_threshold = quantile([sample.peak_mean for sample in samples], args.auto_peak_quantile)
    if peak_iqr_threshold is None:
        peak_iqr_threshold = quantile([sample.peak_iqr for sample in samples], args.auto_peak_iqr_quantile)
    return float(peak_threshold), float(peak_iqr_threshold)


def resolve_q4_stability_threshold(samples: Sequence[Sample], args: argparse.Namespace) -> float:
    if args.q4_stability_threshold is not None:
        return float(args.q4_stability_threshold)
    values = [
        sample.q4_stability
        for sample in samples
        if sample.q4_curve is not None and math.isfinite(sample.q4_stability)
    ]
    return float(quantile(values, args.auto_q4_stability_quantile)) if values else float("inf")


def evaluate_pgar(samples: Sequence[Sample], args: argparse.Namespace, rssi_margin_threshold: float) -> tuple[dict, list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    structures = [sample.structure for sample in samples]
    q4_curves = [sample.q4_curve for sample in samples]
    counts = Counter(labels)
    eval_indices = [idx for idx, label in enumerate(labels) if counts[label] >= 2]
    peak_threshold, peak_iqr_threshold = resolve_auto_thresholds(samples, args)
    q4_stability_threshold = resolve_q4_stability_threshold(samples, args)
    prediction_rows: list[dict] = []
    rssi_correct = pgar_correct = raw_gate_used = topk_contains_true = 0
    rssi_wrong_fixed = rssi_correct_broken = 0
    raw_stage_correct = 0
    q4_gate_used = 0
    q4_wrong_fixed_after_raw = 0
    q4_raw_correct_broken = 0

    for test_index in eval_indices:
        sample = samples[test_index]
        train_indices = [idx for idx in eval_indices if idx != test_index]
        rssi_ranked = class_rank(rssi_rows, labels, train_indices, test_index, args.rssi_class_k)
        candidates = [label for label, _ in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        rssi_pred = candidates[0]
        rssi_ok = int(rssi_pred == sample.label)
        rssi_correct += rssi_ok
        contains = int(sample.label in candidates)
        topk_contains_true += contains
        margin = rssi_ranked[1][1] - rssi_ranked[0][1] if len(rssi_ranked) > 1 else float("inf")
        peak_reliable = sample.peak_mean > peak_threshold and sample.peak_iqr < peak_iqr_threshold
        use_raw = margin <= rssi_margin_threshold and peak_reliable
        raw_pred = rssi_pred
        pgar_pred = rssi_pred
        raw_margin = float("inf")
        q4_stability = sample.q4_stability
        q4_disc = 0.0
        q4_gate_reason = "raw_gate_closed"
        q4_candidate_score_parts: list[str] = []
        candidate_score_parts: list[str] = []
        if use_raw:
            raw_gate_used += 1
            prototypes = build_prototypes(structures, labels, train_indices)
            weights = candidate_weights(prototypes, candidates)
            rssi_norm = normalize_candidate_scores(rssi_ranked, candidates)
            scored = []
            for candidate in candidates:
                if candidate not in prototypes:
                    continue
                d_e = structure_distance(sample.structure, prototypes[candidate], weights)
                d_r = rssi_norm.get(candidate, 0.0)
                score = args.alpha * d_r + args.beta * d_e
                scored.append((candidate, score, d_r, d_e))
                candidate_score_parts.append(f"{candidate}:{score:.6g}:{d_r:.6g}:{d_e:.6g}")
            scored.sort(key=lambda item: (item[1], natural_label_key(item[0])))
            if scored:
                raw_pred = scored[0][0]
                pgar_pred = raw_pred
                raw_margin = scored[1][1] - scored[0][1] if len(scored) > 1 else float("inf")
                q4_gate_reason = "raw_top_margin_not_close"
                if len(scored) < 2:
                    q4_gate_reason = "raw_top2_missing"
                elif raw_margin <= args.q4_raw_margin_threshold:
                    q4_gate_reason = "q4_packet_missing"
                    if sample.q4_curve is not None:
                        q4_gate_reason = "q4_packet_unstable"
                        if q4_stability < q4_stability_threshold:
                            q4_prototypes = build_q4_prototypes(q4_curves, labels, train_indices)
                            q4_disc = q4_discriminability(scored[0][0], scored[1][0], q4_prototypes)
                            q4_gate_reason = "q4_disc_too_low"
                            if q4_disc > args.q4_disc_threshold and args.gamma > 0.0:
                                q4_scores = [
                                    (candidate, q4_distance(sample.q4_curve, q4_prototypes[candidate]))
                                    for candidate in candidates
                                    if candidate in q4_prototypes
                                ]
                                q4_norm = normalize_candidate_scores(q4_scores, candidates)
                                final_scored = []
                                for candidate, raw_score, d_r, d_e in scored:
                                    q4_score = q4_norm.get(candidate, 0.0)
                                    final_score = raw_score + args.gamma * q4_score
                                    final_scored.append((candidate, final_score, raw_score, q4_score, d_r, d_e))
                                    q4_candidate_score_parts.append(
                                        f"{candidate}:{final_score:.6g}:{raw_score:.6g}:{q4_score:.6g}"
                                    )
                                final_scored.sort(key=lambda item: (item[1], natural_label_key(item[0])))
                                if final_scored:
                                    pgar_pred = final_scored[0][0]
                                    q4_gate_used += 1
                                    q4_gate_reason = "q4_used"
                            elif args.gamma <= 0.0:
                                q4_gate_reason = "gamma_zero"
        raw_ok = int(raw_pred == sample.label)
        raw_stage_correct += raw_ok
        pgar_ok = int(pgar_pred == sample.label)
        pgar_correct += pgar_ok
        if (not rssi_ok) and pgar_ok:
            rssi_wrong_fixed += 1
        if rssi_ok and (not pgar_ok):
            rssi_correct_broken += 1
        if q4_gate_reason == "q4_used" and (not raw_ok) and pgar_ok:
            q4_wrong_fixed_after_raw += 1
        if q4_gate_reason == "q4_used" and raw_ok and (not pgar_ok):
            q4_raw_correct_broken += 1
        prediction_rows.append(
            {
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "true_display": point_display(sample.label),
                "rssi_top1_label": rssi_pred,
                "rssi_top1_display": point_display(rssi_pred),
                "rssi_top1_correct": rssi_ok,
                "rssi_margin": margin,
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": contains,
                "peak_mean": sample.peak_mean,
                "peak_iqr": sample.peak_iqr,
                "peak_reliable": int(peak_reliable),
                "raw_gate_used": int(use_raw),
                "raw_rerank_label": raw_pred,
                "raw_rerank_display": point_display(raw_pred),
                "raw_rerank_correct": raw_ok,
                "raw_margin": raw_margin,
                "q4_stability": q4_stability,
                "q4_stability_threshold": q4_stability_threshold,
                "q4_disc_top12": q4_disc,
                "q4_disc_threshold": args.q4_disc_threshold,
                "q4_gate_used": int(q4_gate_reason == "q4_used"),
                "q4_gate_reason": q4_gate_reason,
                "pgar_label": pgar_pred,
                "pgar_display": point_display(pgar_pred),
                "pgar_correct": pgar_ok,
                "candidate_scores": ";".join(candidate_score_parts),
                "q4_candidate_scores": ";".join(q4_candidate_score_parts),
            }
        )
    n = len(prediction_rows)
    metrics = {
        "top_k": args.top_k,
        "rssi_class_k": args.rssi_class_k,
        "alpha": args.alpha,
        "beta": args.beta,
        "rssi_margin_threshold": rssi_margin_threshold,
        "peak_threshold": peak_threshold,
        "peak_iqr_threshold": peak_iqr_threshold,
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "rssi_top1_correct": rssi_correct,
        "rssi_top1_accuracy": rssi_correct / n if n else 0.0,
        "rssi_topk_contains_true": topk_contains_true,
        "rssi_topk_recall": topk_contains_true / n if n else 0.0,
        "raw_rerank_correct": raw_stage_correct,
        "raw_rerank_accuracy": raw_stage_correct / n if n else 0.0,
        "pgar_correct": pgar_correct,
        "pgar_accuracy": pgar_correct / n if n else 0.0,
        "pgar_gain_vs_rssi_top1": pgar_correct - rssi_correct,
        "pgar_gain_vs_raw_rerank": pgar_correct - raw_stage_correct,
        "raw_gate_used": raw_gate_used,
        "raw_gate_rate": raw_gate_used / n if n else 0.0,
        "q4_raw_margin_threshold": args.q4_raw_margin_threshold,
        "q4_stability_threshold": q4_stability_threshold,
        "q4_disc_threshold": args.q4_disc_threshold,
        "gamma": args.gamma,
        "q4_gate_used": q4_gate_used,
        "q4_gate_rate": q4_gate_used / n if n else 0.0,
        "q4_wrong_fixed_after_raw": q4_wrong_fixed_after_raw,
        "q4_raw_correct_broken": q4_raw_correct_broken,
        "rssi_wrong_fixed_by_pgar": rssi_wrong_fixed,
        "rssi_correct_broken_by_pgar": rssi_correct_broken,
    }
    return metrics, prediction_rows


def parse_scan_values(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_or_load_features(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict[PacketKey, dict], list[float]]:
    output_dir = Path(args.output_dir)
    rssi_rows = read_csv_dict(Path(args.rssi_csv))
    raw_source_rows = read_csv_dict(Path(args.raw_feature_csv))
    raw_rows = []
    for row in raw_source_rows:
        raw_bins = [parse_float(row[f"preamble_fft_mag_bin_{int(offset):+d}"]) for offset in RAW_OFFSETS]
        structure = structure_features(raw_bins)
        out = {
            "file_name": row["file_name"],
            "experiment_id": row.get("experiment_id", ""),
            "corridor_id": row["corridor_id"],
            "location_id": row["position_id"],
            "position_key": point_label(row["corridor_id"], row["position_id"]),
            "packet_index": row["packet_index"],
            "sample_start": row.get("sample_start", ""),
            "detect_score_db": row.get("detect_score_db", ""),
            "detect_peak_bin_mean": row.get("detect_peak_bin_mean", ""),
            "detect_peak_bin_std": row.get("detect_peak_bin_std", ""),
            "peak_mean": raw_bins[2],
            "peak_iqr": parse_float(row.get("s17_j_s", row.get("preamble_peak_bin_std", 0.0))),
        }
        out.update({f"raw_bin_{int(offset):+d}": raw_bins[i] for i, offset in enumerate(RAW_OFFSETS)})
        out.update({name: structure[i] for i, name in enumerate(STRUCTURE_COLUMNS)})
        raw_rows.append(out)
    raw_fields = [
        "file_name",
        "experiment_id",
        "corridor_id",
        "location_id",
        "position_key",
        "packet_index",
        "sample_start",
        "detect_score_db",
        "detect_peak_bin_mean",
        "detect_peak_bin_std",
        "peak_mean",
        "peak_iqr",
    ]
    raw_fields += [f"raw_bin_{int(offset):+d}" for offset in RAW_OFFSETS]
    raw_fields += STRUCTURE_COLUMNS
    write_csv(output_dir / "raw_structure_from_20260623.csv", raw_rows, raw_fields)
    q4_packets: dict[PacketKey, dict] = {}
    q4_offsets: list[float] = []
    if not args.disable_q4:
        q4_packets, q4_offsets = read_q4_curves(Path(args.q4_spectrum_csv))
        write_q4_curve_csv(output_dir / "q4_curve_from_20260624.csv", q4_packets, q4_offsets)
    return rssi_rows, raw_rows, q4_packets, q4_offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--raw-feature-csv", type=Path, default=DEFAULT_RAW_FEATURE_CSV)
    parser.add_argument("--q4-spectrum-csv", type=Path, default=DEFAULT_Q4_SPECTRUM_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--rssi-margin-threshold", type=float, default=0.2)
    parser.add_argument("--scan-rssi-margin", default="")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--peak-iqr-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--auto-peak-iqr-quantile", type=float, default=0.75)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--q4-raw-margin-threshold", type=float, default=0.2)
    parser.add_argument("--q4-stability-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-stability-quantile", type=float, default=0.75)
    parser.add_argument("--q4-disc-threshold", type=float, default=0.5)
    parser.add_argument("--disable-q4", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rssi_rows, raw_rows, q4_packets, q4_offsets = build_or_load_features(args)
    samples = align_samples(rssi_rows_to_packets(rssi_rows), raw_rows_to_packets(raw_rows), q4_packets)
    if not samples:
        raise RuntimeError("No aligned origin-data RSSI+/raw packet samples found.")

    scan_values = parse_scan_values(args.scan_rssi_margin) if args.scan_rssi_margin else [args.rssi_margin_threshold]
    summary_rows = []
    selected_predictions: list[dict] = []
    for threshold in scan_values:
        metrics, prediction_rows = evaluate_pgar(samples, args, threshold)
        summary_rows.append(metrics)
        if threshold == scan_values[0]:
            selected_predictions = prediction_rows

    summary_fields = [
        "top_k",
        "rssi_class_k",
        "alpha",
        "beta",
        "rssi_margin_threshold",
        "peak_threshold",
        "peak_iqr_threshold",
        "packet_count",
        "location_count",
        "rssi_top1_correct",
        "rssi_top1_accuracy",
        "rssi_topk_contains_true",
        "rssi_topk_recall",
        "raw_rerank_correct",
        "raw_rerank_accuracy",
        "pgar_correct",
        "pgar_accuracy",
        "pgar_gain_vs_rssi_top1",
        "pgar_gain_vs_raw_rerank",
        "raw_gate_used",
        "raw_gate_rate",
        "q4_raw_margin_threshold",
        "q4_stability_threshold",
        "q4_disc_threshold",
        "gamma",
        "q4_gate_used",
        "q4_gate_rate",
        "q4_wrong_fixed_after_raw",
        "q4_raw_correct_broken",
        "rssi_wrong_fixed_by_pgar",
        "rssi_correct_broken_by_pgar",
    ]
    write_csv(Path(args.output_dir) / "pgar_summary.csv", summary_rows, summary_fields)
    prediction_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "true_display",
        "rssi_top1_label",
        "rssi_top1_display",
        "rssi_top1_correct",
        "rssi_margin",
        "rssi_topk_candidates",
        "true_in_rssi_topk",
        "peak_mean",
        "peak_iqr",
        "peak_reliable",
        "raw_gate_used",
        "raw_rerank_label",
        "raw_rerank_display",
        "raw_rerank_correct",
        "raw_margin",
        "q4_stability",
        "q4_stability_threshold",
        "q4_disc_top12",
        "q4_disc_threshold",
        "q4_gate_used",
        "q4_gate_reason",
        "pgar_label",
        "pgar_display",
        "pgar_correct",
        "candidate_scores",
        "q4_candidate_scores",
    ]
    write_csv(Path(args.output_dir) / "pgar_predictions.csv", selected_predictions, prediction_fields)
    payload = {
        "inputs": {
            "rssi_csv": str(args.rssi_csv),
            "raw_feature_csv": str(args.raw_feature_csv),
            "q4_spectrum_csv": str(args.q4_spectrum_csv),
            "normalized_raw_structure_csv": str(Path(args.output_dir) / "raw_structure_from_20260623.csv"),
            "normalized_q4_curve_csv": str(Path(args.output_dir) / "q4_curve_from_20260624.csv") if not args.disable_q4 else "",
            "aligned_packet_count": len(samples),
            "aligned_location_count": len(Counter(sample.label for sample in samples)),
            "aligned_q4_packet_count": sum(1 for sample in samples if sample.q4_curve is not None),
        },
        "feature_definitions": {
            "rssi_plus": RSSI_PLUS_COLUMNS,
            "raw_offsets": RAW_OFFSETS,
            "structure": STRUCTURE_COLUMNS,
            "q4_offsets": q4_offsets,
            "q4_curve": "median mag_db_rel_peak curve across preamble symbols; stability is median per-offset IQR",
        },
        "method": {
            "name": "PGAR: Physics-Guided Ambiguity Resolution",
            "source_policy": "uses trusted files under v2_output only; v2_output_wrong is not consumed",
            "evaluation": "packet-level leave-one-out over locations with at least two aligned packets",
            "rssi_stage": "class-KNN location ranking in rebuilt RSSI+ space",
            "structure_stage": "median/IQR prototypes over rebuilt raw q=1 structure features",
            "q4_gate": "q4 participates only when raw top1/top2 are close, packet q4 curve is stable, and raw top1/top2 q4 prototypes are discriminable",
            "peak_reliability_note": "peak_iqr uses s17_j_s from lora_frequency_s17_54points.csv as the available packet-level stability proxy",
        },
        "metrics": summary_rows,
    }
    with (Path(args.output_dir) / "pgar_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
