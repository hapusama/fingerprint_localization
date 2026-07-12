#!/usr/bin/env python3
"""Analyze RSSI+ combinations and q=4 separability for RSSI+ confusions."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from analyze_q4_matching_tsne import (
    DEFAULT_RSSI_CSV,
    DEFAULT_SPECTRUM_CSV,
    PacketKey,
    Q4_OFFSETS,
    RAW_OFFSETS,
    RSSI_PLUS_COLUMNS,
    file_stem,
    loo_1nn,
    natural_label_key,
    parse_float,
    parse_int,
    point_display,
    point_label,
    write_csv,
)


DEFAULT_OUTPUT_DIR = (
    "v2_output/20260624_zero_padding_fft_q1_q4_point_compare/"
    "matching_tsne/combined_confusion_analysis"
)
EPS = 1e-12


def read_rssi_plus(path: str) -> Dict[PacketKey, dict]:
    packets: Dict[PacketKey, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (file_stem(row["file_name"]), parse_int(row["packet_index"], "packet_index"))
            label = row.get("position_key") or point_label(row["corridor_id"], row["location_id"])
            packets[key] = {
                "key": key,
                "file_name": row["file_name"],
                "packet_index": key[1],
                "label": label,
                "feature": [parse_float(row[col], col) for col in RSSI_PLUS_COLUMNS],
            }
    return packets


def offset_key(value: str) -> float:
    return round(float(value), 2)


def read_spectrum_packet_features(path: str) -> Dict[PacketKey, dict]:
    raw_offsets = set(RAW_OFFSETS)
    q4_offsets = set(Q4_OFFSETS)
    raw_mag = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    raw_rel = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    q4_rel = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    meta = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = parse_int(row["q"], "q")
            offset = offset_key(row["subbin_offset"])
            key = (file_stem(row["file_name"]), parse_int(row["packet_index"], "packet_index"))
            if q == 1 and offset in raw_offsets:
                raw_mag[key][offset][0] += parse_float(row["mag_raw"], "mag_raw")
                raw_mag[key][offset][1] += 1
                raw_rel[key][offset][0] += parse_float(row["mag_db_rel_peak"], "mag_db_rel_peak")
                raw_rel[key][offset][1] += 1
            elif q == 4 and offset in q4_offsets:
                q4_rel[key][offset][0] += parse_float(row["mag_db_rel_peak"], "mag_db_rel_peak")
                q4_rel[key][offset][1] += 1
            else:
                continue
            meta[key] = {
                "file_name": row["file_name"],
                "packet_index": parse_int(row["packet_index"], "packet_index"),
                "label": point_label(row["corridor_id"], row["position_id"]),
                "preamble_len": parse_int(row["filename_preamble_len"], "filename_preamble_len"),
                "skip_preamble_symbols": parse_int(row["skip_preamble_symbols"], "skip_preamble_symbols"),
                "feature_symbols": parse_int(row["feature_symbols"], "feature_symbols"),
            }

    packets: Dict[PacketKey, dict] = {}
    for key in sorted(set(raw_mag) & set(raw_rel) & set(q4_rel)):
        if any(raw_mag[key][offset][1] == 0 for offset in RAW_OFFSETS):
            continue
        if any(raw_rel[key][offset][1] == 0 for offset in RAW_OFFSETS):
            continue
        if any(q4_rel[key][offset][1] == 0 for offset in Q4_OFFSETS):
            continue
        raw_mag_vec = np.asarray(
            [raw_mag[key][offset][0] / raw_mag[key][offset][1] for offset in RAW_OFFSETS],
            dtype=float,
        )
        raw_rel_vec = np.asarray(
            [raw_rel[key][offset][0] / raw_rel[key][offset][1] for offset in RAW_OFFSETS],
            dtype=float,
        )
        q4_rel_vec = np.asarray(
            [q4_rel[key][offset][0] / q4_rel[key][offset][1] for offset in Q4_OFFSETS],
            dtype=float,
        )
        item = dict(meta[key])
        item.update(
            {
                "key": key,
                "raw_mag_bin_m2_to_p2": raw_mag_vec,
                "raw_rel_bin_m2_to_p2": raw_rel_vec,
                "q4_rel_17subbin": q4_rel_vec,
                "log10_raw_center_mag": float(np.log10(max(raw_mag_vec[2], EPS))),
            }
        )
        packets[key] = item
    return packets


def aligned_matrices(
    rssi: Dict[PacketKey, dict],
    spectrum: Dict[PacketKey, dict],
) -> Tuple[List[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common = sorted(
        set(rssi) & set(spectrum),
        key=lambda k: (natural_label_key(rssi[k]["label"]), rssi[k]["packet_index"], rssi[k]["file_name"]),
    )
    samples = []
    x_rssi = []
    x_raw_mag = []
    x_raw_rel = []
    x_q4_rel = []
    x_log_center = []
    for key in common:
        if rssi[key]["label"] != spectrum[key]["label"]:
            raise ValueError(f"Label mismatch for {key}: {rssi[key]['label']} vs {spectrum[key]['label']}")
        sample = dict(rssi[key])
        sample["file_name"] = spectrum[key]["file_name"]
        sample["preamble_len"] = spectrum[key]["preamble_len"]
        sample["skip_preamble_symbols"] = spectrum[key]["skip_preamble_symbols"]
        sample["feature_symbols"] = spectrum[key]["feature_symbols"]
        samples.append(sample)
        x_rssi.append(rssi[key]["feature"])
        x_raw_mag.append(spectrum[key]["raw_mag_bin_m2_to_p2"])
        x_raw_rel.append(spectrum[key]["raw_rel_bin_m2_to_p2"])
        x_q4_rel.append(spectrum[key]["q4_rel_17subbin"])
        x_log_center.append([spectrum[key]["log10_raw_center_mag"]])
    return (
        samples,
        np.asarray(x_rssi, dtype=float),
        np.asarray(x_raw_mag, dtype=float),
        np.asarray(x_raw_rel, dtype=float),
        np.asarray(x_q4_rel, dtype=float),
        np.asarray(x_log_center, dtype=float),
    )


def enrich_predictions(rows: Iterable[dict], samples: Sequence[dict]) -> List[dict]:
    lookup = {i: sample for i, sample in enumerate(samples)}
    enriched = []
    for row in rows:
        sample = lookup[row["sample_index"]]
        neighbor = lookup[row["neighbor_sample_index"]]
        out = dict(row)
        out.update(
            {
                "file_name": sample["file_name"],
                "packet_index": sample["packet_index"],
                "true_display": point_display(row["true_label"]),
                "pred_display": point_display(row["pred_label"]),
                "neighbor_file_name": neighbor["file_name"],
                "neighbor_packet_index": neighbor["packet_index"],
                "neighbor_display": point_display(neighbor["label"]),
            }
        )
        enriched.append(out)
    return enriched


def build_correction_summary(predictions: Sequence[dict], baseline_feature: str = "rssi_plus") -> List[dict]:
    by_feature_scope = defaultdict(dict)
    for row in predictions:
        if row["scope"] != "all_aligned":
            continue
        by_feature_scope[row["feature_set"]][row["sample_index"]] = int(row["correct"])
    baseline = by_feature_scope[baseline_feature]
    rows = []
    for feature_set in sorted(by_feature_scope):
        if feature_set == baseline_feature:
            continue
        feature = by_feature_scope[feature_set]
        common = sorted(set(baseline) & set(feature))
        both_correct = sum(1 for i in common if baseline[i] and feature[i])
        both_wrong = sum(1 for i in common if (not baseline[i]) and (not feature[i]))
        fixed = sum(1 for i in common if (not baseline[i]) and feature[i])
        broken = sum(1 for i in common if baseline[i] and (not feature[i]))
        baseline_correct = sum(baseline[i] for i in common)
        feature_correct = sum(feature[i] for i in common)
        rows.append(
            {
                "feature_set": feature_set,
                "scope": "all_aligned",
                "packet_count": len(common),
                "rssi_plus_correct": baseline_correct,
                "feature_correct": feature_correct,
                "net_correct_gain_vs_rssi_plus": feature_correct - baseline_correct,
                "rssi_wrong_fixed_by_feature": fixed,
                "rssi_correct_broken_by_feature": broken,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
            }
        )
    return rows


def percentile_curve(x: np.ndarray, q: float) -> np.ndarray:
    if len(x) == 0:
        return np.full(len(Q4_OFFSETS), np.nan)
    return np.percentile(x, q, axis=0)


def undirected_pair(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([a, b], key=natural_label_key))


def label_indices(labels: Sequence[str]) -> Dict[str, List[int]]:
    return {
        label: [i for i, item in enumerate(labels) if item == label]
        for label in sorted(set(labels), key=natural_label_key)
    }


def binary_prediction_map(
    labels: Sequence[str],
    x: np.ndarray,
    pairs: Iterable[Tuple[str, str]],
    feature_name: str,
) -> Tuple[Dict[Tuple[Tuple[str, str], int], str], Dict[Tuple[str, str], dict]]:
    indices_by_label = label_indices(labels)
    pred_map: Dict[Tuple[Tuple[str, str], int], str] = {}
    metrics_map: Dict[Tuple[str, str], dict] = {}
    for pair in sorted(set(pairs), key=lambda p: (natural_label_key(p[0]), natural_label_key(p[1]))):
        a, b = pair
        if min(len(indices_by_label[a]), len(indices_by_label[b])) < 2:
            continue
        pair_indices = indices_by_label[a] + indices_by_label[b]
        metrics, predictions = loo_1nn(x, labels, pair_indices, "rssi_confusion_pair", feature_name)
        metrics_map[pair] = metrics
        for row in predictions:
            pred_map[(pair, row["sample_index"])] = row["pred_label"]
    return pred_map, metrics_map


def build_confusion_pair_rows(
    labels: Sequence[str],
    x_rssi: np.ndarray,
    x_raw_mag: np.ndarray,
    x_raw_rel: np.ndarray,
    x_q4: np.ndarray,
    rssi_predictions: Sequence[dict],
) -> List[dict]:
    pair_counts = Counter()
    directed_counts = Counter()
    for row in rssi_predictions:
        if row["scope"] != "all_aligned" or row["feature_set"] != "rssi_plus":
            continue
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        if true_label == pred_label:
            continue
        pair = undirected_pair(true_label, pred_label)
        pair_counts[pair] += 1
        directed_counts[(true_label, pred_label)] += 1

    rows = []
    all_indices_by_label = label_indices(labels)
    raw_mag_pred_map, raw_mag_metrics_map = binary_prediction_map(
        labels, x_raw_mag, pair_counts, "raw_bin_m2_to_p2_mag"
    )
    raw_rel_pred_map, raw_rel_metrics_map = binary_prediction_map(
        labels, x_raw_rel, pair_counts, "raw_bin_m2_to_p2_rel_db"
    )
    q4_pred_map, q4_metrics_map = binary_prediction_map(
        labels, x_q4, pair_counts, "q4_zero_padding_17subbin"
    )
    rssi_pred_map, rssi_metrics_map = binary_prediction_map(
        labels, x_rssi, pair_counts, "rssi_plus"
    )
    for pair, confusion_count in pair_counts.most_common():
        a, b = pair
        idx_a = all_indices_by_label[a]
        idx_b = all_indices_by_label[b]
        xa = x_q4[idx_a]
        xb = x_q4[idx_b]
        med_a = np.median(xa, axis=0)
        med_b = np.median(xb, axis=0)
        iqr_a = percentile_curve(xa, 75) - percentile_curve(xa, 25)
        iqr_b = percentile_curve(xb, 75) - percentile_curve(xb, 25)
        delta = med_a - med_b
        pooled_iqr = (iqr_a + iqr_b) / 2.0
        best_idx = int(np.argmax(np.abs(delta)))
        q4_binary_accuracy = ""
        rssi_binary_accuracy = ""
        raw_mag_binary_accuracy = ""
        raw_rel_binary_accuracy = ""
        pair_valid = min(len(idx_a), len(idx_b)) >= 2
        if pair_valid:
            q4_binary_accuracy = q4_metrics_map[pair]["micro_accuracy"]
            rssi_binary_accuracy = rssi_metrics_map[pair]["micro_accuracy"]
            raw_mag_binary_accuracy = raw_mag_metrics_map[pair]["micro_accuracy"]
            raw_rel_binary_accuracy = raw_rel_metrics_map[pair]["micro_accuracy"]
        rows.append(
            {
                "point_a": a,
                "point_b": b,
                "point_a_display": point_display(a),
                "point_b_display": point_display(b),
                "point_a_packet_count": len(idx_a),
                "point_b_packet_count": len(idx_b),
                "rssi_confusion_count": confusion_count,
                "rssi_confusion_a_to_b": directed_counts[(a, b)],
                "rssi_confusion_b_to_a": directed_counts[(b, a)],
                "pair_valid_for_binary_loocv": int(pair_valid),
                "rssi_plus_binary_loocv_accuracy": rssi_binary_accuracy,
                "raw_mag_bin_m2_to_p2_binary_loocv_accuracy": raw_mag_binary_accuracy,
                "raw_rel_db_bin_m2_to_p2_binary_loocv_accuracy": raw_rel_binary_accuracy,
                "q4_binary_loocv_accuracy": q4_binary_accuracy,
                "q4_median_curve_rms_delta_db": float(np.sqrt(np.mean(delta**2))),
                "q4_median_curve_max_abs_delta_db": float(np.max(np.abs(delta))),
                "q4_best_subbin_offset": Q4_OFFSETS[best_idx],
                "q4_best_subbin_delta_db": float(delta[best_idx]),
                "q4_pooled_iqr_rms_db": float(np.sqrt(np.mean(pooled_iqr**2))),
                "q4_robust_separation_ratio": float(
                    np.sqrt(np.mean(delta**2)) / max(np.sqrt(np.mean(pooled_iqr**2)), EPS)
                ),
            }
        )
    return rows


def build_stage_cascade(
    labels: Sequence[str],
    x_raw_mag: np.ndarray,
    x_q4: np.ndarray,
    rssi_predictions: Sequence[dict],
) -> Tuple[List[dict], List[dict]]:
    rssi_wrong_pairs = [
        undirected_pair(row["true_label"], row["pred_label"])
        for row in rssi_predictions
        if row["scope"] == "all_aligned"
        and row["feature_set"] == "rssi_plus"
        and row["true_label"] != row["pred_label"]
    ]
    raw_pred_map, _ = binary_prediction_map(labels, x_raw_mag, rssi_wrong_pairs, "raw_bin_m2_to_p2_mag")
    q4_pred_map, _ = binary_prediction_map(labels, x_q4, rssi_wrong_pairs, "q4_zero_padding_17subbin")

    packet_rows = []
    for row in rssi_predictions:
        if row["scope"] != "all_aligned" or row["feature_set"] != "rssi_plus":
            continue
        sample_index = row["sample_index"]
        true_label = row["true_label"]
        rssi_pred = row["pred_label"]
        rssi_correct = int(rssi_pred == true_label)

        direct_q4_pred = ""
        direct_q4_applied = 0
        direct_q4_correct = rssi_correct
        if not rssi_correct:
            pair = undirected_pair(true_label, rssi_pred)
            if (pair, sample_index) in q4_pred_map:
                direct_q4_applied = 1
                direct_q4_pred = q4_pred_map[(pair, sample_index)]
                direct_q4_correct = int(direct_q4_pred == true_label)

        raw_pred = ""
        raw_applied = 0
        after_raw_pred = rssi_pred
        after_raw_correct = rssi_correct
        if not rssi_correct:
            pair = undirected_pair(true_label, rssi_pred)
            if (pair, sample_index) in raw_pred_map:
                raw_applied = 1
                raw_pred = raw_pred_map[(pair, sample_index)]
                after_raw_pred = raw_pred
                after_raw_correct = int(raw_pred == true_label)

        q4_after_raw_pred = ""
        q4_after_raw_applied = 0
        final_pred = after_raw_pred
        final_correct = after_raw_correct
        if not after_raw_correct:
            pair = undirected_pair(true_label, after_raw_pred)
            if (pair, sample_index) in q4_pred_map:
                q4_after_raw_applied = 1
                q4_after_raw_pred = q4_pred_map[(pair, sample_index)]
                final_pred = q4_after_raw_pred
                final_correct = int(final_pred == true_label)

        packet_rows.append(
            {
                "sample_index": sample_index,
                "true_label": true_label,
                "true_display": point_display(true_label),
                "rssi_pred_label": rssi_pred,
                "rssi_pred_display": point_display(rssi_pred),
                "rssi_correct": rssi_correct,
                "q4_after_rssi_applied": direct_q4_applied,
                "q4_after_rssi_pred_label": direct_q4_pred,
                "q4_after_rssi_pred_display": point_display(direct_q4_pred) if direct_q4_pred else "",
                "q4_after_rssi_correct": direct_q4_correct,
                "raw_after_rssi_applied": raw_applied,
                "raw_after_rssi_pred_label": raw_pred,
                "raw_after_rssi_pred_display": point_display(raw_pred) if raw_pred else "",
                "after_raw_correct": after_raw_correct,
                "q4_after_raw_applied": q4_after_raw_applied,
                "q4_after_raw_pred_label": q4_after_raw_pred,
                "q4_after_raw_pred_display": point_display(q4_after_raw_pred) if q4_after_raw_pred else "",
                "final_correct_after_raw_then_q4": final_correct,
            }
        )

    total = len(packet_rows)
    rssi_correct_count = sum(row["rssi_correct"] for row in packet_rows)
    raw_correct_count = sum(row["after_raw_correct"] for row in packet_rows)
    q4_direct_correct_count = sum(row["q4_after_rssi_correct"] for row in packet_rows)
    final_correct_count = sum(row["final_correct_after_raw_then_q4"] for row in packet_rows)
    raw_fixed = sum(
        1 for row in packet_rows
        if not row["rssi_correct"] and row["raw_after_rssi_applied"] and row["after_raw_correct"]
    )
    q4_direct_fixed = sum(
        1 for row in packet_rows
        if not row["rssi_correct"] and row["q4_after_rssi_applied"] and row["q4_after_rssi_correct"]
    )
    q4_after_raw_fixed = sum(
        1 for row in packet_rows
        if not row["after_raw_correct"] and row["q4_after_raw_applied"] and row["final_correct_after_raw_then_q4"]
    )
    summary_rows = [
        {
            "stage": "RSSI+",
            "packet_count": total,
            "correct": rssi_correct_count,
            "accuracy": rssi_correct_count / total,
            "newly_fixed_vs_previous_stage": "",
            "remaining_wrong": total - rssi_correct_count,
            "notes": "baseline all-class LOOCV",
        },
        {
            "stage": "RSSI+ -> q4 pairwise",
            "packet_count": total,
            "correct": q4_direct_correct_count,
            "accuracy": q4_direct_correct_count / total,
            "newly_fixed_vs_previous_stage": q4_direct_fixed,
            "remaining_wrong": total - q4_direct_correct_count,
            "notes": "diagnostic upper bound: q4 binary LOOCV only on RSSI+ wrong true/pred pairs",
        },
        {
            "stage": "RSSI+ -> raw pairwise",
            "packet_count": total,
            "correct": raw_correct_count,
            "accuracy": raw_correct_count / total,
            "newly_fixed_vs_previous_stage": raw_fixed,
            "remaining_wrong": total - raw_correct_count,
            "notes": "diagnostic upper bound: raw mag bin[-2,+2] binary LOOCV only on RSSI+ wrong true/pred pairs",
        },
        {
            "stage": "RSSI+ -> raw pairwise -> q4 pairwise",
            "packet_count": total,
            "correct": final_correct_count,
            "accuracy": final_correct_count / total,
            "newly_fixed_vs_previous_stage": q4_after_raw_fixed,
            "remaining_wrong": total - final_correct_count,
            "notes": "diagnostic upper bound: q4 binary LOOCV on residual wrong pairs after raw stage",
        },
    ]
    return packet_rows, summary_rows


def render_top_confusion_curves_svg(
    path: str,
    rows: Sequence[dict],
    labels: Sequence[str],
    x_q4: np.ndarray,
    top_n: int,
) -> None:
    top_rows = list(rows[:top_n])
    label_indices = {
        label: [i for i, item in enumerate(labels) if item == label]
        for label in sorted(set(labels), key=natural_label_key)
    }
    cols = 3
    rows_count = int(math.ceil(len(top_rows) / cols))
    panel_w, panel_h = 410, 285
    margin_x, margin_y = 70, 95
    width = margin_x * 2 + cols * panel_w
    height = margin_y + rows_count * panel_h + 55
    y_min, y_max = -34.0, 1.0
    x_min, x_max = -2.0, 2.0

    def sx(offset: float, col: int) -> float:
        return margin_x + col * panel_w + 54 + (offset - x_min) / (x_max - x_min) * (panel_w - 90)

    def sy(value: float, row: int) -> float:
        value = max(y_min, min(y_max, value))
        return margin_y + row * panel_h + 42 + (y_max - value) / (y_max - y_min) * (panel_h - 92)

    def polyline(points: Sequence[Tuple[float, float]]) -> str:
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8"/>',
        f'<text x="{margin_x}" y="42" font-family="Arial, sans-serif" font-size="25" font-weight="700" fill="#202124">q=4 sub-bin curves for top RSSI+ confusion pairs</text>',
        f'<text x="{margin_x}" y="68" font-family="Arial, sans-serif" font-size="13" fill="#5f6368">Median curves with IQR bands; pairs are ranked by RSSI+ all-class LOOCV confusion count.</text>',
    ]
    for idx, pair_row in enumerate(top_rows):
        grid_r = idx // cols
        grid_c = idx % cols
        panel_x = margin_x + grid_c * panel_w
        panel_y = margin_y + grid_r * panel_h
        a = pair_row["point_a"]
        b = pair_row["point_b"]
        xa = x_q4[label_indices[a]]
        xb = x_q4[label_indices[b]]
        med_a = np.median(xa, axis=0)
        med_b = np.median(xb, axis=0)
        q25_a = percentile_curve(xa, 25)
        q75_a = percentile_curve(xa, 75)
        q25_b = percentile_curve(xb, 25)
        q75_b = percentile_curve(xb, 75)

        parts.append(f'<rect x="{panel_x + 20}" y="{panel_y + 10}" width="{panel_w - 35}" height="{panel_h - 30}" fill="#ffffff" stroke="#d7d7d2"/>')
        for y_tick in [-30, -20, -10, 0]:
            y = sy(y_tick, grid_r)
            parts.append(f'<line x1="{sx(-2, grid_c):.1f}" y1="{y:.1f}" x2="{sx(2, grid_c):.1f}" y2="{y:.1f}" stroke="#eeeeea"/>')
            parts.append(f'<text x="{panel_x + 24}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="10" fill="#6b6f75">{y_tick}</text>')
        for x_tick in [-2, -1, 0, 1, 2]:
            x = sx(x_tick, grid_c)
            parts.append(f'<line x1="{x:.1f}" y1="{sy(y_min, grid_r):.1f}" x2="{x:.1f}" y2="{sy(y_max, grid_r):.1f}" stroke="#f3f2ef"/>')
            parts.append(f'<text x="{x - 6:.1f}" y="{panel_y + panel_h - 37}" font-family="Arial, sans-serif" font-size="10" fill="#6b6f75">{x_tick}</text>')

        def band(q25: np.ndarray, q75: np.ndarray, color: str) -> None:
            upper = [(sx(o, grid_c), sy(v, grid_r)) for o, v in zip(Q4_OFFSETS, q75)]
            lower = [(sx(o, grid_c), sy(v, grid_r)) for o, v in reversed(list(zip(Q4_OFFSETS, q25)))]
            parts.append(f'<polygon points="{polyline(upper + lower)}" fill="{color}" fill-opacity="0.16" stroke="none"/>')

        band(q25_a, q75_a, "#2563eb")
        band(q25_b, q75_b, "#dc2626")
        pts_a = [(sx(o, grid_c), sy(v, grid_r)) for o, v in zip(Q4_OFFSETS, med_a)]
        pts_b = [(sx(o, grid_c), sy(v, grid_r)) for o, v in zip(Q4_OFFSETS, med_b)]
        parts.append(f'<polyline points="{polyline(pts_a)}" fill="none" stroke="#2563eb" stroke-width="2.2"/>')
        parts.append(f'<polyline points="{polyline(pts_b)}" fill="none" stroke="#dc2626" stroke-width="2.2"/>')
        for x, y in pts_a:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="#2563eb"/>')
        for x, y in pts_b:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="#dc2626"/>')

        q4_acc = pair_row["q4_binary_loocv_accuracy"]
        q4_acc_text = "" if q4_acc == "" else f" q4={float(q4_acc):.0%}"
        title = (
            f'{pair_row["point_a_display"]} vs {pair_row["point_b_display"]} '
            f'conf={pair_row["rssi_confusion_count"]}{q4_acc_text}'
        )
        parts.append(f'<text x="{panel_x + 26}" y="{panel_y + 30}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#202124">{html.escape(title)}</text>')
        parts.append(f'<text x="{panel_x + 26}" y="{panel_y + panel_h - 14}" font-family="Arial, sans-serif" font-size="11" fill="#2563eb">{html.escape(pair_row["point_a_display"])}</text>')
        parts.append(f'<text x="{panel_x + 102}" y="{panel_y + panel_h - 14}" font-family="Arial, sans-serif" font-size="11" fill="#dc2626">{html.escape(pair_row["point_b_display"])}</text>')

    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", default=DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-confusion-pairs", type=int, default=12)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rssi = read_rssi_plus(args.rssi_csv)
    spectrum = read_spectrum_packet_features(args.spectrum_csv)
    samples, x_rssi, x_raw_mag, x_raw_rel, x_q4_rel, x_log_center = aligned_matrices(rssi, spectrum)
    labels = [sample["label"] for sample in samples]
    label_counts = Counter(labels)
    min2_indices = [i for i, label in enumerate(labels) if label_counts[label] >= 2]
    all_indices = list(range(len(samples)))

    feature_sets = [
        ("rssi_plus", x_rssi, RSSI_PLUS_COLUMNS),
        (
            "rssi_plus_raw_bin_m2_to_p2",
            np.column_stack([x_rssi, x_raw_mag]),
            [*RSSI_PLUS_COLUMNS, *[f"raw_mag_bin_{offset:+.0f}" for offset in RAW_OFFSETS]],
        ),
        (
            "rssi_plus_q4_zero_padding_17subbin",
            np.column_stack([x_rssi, x_q4_rel]),
            [*RSSI_PLUS_COLUMNS, *[f"q4_subbin_{offset:+.2f}_db_rel" for offset in Q4_OFFSETS]],
        ),
        (
            "rssi_plus_log_raw_center",
            np.column_stack([x_rssi, x_log_center]),
            [*RSSI_PLUS_COLUMNS, "log10_raw_center_mag"],
        ),
        (
            "raw_bin_m2_to_p2_mag_only",
            x_raw_mag,
            [f"raw_mag_bin_{offset:+.0f}" for offset in RAW_OFFSETS],
        ),
        (
            "raw_bin_m2_to_p2_rel_db_only",
            x_raw_rel,
            [f"raw_bin_{offset:+.0f}_db_rel" for offset in RAW_OFFSETS],
        ),
        (
            "q4_zero_padding_17subbin_only",
            x_q4_rel,
            [f"q4_subbin_{offset:+.2f}_db_rel" for offset in Q4_OFFSETS],
        ),
    ]

    feature_rows = []
    for name, x, columns in feature_sets:
        for i, sample in enumerate(samples):
            row = {
                "feature_set": name,
                "sample_index": i,
                "file_name": sample["file_name"],
                "packet_index": sample["packet_index"],
                "position_key": sample["label"],
                "position_display": point_display(sample["label"]),
            }
            for col, value in zip(columns, x[i]):
                row[col] = value
            feature_rows.append(row)
    all_feature_columns = []
    for _, _, columns in feature_sets:
        for col in columns:
            if col not in all_feature_columns:
                all_feature_columns.append(col)
    write_csv(
        os.path.join(args.output_dir, "combined_aligned_feature_matrix.csv"),
        feature_rows,
        [
            "feature_set",
            "sample_index",
            "file_name",
            "packet_index",
            "position_key",
            "position_display",
            *all_feature_columns,
        ],
    )

    metrics_rows = []
    prediction_rows = []
    metrics_json = {
        "inputs": {
            "rssi_csv": args.rssi_csv,
            "spectrum_csv": args.spectrum_csv,
            "aligned_packet_count": len(samples),
            "aligned_location_count": len(label_counts),
        },
        "feature_definitions": {
            "rssi_plus_raw_bin_m2_to_p2": "RSSI+ plus q=1 mag_raw packet means at offsets [-2,-1,0,+1,+2].",
            "rssi_plus_q4_zero_padding_17subbin": "RSSI+ plus q=4 relative dB sub-bin curve at offsets -2:0.25:+2.",
            "rssi_plus_log_raw_center": "RSSI+ plus log10 of q=1 center-bin mag_raw packet mean. Log base is immaterial after per-fold z-scoring.",
        },
        "metrics": [],
    }
    rssi_predictions_all = []
    for feature_name, x, _ in feature_sets:
        for scope_name, indices in [
            ("valid_min2_locations", min2_indices),
            ("all_aligned", all_indices),
        ]:
            metrics, predictions = loo_1nn(x, labels, indices, scope_name, feature_name)
            metrics_json["metrics"].append(metrics)
            metrics_rows.append(
                {
                    "feature_set": metrics["feature_set"],
                    "scope": metrics["scope"],
                    "packet_count": metrics["packet_count"],
                    "location_count": metrics["location_count"],
                    "feature_dim": metrics["feature_dim"],
                    "correct": metrics["correct"],
                    "micro_accuracy": metrics["micro_accuracy"],
                    "macro_accuracy": metrics["macro_accuracy"],
                }
            )
            prediction_rows.extend(predictions)
            if feature_name == "rssi_plus" and scope_name == "all_aligned":
                rssi_predictions_all = predictions

    write_csv(
        os.path.join(args.output_dir, "combined_matching_comparison.csv"),
        metrics_rows,
        [
            "feature_set",
            "scope",
            "packet_count",
            "location_count",
            "feature_dim",
            "correct",
            "micro_accuracy",
            "macro_accuracy",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "combined_loocv_predictions.csv"),
        enrich_predictions(prediction_rows, samples),
        [
            "feature_set",
            "scope",
            "sample_index",
            "file_name",
            "packet_index",
            "true_label",
            "true_display",
            "pred_label",
            "pred_display",
            "correct",
            "neighbor_sample_index",
            "neighbor_file_name",
            "neighbor_packet_index",
            "neighbor_display",
            "neighbor_distance_sq",
        ],
    )
    correction_rows = build_correction_summary(prediction_rows)
    write_csv(
        os.path.join(args.output_dir, "combined_vs_rssi_correction_summary.csv"),
        correction_rows,
        [
            "feature_set",
            "scope",
            "packet_count",
            "rssi_plus_correct",
            "feature_correct",
            "net_correct_gain_vs_rssi_plus",
            "rssi_wrong_fixed_by_feature",
            "rssi_correct_broken_by_feature",
            "both_correct",
            "both_wrong",
        ],
    )

    pair_rows = build_confusion_pair_rows(labels, x_rssi, x_raw_mag, x_raw_rel, x_q4_rel, rssi_predictions_all)
    pair_fieldnames = [
        "point_a",
        "point_b",
        "point_a_display",
        "point_b_display",
        "point_a_packet_count",
        "point_b_packet_count",
        "rssi_confusion_count",
        "rssi_confusion_a_to_b",
        "rssi_confusion_b_to_a",
        "pair_valid_for_binary_loocv",
        "rssi_plus_binary_loocv_accuracy",
        "raw_mag_bin_m2_to_p2_binary_loocv_accuracy",
        "raw_rel_db_bin_m2_to_p2_binary_loocv_accuracy",
        "q4_binary_loocv_accuracy",
        "q4_median_curve_rms_delta_db",
        "q4_median_curve_max_abs_delta_db",
        "q4_best_subbin_offset",
        "q4_best_subbin_delta_db",
        "q4_pooled_iqr_rms_db",
        "q4_robust_separation_ratio",
    ]
    write_csv(
        os.path.join(args.output_dir, "rssi_confusion_q4_pair_separability.csv"),
        pair_rows,
        pair_fieldnames,
    )
    write_csv(
        os.path.join(args.output_dir, "rssi_confusion_pair_binary_separability.csv"),
        pair_rows,
        pair_fieldnames,
    )

    cascade_packet_rows, cascade_summary_rows = build_stage_cascade(
        labels, x_raw_mag, x_q4_rel, rssi_predictions_all
    )
    write_csv(
        os.path.join(args.output_dir, "rssi_raw_q4_stage_cascade_packets.csv"),
        cascade_packet_rows,
        [
            "sample_index",
            "true_label",
            "true_display",
            "rssi_pred_label",
            "rssi_pred_display",
            "rssi_correct",
            "q4_after_rssi_applied",
            "q4_after_rssi_pred_label",
            "q4_after_rssi_pred_display",
            "q4_after_rssi_correct",
            "raw_after_rssi_applied",
            "raw_after_rssi_pred_label",
            "raw_after_rssi_pred_display",
            "after_raw_correct",
            "q4_after_raw_applied",
            "q4_after_raw_pred_label",
            "q4_after_raw_pred_display",
            "final_correct_after_raw_then_q4",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "rssi_raw_q4_stage_cascade_summary.csv"),
        cascade_summary_rows,
        [
            "stage",
            "packet_count",
            "correct",
            "accuracy",
            "newly_fixed_vs_previous_stage",
            "remaining_wrong",
            "notes",
        ],
    )

    svg_path = os.path.join(args.output_dir, "rssi_confusion_q4_pair_curves_top12.svg")
    render_top_confusion_curves_svg(svg_path, pair_rows, labels, x_q4_rel, args.top_confusion_pairs)

    valid_pairs = [row for row in pair_rows if row["pair_valid_for_binary_loocv"]]
    high_pairs = [
        row for row in valid_pairs
        if row["q4_binary_loocv_accuracy"] != "" and float(row["q4_binary_loocv_accuracy"]) >= 0.75
    ]
    low_pairs = [
        row for row in valid_pairs
        if row["q4_binary_loocv_accuracy"] != "" and float(row["q4_binary_loocv_accuracy"]) <= 0.60
    ]
    raw_high_pairs = [
        row for row in valid_pairs
        if row["raw_mag_bin_m2_to_p2_binary_loocv_accuracy"] != ""
        and float(row["raw_mag_bin_m2_to_p2_binary_loocv_accuracy"]) >= 0.75
    ]
    raw_low_pairs = [
        row for row in valid_pairs
        if row["raw_mag_bin_m2_to_p2_binary_loocv_accuracy"] != ""
        and float(row["raw_mag_bin_m2_to_p2_binary_loocv_accuracy"]) <= 0.60
    ]
    metrics_json["rssi_confusion_q4_separability_summary"] = {
        "rssi_all_aligned_wrong_packets": int(sum(1 for row in rssi_predictions_all if not row["correct"])),
        "rssi_confusion_pair_count": len(pair_rows),
        "binary_valid_pair_count": len(valid_pairs),
        "median_raw_mag_binary_accuracy": (
            float(np.median([float(row["raw_mag_bin_m2_to_p2_binary_loocv_accuracy"]) for row in valid_pairs]))
            if valid_pairs else None
        ),
        "median_q4_binary_accuracy": (
            float(np.median([float(row["q4_binary_loocv_accuracy"]) for row in valid_pairs]))
            if valid_pairs else None
        ),
        "pairs_with_raw_mag_binary_accuracy_ge_0_75": len(raw_high_pairs),
        "pairs_with_raw_mag_binary_accuracy_le_0_60": len(raw_low_pairs),
        "pairs_with_q4_binary_accuracy_ge_0_75": len(high_pairs),
        "pairs_with_q4_binary_accuracy_le_0_60": len(low_pairs),
        "top_pair_curve_plot_svg": svg_path,
    }
    metrics_json["rssi_raw_q4_stage_cascade_summary"] = cascade_summary_rows
    with open(os.path.join(args.output_dir, "combined_confusion_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics_rows, indent=2, ensure_ascii=False))
    print(json.dumps(metrics_json["rssi_confusion_q4_separability_summary"], indent=2, ensure_ascii=False))
    print(json.dumps(cascade_summary_rows, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
