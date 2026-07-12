#!/usr/bin/env python3
"""KNN and RSSI+ Top-K reranking with raw LoRa bin[-2,+2].

Pure standard-library implementation so the analysis can run even when NumPy is
not available in the current shell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


RSSI_PLUS_COLUMNS = [
    "snr",
    "realtime_average_rssi",
    "median_rssi",
    "mode_rssi",
    "rssi_variance",
    "residual",
]
RAW_OFFSETS = [-2.0, -1.0, 0.0, 1.0, 2.0]
DEFAULT_RSSI_CSV = (
    "v2_output/20260623_from_raw/experiment_32_lora/input/"
    "rssi_plus_packet_level_32points.csv"
)
DEFAULT_SPECTRUM_CSV = (
    "v2_output/20260624_zero_padding_fft_q1_q4_point_compare/"
    "subbin_spectrum_long.csv"
)
DEFAULT_OUTPUT_DIR = (
    "v2_output/20260624_zero_padding_fft_q1_q4_point_compare/"
    "matching_tsne/knn_topk_analysis"
)
PACKET_K_VALUES = [1, 3, 5, 7, 9, 15]
CLASS_K_VALUES = [1, 3, 5]
TOPK_VALUES = [1, 2, 3, 5, 10]

PacketKey = Tuple[str, int]


def file_stem(file_name: str) -> str:
    return os.path.splitext(os.path.basename(file_name))[0]


def parse_float(value: str) -> float:
    return float(value)


def parse_int(value: str) -> int:
    return int(float(value))


def point_label(corridor_id: str, position_id: str) -> str:
    return f"{parse_int(corridor_id)}_{parse_int(position_id)}"


def point_display(label: str) -> str:
    corridor, position = label.split("_", 1)
    return f"c{corridor}p{position}"


def natural_label_key(label: str) -> Tuple[int, int]:
    corridor, position = label.split("_", 1)
    return int(corridor), int(position)


def write_csv(path: str, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rssi_plus(path: str) -> Dict[PacketKey, dict]:
    packets: Dict[PacketKey, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
            label = row.get("position_key") or point_label(row["corridor_id"], row["location_id"])
            packets[key] = {
                "key": key,
                "file_name": row["file_name"],
                "packet_index": key[1],
                "label": label,
                "rssi_plus": [parse_float(row[col]) for col in RSSI_PLUS_COLUMNS],
            }
    return packets


def offset_key(value: str) -> float:
    return round(float(value), 2)


def read_raw_mag_features(path: str) -> Dict[PacketKey, dict]:
    wanted = set(RAW_OFFSETS)
    sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    meta = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if parse_int(row["q"]) != 1:
                continue
            offset = offset_key(row["subbin_offset"])
            if offset not in wanted:
                continue
            key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
            sums[key][offset][0] += parse_float(row["mag_raw"])
            sums[key][offset][1] += 1
            meta[key] = {
                "file_name": row["file_name"],
                "packet_index": key[1],
                "label": point_label(row["corridor_id"], row["position_id"]),
                "preamble_len": parse_int(row["filename_preamble_len"]),
                "skip_preamble_symbols": parse_int(row["skip_preamble_symbols"]),
                "feature_symbols": parse_int(row["feature_symbols"]),
            }

    packets: Dict[PacketKey, dict] = {}
    for key, offset_sums in sums.items():
        if any(offset_sums[offset][1] == 0 for offset in RAW_OFFSETS):
            continue
        item = dict(meta[key])
        item["key"] = key
        item["raw_mag_bin_m2_to_p2"] = [
            offset_sums[offset][0] / offset_sums[offset][1] for offset in RAW_OFFSETS
        ]
        packets[key] = item
    return packets


def align_samples(
    rssi: Dict[PacketKey, dict],
    raw: Dict[PacketKey, dict],
) -> Tuple[List[dict], List[List[float]], List[List[float]], List[str]]:
    samples = []
    x_rssi = []
    x_raw = []
    labels = []
    common = sorted(
        set(rssi) & set(raw),
        key=lambda key: (natural_label_key(rssi[key]["label"]), rssi[key]["packet_index"], rssi[key]["file_name"]),
    )
    for key in common:
        if rssi[key]["label"] != raw[key]["label"]:
            raise ValueError(f"Label mismatch for {key}: {rssi[key]['label']} vs {raw[key]['label']}")
        sample = {
            "key": key,
            "file_name": raw[key]["file_name"],
            "packet_index": key[1],
            "label": rssi[key]["label"],
        }
        samples.append(sample)
        x_rssi.append(rssi[key]["rssi_plus"])
        x_raw.append(raw[key]["raw_mag_bin_m2_to_p2"])
        labels.append(rssi[key]["label"])
    return samples, x_rssi, x_raw, labels


def feature_stats(rows: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    dim = len(rows[0])
    means = [sum(row[j] for row in rows) / len(rows) for j in range(dim)]
    stds = []
    for j in range(dim):
        var = sum((row[j] - means[j]) ** 2 for row in rows) / len(rows)
        std = math.sqrt(var)
        stds.append(std if std > 1e-12 else 1.0)
    return means, stds


def squared_distance(a: Sequence[float], b: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> float:
    return sum((((a[j] - means[j]) / stds[j]) - ((b[j] - means[j]) / stds[j])) ** 2 for j in range(len(a)))


def label_count(labels: Sequence[str], indices: Sequence[int]) -> Counter:
    return Counter(labels[i] for i in indices)


def vote_rank(neighbor_labels: Sequence[str], neighbor_distances: Sequence[float]) -> List[str]:
    stats = defaultdict(lambda: {"count": 0, "dist_sum": 0.0, "nearest": float("inf")})
    for label, distance in zip(neighbor_labels, neighbor_distances):
        stats[label]["count"] += 1
        stats[label]["dist_sum"] += distance
        stats[label]["nearest"] = min(stats[label]["nearest"], distance)
    return sorted(
        stats,
        key=lambda label: (
            -stats[label]["count"],
            stats[label]["dist_sum"] / stats[label]["count"],
            stats[label]["nearest"],
            natural_label_key(label),
        ),
    )


def packet_knn(
    x: Sequence[Sequence[float]],
    labels: Sequence[str],
    sample_indices: Sequence[int],
    k: int,
) -> Tuple[dict, List[dict]]:
    predictions = []
    correct = 0
    for global_i in sample_indices:
        train_indices = [idx for idx in sample_indices if idx != global_i]
        train_rows = [x[idx] for idx in train_indices]
        means, stds = feature_stats(train_rows)
        distances = [
            (idx, squared_distance(x[idx], x[global_i], means, stds))
            for idx in train_indices
        ]
        distances.sort(key=lambda item: (item[1], item[0]))
        k_eff = min(k, len(distances))
        nearest = distances[:k_eff]
        neighbor_labels = [labels[idx] for idx, _ in nearest]
        ranked = vote_rank(neighbor_labels, [dist for _, dist in nearest])
        pred = ranked[0]
        is_correct = int(pred == labels[global_i])
        correct += is_correct
        predictions.append(
            {
                "sample_index": global_i,
                "true_label": labels[global_i],
                "pred_label": pred,
                "correct": is_correct,
                "packet_neighbor_k": k,
                "effective_neighbor_k": k_eff,
                "neighbor_labels": ";".join(neighbor_labels),
            }
        )
    metrics = {
        "packet_neighbor_k": k,
        "packet_count": len(sample_indices),
        "location_count": len(label_count(labels, sample_indices)),
        "correct": correct,
        "accuracy": correct / len(sample_indices),
    }
    return metrics, predictions


def class_knn_rank_for_fold(
    x: Sequence[Sequence[float]],
    labels: Sequence[str],
    sample_indices: Sequence[int],
    test_index: int,
    class_neighbor_k: int,
    candidate_labels: Sequence[str] | None = None,
) -> List[Tuple[str, float]]:
    train_indices = [idx for idx in sample_indices if idx != test_index]
    train_rows = [x[idx] for idx in train_indices]
    means, stds = feature_stats(train_rows)
    if candidate_labels is None:
        candidate_labels = sorted({labels[idx] for idx in train_indices}, key=natural_label_key)
    ranked = []
    for label in candidate_labels:
        label_distances = [
            squared_distance(x[idx], x[test_index], means, stds)
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


def topk_raw_rerank(
    x_rssi: Sequence[Sequence[float]],
    x_raw: Sequence[Sequence[float]],
    labels: Sequence[str],
    sample_indices: Sequence[int],
    rssi_class_k: int,
    raw_class_k: int,
    top_k: int,
) -> Tuple[dict, List[dict]]:
    rows = []
    rssi_top1_correct = 0
    raw_correct = 0
    topk_contains_true = 0
    raw_conditional_correct = 0
    rssi_wrong_fixed = 0
    rssi_correct_broken = 0
    for global_i in sample_indices:
        true_label = labels[global_i]
        rssi_ranked = class_knn_rank_for_fold(x_rssi, labels, sample_indices, global_i, rssi_class_k)
        candidates = [label for label, _ in rssi_ranked[:top_k]]
        rssi_pred = candidates[0] if candidates else ""
        rssi_ok = int(rssi_pred == true_label)
        rssi_top1_correct += rssi_ok
        contains = int(true_label in candidates)
        topk_contains_true += contains

        raw_pred = ""
        raw_ok = 0
        if candidates:
            raw_ranked = class_knn_rank_for_fold(
                x_raw,
                labels,
                sample_indices,
                global_i,
                raw_class_k,
                candidate_labels=candidates,
            )
            raw_pred = raw_ranked[0][0] if raw_ranked else ""
            raw_ok = int(raw_pred == true_label)
            raw_correct += raw_ok
            if contains:
                raw_conditional_correct += raw_ok
            if (not rssi_ok) and raw_ok:
                rssi_wrong_fixed += 1
            if rssi_ok and (not raw_ok):
                rssi_correct_broken += 1

        rows.append(
            {
                "sample_index": global_i,
                "true_label": true_label,
                "true_display": point_display(true_label),
                "rssi_class_k": rssi_class_k,
                "raw_class_k": raw_class_k,
                "top_k": top_k,
                "rssi_top1_label": rssi_pred,
                "rssi_top1_display": point_display(rssi_pred) if rssi_pred else "",
                "rssi_top1_correct": rssi_ok,
                "rssi_topk_candidates": ";".join(candidates),
                "rssi_topk_candidate_displays": ";".join(point_display(label) for label in candidates),
                "true_in_rssi_topk": contains,
                "raw_rerank_label": raw_pred,
                "raw_rerank_display": point_display(raw_pred) if raw_pred else "",
                "raw_rerank_correct": raw_ok,
            }
        )

    n = len(sample_indices)
    metrics = {
        "rssi_class_k": rssi_class_k,
        "raw_class_k": raw_class_k,
        "top_k": top_k,
        "packet_count": n,
        "location_count": len(label_count(labels, sample_indices)),
        "rssi_top1_correct": rssi_top1_correct,
        "rssi_top1_accuracy": rssi_top1_correct / n,
        "rssi_topk_contains_true": topk_contains_true,
        "rssi_topk_recall": topk_contains_true / n,
        "raw_rerank_correct": raw_correct,
        "raw_rerank_accuracy": raw_correct / n,
        "raw_rerank_conditional_correct": raw_conditional_correct,
        "raw_rerank_conditional_accuracy_when_true_in_topk": (
            raw_conditional_correct / topk_contains_true if topk_contains_true else ""
        ),
        "raw_gain_vs_rssi_top1": raw_correct - rssi_top1_correct,
        "rssi_wrong_fixed_by_raw": rssi_wrong_fixed,
        "rssi_correct_broken_by_raw": rssi_correct_broken,
    }
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", default=DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rssi = read_rssi_plus(args.rssi_csv)
    raw = read_raw_mag_features(args.spectrum_csv)
    samples, x_rssi, x_raw, labels = align_samples(rssi, raw)
    counts = Counter(labels)
    scopes = {
        "valid_min2_locations": [i for i, label in enumerate(labels) if counts[label] >= 2],
        "all_aligned": list(range(len(samples))),
    }

    packet_metric_rows = []
    packet_prediction_rows = []
    for scope, indices in scopes.items():
        for k in PACKET_K_VALUES:
            metrics, predictions = packet_knn(x_rssi, labels, indices, k)
            metrics["scope"] = scope
            packet_metric_rows.append(metrics)
            for row in predictions:
                sample = samples[row["sample_index"]]
                out = dict(row)
                out.update(
                    {
                        "scope": scope,
                        "file_name": sample["file_name"],
                        "packet_index": sample["packet_index"],
                        "true_display": point_display(row["true_label"]),
                        "pred_display": point_display(row["pred_label"]),
                    }
                )
                packet_prediction_rows.append(out)

    write_csv(
        os.path.join(args.output_dir, "rssi_packet_knn_accuracy.csv"),
        packet_metric_rows,
        ["scope", "packet_neighbor_k", "packet_count", "location_count", "correct", "accuracy"],
    )
    write_csv(
        os.path.join(args.output_dir, "rssi_packet_knn_predictions.csv"),
        packet_prediction_rows,
        [
            "scope",
            "sample_index",
            "file_name",
            "packet_index",
            "true_label",
            "true_display",
            "pred_label",
            "pred_display",
            "correct",
            "packet_neighbor_k",
            "effective_neighbor_k",
            "neighbor_labels",
        ],
    )

    topk_metric_rows = []
    topk_prediction_rows_default = []
    for scope, indices in scopes.items():
        for rssi_class_k in CLASS_K_VALUES:
            for raw_class_k in CLASS_K_VALUES:
                for top_k in TOPK_VALUES:
                    metrics, rows = topk_raw_rerank(
                        x_rssi,
                        x_raw,
                        labels,
                        indices,
                        rssi_class_k,
                        raw_class_k,
                        top_k,
                    )
                    metrics["scope"] = scope
                    topk_metric_rows.append(metrics)
                    if scope == "valid_min2_locations" and rssi_class_k == 3 and raw_class_k == 3:
                        for row in rows:
                            sample = samples[row["sample_index"]]
                            out = dict(row)
                            out.update(
                                {
                                    "scope": scope,
                                    "file_name": sample["file_name"],
                                    "packet_index": sample["packet_index"],
                                }
                            )
                            topk_prediction_rows_default.append(out)

    write_csv(
        os.path.join(args.output_dir, "rssi_topk_raw_rerank_summary.csv"),
        topk_metric_rows,
        [
            "scope",
            "rssi_class_k",
            "raw_class_k",
            "top_k",
            "packet_count",
            "location_count",
            "rssi_top1_correct",
            "rssi_top1_accuracy",
            "rssi_topk_contains_true",
            "rssi_topk_recall",
            "raw_rerank_correct",
            "raw_rerank_accuracy",
            "raw_rerank_conditional_correct",
            "raw_rerank_conditional_accuracy_when_true_in_topk",
            "raw_gain_vs_rssi_top1",
            "rssi_wrong_fixed_by_raw",
            "rssi_correct_broken_by_raw",
        ],
    )
    write_csv(
        os.path.join(args.output_dir, "rssi_topk_raw_rerank_predictions_k3.csv"),
        topk_prediction_rows_default,
        [
            "scope",
            "sample_index",
            "file_name",
            "packet_index",
            "true_label",
            "true_display",
            "rssi_class_k",
            "raw_class_k",
            "top_k",
            "rssi_top1_label",
            "rssi_top1_display",
            "rssi_top1_correct",
            "rssi_topk_candidates",
            "rssi_topk_candidate_displays",
            "true_in_rssi_topk",
            "raw_rerank_label",
            "raw_rerank_display",
            "raw_rerank_correct",
        ],
    )

    metrics_json = {
        "inputs": {
            "rssi_csv": args.rssi_csv,
            "spectrum_csv": args.spectrum_csv,
            "aligned_packet_count": len(samples),
            "aligned_location_count": len(counts),
        },
        "feature_definitions": {
            "rssi_plus": RSSI_PLUS_COLUMNS,
            "raw_bin_m2_to_p2": f"q=1 mag_raw packet means at offsets {RAW_OFFSETS}",
        },
        "method": {
            "packet_knn": "LOOCV packet-level KNN majority vote with per-fold z-score; ties by mean distance.",
            "rssi_topk": "Location-level KNN ranking: each point score is mean distance to its class_neighbor_k nearest training packets in RSSI+ space.",
            "raw_rerank": "Within RSSI+ Top-K point candidates, choose point with lowest class-KNN score in raw mag_raw bin[-2,+2] space.",
        },
        "packet_knn_accuracy": packet_metric_rows,
        "topk_raw_rerank": topk_metric_rows,
    }
    with open(os.path.join(args.output_dir, "knn_topk_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    print(json.dumps(packet_metric_rows, indent=2, ensure_ascii=False))
    default_rows = [
        row for row in topk_metric_rows
        if row["scope"] == "valid_min2_locations"
        and row["rssi_class_k"] == 3
        and row["raw_class_k"] == 3
    ]
    print(json.dumps(default_rows, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
