#!/usr/bin/env python3
"""Compare RSSI+ Top-K reranking with raw bin[-2,+2] and q=4 FFT features."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from analyze_knn_topk_raw import (
    DEFAULT_RSSI_CSV,
    DEFAULT_SPECTRUM_CSV,
    RAW_OFFSETS,
    class_knn_rank_for_fold,
    file_stem,
    label_count,
    natural_label_key,
    parse_float,
    parse_int,
    point_display,
    point_label,
    read_rssi_plus,
    write_csv,
)


DEFAULT_OUTPUT_DIR = (
    "v2_output/20260624_zero_padding_fft_q1_q4_point_compare/"
    "matching_tsne/knn_topk_analysis"
)
Q4_OFFSETS = [x / 4.0 for x in range(-8, 9)]
TOPK_VALUES = [1, 2, 3, 5, 10]
Q4_CLASS_K_VALUES = [1, 3, 5]
RSSI_CLASS_K = 3
RAW_CLASS_K = 1
RAW_KEEP_K_VALUES = [1, 2, 3]


PacketKey = Tuple[str, int]


def offset_key(value: str) -> float:
    return round(float(value), 2)


def read_spectrum_raw_q4(path: str) -> Dict[PacketKey, dict]:
    raw_wanted = set(RAW_OFFSETS)
    q4_wanted = set(Q4_OFFSETS)
    raw_sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    q4_sums = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    meta = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = parse_int(row["q"])
            offset = offset_key(row["subbin_offset"])
            key = (file_stem(row["file_name"]), parse_int(row["packet_index"]))
            if q == 1 and offset in raw_wanted:
                raw_sums[key][offset][0] += parse_float(row["mag_raw"])
                raw_sums[key][offset][1] += 1
            elif q == 4 and offset in q4_wanted:
                q4_sums[key][offset][0] += parse_float(row["mag_db_rel_peak"])
                q4_sums[key][offset][1] += 1
            else:
                continue
            meta[key] = {
                "file_name": row["file_name"],
                "packet_index": key[1],
                "label": point_label(row["corridor_id"], row["position_id"]),
            }

    packets = {}
    for key in sorted(set(raw_sums) & set(q4_sums)):
        if any(raw_sums[key][offset][1] == 0 for offset in RAW_OFFSETS):
            continue
        if any(q4_sums[key][offset][1] == 0 for offset in Q4_OFFSETS):
            continue
        item = dict(meta[key])
        item["key"] = key
        item["raw_mag_bin_m2_to_p2"] = [
            raw_sums[key][offset][0] / raw_sums[key][offset][1]
            for offset in RAW_OFFSETS
        ]
        item["q4_rel_17subbin"] = [
            q4_sums[key][offset][0] / q4_sums[key][offset][1]
            for offset in Q4_OFFSETS
        ]
        packets[key] = item
    return packets


def align_samples(rssi: Dict[PacketKey, dict], spec: Dict[PacketKey, dict]):
    common = sorted(
        set(rssi) & set(spec),
        key=lambda key: (natural_label_key(rssi[key]["label"]), rssi[key]["packet_index"], rssi[key]["file_name"]),
    )
    samples = []
    x_rssi = []
    x_raw = []
    x_q4 = []
    labels = []
    for key in common:
        if rssi[key]["label"] != spec[key]["label"]:
            raise ValueError(f"Label mismatch for {key}: {rssi[key]['label']} vs {spec[key]['label']}")
        samples.append(
            {
                "key": key,
                "file_name": spec[key]["file_name"],
                "packet_index": key[1],
                "label": rssi[key]["label"],
            }
        )
        x_rssi.append(rssi[key]["rssi_plus"])
        x_raw.append(spec[key]["raw_mag_bin_m2_to_p2"])
        x_q4.append(spec[key]["q4_rel_17subbin"])
        labels.append(rssi[key]["label"])
    return samples, x_rssi, x_raw, x_q4, labels


def evaluate_topk(
    x_rssi: Sequence[Sequence[float]],
    x_raw: Sequence[Sequence[float]],
    x_q4: Sequence[Sequence[float]],
    labels: Sequence[str],
    sample_indices: Sequence[int],
    top_k: int,
    q4_class_k: int,
    raw_keep_k: int,
) -> Tuple[dict, List[dict]]:
    rows = []
    rssi_correct = 0
    topk_contains = 0
    raw_correct = 0
    q4_correct = 0
    raw_then_q4_correct = 0
    raw_fixed = raw_broke = 0
    q4_fixed = q4_broke = 0
    raw_then_q4_fixed_vs_raw = raw_then_q4_broke_vs_raw = 0

    for sample_index in sample_indices:
        true_label = labels[sample_index]
        rssi_rank = class_knn_rank_for_fold(
            x_rssi,
            labels,
            sample_indices,
            sample_index,
            RSSI_CLASS_K,
        )
        rssi_candidates = [label for label, _ in rssi_rank[:top_k]]
        rssi_pred = rssi_candidates[0] if rssi_candidates else ""
        rssi_ok = int(rssi_pred == true_label)
        rssi_correct += rssi_ok
        contains = int(true_label in rssi_candidates)
        topk_contains += contains

        raw_rank = class_knn_rank_for_fold(
            x_raw,
            labels,
            sample_indices,
            sample_index,
            RAW_CLASS_K,
            candidate_labels=rssi_candidates,
        )
        raw_candidates = [label for label, _ in raw_rank]
        raw_pred = raw_candidates[0] if raw_candidates else ""
        raw_ok = int(raw_pred == true_label)
        raw_correct += raw_ok
        if not rssi_ok and raw_ok:
            raw_fixed += 1
        if rssi_ok and not raw_ok:
            raw_broke += 1

        q4_rank = class_knn_rank_for_fold(
            x_q4,
            labels,
            sample_indices,
            sample_index,
            q4_class_k,
            candidate_labels=rssi_candidates,
        )
        q4_candidates = [label for label, _ in q4_rank]
        q4_pred = q4_candidates[0] if q4_candidates else ""
        q4_ok = int(q4_pred == true_label)
        q4_correct += q4_ok
        if not rssi_ok and q4_ok:
            q4_fixed += 1
        if rssi_ok and not q4_ok:
            q4_broke += 1

        narrowed = raw_candidates[: min(raw_keep_k, len(raw_candidates))]
        raw_then_q4_rank = class_knn_rank_for_fold(
            x_q4,
            labels,
            sample_indices,
            sample_index,
            q4_class_k,
            candidate_labels=narrowed,
        )
        raw_then_q4_pred = raw_then_q4_rank[0][0] if raw_then_q4_rank else ""
        raw_then_q4_ok = int(raw_then_q4_pred == true_label)
        raw_then_q4_correct += raw_then_q4_ok
        if not raw_ok and raw_then_q4_ok:
            raw_then_q4_fixed_vs_raw += 1
        if raw_ok and not raw_then_q4_ok:
            raw_then_q4_broke_vs_raw += 1

        rows.append(
            {
                "sample_index": sample_index,
                "true_label": true_label,
                "true_display": point_display(true_label),
                "rssi_top_k": top_k,
                "q4_class_k": q4_class_k,
                "raw_keep_k": raw_keep_k,
                "rssi_top1_label": rssi_pred,
                "rssi_top1_display": point_display(rssi_pred) if rssi_pred else "",
                "rssi_top1_correct": rssi_ok,
                "rssi_topk_candidates": ";".join(rssi_candidates),
                "rssi_topk_candidate_displays": ";".join(point_display(label) for label in rssi_candidates),
                "true_in_rssi_topk": contains,
                "raw_rerank_label": raw_pred,
                "raw_rerank_display": point_display(raw_pred) if raw_pred else "",
                "raw_rerank_correct": raw_ok,
                "q4_rerank_label": q4_pred,
                "q4_rerank_display": point_display(q4_pred) if q4_pred else "",
                "q4_rerank_correct": q4_ok,
                "raw_then_q4_candidate_labels": ";".join(narrowed),
                "raw_then_q4_candidate_displays": ";".join(point_display(label) for label in narrowed),
                "raw_then_q4_label": raw_then_q4_pred,
                "raw_then_q4_display": point_display(raw_then_q4_pred) if raw_then_q4_pred else "",
                "raw_then_q4_correct": raw_then_q4_ok,
            }
        )

    n = len(sample_indices)
    metrics = {
        "rssi_class_k": RSSI_CLASS_K,
        "raw_class_k": RAW_CLASS_K,
        "q4_class_k": q4_class_k,
        "rssi_top_k": top_k,
        "raw_keep_k": raw_keep_k,
        "packet_count": n,
        "location_count": len(label_count(labels, sample_indices)),
        "rssi_top1_correct": rssi_correct,
        "rssi_top1_accuracy": rssi_correct / n,
        "rssi_topk_contains_true": topk_contains,
        "rssi_topk_recall": topk_contains / n,
        "raw_rerank_correct": raw_correct,
        "raw_rerank_accuracy": raw_correct / n,
        "raw_gain_vs_rssi": raw_correct - rssi_correct,
        "raw_fixed_rssi_wrong": raw_fixed,
        "raw_broke_rssi_correct": raw_broke,
        "q4_rerank_correct": q4_correct,
        "q4_rerank_accuracy": q4_correct / n,
        "q4_gain_vs_rssi": q4_correct - rssi_correct,
        "q4_fixed_rssi_wrong": q4_fixed,
        "q4_broke_rssi_correct": q4_broke,
        "raw_then_q4_correct": raw_then_q4_correct,
        "raw_then_q4_accuracy": raw_then_q4_correct / n,
        "raw_then_q4_gain_vs_rssi": raw_then_q4_correct - rssi_correct,
        "raw_then_q4_gain_vs_raw": raw_then_q4_correct - raw_correct,
        "q4_fixed_raw_wrong": raw_then_q4_fixed_vs_raw,
        "q4_broke_raw_correct": raw_then_q4_broke_vs_raw,
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
    spec = read_spectrum_raw_q4(args.spectrum_csv)
    samples, x_rssi, x_raw, x_q4, labels = align_samples(rssi, spec)
    counts = Counter(labels)
    scopes = {
        "valid_min2_locations": [i for i, label in enumerate(labels) if counts[label] >= 2],
        "all_aligned": list(range(len(samples))),
    }

    summary_rows = []
    prediction_rows = []
    for scope, indices in scopes.items():
        for top_k in TOPK_VALUES:
            for q4_class_k in Q4_CLASS_K_VALUES:
                for raw_keep_k in RAW_KEEP_K_VALUES:
                    if raw_keep_k > top_k:
                        continue
                    metrics, rows = evaluate_topk(
                        x_rssi,
                        x_raw,
                        x_q4,
                        labels,
                        indices,
                        top_k,
                        q4_class_k,
                        raw_keep_k,
                    )
                    metrics["scope"] = scope
                    summary_rows.append(metrics)
                    if (
                        scope == "valid_min2_locations"
                        and q4_class_k == 1
                        and raw_keep_k == min(2, top_k)
                    ):
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
                            prediction_rows.append(out)

    fieldnames = [
        "scope",
        "rssi_class_k",
        "raw_class_k",
        "q4_class_k",
        "rssi_top_k",
        "raw_keep_k",
        "packet_count",
        "location_count",
        "rssi_top1_correct",
        "rssi_top1_accuracy",
        "rssi_topk_contains_true",
        "rssi_topk_recall",
        "raw_rerank_correct",
        "raw_rerank_accuracy",
        "raw_gain_vs_rssi",
        "raw_fixed_rssi_wrong",
        "raw_broke_rssi_correct",
        "q4_rerank_correct",
        "q4_rerank_accuracy",
        "q4_gain_vs_rssi",
        "q4_fixed_rssi_wrong",
        "q4_broke_rssi_correct",
        "raw_then_q4_correct",
        "raw_then_q4_accuracy",
        "raw_then_q4_gain_vs_rssi",
        "raw_then_q4_gain_vs_raw",
        "q4_fixed_raw_wrong",
        "q4_broke_raw_correct",
    ]
    write_csv(
        os.path.join(args.output_dir, "rssi_topk_raw_q4_compare_summary.csv"),
        summary_rows,
        fieldnames,
    )
    write_csv(
        os.path.join(args.output_dir, "rssi_topk_raw_q4_compare_predictions_q4k1.csv"),
        prediction_rows,
        [
            "scope",
            "sample_index",
            "file_name",
            "packet_index",
            "true_label",
            "true_display",
            "rssi_top_k",
            "q4_class_k",
            "raw_keep_k",
            "rssi_top1_label",
            "rssi_top1_display",
            "rssi_top1_correct",
            "rssi_topk_candidates",
            "rssi_topk_candidate_displays",
            "true_in_rssi_topk",
            "raw_rerank_label",
            "raw_rerank_display",
            "raw_rerank_correct",
            "q4_rerank_label",
            "q4_rerank_display",
            "q4_rerank_correct",
            "raw_then_q4_candidate_labels",
            "raw_then_q4_candidate_displays",
            "raw_then_q4_label",
            "raw_then_q4_display",
            "raw_then_q4_correct",
        ],
    )

    metrics_json = {
        "inputs": {
            "rssi_csv": args.rssi_csv,
            "spectrum_csv": args.spectrum_csv,
            "aligned_packet_count": len(samples),
            "aligned_location_count": len(counts),
        },
        "method": {
            "rssi_topk": "RSSI+ location-level KNN ranking with rssi_class_k=3.",
            "raw_rerank": "raw mag_raw bin[-2,+2] reranks all RSSI+ Top-K candidates with raw_class_k=1.",
            "q4_rerank": "q=4 relative 17-subbin curve reranks all RSSI+ Top-K candidates.",
            "raw_then_q4": "raw first ranks RSSI+ Top-K candidates, keeps raw_keep_k candidates, then q=4 reranks that narrowed set.",
        },
        "summary": summary_rows,
    }
    with open(os.path.join(args.output_dir, "topk_raw_q4_compare_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    main_rows = [
        row for row in summary_rows
        if row["scope"] == "valid_min2_locations"
        and row["q4_class_k"] == 1
        and row["raw_keep_k"] == min(2, row["rssi_top_k"])
    ]
    print(json.dumps(main_rows, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
