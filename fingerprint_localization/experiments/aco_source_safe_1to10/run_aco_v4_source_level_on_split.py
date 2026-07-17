#!/usr/bin/env python3
"""Run ACO v4 with source-level templates and source-safe T_seg calibration."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import run_aco_v4_on_split as runner


aco4 = runner.aco4
aco2 = aco4.aco2
base = aco2.base
ORIGINAL_BUILD_TEMPLATES = aco2.build_templates
ORIGINAL_BUILD_PROTOTYPES = aco2.build_segment_prototypes
ORIGINAL_CLASS_RANK = base.class_rank
AUGMENTATION_SUFFIX = re.compile(r"_aug\d+$")


def source_id(sample) -> tuple[str, int]:
    stem = Path(sample.file_name).stem
    return AUGMENTATION_SUFFIX.sub("", stem), int(sample.packet_index)


def vector_median(rows: Sequence[Sequence[float]]) -> list[float]:
    return [aco2.median([row[index] for row in rows]) for index in range(len(rows[0]))]


def aggregate_group(group: Sequence) -> object:
    if not group:
        raise ValueError("Cannot aggregate an empty source group")
    labels = {sample.label for sample in group}
    if len(labels) != 1:
        raise ValueError(f"Source group label mismatch: {labels}")
    segment_count = len(group[0].segment_shapes)
    if any(len(sample.segment_shapes) != segment_count for sample in group):
        raise ValueError("Source group segment count mismatch")
    stem, packet_index = source_id(group[0])
    segment_shapes = [
        vector_median([sample.segment_shapes[segment] for sample in group])
        for segment in range(segment_count)
    ]
    segment_zw = [
        vector_median([sample.segment_zw[segment] for sample in group])
        for segment in range(segment_count)
    ]
    segment_q4_curves = [
        vector_median([sample.segment_q4_curves[segment] for sample in group])
        for segment in range(segment_count)
    ]
    return aco2.SegmentPacket(
        key=(stem, packet_index),
        file_name=f"{stem}.bin",
        packet_index=packet_index,
        label=group[0].label,
        rssi_plus=vector_median([sample.rssi_plus for sample in group]),
        segment_shapes=segment_shapes,
        segment_zw=segment_zw,
        segment_q4_curves=segment_q4_curves,
        segment_q4_reliable=[
            sum(bool(sample.segment_q4_reliable[segment]) for sample in group) * 2 >= len(group)
            for segment in range(segment_count)
        ],
        segment_q4_peak_offsets=[
            aco2.median([sample.segment_q4_peak_offsets[segment] for sample in group])
            for segment in range(segment_count)
        ],
        segment_q4_peak_to_side_db=[
            aco2.median([sample.segment_q4_peak_to_side_db[segment] for sample in group])
            for segment in range(segment_count)
        ],
        segment_q4_dev_from_packet=[
            aco2.median([sample.segment_q4_dev_from_packet[segment] for sample in group])
            for segment in range(segment_count)
        ],
    )


def aggregate_sources(samples: Sequence, indices: Sequence[int]) -> tuple[list, list[str]]:
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for index in indices:
        groups[source_id(samples[index])].append(samples[index])
    aggregated = [aggregate_group(groups[key]) for key in sorted(groups)]
    labels = [sample.label for sample in aggregated]
    return aggregated, labels


def source_level_build_templates(packets, labels, train_indices, chirp_shapes, chirp_struct, args):
    aggregated, aggregated_labels = aggregate_sources(packets, train_indices)
    return ORIGINAL_BUILD_TEMPLATES(
        aggregated,
        aggregated_labels,
        list(range(len(aggregated))),
        chirp_shapes,
        chirp_struct,
        args,
    )


def source_level_build_prototypes(packets, labels, train_indices):
    aggregated, aggregated_labels = aggregate_sources(packets, train_indices)
    return ORIGINAL_BUILD_PROTOTYPES(
        aggregated,
        aggregated_labels,
        list(range(len(aggregated))),
    )


def make_source_level_class_rank(samples: Sequence):
    source_ids = [source_id(sample) for sample in samples]

    def class_rank(
        rows: Sequence[Sequence[float]],
        labels: Sequence[str],
        train_indices: Sequence[int],
        test_index: int,
        class_neighbor_k: int,
        candidate_labels: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        if len(rows) != len(source_ids):
            return ORIGINAL_CLASS_RANK(
                rows,
                labels,
                train_indices,
                test_index,
                class_neighbor_k,
                candidate_labels,
            )
        grouped_rows: dict[tuple[str, int], list[Sequence[float]]] = defaultdict(list)
        grouped_labels: dict[tuple[str, int], str] = {}
        for index in train_indices:
            key = source_ids[index]
            grouped_rows[key].append(rows[index])
            grouped_labels[key] = labels[index]
        aggregated_rows = []
        aggregated_labels = []
        for key in sorted(grouped_rows):
            aggregated_rows.append(vector_median(grouped_rows[key]))
            aggregated_labels.append(grouped_labels[key])
        means, stds = base.zscore_stats(aggregated_rows)
        if candidate_labels is None:
            candidate_labels = sorted(set(aggregated_labels), key=base.natural_label_key)
        ranked = []
        for label in candidate_labels:
            distances = [
                base.squared_distance(row, rows[test_index], means, stds)
                for row, row_label in zip(aggregated_rows, aggregated_labels)
                if row_label == label
            ]
            if not distances:
                continue
            distances.sort()
            k_eff = min(class_neighbor_k, len(distances))
            ranked.append((label, sum(distances[:k_eff]) / k_eff))
        ranked.sort(key=lambda item: (item[1], base.natural_label_key(item[0])))
        return ranked

    return class_rank


def stratified_source_folds(labels: Sequence[str], fold_count: int, seed: int) -> list[list[int]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[label].append(index)
    folds = [[] for _ in range(fold_count)]
    rng = random.Random(seed)
    for label in sorted(by_label, key=base.natural_label_key):
        indices = list(by_label[label])
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            folds[offset % fold_count].append(index)
    return [sorted(fold) for fold in folds]


def calibrate_t_seg(
    samples: Sequence,
    train_indices: Sequence[int],
    q4_offsets: Sequence[float],
    chirp_shapes,
    chirp_struct,
    args: argparse.Namespace,
    fold_count: int,
) -> tuple[float, dict]:
    aggregated, labels = aggregate_sources(samples, train_indices)
    folds = stratified_source_folds(labels, fold_count, args.seed + 707)
    all_indices = set(range(len(aggregated)))
    rssi_rows = [sample.rssi_plus for sample in aggregated]
    values = []
    old_t = args.t_seg_resolved
    args.t_seg_resolved = 1.0
    for fold in folds:
        fit_indices = sorted(all_indices - set(fold))
        templates = ORIGINAL_BUILD_TEMPLATES(
            aggregated,
            labels,
            fit_indices,
            chirp_shapes,
            chirp_struct,
            args,
        )
        prototypes = ORIGINAL_BUILD_PROTOTYPES(aggregated, labels, fit_indices)
        for test_index in fold:
            ranked = ORIGINAL_CLASS_RANK(
                rssi_rows,
                labels,
                fit_indices,
                test_index,
                args.rssi_class_k,
            )
            candidates = [label for label, _score in ranked[: args.top_k]]
            if not candidates:
                continue
            rssi_costs = {label: score for label, score in ranked if label in candidates}
            _costs, _rows, meta = aco4.build_observation_costs_v4(
                aggregated[test_index],
                candidates,
                rssi_costs,
                templates,
                prototypes,
                q4_offsets,
                args,
            )
            values.append(float(meta["segment_cost_std"]))
    args.t_seg_resolved = old_t
    if len(values) != len(aggregated):
        raise RuntimeError(f"T_seg calibration coverage mismatch: {len(values)} != {len(aggregated)}")
    ordered = sorted(values)
    quantile = min(1.0, max(0.0, args.t_seg_quantile))
    resolved = ordered[round((len(ordered) - 1) * quantile)]
    resolved = resolved if resolved > aco4.EPS else 1.0
    metadata = {
        "method": "stratified source-level out-of-fold segment-cost-std quantile",
        "source_count": len(aggregated),
        "fold_count": fold_count,
        "quantile": quantile,
        "resolved_T_seg": resolved,
        "distribution": {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "p90": ordered[round((len(ordered) - 1) * 0.90)],
            "p95": ordered[round((len(ordered) - 1) * 0.95)],
            "max": ordered[-1],
        },
    }
    return resolved, metadata


def load_samples(args: argparse.Namespace):
    aco_args = runner.build_args(args)
    rssi_packets = base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, _thresholds = base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = base.align_samples(rssi_packets, symbol_packets)
    samples = aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = runner.split_runner.load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = aco2.prepare_chirp_fields(aco_args, labels)
    return aco_args, samples, split_indices, q4_offsets, chirp_shapes, chirp_struct, chirp_metadata


def run(args: argparse.Namespace) -> dict:
    if args.t_seg is not None:
        raise ValueError("Source-level runner calibrates T_seg; do not pass --t-seg")
    aco_args, samples, split_indices, q4_offsets, chirp_shapes, chirp_struct, chirp_metadata = load_samples(args)
    train_source_count = len({source_id(samples[index]) for index in split_indices["train"]})
    resolved_t_seg, calibration = calibrate_t_seg(
        samples,
        split_indices["train"],
        q4_offsets,
        chirp_shapes,
        chirp_struct,
        aco_args,
        args.source_folds,
    )
    args.t_seg = resolved_t_seg
    aco2.build_templates = source_level_build_templates
    aco2.build_segment_prototypes = source_level_build_prototypes
    if args.source_level_rssi:
        base.class_rank = make_source_level_class_rank(samples)
    metadata = runner.run(args)
    source_metadata = {
        "method": "ACO v4 source-level templates with source-safe T_seg calibration",
        "source_level_rssi": args.source_level_rssi,
        "augmentation_suffix_pattern": AUGMENTATION_SUFFIX.pattern,
        "train_row_count": len(split_indices["train"]),
        "train_source_count": train_source_count,
        "source_aggregation": "component-wise median across all augmented rows for one source packet",
        "template_n_definition": "independent source packets after aggregation",
        "T_seg_calibration": calibration,
        "chirp_template_field": chirp_metadata,
        "summary": metadata["summary"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_level_metadata.json").write_text(
        json.dumps(source_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(source_metadata, indent=2, ensure_ascii=False))
    return source_metadata


def parse_args() -> argparse.Namespace:
    parser = runner.parse_args.__wrapped__() if hasattr(runner.parse_args, "__wrapped__") else None
    if parser is not None:
        raise RuntimeError("Unexpected wrapped parser")
    # Mirror the established runner surface, then add source-level controls.
    original_argv = __import__("sys").argv
    try:
        __import__("sys").argv = [original_argv[0]]
        defaults = runner.parse_args()
    finally:
        __import__("sys").argv = original_argv
    parser = argparse.ArgumentParser()
    for name, value in vars(defaults).items():
        option = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(option, action="store_true", default=value)
        elif isinstance(value, Path):
            parser.add_argument(option, type=Path, default=value)
        elif isinstance(value, int):
            parser.add_argument(option, type=int, default=value)
        elif isinstance(value, float):
            parser.add_argument(option, type=float, default=value)
        else:
            parser.add_argument(option, default=value)
    parser.add_argument("--source-folds", type=int, default=5)
    parser.add_argument("--source-level-rssi", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
