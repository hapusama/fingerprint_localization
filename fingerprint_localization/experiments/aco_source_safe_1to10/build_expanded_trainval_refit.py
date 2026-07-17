#!/usr/bin/env python3
"""Build the formal ExpandedReal train+validation refit dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from build_expanded_source_safe_1to10 import (
    RSSI_COLUMNS,
    S17_COLUMNS,
    add_metadata_fields,
    augment_packet_rows,
    augment_spectrum_packet,
    file_stem,
    iter_spectrum_packets,
    packet_key,
    read_csv,
    split_assignment_rows,
    write_checksums,
    write_csv,
)


VERSION = "ExpandedReal-649-v1-source-safe-trainval-refit-v1"


def source_key(row: dict) -> tuple[str, int]:
    return file_stem(row["source_file_name"]), int(float(row["source_packet_index"]))


def write_refit_spectrum(
    parent_path: Path,
    split_path: Path,
    output_path: Path,
    original_train: set[tuple[str, int]],
    original_val: set[tuple[str, int]],
    original_test: set[tuple[str, int]],
    augment_factor: int,
    rng: random.Random,
    spectrum_stats: dict,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    writer = None
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        for fields, _key, rows in iter_spectrum_packets(split_path):
            if writer is None:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=add_metadata_fields(fields),
                    extrasaction="ignore",
                )
                writer.writeheader()
            row_source = source_key(rows[0])
            augmentation_id = rows[0]["augmentation_id"]
            keep_existing_train = row_source in original_train and augmentation_id != "orig"
            keep_test = row_source in original_test and augmentation_id == "orig"
            if keep_existing_train or keep_test:
                writer.writerows(rows)
                row_count += len(rows)

        for fields, key, rows in iter_spectrum_packets(parent_path):
            if key not in original_val:
                continue
            if writer is None:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=add_metadata_fields(fields),
                    extrasaction="ignore",
                )
                writer.writeheader()
            for augmentation_index in range(augment_factor):
                augmented = augment_spectrum_packet(
                    rows,
                    augmentation_index,
                    rng,
                    spectrum_stats,
                )
                writer.writerows(augmented)
                row_count += len(augmented)
    return row_count


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parent_rssi = args.parent_dir / "inputs" / "rssi_plus_packet_level_32points_649.csv"
    parent_s17 = args.parent_dir / "inputs" / "lora_frequency_s17_32points_649.csv"
    parent_spectrum = args.parent_dir / "inputs" / "subbin_spectrum_long_q1q4_32points_649.csv"
    split_rssi = args.split_dir / "data" / "noisy_rssi_plus_packet_level_32points_649.csv"
    split_s17 = args.split_dir / "data" / "noisy_lora_frequency_s17_32points_649.csv"
    split_spectrum = args.split_dir / "data" / "noisy_subbin_spectrum_long_32points_649.csv"
    source_split_path = args.split_dir / "data" / "source_packet_split.csv"
    metadata_path = args.split_dir / "metadata" / "split_augmentation_metadata.json"

    source_fields, source_rows = read_csv(source_split_path)
    source_sets = {
        name: {
            (row["source_file_stem"], int(float(row["source_packet_index"])))
            for row in source_rows
            if row["split"] == name
        }
        for name in ["train", "val", "test"]
    }
    if {name: len(keys) for name, keys in source_sets.items()} != {
        "train": 393,
        "val": 128,
        "test": 128,
    }:
        raise RuntimeError("Unexpected source split counts")
    refit_train_sources = source_sets["train"] | source_sets["val"]
    if refit_train_sources & source_sets["test"]:
        raise RuntimeError("Refit train sources overlap test sources")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rssi_stats = metadata["noise_stats"]["rssi"]
    s17_stats = metadata["noise_stats"]["s17"]
    spectrum_stats = metadata["noise_stats"]["spectrum"]
    if spectrum_stats["train_packet_count"] != 393:
        raise RuntimeError("Spectrum noise statistics are not train-source-only")

    rssi_fields, split_rssi_rows = read_csv(split_rssi)
    s17_fields, split_s17_rows = read_csv(split_s17)
    parent_rssi_fields, parent_rssi_rows = read_csv(parent_rssi)
    parent_s17_fields, parent_s17_rows = read_csv(parent_s17)

    existing_train_rssi = [
        row for row in split_rssi_rows
        if source_key(row) in source_sets["train"] and row["augmentation_id"] != "orig"
    ]
    existing_train_s17 = [
        row for row in split_s17_rows
        if source_key(row) in source_sets["train"] and row["augmentation_id"] != "orig"
    ]
    test_rssi = [
        row for row in split_rssi_rows
        if source_key(row) in source_sets["test"] and row["augmentation_id"] == "orig"
    ]
    test_s17 = [
        row for row in split_s17_rows
        if source_key(row) in source_sets["test"] and row["augmentation_id"] == "orig"
    ]
    val_parent_rssi = [row for row in parent_rssi_rows if packet_key(row) in source_sets["val"]]
    val_parent_s17 = [row for row in parent_s17_rows if packet_key(row) in source_sets["val"]]
    val_augmented_rssi = augment_packet_rows(
        val_parent_rssi,
        RSSI_COLUMNS,
        rssi_stats,
        random.Random(args.seed + 404),
        args.augment_factor,
    )
    val_augmented_s17 = augment_packet_rows(
        val_parent_s17,
        S17_COLUMNS,
        s17_stats,
        random.Random(args.seed + 505),
        args.augment_factor,
    )

    combined_rssi = existing_train_rssi + val_augmented_rssi + test_rssi
    combined_s17 = existing_train_s17 + val_augmented_s17 + test_s17
    if len(combined_rssi) != 5_338 or len(combined_s17) != 5_338:
        raise RuntimeError("Unexpected refit packet-row count")
    data_dir = args.output_dir / "data"
    output_rssi = data_dir / "noisy_rssi_plus_packet_level_32points_649.csv"
    output_s17 = data_dir / "noisy_lora_frequency_s17_32points_649.csv"
    output_spectrum = data_dir / "noisy_subbin_spectrum_long_32points_649.csv"
    write_csv(output_rssi, add_metadata_fields(parent_rssi_fields), combined_rssi)
    write_csv(output_s17, add_metadata_fields(parent_s17_fields), combined_s17)
    spectrum_row_count = write_refit_spectrum(
        parent_spectrum,
        split_spectrum,
        output_spectrum,
        source_sets["train"],
        source_sets["val"],
        source_sets["test"],
        args.augment_factor,
        random.Random(args.seed + 606),
        spectrum_stats,
    )
    if spectrum_row_count != 5_338 * 352:
        raise RuntimeError(f"Unexpected refit spectrum row count: {spectrum_row_count}")

    refit_split = {"train": refit_train_sources, "val": set(), "test": source_sets["test"]}
    assignments = split_assignment_rows(combined_rssi, refit_split)
    assignment_fields = [
        "split",
        "position_key",
        "file_stem",
        "packet_index",
        "source_file_stem",
        "source_packet_index",
        "augmentation_id",
    ]
    write_csv(data_dir / "split_assignments.csv", assignment_fields, assignments)
    refit_source_rows = []
    for row in source_rows:
        original_split = row["split"]
        refit_source_rows.append(
            {
                **row,
                "original_split": original_split,
                "refit_split": "test" if original_split == "test" else "train",
            }
        )
    write_csv(
        data_dir / "source_packet_split.csv",
        list(source_fields) + ["original_split", "refit_split"],
        refit_source_rows,
    )

    assignment_counts = Counter(row["split"] for row in assignments)
    output_metadata = {
        "version": VERSION,
        "parent_split_version": metadata["derived_version"],
        "seed": args.seed,
        "protocol": (
            "After the frozen mainline configuration is selected, original train and validation sources "
            "form the refit training set. Existing train augmentations are reused; former validation "
            "sources receive 10 augmentations using statistics estimated from the original 393 train "
            "sources only. Test sources remain untouched originals."
        ),
        "source_counts": {"train_refit": len(refit_train_sources), "test": len(source_sets["test"])},
        "packet_row_counts": dict(assignment_counts),
        "spectrum_row_count": spectrum_row_count,
        "test_source_overlap_with_refit_train": len(refit_train_sources & source_sets["test"]),
        "former_validation_source_count": len(source_sets["val"]),
        "former_validation_augmented_rows": len(val_augmented_rssi),
        "noise_statistics_source_count": spectrum_stats["train_packet_count"],
        "noise_statistics_include_former_validation": False,
        "test_augmentation": False,
    }
    metadata_dir = args.output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "refit_metadata.json").write_text(
        json.dumps(output_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    scripts_dir = args.output_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), scripts_dir / Path(__file__).name)
    (args.output_dir / "README.md").write_text(
        "# ExpandedReal train+validation refit\n\n"
        "This dataset is only for the frozen final ACO refit/test run. Former validation sources are no "
        "longer an independent evaluation set. The 128 test sources remain untouched and source-disjoint.\n",
        encoding="utf-8",
    )
    write_checksums(args.output_dir)
    print(json.dumps(output_metadata, indent=2, ensure_ascii=False))
    return output_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--augment-factor", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
