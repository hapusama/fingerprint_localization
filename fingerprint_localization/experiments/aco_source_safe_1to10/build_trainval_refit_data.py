#!/usr/bin/env python3
"""Build the final train+validation refit dataset used by the paper mainline.

The source-safe split must already exist.  This builder merges its original
train and validation source packets for the final augmented training set while
leaving the original test source packets untouched.  Original validation rows
are retained only as a diagnostic split; they overlap the refit training set
and must not be used for configuration selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import build_group_safe_1to10_data as group_safe


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
DATA_ROOT = PROJECT_ROOT / "fingerprint_localization" / "data" / "mainline_202607"
DEFAULT_GROUP_SAFE_DIR = EXPERIMENT_DIR / "group_safe_1to10"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "group_safe_trainval_refit"

DEFAULT_ORIG_RSSI = DATA_ROOT / "inputs" / "rssi_plus_packet_level_54points.csv"
DEFAULT_ORIG_RAW = DATA_ROOT / "inputs" / "lora_frequency_s17_54points.csv"
DEFAULT_ORIG_SPECTRUM = DATA_ROOT / "external" / "subbin_spectrum_long.csv"
DEFAULT_NOISY_RSSI = EXPERIMENT_DIR / "data" / "noisy_rssi_plus_packet_level_54points.csv"
DEFAULT_NOISY_RAW = EXPERIMENT_DIR / "data" / "noisy_lora_frequency_s17_54points.csv"
DEFAULT_NOISY_SPECTRUM = EXPERIMENT_DIR / "data" / "noisy_subbin_spectrum_long.csv"


def read_source_split(path: Path) -> dict[str, set[tuple[str, int]]]:
    rows, _ = group_safe.read_csv(path)
    sources: dict[str, set[tuple[str, int]]] = {"train": set(), "val": set(), "test": set()}
    for row in rows:
        split = row["split"]
        if split not in sources:
            raise ValueError(f"Unexpected split {split!r} in {path}")
        sources[split].add((row["source_file_stem"], int(float(row["source_packet_index"]))))
    return sources


def run(args: argparse.Namespace) -> dict:
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source_split_path = args.group_safe_dir / "data" / "source_packet_split.csv"
    sources = read_source_split(source_split_path)
    train_refit_sources = sources["train"] | sources["val"]
    test_sources = sources["test"]
    if train_refit_sources & test_sources:
        raise ValueError("Test source packets overlap the train+validation refit sources")

    orig_rssi, orig_rssi_fields = group_safe.read_csv(args.orig_rssi)
    orig_raw, orig_raw_fields = group_safe.read_csv(args.orig_raw)
    orig_spectrum, orig_spectrum_fields = group_safe.read_csv(args.orig_spectrum)
    noisy_rssi, _ = group_safe.read_csv(args.noisy_rssi)
    noisy_raw, _ = group_safe.read_csv(args.noisy_raw)
    noisy_spectrum, _ = group_safe.read_csv(args.noisy_spectrum)

    train_rssi = group_safe.filter_noisy_rows(noisy_rssi, train_refit_sources)
    train_raw = group_safe.filter_noisy_rows(noisy_raw, train_refit_sources)
    train_spectrum = group_safe.filter_noisy_rows(noisy_spectrum, train_refit_sources)

    # The old validation split is diagnostic after refit because its sources
    # now occur in training.  Keep the original, unaugmented rows for auditing.
    val_rssi = group_safe.filter_original_rows(orig_rssi, sources["val"])
    val_raw = group_safe.filter_original_rows(orig_raw, sources["val"])
    val_spectrum = group_safe.filter_original_rows(orig_spectrum, sources["val"])
    test_rssi = group_safe.filter_original_rows(orig_rssi, test_sources)
    test_raw = group_safe.filter_original_rows(orig_raw, test_sources)
    test_spectrum = group_safe.filter_original_rows(orig_spectrum, test_sources)

    combined_rssi = train_rssi + val_rssi + test_rssi
    combined_raw = train_raw + val_raw + test_raw
    combined_spectrum = train_spectrum + val_spectrum + test_spectrum
    split_rows = group_safe.build_split_assignments(train_rssi, val_rssi, test_rssi)
    split_counts = Counter(row["split"] for row in split_rows)

    outputs = {
        "rssi": data_dir / "noisy_rssi_plus_packet_level_54points.csv",
        "raw": data_dir / "noisy_lora_frequency_s17_54points.csv",
        "spectrum": data_dir / "noisy_subbin_spectrum_long.csv",
        "split": data_dir / "split_assignments.csv",
    }
    group_safe.write_csv(outputs["rssi"], combined_rssi, group_safe.add_metadata_fields(orig_rssi_fields))
    group_safe.write_csv(outputs["raw"], combined_raw, group_safe.add_metadata_fields(orig_raw_fields))
    group_safe.write_csv(
        outputs["spectrum"],
        combined_spectrum,
        group_safe.add_metadata_fields(orig_spectrum_fields),
    )
    group_safe.write_csv(
        outputs["split"],
        split_rows,
        [
            "split",
            "position_key",
            "file_stem",
            "packet_index",
            "source_file_stem",
            "source_packet_index",
            "augmentation_id",
        ],
    )

    metadata = {
        "protocol": (
            "Final refit dataset: train uses original train+val source packets with 1:10 augmentation; "
            "test remains original source packets. Validation rows are retained only for diagnostics and "
            "are not independent of train."
        ),
        "source_split_path": group_safe.portable_path(source_split_path),
        "source_counts": {"train_refit": len(train_refit_sources), "test": len(test_sources)},
        "row_counts": {name: split_counts[name] for name in ("train", "val", "test")},
        "test_source_overlap_with_train": len(train_refit_sources & test_sources),
        "paths": {name: group_safe.portable_path(path) for name, path in outputs.items()},
    }
    with (data_dir / "refit_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-safe-dir", type=Path, default=DEFAULT_GROUP_SAFE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--orig-rssi", type=Path, default=DEFAULT_ORIG_RSSI)
    parser.add_argument("--orig-raw", type=Path, default=DEFAULT_ORIG_RAW)
    parser.add_argument("--orig-spectrum", type=Path, default=DEFAULT_ORIG_SPECTRUM)
    parser.add_argument("--noisy-rssi", type=Path, default=DEFAULT_NOISY_RSSI)
    parser.add_argument("--noisy-raw", type=Path, default=DEFAULT_NOISY_RAW)
    parser.add_argument("--noisy-spectrum", type=Path, default=DEFAULT_NOISY_SPECTRUM)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
