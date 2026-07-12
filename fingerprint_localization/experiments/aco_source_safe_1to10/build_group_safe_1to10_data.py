#!/usr/bin/env python3
"""Build a group-safe train-augmented split for the Gaussian-noise data.

The original experiment augmented all packets first and then split augmented
copies, which leaks source packets across train/val/test.  This builder first
splits original packet keys by label, then keeps only train source packets'
1:10 augmented copies.  Validation and test are original, unaugmented packets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
DATA_ROOT = PROJECT_ROOT / "fingerprint_localization" / "data" / "mainline_202607"
DEFAULT_ORIG_RSSI = DATA_ROOT / "inputs" / "rssi_plus_packet_level_54points.csv"
DEFAULT_ORIG_RAW = DATA_ROOT / "inputs" / "lora_frequency_s17_54points.csv"
DEFAULT_ORIG_SPECTRUM = DATA_ROOT / "external" / "subbin_spectrum_long.csv"
DEFAULT_NOISY_RSSI = EXPERIMENT_DIR / "data" / "noisy_rssi_plus_packet_level_54points.csv"
DEFAULT_NOISY_RAW = EXPERIMENT_DIR / "data" / "noisy_lora_frequency_s17_54points.csv"
DEFAULT_NOISY_SPECTRUM = EXPERIMENT_DIR / "data" / "noisy_subbin_spectrum_long.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "group_safe_1to10"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def file_stem(file_name: str) -> str:
    return Path(file_name).stem


def packet_key(row: dict) -> tuple[str, int]:
    return file_stem(row["file_name"]), int(float(row["packet_index"]))


def source_key(row: dict) -> tuple[str, int]:
    file_name = row.get("source_file_name") or row.get("file_name", "")
    packet_index = row.get("source_packet_index") or row.get("packet_index", "")
    return file_stem(file_name), int(float(packet_index))


def add_metadata(row: dict, augmentation_id: str = "orig") -> dict:
    out = dict(row)
    out.setdefault("source_file_name", out.get("file_name", ""))
    out.setdefault("source_packet_index", out.get("packet_index", ""))
    out.setdefault("augmentation_id", augmentation_id)
    return out


def add_metadata_fields(fieldnames: Sequence[str]) -> list[str]:
    out = list(fieldnames)
    for field in ["source_file_name", "source_packet_index", "augmentation_id"]:
        if field not in out:
            out.append(field)
    return out


def label_from_rssi(row: dict) -> str:
    return row["position_key"]


def stratified_source_split(rssi_rows: Sequence[dict], seed: int) -> tuple[dict[str, set[tuple[str, int]]], list[dict]]:
    by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in rssi_rows:
        by_label[label_from_rssi(row)].append(packet_key(row))

    rng = random.Random(seed)
    split: dict[str, set[tuple[str, int]]] = {"train": set(), "val": set(), "test": set()}
    rows: list[dict] = []
    for label in sorted(by_label):
        keys = list(dict.fromkeys(by_label[label]))
        rng.shuffle(keys)
        n = len(keys)
        if n == 1:
            n_train = 1
            n_val = 0
        elif n == 2:
            n_train = 1
            n_val = 0
        elif n == 3:
            n_train = 1
            n_val = 1
        else:
            n_train = max(2, int(round(n * 0.6)))
            n_val = max(1, int(round(n * 0.2)))
            if n_train + n_val >= n:
                n_val = 1
                n_train = n - 2
        parts = {
            "train": keys[:n_train],
            "val": keys[n_train : n_train + n_val],
            "test": keys[n_train + n_val :],
        }
        for split_name, split_keys in parts.items():
            split[split_name].update(split_keys)
            for stem, packet_index in split_keys:
                rows.append(
                    {
                        "split": split_name,
                        "position_key": label,
                        "source_file_stem": stem,
                        "source_packet_index": packet_index,
                    }
                )
    return split, rows


def filter_noisy_rows(rows: Sequence[dict], train_sources: set[tuple[str, int]]) -> list[dict]:
    return [dict(row) for row in rows if source_key(row) in train_sources]


def filter_original_rows(rows: Sequence[dict], sources: set[tuple[str, int]]) -> list[dict]:
    return [add_metadata(row) for row in rows if packet_key(row) in sources]


def build_split_assignments(
    train_rssi: Sequence[dict],
    val_rssi: Sequence[dict],
    test_rssi: Sequence[dict],
) -> list[dict]:
    rows: list[dict] = []
    for split_name, split_rows in [("train", train_rssi), ("val", val_rssi), ("test", test_rssi)]:
        for row in split_rows:
            rows.append(
                {
                    "split": split_name,
                    "position_key": row["position_key"],
                    "file_stem": file_stem(row["file_name"]),
                    "packet_index": int(float(row["packet_index"])),
                    "source_file_stem": file_stem(row.get("source_file_name") or row["file_name"]),
                    "source_packet_index": int(float(row.get("source_packet_index") or row["packet_index"])),
                    "augmentation_id": row.get("augmentation_id", "orig"),
                }
            )
    return rows


def split_sources_from_assignments(rows: Sequence[dict]) -> dict[str, set[tuple[str, int]]]:
    out: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        out[row["split"]].add((row["source_file_stem"], int(float(row["source_packet_index"]))))
    return out


def run(args: argparse.Namespace) -> dict:
    data_dir = args.output_dir / "data"
    results_dir = args.output_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    orig_rssi, orig_rssi_fields = read_csv(args.orig_rssi)
    orig_raw, orig_raw_fields = read_csv(args.orig_raw)
    orig_spectrum, orig_spectrum_fields = read_csv(args.orig_spectrum)
    noisy_rssi, noisy_rssi_fields = read_csv(args.noisy_rssi)
    noisy_raw, noisy_raw_fields = read_csv(args.noisy_raw)
    noisy_spectrum, noisy_spectrum_fields = read_csv(args.noisy_spectrum)

    available_sources = set(packet_key(row) for row in orig_rssi)
    available_sources &= set(packet_key(row) for row in orig_raw)
    available_sources &= set(packet_key(row) for row in orig_spectrum)
    aligned_orig_rssi = [row for row in orig_rssi if packet_key(row) in available_sources]

    source_split, source_rows = stratified_source_split(aligned_orig_rssi, args.seed)
    train_sources = source_split["train"]
    val_sources = source_split["val"]
    test_sources = source_split["test"]

    train_rssi = filter_noisy_rows(noisy_rssi, train_sources)
    train_raw = filter_noisy_rows(noisy_raw, train_sources)
    train_spectrum = filter_noisy_rows(noisy_spectrum, train_sources)
    val_rssi = filter_original_rows(orig_rssi, val_sources)
    val_raw = filter_original_rows(orig_raw, val_sources)
    val_spectrum = filter_original_rows(orig_spectrum, val_sources)
    test_rssi = filter_original_rows(orig_rssi, test_sources)
    test_raw = filter_original_rows(orig_raw, test_sources)
    test_spectrum = filter_original_rows(orig_spectrum, test_sources)

    combined_rssi = train_rssi + val_rssi + test_rssi
    combined_raw = train_raw + val_raw + test_raw
    combined_spectrum = train_spectrum + val_spectrum + test_spectrum

    split_rows = build_split_assignments(train_rssi, val_rssi, test_rssi)
    split_sources = split_sources_from_assignments(split_rows)
    source_overlap = {
        "train_val": len(split_sources["train"] & split_sources["val"]),
        "train_test": len(split_sources["train"] & split_sources["test"]),
        "val_test": len(split_sources["val"] & split_sources["test"]),
    }

    rssi_path = data_dir / "noisy_rssi_plus_packet_level_54points.csv"
    raw_path = data_dir / "noisy_lora_frequency_s17_54points.csv"
    spectrum_path = data_dir / "noisy_subbin_spectrum_long.csv"
    split_path = data_dir / "split_assignments.csv"
    source_split_path = data_dir / "source_packet_split.csv"

    write_csv(rssi_path, combined_rssi, add_metadata_fields(orig_rssi_fields))
    write_csv(raw_path, combined_raw, add_metadata_fields(orig_raw_fields))
    write_csv(spectrum_path, combined_spectrum, add_metadata_fields(orig_spectrum_fields))
    write_csv(
        split_path,
        split_rows,
        ["split", "position_key", "file_stem", "packet_index", "source_file_stem", "source_packet_index", "augmentation_id"],
    )
    write_csv(source_split_path, source_rows, ["split", "position_key", "source_file_stem", "source_packet_index"])

    split_counts = Counter(row["split"] for row in split_rows)
    split_location_counts = {
        split_name: len({row["position_key"] for row in split_rows if row["split"] == split_name})
        for split_name in ["train", "val", "test"]
    }
    source_counts = {name: len(values) for name, values in source_split.items()}
    metadata = {
        "protocol": "Split original packet sources first; use 1:10 augmented copies only for train; val/test are original packets.",
        "seed": args.seed,
        "input_paths": {
            "orig_rssi": portable_path(args.orig_rssi),
            "orig_raw": portable_path(args.orig_raw),
            "orig_spectrum": portable_path(args.orig_spectrum),
            "noisy_rssi": portable_path(args.noisy_rssi),
            "noisy_raw": portable_path(args.noisy_raw),
            "noisy_spectrum": portable_path(args.noisy_spectrum),
        },
        "output_paths": {
            "rssi": portable_path(rssi_path),
            "raw": portable_path(raw_path),
            "spectrum": portable_path(spectrum_path),
            "split": portable_path(split_path),
            "source_split": portable_path(source_split_path),
        },
        "source_packet_counts": source_counts,
        "row_counts": {
            "rssi": len(combined_rssi),
            "raw": len(combined_raw),
            "spectrum": len(combined_spectrum),
            "split_assignments": len(split_rows),
            "train_assignments": split_counts["train"],
            "val_assignments": split_counts["val"],
            "test_assignments": split_counts["test"],
        },
        "location_counts": split_location_counts,
        "source_overlap": source_overlap,
    }
    with (data_dir / "group_safe_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--orig-rssi", type=Path, default=DEFAULT_ORIG_RSSI)
    parser.add_argument("--orig-raw", type=Path, default=DEFAULT_ORIG_RAW)
    parser.add_argument("--orig-spectrum", type=Path, default=DEFAULT_ORIG_SPECTRUM)
    parser.add_argument("--noisy-rssi", type=Path, default=DEFAULT_NOISY_RSSI)
    parser.add_argument("--noisy-raw", type=Path, default=DEFAULT_NOISY_RAW)
    parser.add_argument("--noisy-spectrum", type=Path, default=DEFAULT_NOISY_SPECTRUM)
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
