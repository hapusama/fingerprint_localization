#!/usr/bin/env python3
"""Split ExpandedReal-649-v1 by source, then augment training sources 1:10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DATASET_VERSION = "ExpandedReal-649-v1"
DERIVED_VERSION = "ExpandedReal-649-v1-source-safe-1to10-v1"
RSSI_COLUMNS = [
    "snr",
    "realtime_average_rssi",
    "median_rssi",
    "mode_rssi",
    "rssi_variance",
    "residual",
]
S17_COLUMNS = [f"preamble_fft_mag_bin_{offset:+d}" for offset in range(-8, 9)] + [
    "preamble_peak_to_residual_db",
    "detect_score_db",
    "s17_c_s",
    "s17_j_s",
]
POSITIVE_COLUMNS = {
    "rssi_variance",
    *[f"preamble_fft_mag_bin_{offset:+d}" for offset in range(-8, 9)],
    "s17_c_s",
    "s17_j_s",
}
RAW_OFFSETS = [-2.0, -1.0, 0.0, 1.0, 2.0]
Q4_OFFSETS = [round(-2.0 + 0.25 * index, 6) for index in range(17)]
EPS = 1e-12


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_stem(file_name: str) -> str:
    return Path(file_name).stem


def packet_key(row: dict) -> tuple[str, int]:
    return file_stem(row["file_name"]), int(float(row["packet_index"]))


def natural_label_key(label: str) -> tuple[int, int]:
    corridor, location = label.split("_", 1)
    return int(corridor), int(location)


def parse_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def add_metadata_fields(fields: Sequence[str]) -> list[str]:
    output = list(fields)
    for field in ["source_file_name", "source_packet_index", "augmentation_id"]:
        if field not in output:
            output.append(field)
    return output


def add_original_metadata(row: dict) -> dict:
    output = dict(row)
    output["source_file_name"] = row["file_name"]
    output["source_packet_index"] = row["packet_index"]
    output["augmentation_id"] = "orig"
    return output


def augmented_file_name(file_name: str, augmentation_index: int) -> str:
    path = Path(file_name)
    return f"{path.stem}_aug{augmentation_index:02d}{path.suffix}"


def split_sources(
    rssi_rows: Sequence[dict],
    seed: int,
) -> tuple[dict[str, set[tuple[str, int]]], list[dict]]:
    by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in rssi_rows:
        by_label[row["position_key"]].append(packet_key(row))
    rng = random.Random(seed)
    split: dict[str, set[tuple[str, int]]] = {"train": set(), "val": set(), "test": set()}
    rows: list[dict] = []
    for label in sorted(by_label, key=natural_label_key):
        keys = list(dict.fromkeys(by_label[label]))
        rng.shuffle(keys)
        n = len(keys)
        n_train = max(2, int(round(n * 0.6)))
        n_val = max(1, int(round(n * 0.2)))
        if n_train + n_val >= n:
            n_train = n - 2
            n_val = 1
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


def column_noise_stats(
    rows: Sequence[dict],
    columns: Sequence[str],
    noise_ratio: float,
) -> dict[str, dict[str, float]]:
    output = {}
    for column in columns:
        values = [parse_float(row.get(column)) for row in rows]
        clean = [value for value in values if value is not None]
        mean = sum(clean) / len(clean)
        variance = sum((value - mean) ** 2 for value in clean) / len(clean)
        source_std = math.sqrt(variance)
        output[column] = {
            "train_mean": mean,
            "train_std": source_std,
            "noise_std": source_std / noise_ratio,
            "noise_ratio": noise_ratio,
        }
    return output


def augment_packet_rows(
    rows: Sequence[dict],
    columns: Sequence[str],
    stats: dict[str, dict[str, float]],
    rng: random.Random,
    augment_factor: int,
) -> list[dict]:
    output = []
    for row in rows:
        for augmentation_index in range(augment_factor):
            augmented = dict(row)
            augmented["source_file_name"] = row["file_name"]
            augmented["source_packet_index"] = row["packet_index"]
            augmented["augmentation_id"] = f"aug{augmentation_index:02d}"
            augmented["file_name"] = augmented_file_name(row["file_name"], augmentation_index)
            for column in columns:
                value = parse_float(row.get(column))
                if value is None:
                    continue
                noisy = value + rng.gauss(0.0, stats[column]["noise_std"])
                if column in POSITIVE_COLUMNS:
                    noisy = max(noisy, EPS)
                augmented[column] = f"{noisy:.12g}"
            output.append(augmented)
    return output


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else 0.0


def iter_spectrum_packets(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        current_key = None
        rows = []
        for row in reader:
            key = packet_key(row)
            if current_key is not None and key != current_key:
                yield fields, current_key, rows
                rows = []
            current_key = key
            rows.append(row)
        if current_key is not None:
            yield fields, current_key, rows


def spectrum_train_stats(
    spectrum_path: Path,
    train_sources: set[tuple[str, int]],
    noise_ratio: float,
) -> dict:
    shape = RunningStats()
    peak = RunningStats()
    packet_count = 0
    for _fields, key, rows in iter_spectrum_packets(spectrum_path):
        if key not in train_sources:
            continue
        packet_count += 1
        groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for row in rows:
            symbol = int(float(row["local_symbol_index"]))
            q = int(float(row["q"]))
            groups[(symbol, q)].append(row)
            if q == 4:
                shape.add(float(row["mag_db_rel_peak"]))
        for (_symbol, q), group in groups.items():
            if q != 4:
                continue
            peak_raw = max(float(row["mag_raw"]) for row in group)
            peak.add(20.0 * math.log10(max(peak_raw, EPS)))
    return {
        "train_packet_count": packet_count,
        "q4_shape_db_train_mean": shape.mean,
        "q4_shape_db_train_std": shape.std,
        "q4_shape_noise_std_db": shape.std / noise_ratio,
        "q4_peak_db_train_mean": peak.mean,
        "q4_peak_db_train_std": peak.std,
        "q4_peak_gain_noise_std_db": peak.std / noise_ratio,
        "noise_ratio": noise_ratio,
    }


def smooth_noise(values: list[float]) -> list[float]:
    if len(values) <= 2:
        return values
    output = []
    for index, value in enumerate(values):
        left = values[max(0, index - 1)]
        right = values[min(len(values) - 1, index + 1)]
        output.append(0.25 * left + 0.5 * value + 0.25 * right)
    return output


def set_spectrum_values(row: dict, raw: float, normalized: float, db: float) -> None:
    row["mag_raw"] = f"{raw:.12g}"
    row["mag_norm"] = f"{normalized:.12g}"
    row["mag_db_rel_peak"] = f"{db:.12g}"
    phase = float(row["phase_rad_rel_center"])
    row["real_norm"] = f"{normalized * math.cos(phase):.12g}"
    row["imag_norm"] = f"{normalized * math.sin(phase):.12g}"


def augment_spectrum_packet(
    rows: Sequence[dict],
    augmentation_index: int,
    rng: random.Random,
    stats: dict,
) -> list[dict]:
    output = [dict(row) for row in rows]
    by_symbol_q: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row_index, row in enumerate(output):
        by_symbol_q[(int(float(row["local_symbol_index"])), int(float(row["q"])))].append(row_index)

    for symbol in range(16):
        q4_indices = sorted(
            by_symbol_q[(symbol, 4)],
            key=lambda index: float(output[index]["subbin_offset"]),
        )
        if len(q4_indices) != 17:
            raise RuntimeError(f"Expected 17 q4 rows for symbol {symbol}, got {len(q4_indices)}")
        original_db = [float(output[index]["mag_db_rel_peak"]) for index in q4_indices]
        raw_noise = [
            rng.gauss(0.0, stats["q4_shape_noise_std_db"])
            for _ in q4_indices
        ]
        noise = smooth_noise(raw_noise)
        perturbed_db = [left + right for left, right in zip(original_db, noise)]
        peak_db = max(perturbed_db)
        perturbed_db = [value - peak_db for value in perturbed_db]
        original_peak_raw = max(float(output[index]["mag_raw"]) for index in q4_indices)
        gain_db = rng.gauss(0.0, stats["q4_peak_gain_noise_std_db"])
        augmented_peak_raw = original_peak_raw * (10.0 ** (gain_db / 20.0))
        q4_values = {}
        for row_index, db in zip(q4_indices, perturbed_db):
            normalized = 10.0 ** (db / 20.0)
            raw = augmented_peak_raw * normalized
            set_spectrum_values(output[row_index], raw, normalized, db)
            offset = round(float(output[row_index]["subbin_offset"]), 6)
            q4_values[offset] = (raw, normalized, db)

        q1_indices = sorted(
            by_symbol_q[(symbol, 1)],
            key=lambda index: float(output[index]["subbin_offset"]),
        )
        if len(q1_indices) != 5:
            raise RuntimeError(f"Expected 5 q1 rows for symbol {symbol}, got {len(q1_indices)}")
        for row_index in q1_indices:
            offset = round(float(output[row_index]["subbin_offset"]), 6)
            raw, normalized, db = q4_values[offset]
            set_spectrum_values(output[row_index], raw, normalized, db)

    for row in output:
        row["source_file_name"] = row["file_name"]
        row["source_packet_index"] = row["packet_index"]
        row["augmentation_id"] = f"aug{augmentation_index:02d}"
        row["file_name"] = augmented_file_name(row["file_name"], augmentation_index)
    return output


def write_augmented_spectrum(
    input_path: Path,
    output_path: Path,
    source_split: dict[str, set[tuple[str, int]]],
    augment_factor: int,
    rng: random.Random,
    stats: dict,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for fields, key, rows in iter_spectrum_packets(input_path):
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=add_metadata_fields(fields), extrasaction="ignore")
                writer.writeheader()
            if key in source_split["train"]:
                for augmentation_index in range(augment_factor):
                    augmented = augment_spectrum_packet(rows, augmentation_index, rng, stats)
                    writer.writerows(augmented)
                    row_count += len(augmented)
            elif key in source_split["val"] or key in source_split["test"]:
                original = [add_original_metadata(row) for row in rows]
                writer.writerows(original)
                row_count += len(original)
            else:
                raise RuntimeError(f"Spectrum packet not assigned to a split: {key}")
    return row_count


def split_assignment_rows(
    combined_rssi_rows: Sequence[dict],
    source_split: dict[str, set[tuple[str, int]]],
) -> list[dict]:
    output = []
    source_to_split = {
        key: split_name for split_name, keys in source_split.items() for key in keys
    }
    for row in combined_rssi_rows:
        source_key = (
            file_stem(row["source_file_name"]),
            int(float(row["source_packet_index"])),
        )
        output.append(
            {
                "split": source_to_split[source_key],
                "position_key": row["position_key"],
                "file_stem": file_stem(row["file_name"]),
                "packet_index": int(float(row["packet_index"])),
                "source_file_stem": source_key[0],
                "source_packet_index": source_key[1],
                "augmentation_id": row["augmentation_id"],
            }
        )
    return output


def pgar_structure(raw_bins: Sequence[float]) -> list[float]:
    a_m2, a_m1, a0, a_p1, a_p2 = [max(0.0, float(value)) for value in raw_bins]
    total = sum(raw_bins)
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


def build_arrays(
    rssi_path: Path,
    s17_path: Path,
    spectrum_path: Path,
    split_rows: Sequence[dict],
    output_path: Path,
    feature_csv_path: Path,
) -> dict:
    _rssi_fields, rssi_rows = read_csv(rssi_path)
    _s17_fields, s17_rows = read_csv(s17_path)
    rssi_by_key = {packet_key(row): row for row in rssi_rows}
    s17_by_key = {packet_key(row): row for row in s17_rows}
    split_by_key = {
        (row["file_stem"], int(float(row["packet_index"]))): row for row in split_rows
    }
    ordered_keys = [packet_key(row) for row in rssi_rows]
    if len(ordered_keys) != len(set(ordered_keys)):
        raise RuntimeError("Augmented packet keys are not unique")
    if set(ordered_keys) != set(s17_by_key) or set(ordered_keys) != set(split_by_key):
        raise RuntimeError("RSSI, S17, and split key sets do not match")
    key_to_index = {key: index for index, key in enumerate(ordered_keys)}
    labels = sorted({row["position_key"] for row in rssi_rows}, key=natural_label_key)
    label_to_id = {label: index for index, label in enumerate(labels)}
    n = len(ordered_keys)
    raw_tensor = np.full((n, 16, 5), np.nan, dtype=np.float32)
    q4_db_tensor = np.full((n, 16, 17), np.nan, dtype=np.float32)
    q4_mag_tensor = np.full((n, 16, 17), np.nan, dtype=np.float32)
    seen = set()
    for _fields, key, rows in iter_spectrum_packets(spectrum_path):
        if key not in key_to_index:
            raise RuntimeError(f"Spectrum key is missing from packet tables: {key}")
        index = key_to_index[key]
        by_symbol_q: dict[tuple[int, int], dict[float, dict]] = defaultdict(dict)
        for row in rows:
            symbol = int(float(row["local_symbol_index"]))
            q = int(float(row["q"]))
            offset = round(float(row["subbin_offset"]), 6)
            by_symbol_q[(symbol, q)][offset] = row
        for symbol in range(16):
            raw_tensor[index, symbol] = [
                float(by_symbol_q[(symbol, 1)][offset]["mag_raw"])
                for offset in RAW_OFFSETS
            ]
            q4_db_tensor[index, symbol] = [
                float(by_symbol_q[(symbol, 4)][offset]["mag_db_rel_peak"])
                for offset in Q4_OFFSETS
            ]
            q4_mag_tensor[index, symbol] = [
                float(by_symbol_q[(symbol, 4)][offset]["mag_raw"])
                for offset in Q4_OFFSETS
            ]
        seen.add(key)
    if seen != set(ordered_keys):
        raise RuntimeError(f"Spectrum packet coverage mismatch: {len(seen)} != {n}")
    if not np.isfinite(raw_tensor).all() or not np.isfinite(q4_db_tensor).all():
        raise RuntimeError("Non-finite tensor values after augmentation")

    x_rssi = np.asarray(
        [[float(rssi_by_key[key][field]) for field in RSSI_COLUMNS] for key in ordered_keys],
        dtype=np.float32,
    )
    x_s17 = np.asarray(
        [[float(s17_by_key[key][field]) for field in S17_COLUMNS] for key in ordered_keys],
        dtype=np.float32,
    )
    x_rssi_s17 = np.concatenate([x_rssi, x_s17], axis=1)
    raw_packet_median = np.median(raw_tensor, axis=1)
    q4_packet_median = np.median(q4_db_tensor, axis=1)
    q4_iqr = np.quantile(q4_db_tensor, 0.75, axis=1) - np.quantile(q4_db_tensor, 0.25, axis=1)
    q4_stability = np.median(q4_iqr, axis=1).reshape(-1, 1)
    structure = np.asarray(
        [pgar_structure(row.tolist()) for row in raw_packet_median],
        dtype=np.float32,
    )
    x_pgar = np.concatenate([x_rssi, structure, q4_stability, q4_packet_median], axis=1)
    split = np.asarray([split_by_key[key]["split"] for key in ordered_keys])
    train_indices = np.flatnonzero(split == "train")
    val_indices = np.flatnonzero(split == "val")
    test_indices = np.flatnonzero(split == "test")

    def scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = matrix[train_indices].mean(axis=0, dtype=np.float64)
        std = matrix[train_indices].std(axis=0, dtype=np.float64)
        std[std <= EPS] = 1.0
        return ((matrix - mean) / std).astype(np.float32), mean, std

    x_rssi_scaled, rssi_mean, rssi_std = scale(x_rssi)
    x_rssi_s17_scaled, combined_mean, combined_std = scale(x_rssi_s17)
    x_pgar_scaled, pgar_mean, pgar_std = scale(x_pgar)
    y = np.asarray(
        [label_to_id[rssi_by_key[key]["position_key"]] for key in ordered_keys],
        dtype=np.int64,
    )
    source_key_values = np.asarray(
        [
            f"{split_by_key[key]['source_file_stem']}|{int(float(split_by_key[key]['source_packet_index']))}"
            for key in ordered_keys
        ]
    )
    augmentation_ids = np.asarray([split_by_key[key]["augmentation_id"] for key in ordered_keys])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_rssi=x_rssi,
        X_rssi_scaled=x_rssi_scaled,
        X_rssi_s17=x_rssi_s17,
        X_rssi_s17_scaled=x_rssi_s17_scaled,
        X_pgar=x_pgar,
        X_pgar_scaled=x_pgar_scaled,
        raw_symbol_mag=raw_tensor,
        q4_symbol_db=q4_db_tensor,
        q4_symbol_mag=q4_mag_tensor,
        y=y,
        split=split,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        source_key=source_key_values,
        augmentation_id=augmentation_ids,
        file_stem=np.asarray([key[0] for key in ordered_keys]),
        packet_index=np.asarray([key[1] for key in ordered_keys], dtype=np.int64),
        position_key=np.asarray([rssi_by_key[key]["position_key"] for key in ordered_keys]),
        rssi_feature_names=np.asarray(RSSI_COLUMNS),
        rssi_s17_feature_names=np.asarray(RSSI_COLUMNS + S17_COLUMNS),
        pgar_feature_names=np.asarray(
            RSSI_COLUMNS
            + [
                "pgar_log_center",
                "pgar_center_fraction",
                "pgar_log_side_to_center",
                "pgar_log_right_to_left",
                "pgar_center_minus_left1",
                "pgar_center_minus_right1",
                "q4_stability",
            ]
            + [f"q4_packet_median_db_{offset:+.2f}" for offset in Q4_OFFSETS]
        ),
        raw_offsets=np.asarray(RAW_OFFSETS, dtype=np.float32),
        q4_offsets=np.asarray(Q4_OFFSETS, dtype=np.float32),
        scaler_rssi_mean=rssi_mean,
        scaler_rssi_std=rssi_std,
        scaler_rssi_s17_mean=combined_mean,
        scaler_rssi_s17_std=combined_std,
        scaler_pgar_mean=pgar_mean,
        scaler_pgar_std=pgar_std,
    )
    feature_rows = []
    fields = [
        "row_index",
        "split",
        "source_key",
        "augmentation_id",
        "file_stem",
        "packet_index",
        "position_key",
        "label_id",
        *RSSI_COLUMNS,
        *S17_COLUMNS,
    ]
    for index, key in enumerate(ordered_keys):
        feature_rows.append(
            {
                "row_index": index,
                "split": split[index],
                "source_key": source_key_values[index],
                "augmentation_id": augmentation_ids[index],
                "file_stem": key[0],
                "packet_index": key[1],
                "position_key": rssi_by_key[key]["position_key"],
                "label_id": y[index],
                **{
                    name: value
                    for name, value in zip(RSSI_COLUMNS + S17_COLUMNS, x_rssi_s17[index].tolist())
                },
            }
        )
    write_csv(feature_csv_path, fields, feature_rows)
    return {
        "row_count": n,
        "array_shapes": {
            "X_rssi": list(x_rssi.shape),
            "X_rssi_s17": list(x_rssi_s17.shape),
            "X_pgar": list(x_pgar.shape),
            "raw_symbol_mag": list(raw_tensor.shape),
            "q4_symbol_db": list(q4_db_tensor.shape),
        },
        "split_row_counts": {
            "train": int(train_indices.size),
            "val": int(val_indices.size),
            "test": int(test_indices.size),
        },
        "label_count": len(labels),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path, include_manifest: bool = True) -> list[dict]:
    checksum_path = output_dir / "CHECKSUMS.sha256"
    manifest_path = output_dir / "metadata" / "file_manifest.csv"
    excluded = {checksum_path}
    if not include_manifest:
        excluded.add(manifest_path)
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file()
        and path not in excluded
        and "__pycache__" not in path.parts
    )
    rows = []
    lines = []
    for path in files:
        relative = path.relative_to(output_dir).as_posix()
        digest = sha256(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
        lines.append(f"{digest}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rssi_path = args.input_dir / "inputs" / "rssi_plus_packet_level_32points_649.csv"
    s17_path = args.input_dir / "inputs" / "lora_frequency_s17_32points_649.csv"
    spectrum_path = args.input_dir / "inputs" / "subbin_spectrum_long_q1q4_32points_649.csv"
    rssi_fields, rssi_rows = read_csv(rssi_path)
    s17_fields, s17_rows = read_csv(s17_path)
    if len(rssi_rows) != 649 or len(s17_rows) != 649:
        raise RuntimeError("ExpandedReal-649-v1 packet tables must each contain 649 rows")
    rssi_keys = {packet_key(row) for row in rssi_rows}
    if rssi_keys != {packet_key(row) for row in s17_rows}:
        raise RuntimeError("RSSI and S17 source keys do not match")

    source_split, source_rows = split_sources(rssi_rows, args.seed)
    source_counts = {name: len(keys) for name, keys in source_split.items()}
    if source_counts != {"train": 393, "val": 128, "test": 128}:
        raise RuntimeError(f"Unexpected source split counts: {source_counts}")
    overlap = {
        "train_val": len(source_split["train"] & source_split["val"]),
        "train_test": len(source_split["train"] & source_split["test"]),
        "val_test": len(source_split["val"] & source_split["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Source overlap detected: {overlap}")

    train_rssi = [row for row in rssi_rows if packet_key(row) in source_split["train"]]
    train_s17 = [row for row in s17_rows if packet_key(row) in source_split["train"]]
    rssi_stats = column_noise_stats(train_rssi, RSSI_COLUMNS, args.noise_ratio)
    s17_stats = column_noise_stats(train_s17, S17_COLUMNS, args.noise_ratio)
    spectrum_stats = spectrum_train_stats(spectrum_path, source_split["train"], args.noise_ratio)

    augmented_train_rssi = augment_packet_rows(
        train_rssi,
        RSSI_COLUMNS,
        rssi_stats,
        random.Random(args.seed + 101),
        args.augment_factor,
    )
    augmented_train_s17 = augment_packet_rows(
        train_s17,
        S17_COLUMNS,
        s17_stats,
        random.Random(args.seed + 202),
        args.augment_factor,
    )
    original_eval_rssi = [
        add_original_metadata(row)
        for row in rssi_rows
        if packet_key(row) in source_split["val"] or packet_key(row) in source_split["test"]
    ]
    original_eval_s17 = [
        add_original_metadata(row)
        for row in s17_rows
        if packet_key(row) in source_split["val"] or packet_key(row) in source_split["test"]
    ]
    combined_rssi = augmented_train_rssi + original_eval_rssi
    combined_s17 = augmented_train_s17 + original_eval_s17

    data_dir = args.output_dir / "data"
    output_rssi = data_dir / "noisy_rssi_plus_packet_level_32points_649.csv"
    output_s17 = data_dir / "noisy_lora_frequency_s17_32points_649.csv"
    output_spectrum = data_dir / "noisy_subbin_spectrum_long_32points_649.csv"
    write_csv(output_rssi, add_metadata_fields(rssi_fields), combined_rssi)
    write_csv(output_s17, add_metadata_fields(s17_fields), combined_s17)
    spectrum_rows = write_augmented_spectrum(
        spectrum_path,
        output_spectrum,
        source_split,
        args.augment_factor,
        random.Random(args.seed + 303),
        spectrum_stats,
    )
    assignments = split_assignment_rows(combined_rssi, source_split)
    write_csv(
        data_dir / "split_assignments.csv",
        [
            "split",
            "position_key",
            "file_stem",
            "packet_index",
            "source_file_stem",
            "source_packet_index",
            "augmentation_id",
        ],
        assignments,
    )
    write_csv(
        data_dir / "source_packet_split.csv",
        ["split", "position_key", "source_file_stem", "source_packet_index"],
        source_rows,
    )

    arrays = build_arrays(
        output_rssi,
        output_s17,
        output_spectrum,
        assignments,
        args.output_dir / "arrays" / "source_safe_1to10_arrays.npz",
        args.output_dir / "features" / "source_safe_1to10_ml_features.csv",
    )
    expected_spectrum_rows = arrays["row_count"] * 16 * (5 + 17)
    if spectrum_rows != expected_spectrum_rows:
        raise RuntimeError(f"Unexpected spectrum row count: {spectrum_rows} != {expected_spectrum_rows}")

    assignment_counts = Counter(row["split"] for row in assignments)
    split_locations = {
        name: len({row["position_key"] for row in source_rows if row["split"] == name})
        for name in ["train", "val", "test"]
    }
    per_label_rows = []
    for label in sorted({row["position_key"] for row in source_rows}, key=natural_label_key):
        counts = Counter(row["split"] for row in source_rows if row["position_key"] == label)
        per_label_rows.append(
            {
                "position_key": label,
                "train_sources": counts["train"],
                "val_sources": counts["val"],
                "test_sources": counts["test"],
                "train_augmented_rows": counts["train"] * args.augment_factor,
            }
        )
    write_csv(
        args.output_dir / "metadata" / "per_location_split_counts.csv",
        ["position_key", "train_sources", "val_sources", "test_sources", "train_augmented_rows"],
        per_label_rows,
    )

    metadata = {
        "dataset_version": DATASET_VERSION,
        "derived_version": DERIVED_VERSION,
        "seed": args.seed,
        "protocol": (
            "Stratified source-packet-safe 60/20/20 split. Only training sources are represented by "
            "10 Gaussian-perturbed copies; validation and test are untouched original packets."
        ),
        "source_packet_counts": source_counts,
        "augmented_row_counts": dict(assignment_counts),
        "total_packet_rows": len(assignments),
        "spectrum_rows": spectrum_rows,
        "location_counts": split_locations,
        "source_overlap": overlap,
        "augmentation": {
            "factor": args.augment_factor,
            "noise_ratio": args.noise_ratio,
            "rssi_and_s17": (
                "Independent Gaussian feature noise with sigma=train-column-std/noise_ratio; "
                "statistics use training sources only."
            ),
            "spectrum": (
                "Smooth Gaussian perturbation in q4 dB shape using training-only sigma, with peak-gain "
                "jitter. q1 integer bins are tied to q4; magnitude, normalized magnitude, dB, real, and "
                "imaginary fields are recomputed consistently."
            ),
            "train_original_in_augmented_pool": False,
            "note": "The original 393 training source packets remain recoverable from the parent ExpandedReal-649-v1 package.",
        },
        "noise_stats": {
            "rssi": rssi_stats,
            "s17": s17_stats,
            "spectrum": spectrum_stats,
        },
        "arrays": arrays,
    }
    metadata_dir = args.output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "split_augmentation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), args.output_dir / "scripts" / Path(__file__).name)
    readme = f"""# {DERIVED_VERSION}

This dataset is derived from `{DATASET_VERSION}`.

## Split

- Seed: {args.seed}
- Train sources: {source_counts['train']}
- Validation sources: {source_counts['val']}
- Test sources: {source_counts['test']}
- Source overlap: 0 for every split pair
- All 32 positions occur in train, validation, and test

## Augmentation

- Only training sources are augmented.
- Each train source produces {args.augment_factor} Gaussian-perturbed copies.
- Augmented train rows: {assignment_counts['train']}
- Validation rows: {assignment_counts['val']} untouched originals
- Test rows: {assignment_counts['test']} untouched originals
- Noise statistics are estimated from the 393 training sources only.
- Spectrum augmentation keeps q1/q4 integer bins and magnitude/dB/complex fields consistent for training copies.
- Validation/test rows are byte-for-field copies of the parent packet tables; parent q1/q4 differences are not rewritten.

## Algorithm inputs

- `data/noisy_rssi_plus_packet_level_32points_649.csv`
- `data/noisy_lora_frequency_s17_32points_649.csv`
- `data/noisy_subbin_spectrum_long_32points_649.csv`
- `data/split_assignments.csv`
- `data/source_packet_split.csv`
- `arrays/source_safe_1to10_arrays.npz`
- `features/source_safe_1to10_ml_features.csv`
- `metadata/validation_report.json`
- `reports/ExpandedReal649_SourceSafe_1to10_Report.xlsx`

The NPZ contains raw and train-standardized RSSI, RSSI+S17, and PGAR matrices, plus q1/q4 symbol tensors.
Do not use validation or test data to refit scalers, augmentation statistics, thresholds, or prototypes.

## Validation

The independent validation report must have `status: PASS`. Full revalidation of evaluation-field identity
also requires the parent `LoRaMorph_ExpandedReal649_v1_20260716` package.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    workbook_payload = {
        "metadata": metadata,
        "per_location": per_label_rows,
        "noise_rows": [
            {"group": "RSSI+", "feature": name, **values}
            for name, values in rssi_stats.items()
        ]
        + [
            {"group": "S17", "feature": name, **values}
            for name, values in s17_stats.items()
        ],
    }
    (metadata_dir / "workbook_payload.json").write_text(
        json.dumps(workbook_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    file_rows = write_checksums(args.output_dir, include_manifest=False)
    write_csv(metadata_dir / "file_manifest.csv", ["path", "bytes", "sha256"], file_rows)
    write_checksums(args.output_dir)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--augment-factor", type=int, default=10)
    parser.add_argument("--noise-ratio", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
