#!/usr/bin/env python3
"""Validate the ExpandedReal-649 source-safe 1:10 delivery dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


EXTRA_FIELDS = {"source_file_name", "source_packet_index", "augmentation_id"}
EPS = 1e-12
RAW_OFFSETS = {-2.0, -1.0, 0.0, 1.0, 2.0}


def file_stem(file_name: str) -> str:
    return Path(file_name).stem


def packet_key(row: dict) -> tuple[str, int]:
    return file_stem(row["file_name"]), int(float(row["packet_index"]))


def source_key(row: dict) -> tuple[str, int]:
    return row["source_file_stem"], int(float(row["source_packet_index"]))


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def assert_exact_eval_table(parent_path: Path, derived_path: Path, eval_sources: set[tuple[str, int]]) -> int:
    parent_fields, parent_rows = read_csv(parent_path)
    derived_fields, derived_rows = read_csv(derived_path)
    if [field for field in derived_fields if field not in EXTRA_FIELDS] != parent_fields:
        raise AssertionError(f"Field mismatch: {derived_path.name}")
    expected = {packet_key(row): row for row in parent_rows if packet_key(row) in eval_sources}
    observed = {packet_key(row): row for row in derived_rows if row["augmentation_id"] == "orig"}
    if set(expected) != set(observed):
        raise AssertionError(f"Evaluation key mismatch: {derived_path.name}")
    for key, original in expected.items():
        actual = observed[key]
        for field in parent_fields:
            if original[field] != actual[field]:
                raise AssertionError(f"Evaluation value changed in {derived_path.name}: {key} {field}")
    return len(observed)


def load_eval_spectrum_rows(
    parent_path: Path,
    eval_sources: set[tuple[str, int]],
) -> tuple[list[str], dict[tuple[str, int], list[dict]]]:
    output: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with parent_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        for row in reader:
            key = packet_key(row)
            if key in eval_sources:
                output[key].append(row)
    return fields, output


def validate_spectrum(
    parent_path: Path,
    derived_path: Path,
    eval_sources: set[tuple[str, int]],
) -> dict:
    parent_fields, expected_eval = load_eval_spectrum_rows(parent_path, eval_sources)
    packet_rows: Counter[tuple[str, int]] = Counter()
    packet_symbols: dict[tuple[str, int], set[int]] = defaultdict(set)
    q_values: dict[
        tuple[tuple[str, int], int, float],
        dict[int, tuple[float, float, float, str]],
    ] = defaultdict(dict)
    observed_eval: dict[tuple[str, int], list[dict]] = defaultdict(list)
    max_norm_error = 0.0
    max_real_error = 0.0
    max_imag_error = 0.0
    row_count = 0
    with derived_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if [field for field in fields if field not in EXTRA_FIELDS] != parent_fields:
            raise AssertionError("Spectrum field mismatch")
        for row in reader:
            row_count += 1
            key = packet_key(row)
            symbol = int(float(row["local_symbol_index"]))
            q = int(float(row["q"]))
            offset = round(float(row["subbin_offset"]), 6)
            raw = float(row["mag_raw"])
            norm = float(row["mag_norm"])
            db = float(row["mag_db_rel_peak"])
            phase = float(row["phase_rad_rel_center"])
            real = float(row["real_norm"])
            imag = float(row["imag_norm"])
            values = [raw, norm, db, phase, real, imag]
            if not all(math.isfinite(value) for value in values):
                raise AssertionError(f"Non-finite spectrum value at {key}")
            max_norm_error = max(max_norm_error, abs(norm - 10.0 ** (db / 20.0)))
            max_real_error = max(max_real_error, abs(real - norm * math.cos(phase)))
            max_imag_error = max(max_imag_error, abs(imag - norm * math.sin(phase)))
            packet_rows[key] += 1
            packet_symbols[key].add(symbol)
            if offset in RAW_OFFSETS:
                q_values[(key, symbol, offset)][q] = (raw, norm, db, row["augmentation_id"])
            if row["augmentation_id"] == "orig":
                observed_eval[key].append(row)

    if row_count != 1_473_472 or len(packet_rows) != 4_186:
        raise AssertionError(f"Unexpected spectrum coverage: rows={row_count}, packets={len(packet_rows)}")
    if set(packet_rows.values()) != {352} or any(len(symbols) != 16 for symbols in packet_symbols.values()):
        raise AssertionError("Each packet must contain 352 rows and 16 symbols")
    if max_norm_error > 1e-9 or max_real_error > 1e-9 or max_imag_error > 1e-9:
        raise AssertionError(
            f"Spectrum magnitude/phase inconsistency: {max_norm_error}, {max_real_error}, {max_imag_error}"
        )
    if len(q_values) != 4_186 * 16 * 5:
        raise AssertionError("Incomplete integer-bin q1/q4 coverage")
    max_q1_q4_error = 0.0
    max_augmented_q1_q4_error = 0.0
    for key, values in q_values.items():
        if set(values) != {1, 4}:
            raise AssertionError(f"Missing q1/q4 pair: {key}")
        errors = [abs(left - right) for left, right in zip(values[1][:3], values[4][:3])]
        max_q1_q4_error = max(max_q1_q4_error, *errors)
        if values[1][3] != "orig":
            max_augmented_q1_q4_error = max(max_augmented_q1_q4_error, *errors)
    if max_augmented_q1_q4_error > 1e-12:
        raise AssertionError(f"Augmented q1/q4 integer-bin mismatch: {max_augmented_q1_q4_error}")

    if set(expected_eval) != set(observed_eval):
        raise AssertionError("Evaluation spectrum packet keys changed")
    for key, expected_rows in expected_eval.items():
        actual_rows = observed_eval[key]
        if len(expected_rows) != len(actual_rows):
            raise AssertionError(f"Evaluation spectrum row count changed: {key}")
        for expected, actual in zip(expected_rows, actual_rows):
            for field in parent_fields:
                if expected[field] != actual[field]:
                    raise AssertionError(f"Evaluation spectrum changed: {key} {field}")
    return {
        "row_count": row_count,
        "packet_count": len(packet_rows),
        "rows_per_packet": 352,
        "symbols_per_packet": 16,
        "eval_packets_exact": len(observed_eval),
        "max_mag_norm_identity_error": max_norm_error,
        "max_real_identity_error": max_real_error,
        "max_imag_identity_error": max_imag_error,
        "max_q1_q4_integer_bin_error_all_rows": max_q1_q4_error,
        "max_q1_q4_integer_bin_error_augmented_train": max_augmented_q1_q4_error,
        "q1_q4_note": "Evaluation originals preserve the parent dataset exactly; the q1/q4 equality constraint applies to augmented training copies.",
    }


def validate_npz(path: Path) -> dict:
    expected_shapes = {
        "X_rssi": (4186, 6),
        "X_rssi_scaled": (4186, 6),
        "X_rssi_s17": (4186, 27),
        "X_rssi_s17_scaled": (4186, 27),
        "X_pgar": (4186, 30),
        "X_pgar_scaled": (4186, 30),
        "raw_symbol_mag": (4186, 16, 5),
        "q4_symbol_db": (4186, 16, 17),
        "q4_symbol_mag": (4186, 16, 17),
    }
    with np.load(path) as arrays:
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise AssertionError(f"Invalid NPZ array: {name} {arrays[name].shape}")
        split_counts = Counter(arrays["split"].tolist())
        if split_counts != Counter({"train": 3930, "val": 128, "test": 128}):
            raise AssertionError(f"Invalid NPZ split counts: {split_counts}")
        train_indices = arrays["train_indices"]
        scaler_checks = {}
        for matrix_name, mean_name, std_name, scaled_name in [
            ("X_rssi", "scaler_rssi_mean", "scaler_rssi_std", "X_rssi_scaled"),
            ("X_rssi_s17", "scaler_rssi_s17_mean", "scaler_rssi_s17_std", "X_rssi_s17_scaled"),
            ("X_pgar", "scaler_pgar_mean", "scaler_pgar_std", "X_pgar_scaled"),
        ]:
            matrix = arrays[matrix_name]
            mean = matrix[train_indices].mean(axis=0, dtype=np.float64)
            std = matrix[train_indices].std(axis=0, dtype=np.float64)
            std[std <= EPS] = 1.0
            mean_error = float(np.max(np.abs(mean - arrays[mean_name])))
            std_error = float(np.max(np.abs(std - arrays[std_name])))
            scaled_error = float(
                np.max(np.abs(((matrix - mean) / std).astype(np.float32) - arrays[scaled_name]))
            )
            if mean_error > 1e-12 or std_error > 1e-12 or scaled_error > 1e-6:
                raise AssertionError(f"Train-only scaler mismatch: {matrix_name}")
            scaler_checks[matrix_name] = {
                "mean_max_abs_error": mean_error,
                "std_max_abs_error": std_error,
                "scaled_max_abs_error": scaled_error,
            }
    return {"expected_shapes": {name: list(shape) for name, shape in expected_shapes.items()}, "scalers": scaler_checks}


def validate_native_loaders(project_root: Path, derived_dir: Path) -> dict:
    model_v3 = project_root / "model" / "v3"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(model_v3) not in sys.path:
        sys.path.insert(0, str(model_v3))
    from model.v3 import pgar_heuristic as pgar
    from model.v3 import aco_packet_path as aco_base
    from model.v3 import aco_packet_path_v2 as aco_v2

    rssi_path = derived_dir / "data" / "noisy_rssi_plus_packet_level_32points_649.csv"
    s17_path = derived_dir / "data" / "noisy_lora_frequency_s17_32points_649.csv"
    spectrum_path = derived_dir / "data" / "noisy_subbin_spectrum_long_32points_649.csv"
    with tempfile.TemporaryDirectory(prefix="expanded649_loader_smoke_") as temp_dir:
        pgar_args = SimpleNamespace(
            rssi_csv=rssi_path,
            raw_feature_csv=s17_path,
            q4_spectrum_csv=spectrum_path,
            output_dir=Path(temp_dir),
            disable_q4=False,
        )
        rssi_rows, raw_rows, q4_packets, q4_offsets = pgar.build_or_load_features(pgar_args)
        pgar_samples = pgar.align_samples(
            pgar.rssi_rows_to_packets(rssi_rows),
            pgar.raw_rows_to_packets(raw_rows),
            q4_packets,
        )
    if len(pgar_samples) != 4_186 or len(q4_offsets) != 17:
        raise AssertionError(f"PGAR loader mismatch: {len(pgar_samples)}, {len(q4_offsets)}")

    aco_args = SimpleNamespace(
        peak_threshold=None,
        auto_peak_quantile=0.10,
        q4_dev_threshold=None,
        auto_q4_dev_quantile=0.75,
        q4_peak_offset_max=0.50,
        q4_peak_to_side_threshold=6.0,
    )
    rssi_packets = aco_base.read_rssi_packets(rssi_path)
    symbol_packets, aco_offsets, thresholds = aco_base.read_symbol_packets(spectrum_path, aco_args)
    base_samples = aco_base.align_samples(rssi_packets, symbol_packets)
    segment_packets = aco_v2.build_segment_packets(base_samples, 4)
    if not (len(rssi_packets) == len(symbol_packets) == len(base_samples) == len(segment_packets) == 4_186):
        raise AssertionError("ACO v2 loader packet count mismatch")
    return {
        "pgar_sample_count": len(pgar_samples),
        "pgar_q4_offset_count": len(q4_offsets),
        "aco_rssi_packet_count": len(rssi_packets),
        "aco_symbol_packet_count": len(symbol_packets),
        "aco_segment_packet_count": len(segment_packets),
        "aco_q4_offset_count": len(aco_offsets),
        "aco_thresholds": thresholds,
    }


def run(args: argparse.Namespace) -> dict:
    parent = args.parent_dir
    derived = args.derived_dir
    _source_fields, source_rows = read_csv(derived / "data" / "source_packet_split.csv")
    split_sets = {
        name: {source_key(row) for row in source_rows if row["split"] == name}
        for name in ["train", "val", "test"]
    }
    source_counts = {name: len(values) for name, values in split_sets.items()}
    overlaps = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    if source_counts != {"train": 393, "val": 128, "test": 128} or any(overlaps.values()):
        raise AssertionError(f"Invalid source split: {source_counts}, {overlaps}")

    _assignment_fields, assignments = read_csv(derived / "data" / "split_assignments.csv")
    row_counts = Counter(row["split"] for row in assignments)
    if row_counts != Counter({"train": 3930, "val": 128, "test": 128}):
        raise AssertionError(f"Invalid augmented split: {row_counts}")
    by_source = Counter(source_key(row) for row in assignments)
    if any(by_source[key] != 10 for key in split_sets["train"]):
        raise AssertionError("Every training source must have exactly 10 augmented rows")
    if any(by_source[key] != 1 for key in split_sets["val"] | split_sets["test"]):
        raise AssertionError("Every evaluation source must have exactly one original row")
    eval_sources = split_sets["val"] | split_sets["test"]
    rssi_exact = assert_exact_eval_table(
        parent / "inputs" / "rssi_plus_packet_level_32points_649.csv",
        derived / "data" / "noisy_rssi_plus_packet_level_32points_649.csv",
        eval_sources,
    )
    s17_exact = assert_exact_eval_table(
        parent / "inputs" / "lora_frequency_s17_32points_649.csv",
        derived / "data" / "noisy_lora_frequency_s17_32points_649.csv",
        eval_sources,
    )
    spectrum = validate_spectrum(
        parent / "inputs" / "subbin_spectrum_long_q1q4_32points_649.csv",
        derived / "data" / "noisy_subbin_spectrum_long_32points_649.csv",
        eval_sources,
    )
    arrays = validate_npz(derived / "arrays" / "source_safe_1to10_arrays.npz")
    loaders = validate_native_loaders(args.project_root, derived)
    report = {
        "status": "PASS",
        "source_packet_counts": source_counts,
        "augmented_row_counts": dict(row_counts),
        "source_overlap": overlaps,
        "evaluation_exact_match": {"rssi_packets": rssi_exact, "s17_packets": s17_exact, "spectrum_packets": spectrum["eval_packets_exact"]},
        "spectrum": spectrum,
        "npz": arrays,
        "native_loaders": loaders,
    }
    report_path = derived / "metadata" / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    scripts_dir = derived / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_target = scripts_dir / Path(__file__).name
    if Path(__file__).resolve() != script_target.resolve():
        shutil.copy2(Path(__file__), script_target)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--derived-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
