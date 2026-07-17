#!/usr/bin/env python3
"""Validate the formal ExpandedReal train+validation refit dataset and run."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from build_expanded_source_safe_1to10 import file_stem, iter_spectrum_packets, packet_key, read_csv


EXTRA_FIELDS = {"source_file_name", "source_packet_index", "augmentation_id"}
INTEGER_OFFSETS = {-2.0, -1.0, 0.0, 1.0, 2.0}


def source_key(row: dict) -> tuple[str, int]:
    return file_stem(row["source_file_name"]), int(float(row["source_packet_index"]))


def exact_test_table(parent_path: Path, refit_path: Path, test_sources: set[tuple[str, int]]) -> int:
    parent_fields, parent_rows = read_csv(parent_path)
    refit_fields, refit_rows = read_csv(refit_path)
    if [field for field in refit_fields if field not in EXTRA_FIELDS] != parent_fields:
        raise AssertionError(f"Field mismatch: {refit_path.name}")
    expected = {packet_key(row): row for row in parent_rows if packet_key(row) in test_sources}
    observed = {
        packet_key(row): row
        for row in refit_rows
        if row["augmentation_id"] == "orig" and source_key(row) in test_sources
    }
    if set(expected) != set(observed):
        raise AssertionError(f"Test packet key mismatch: {refit_path.name}")
    for key, expected_row in expected.items():
        for field in parent_fields:
            if expected_row[field] != observed[key][field]:
                raise AssertionError(f"Test value changed: {refit_path.name} {key} {field}")
    return len(observed)


def load_parent_test_spectrum(
    path: Path,
    test_sources: set[tuple[str, int]],
) -> tuple[list[str], dict[tuple[str, int], list[dict]]]:
    fields: list[str] = []
    packets: dict[tuple[str, int], list[dict]] = {}
    for packet_fields, key, rows in iter_spectrum_packets(path):
        fields = packet_fields
        if key in test_sources:
            packets[key] = rows
    return fields, packets


def validate_spectrum(
    parent_path: Path,
    refit_path: Path,
    test_sources: set[tuple[str, int]],
) -> dict:
    parent_fields, expected_test = load_parent_test_spectrum(parent_path, test_sources)
    observed_test: dict[tuple[str, int], list[dict]] = {}
    row_count = 0
    packet_count = 0
    train_packet_count = 0
    test_packet_count = 0
    max_q1_q4_error = 0.0
    max_norm_error = 0.0
    max_real_error = 0.0
    max_imag_error = 0.0
    for fields, key, rows in iter_spectrum_packets(refit_path):
        if [field for field in fields if field not in EXTRA_FIELDS] != parent_fields:
            raise AssertionError("Spectrum fields differ from parent")
        packet_count += 1
        row_count += len(rows)
        if len(rows) != 352:
            raise AssertionError(f"Spectrum packet does not have 352 rows: {key}")
        augmentation_id = rows[0]["augmentation_id"]
        if augmentation_id == "orig":
            if source_key(rows[0]) not in test_sources:
                raise AssertionError(f"Only test packets may remain original: {key}")
            observed_test[key] = rows
            test_packet_count += 1
            continue

        train_packet_count += 1
        by_bin: dict[tuple[int, int, float], dict] = {}
        symbols = set()
        for row in rows:
            symbol = int(float(row["local_symbol_index"]))
            q = int(float(row["q"]))
            offset = round(float(row["subbin_offset"]), 6)
            symbols.add(symbol)
            by_bin[(symbol, q, offset)] = row
            norm = float(row["mag_norm"])
            db = float(row["mag_db_rel_peak"])
            phase = float(row["phase_rad_rel_center"])
            real = float(row["real_norm"])
            imag = float(row["imag_norm"])
            if not all(math.isfinite(value) for value in [norm, db, phase, real, imag]):
                raise AssertionError(f"Non-finite augmented spectrum value: {key}")
            max_norm_error = max(max_norm_error, abs(norm - 10.0 ** (db / 20.0)))
            max_real_error = max(max_real_error, abs(real - norm * math.cos(phase)))
            max_imag_error = max(max_imag_error, abs(imag - norm * math.sin(phase)))
        if len(symbols) != 16:
            raise AssertionError(f"Expected 16 symbols: {key}")
        for symbol in symbols:
            for offset in INTEGER_OFFSETS:
                q1 = by_bin[(symbol, 1, offset)]
                q4 = by_bin[(symbol, 4, offset)]
                for field in ["mag_raw", "mag_norm", "mag_db_rel_peak"]:
                    max_q1_q4_error = max(
                        max_q1_q4_error,
                        abs(float(q1[field]) - float(q4[field])),
                    )

    if row_count != 1_878_976 or packet_count != 5_338:
        raise AssertionError(f"Unexpected refit spectrum coverage: {row_count}, {packet_count}")
    if train_packet_count != 5_210 or test_packet_count != 128:
        raise AssertionError(f"Unexpected refit spectrum split: {train_packet_count}, {test_packet_count}")
    if max_q1_q4_error > 1e-12:
        raise AssertionError(f"Augmented q1/q4 mismatch: {max_q1_q4_error}")
    if max(max_norm_error, max_real_error, max_imag_error) > 1e-9:
        raise AssertionError("Augmented spectrum magnitude/phase identity failed")
    if set(expected_test) != set(observed_test):
        raise AssertionError("Test spectrum packet keys changed")
    for key, expected_rows in expected_test.items():
        observed_rows = observed_test[key]
        if len(expected_rows) != len(observed_rows):
            raise AssertionError(f"Test spectrum row count changed: {key}")
        for expected, observed in zip(expected_rows, observed_rows):
            for field in parent_fields:
                if expected[field] != observed[field]:
                    raise AssertionError(f"Test spectrum value changed: {key} {field}")
    return {
        "row_count": row_count,
        "packet_count": packet_count,
        "train_packet_count": train_packet_count,
        "test_packet_count": test_packet_count,
        "rows_per_packet": 352,
        "max_augmented_q1_q4_error": max_q1_q4_error,
        "max_mag_norm_identity_error": max_norm_error,
        "max_real_identity_error": max_real_error,
        "max_imag_identity_error": max_imag_error,
        "test_packets_exact": len(observed_test),
    }


def run(args: argparse.Namespace) -> dict:
    _source_fields, source_rows = read_csv(args.refit_dir / "data" / "source_packet_split.csv")
    train_sources = {
        (row["source_file_stem"], int(float(row["source_packet_index"])))
        for row in source_rows
        if row["refit_split"] == "train"
    }
    test_sources = {
        (row["source_file_stem"], int(float(row["source_packet_index"])))
        for row in source_rows
        if row["refit_split"] == "test"
    }
    if len(train_sources) != 521 or len(test_sources) != 128 or train_sources & test_sources:
        raise AssertionError("Invalid refit source split")
    _assignment_fields, assignments = read_csv(args.refit_dir / "data" / "split_assignments.csv")
    assignment_counts = Counter(row["split"] for row in assignments)
    if assignment_counts != Counter({"train": 5210, "test": 128}):
        raise AssertionError(f"Invalid refit row split: {assignment_counts}")

    rssi_exact = exact_test_table(
        args.parent_dir / "inputs" / "rssi_plus_packet_level_32points_649.csv",
        args.refit_dir / "data" / "noisy_rssi_plus_packet_level_32points_649.csv",
        test_sources,
    )
    s17_exact = exact_test_table(
        args.parent_dir / "inputs" / "lora_frequency_s17_32points_649.csv",
        args.refit_dir / "data" / "noisy_lora_frequency_s17_32points_649.csv",
        test_sources,
    )
    spectrum = validate_spectrum(
        args.parent_dir / "inputs" / "subbin_spectrum_long_q1q4_32points_649.csv",
        args.refit_dir / "data" / "noisy_subbin_spectrum_long_32points_649.csv",
        test_sources,
    )
    metrics = json.loads((args.result_dir / "aco_v4_split_metrics.json").read_text(encoding="utf-8"))
    summary = metrics["summary"]
    if metrics["sample_counts"] != {
        "aligned": 5338,
        "train": 5210,
        "val": 0,
        "test": 128,
        "locations": 32,
    }:
        raise AssertionError(f"ACO loader/sample mismatch: {metrics['sample_counts']}")
    if len(summary) != 1 or summary[0]["final_correct"] != 98:
        raise AssertionError("Unexpected formal ACO result")
    report = {
        "status": "PASS",
        "source_counts": {"train_refit": len(train_sources), "test": len(test_sources)},
        "source_overlap": len(train_sources & test_sources),
        "row_counts": dict(assignment_counts),
        "test_exact_match": {"rssi": rssi_exact, "s17": s17_exact, "spectrum": spectrum["test_packets_exact"]},
        "spectrum": spectrum,
        "aco_loader_counts": metrics["sample_counts"],
        "aco_result": {
            "correct": summary[0]["final_correct"],
            "packet_count": summary[0]["packet_count"],
            "accuracy": summary[0]["final_accuracy"],
        },
    }
    metadata_dir = args.refit_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    scripts_dir = args.refit_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    target = scripts_dir / Path(__file__).name
    if Path(__file__).resolve() != target.resolve():
        shutil.copy2(Path(__file__), target)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--refit-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
