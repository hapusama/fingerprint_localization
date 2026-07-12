#!/usr/bin/env python3
"""Verify the published mainline files and frozen protocol invariants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "fingerprint_localization" / "docs" / "mainline_202607" / "DATA_MANIFEST.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(check_hashes: bool) -> list[str]:
    errors: list[str] = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        path = ROOT / row["path"]
        required = row["publish_policy"] == "git"
        if not path.exists():
            if required:
                errors.append(f"missing required Git file: {row['path']}")
            continue
        actual_size = path.stat().st_size
        expected_size = int(row["bytes"])
        if actual_size != expected_size:
            errors.append(f"size mismatch: {row['path']} ({actual_size} != {expected_size})")
        if check_hashes:
            actual_hash = sha256(path)
            if actual_hash != row["sha256"]:
                errors.append(f"SHA-256 mismatch: {row['path']}")
    return errors


def verify_protocol() -> list[str]:
    errors: list[str] = []
    group_meta_path = (
        ROOT / "fingerprint_localization/data/mainline_202607/splits/source_safe/group_safe_metadata.json"
    )
    refit_meta_path = (
        ROOT / "fingerprint_localization/data/mainline_202607/splits/refit/refit_metadata.json"
    )
    final_metrics_path = (
        ROOT
        / "fingerprint_localization/experiments/aco_source_safe_1to10/"
        "results/frozen/test_metrics.json"
    )
    for path in (group_meta_path, refit_meta_path, final_metrics_path):
        if not path.exists():
            errors.append(f"missing protocol evidence: {path.relative_to(ROOT)}")
    if errors:
        return errors

    group_meta = json.loads(group_meta_path.read_text(encoding="utf-8"))
    refit_meta = json.loads(refit_meta_path.read_text(encoding="utf-8"))
    final_metrics = json.loads(final_metrics_path.read_text(encoding="utf-8"))

    expected = {
        "source_packet_counts": {"train": 223, "val": 73, "test": 74},
        "source_overlap": {"train_val": 0, "train_test": 0, "val_test": 0},
    }
    for key, value in expected.items():
        if group_meta.get(key) != value:
            errors.append(f"group-safe invariant mismatch for {key}: {group_meta.get(key)!r}")
    if refit_meta.get("source_counts") != {"train_refit": 296, "test": 74}:
        errors.append(f"refit source count mismatch: {refit_meta.get('source_counts')!r}")
    if refit_meta.get("row_counts") != {"train": 2960, "val": 73, "test": 74}:
        errors.append(f"refit row count mismatch: {refit_meta.get('row_counts')!r}")
    if refit_meta.get("test_source_overlap_with_train") != 0:
        errors.append("refit test source overlaps training")
    if final_metrics.get("packet_count") != 74 or final_metrics.get("aco_score4_correct") != 67:
        errors.append("final ACO v4 result is not the frozen 67/74")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Also hash all present manifest files; this reads about 1.1 GB locally.",
    )
    args = parser.parse_args()
    errors = verify_manifest(args.hash) + verify_protocol()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    mode = "sizes, hashes, and protocol" if args.hash else "sizes and protocol"
    print(f"Handoff verification passed ({mode}).")


if __name__ == "__main__":
    main()
