#!/usr/bin/env python3
"""Verify the lightweight Expanded-649 GitHub handoff without extracting data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_ARCHIVES = {
    "fingerprint_localization/data/expanded_real_32points_20260716/algorithm_ready/"
    "LoRaMorph_ExpandedReal649_v1_20260716.tar.gz": (
        20_118_121,
        "21dac9b8ad448211de8faef5a3748cfc3896a4f50b025f2e9a0c00642e860210",
    ),
    "fingerprint_localization/data/expanded_real_32points_20260716/source_safe_1to10/"
    "deliverables/ExpandedReal649_source_safe_1to10_seed20260626_partner_20260716.tar.gz": (
        80_517_671,
        "625723f2f0a1769554c25bb9187202c41e53d8acba147f6c8b46d2c72fffa81a",
    ),
}

REQUIRED_PATHS = [
    "fingerprint_localization/HANDOFF.md",
    "fingerprint_localization/docs/mainline_202607/EXPANDED_649_PARTNER_HANDOFF_20260717.md",
    "fingerprint_localization/docs/mainline_202607/EXPANDED649_VS_OLD370_MAINLINE_20260717.md",
    "fingerprint_localization/docs/mainline_202607/EXPANDED_LDA_ACO_NO_ALPHA_REFREEZE_20260717.md",
    "fingerprint_localization/experiments/aco_source_safe_1to10/run_no_alpha_validation_refreeze.py",
    "fingerprint_localization/experiments/aco_source_safe_1to10/run_expanded_lda_aco_mainline.py",
    "fingerprint_localization/experiments/aco_source_safe_1to10/run_candidate_recall_and_controlled_weakness.py",
    "fingerprint_localization/experiments/aco_source_safe_1to10/run_search_mechanism_ablation.py",
    "fingerprint_localization/results/expanded_source_safe_1to10/aco_lda_only_mainline/validation_lda_model.joblib",
    "fingerprint_localization/results/expanded_source_safe_1to10/aco_lda_only_mainline/formal_lda_model.joblib",
    "fingerprint_localization/results/expanded_source_safe_1to10/aco_lda_only_no_alpha_refrozen_20260717/FROZEN_CONFIG.json",
    "fingerprint_localization/results/expanded_source_safe_1to10/aco_lda_only_no_alpha_refrozen_20260717/MANIFEST.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(repo_root: Path, skip_hash: bool) -> dict:
    missing = [relative for relative in REQUIRED_PATHS if not (repo_root / relative).is_file()]
    archives = []
    for relative, (expected_size, expected_hash) in EXPECTED_ARCHIVES.items():
        path = repo_root / relative
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256(path) if exists and not skip_hash else None
        archives.append(
            {
                "path": relative,
                "exists": exists,
                "size": actual_size,
                "expected_size": expected_size,
                "size_ok": actual_size == expected_size,
                "sha256": actual_hash,
                "expected_sha256": expected_hash,
                "hash_ok": None if skip_hash else actual_hash == expected_hash,
            }
        )

    config_path = repo_root / (
        "fingerprint_localization/results/expanded_source_safe_1to10/"
        "aco_lda_only_no_alpha_refrozen_20260717/FROZEN_CONFIG.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    config_ok = (
        config.get("candidate_policy") == "direct LDA Top-5"
        and config.get("alpha_fusion") is False
        and config.get("beta") == 0.6
        and config.get("aco", {}).get("seed") == 20260626
    )
    status = "PASS" if not missing and all(row["size_ok"] for row in archives) and config_ok else "FAIL"
    if not skip_hash and not all(row["hash_ok"] for row in archives):
        status = "FAIL"
    report = {
        "status": status,
        "repo_root": str(repo_root),
        "missing_required_paths": missing,
        "archives": archives,
        "frozen_config_ok": config_ok,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--skip-hash", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(**vars(parse_args()))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
