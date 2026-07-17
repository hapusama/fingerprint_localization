#!/usr/bin/env python3
"""Run the adopted Expanded-649 LDA-only full ACO mainline.

The learned prior is standardized LDA(svd) only. Alpha=0.3, beta=0.5,
source-level T_seg, ACO settings, split, and seed remain frozen from the prior
Expanded mainline. The historical LDA+RF outputs are not overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import joblib

import finalize_expanded_aco_ml_score4 as finalizer
import run_aco_v4_source_level_on_split as source
import run_expanded_aco_ml_prior as prior
import run_expanded_supervised_ensemble as supervised


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
HISTORICAL_ROOT = (
    PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "aco_ml_candidate_prior"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "aco_lda_only_mainline"
)
ALPHA = 0.3
BETA = 0.5
VALIDATION_T_SEG = 0.009161130588433
FORMAL_T_SEG = 0.009034126697630788


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def packet_key(row: dict) -> tuple[str, int]:
    return Path(row["file_name"]).stem, int(float(row["packet_index"]))


def load_samples(args: argparse.Namespace) -> list:
    _aco_args, samples, _indices, _q4, _shapes, _struct, _metadata = source.load_samples(args)
    return samples


def run_stage(
    base_args: argparse.Namespace,
    samples: Sequence,
    probabilities: dict,
    rssi_csv: Path,
    spectrum_csv: Path,
    split_csv: Path,
    output_dir: Path,
    split: str,
    t_seg: float,
) -> tuple[dict, dict, list[dict]]:
    stage_args = prior.prepare_stage_args(
        base_args,
        rssi_csv,
        spectrum_csv,
        split_csv,
        output_dir,
        split,
        t_seg,
    )
    aco_metrics = prior.run_aco_stage(stage_args, samples, probabilities, ALPHA)
    groups = finalizer.load_groups(
        output_dir / f"{split}_predictions.csv",
        output_dir / f"{split}_candidate_scores.csv",
        probabilities,
    )
    final_metrics, final_rows = finalizer.evaluate(groups, BETA, split)
    prior.write_csv(output_dir / f"fixed_beta_final_{split}_predictions.csv", final_rows)
    return aco_metrics, final_metrics, final_rows


def compare_to_historical(rows: Sequence[dict], split: str) -> dict:
    historical_path = (
        HISTORICAL_ROOT / "final_validation_predictions.csv"
        if split == "val"
        else HISTORICAL_ROOT / "final_test_predictions.csv"
    )
    historical = {packet_key(row): row for row in read_csv(historical_path)}
    changed = candidate_changed = w2r = r2w = both_wrong = 0
    for row in rows:
        old = historical[packet_key(row)]
        old_ok = old["final_label"] == old["true_label"]
        new_ok = row["final_label"] == row["true_label"]
        changed += int(old["final_label"] != row["final_label"])
        candidate_changed += int(old["candidate_labels"] != row["candidate_labels"])
        w2r += int((not old_ok) and new_ok)
        r2w += int(old_ok and (not new_ok))
        both_wrong += int(
            old["final_label"] != row["final_label"]
            and (not old_ok)
            and (not new_ok)
        )
    return {
        "split": split,
        "reference": "historical_lda_rf_equal",
        "reference_correct": sum(int(row["final_correct"]) for row in historical.values()),
        "lda_only_correct": sum(int(row["final_correct"]) for row in rows),
        "final_label_change_count": changed,
        "candidate_list_change_count": candidate_changed,
        "W2R": w2r,
        "R2W": r2w,
        "both_wrong_changed": both_wrong,
        "net_gain": w2r - r2w,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path) -> None:
    target = output_dir / "MAINLINE_CHECKSUMS.sha256"
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_args = prior.default_runner_args()

    validation_table = supervised.load_validation_table(supervised.DEFAULT_VALIDATION_FEATURES)
    validation_train = [row for row in validation_table if row["split"] == "train"]
    validation_eval = [row for row in validation_table if row["split"] == "val"]
    validation_probabilities, validation_classes, validation_models = prior.fit_prior(
        validation_train,
        validation_eval,
        args.seed,
        ["lda_svd"],
    )
    validation_args = prior.prepare_stage_args(
        base_args,
        supervised.SOURCE_SAFE_DIR / "data" / "noisy_rssi_plus_packet_level_32points_649.csv",
        supervised.SOURCE_SAFE_DIR / "data" / "noisy_subbin_spectrum_long_32points_649.csv",
        supervised.SOURCE_SAFE_DIR / "data" / "split_assignments.csv",
        args.output_dir / "validation",
        "val",
        VALIDATION_T_SEG,
    )
    validation_samples = load_samples(validation_args)
    val_aco, val_final, val_rows = run_stage(
        base_args,
        validation_samples,
        validation_probabilities,
        validation_args.rssi_csv,
        validation_args.spectrum_csv,
        validation_args.split_csv,
        args.output_dir / "validation",
        "val",
        VALIDATION_T_SEG,
    )
    joblib.dump(validation_models, args.output_dir / "validation_lda_model.joblib")

    refit_table = supervised.load_refit_table(
        supervised.DEFAULT_REFIT_RSSI,
        supervised.DEFAULT_REFIT_RAW,
        supervised.DEFAULT_REFIT_SPLIT,
    )
    refit_train = [row for row in refit_table if row["split"] == "train"]
    formal_eval = [row for row in refit_table if row["split"] == "test"]
    formal_probabilities, formal_classes, formal_models = prior.fit_prior(
        refit_train,
        formal_eval,
        args.seed,
        ["lda_svd"],
    )
    if validation_classes != formal_classes:
        raise RuntimeError("Validation and formal LDA class orders differ")
    formal_args = prior.prepare_stage_args(
        base_args,
        supervised.DEFAULT_REFIT_RSSI,
        supervised.REFIT_DIR / "data" / "noisy_subbin_spectrum_long_32points_649.csv",
        supervised.DEFAULT_REFIT_SPLIT,
        args.output_dir / "formal_test",
        "test",
        FORMAL_T_SEG,
    )
    formal_samples = load_samples(formal_args)
    test_aco, test_final, test_rows = run_stage(
        base_args,
        formal_samples,
        formal_probabilities,
        formal_args.rssi_csv,
        formal_args.spectrum_csv,
        formal_args.split_csv,
        args.output_dir / "formal_test",
        "test",
        FORMAL_T_SEG,
    )
    joblib.dump(formal_models, args.output_dir / "formal_lda_model.joblib")

    if val_final["final_correct"] != 117 or test_final["final_correct"] != 120:
        raise RuntimeError(
            "Adopted LDA-only result mismatch: "
            f"validation={val_final['final_correct']}/128, formal={test_final['final_correct']}/128"
        )
    comparisons = [
        compare_to_historical(val_rows, "val"),
        compare_to_historical(test_rows, "test"),
    ]
    prior.write_csv(args.output_dir / "comparisons_vs_historical_lda_rf.csv", comparisons)

    payload = {
        "status": "ADOPTED_MAINLINE",
        "method": "Standardized LDA(svd) prior + source-level ACO v4 + fixed-beta fusion",
        "adoption_date": "2026-07-16",
        "protocol": {
            "dataset": "ExpandedReal-649-v1",
            "seed": args.seed,
            "source_split": {"train": 393, "validation": 128, "test": 128},
            "formal_refit_sources": 521,
            "formal_refit_rows": 5210,
            "formal_test_packets": 128,
            "source_overlap": 0,
            "validation_and_test_augmentation": False,
            "posterior_components": ["lda_svd"],
            "RF_used": False,
            "alpha": ALPHA,
            "beta": BETA,
            "validation_T_seg": VALIDATION_T_SEG,
            "formal_T_seg": FORMAL_T_SEG,
            "ACO": {
                "top_k": 5,
                "rssi_class_k": 3,
                "segments": 4,
                "ants": 16,
                "iterations": 12,
                "elite_ants": 4,
            },
            "test_status": "exploratory because the Expanded test was previously inspected",
        },
        "validation": {
            "ml_top1_correct": val_final["ml_prior_top1_correct"],
            "candidate_rank_top1_correct": val_aco["rssi_top1_correct"],
            "candidate_true_top5_recall": val_aco["rssi_topk_recall"],
            "aco_score4_correct": val_final["aco_score4_correct"],
            "final_correct": val_final["final_correct"],
            "final_accuracy": val_final["final_accuracy"],
        },
        "formal_test": {
            "ml_top1_correct": test_final["ml_prior_top1_correct"],
            "candidate_rank_top1_correct": test_aco["rssi_top1_correct"],
            "candidate_true_top5_recall": test_aco["rssi_topk_recall"],
            "aco_score4_correct": test_final["aco_score4_correct"],
            "final_correct": test_final["final_correct"],
            "final_accuracy": test_final["final_accuracy"],
        },
        "comparisons_vs_historical_lda_rf": comparisons,
        "historical_configuration": {
            "method": "LDA+RF(700) equal posterior + ACO",
            "validation_correct": 116,
            "formal_correct": 119,
            "status": "superseded; retained for comparison",
        },
    }
    (args.output_dir / "mainline_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "mainline_report.md").write_text(
        "# Expanded-649 Adopted LDA-only ACO Mainline\n\n"
        "- Posterior: standardized LDA(svd), no RF.\n"
        f"- Alpha: {ALPHA}; beta: {BETA}.\n"
        f"- Validation: {val_final['final_correct']}/128 = {val_final['final_accuracy']:.2%}.\n"
        f"- Formal test: {test_final['final_correct']}/128 = {test_final['final_accuracy']:.2%}.\n"
        f"- Formal true Top-5 recall: {round(test_aco['rssi_topk_recall'] * 128)}/128.\n"
        "- Historical LDA+RF formal result: 119/128; superseded.\n\n"
        "The Expanded formal test was previously inspected, so this adopted result remains exploratory.\n",
        encoding="utf-8",
    )
    write_checksums(args.output_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
