#!/usr/bin/env python3
"""Retrain the no-alpha LDA/ACO pipeline with RSSI+ features only.

This is a controlled feature ablation of the 2026-07-17 no-alpha refreeze:
the LDA candidate model sees only the six RSSI+ columns, while the ACO search
keeps the same segment observations, templates, hyperparameters, and seed.
Beta is selected again on validation and frozen before formal evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_candidate_recall_and_controlled_weakness as weakness
import run_expanded_supervised_ensemble as supervised
import run_no_alpha_validation_refreeze as no_alpha


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "aco_rssi_only_lda_no_alpha_refrozen_20260717"
)
REFERENCE_ROOT = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "aco_lda_only_no_alpha_refrozen_20260717"
)
SEED = weakness.SEED


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    first = list(rows[0])
    fields = first + sorted({field for row in rows for field in row} - set(first))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rssi_matrix(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray(
        [
            [supervised.number(row[column]) for column in supervised.RSSI_COLUMNS]
            for row in rows
        ],
        dtype=float,
    )


def fit_rssi_only_lda(
    train_rows: Sequence[dict], eval_rows: Sequence[dict]
) -> tuple[dict, dict[tuple[str, int], dict[str, float]], list[str]]:
    """Fit the same standardized LDA(svd), changing only its feature columns."""
    x_train = rssi_matrix(train_rows)
    y_train = supervised.labels(train_rows)
    x_eval = rssi_matrix(eval_rows)
    class_order = sorted(set(y_train), key=supervised.natural_label_key)
    model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="svd"),
    )
    model.fit(x_train, y_train)
    common = supervised.common_probabilities(
        model,
        model.predict_proba(x_eval),
        class_order,
    )
    probabilities = {}
    for row, row_probabilities in zip(eval_rows, common):
        key = (row["file_stem"], int(float(row["packet_index"])))
        probabilities[key] = {
            label: float(row_probabilities[index])
            for index, label in enumerate(class_order)
        }
    return {"lda_svd": model}, probabilities, class_order


def topk_summary(rows: Sequence[dict], feature_set: str) -> dict:
    full = [row for row in rows if row["method"] == "full_aco"]
    counts = {}
    for k in (1, 3, 5):
        counts[k] = sum(
            row["true_label"] in row["candidate_labels"].split(";")[:k]
            for row in full
        )
    return {
        "split": full[0]["split"],
        "lda_feature_set": feature_set,
        "packets": len(full),
        "lda_top1_recall_count": counts[1],
        "lda_top1_recall": counts[1] / len(full),
        "lda_top3_recall_count": counts[3],
        "lda_top3_recall": counts[3] / len(full),
        "lda_top5_recall_count": counts[5],
        "lda_top5_recall": counts[5] / len(full),
        "candidate_truncation_errors": len(full) - counts[5],
    }


def compare_stages(
    new_rows: Sequence[dict], reference_path: Path
) -> list[dict]:
    reference = {
        (Path(row["file_name"]).stem, int(float(row["packet_index"]))): row
        for row in read_csv(reference_path)
        if row["method"] == "full_aco"
    }
    new = [row for row in new_rows if row["method"] == "full_aco"]
    output = []
    for stage in ("lda", "search", "final"):
        column = f"{stage}_label"
        changed = w2r = r2w = changed_both_wrong = 0
        reference_correct = new_correct = 0
        for row in new:
            key = (Path(row["file_name"]).stem, int(row["packet_index"]))
            reference_row = reference[key]
            old_label = reference_row[column]
            new_label = row[column]
            true_label = row["true_label"]
            old_ok = old_label == true_label
            new_ok = new_label == true_label
            reference_correct += int(old_ok)
            new_correct += int(new_ok)
            changed += int(old_label != new_label)
            w2r += int(not old_ok and new_ok)
            r2w += int(old_ok and not new_ok)
            changed_both_wrong += int(
                old_label != new_label and not old_ok and not new_ok
            )
        output.append(
            {
                "split": new[0]["split"],
                "stage": stage,
                "reference_feature_set": "RSSI+S17",
                "new_feature_set": "RSSI+ only",
                "reference_correct": reference_correct,
                "new_correct": new_correct,
                "accuracy_delta_pp": 100.0 * (new_correct - reference_correct) / len(new),
                "changed": changed,
                "W2R": w2r,
                "R2W": r2w,
                "net_gain": w2r - r2w,
                "changed_both_wrong": changed_both_wrong,
                "mcnemar_exact_two_sided_p": no_alpha.mcnemar_exact(w2r, r2w),
            }
        )
    return output


def reference_topk(path: Path) -> dict:
    return topk_summary(read_csv(path), "RSSI+S17")


def render_report(
    output_dir: Path,
    frozen_beta: float,
    beta_rows: Sequence[dict],
    recall_rows: Sequence[dict],
    metrics: Sequence[dict],
    comparisons: Sequence[dict],
) -> None:
    lines = [
        "# RSSI-only LDA feature ablation (no alpha)",
        "",
        "Date: 2026-07-17",
        "",
        "Controlled change: LDA uses only the six RSSI+ features instead of RSSI+ plus 21 S17/raw features. ACO segment observations and all search parameters are unchanged.",
        "",
        "## Validation beta refreeze",
        "",
        "| beta | correct | accuracy | changes vs RSSI-only LDA |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in beta_rows:
        mark = " **(selected)**" if float(row["beta"]) == frozen_beta else ""
        lines.append(
            f"| {float(row['beta']):.1f}{mark} | {row['final_correct']}/128 | "
            f"{float(row['final_accuracy']):.2%} | {row['changes_vs_lda']} |"
        )
    lines.extend(
        [
            "",
            "Selection rule: maximum validation accuracy, then smaller beta on ties.",
            "",
            "## LDA candidate recall",
            "",
            "| Split | LDA inputs | Top-1 | Top-3 | Top-5 | cutoff errors |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in recall_rows:
        lines.append(
            f"| {row['split']} | {row['lda_feature_set']} | "
            f"{row['lda_top1_recall_count']}/128 | {row['lda_top3_recall_count']}/128 | "
            f"{row['lda_top5_recall_count']}/128 | {row['candidate_truncation_errors']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-beta full ACO",
            "",
            "| Split | LDA inputs | LDA/search/final correct | final accuracy | MAE/P95 m | severe >10 m | correction precision |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ("validation", "formal_test"):
        row = next(
            item
            for item in metrics
            if item["split"] == split and item["method"] == "full_aco"
        )
        precision = row["correction_precision"]
        precision_text = "n/a" if precision == "" else f"{float(precision):.2%}"
        lines.append(
            f"| {split} | RSSI+ only | {row['lda_correct']}/{row['search_correct']}/{row['final_correct']} | "
            f"{float(row['final_accuracy']):.2%} | {float(row['final_mae_m']):.3f}/{float(row['final_p95_m']):.3f} | "
            f"{float(row['severe_error_rate']):.2%} | {precision_text} |"
        )
    lines.extend(
        [
            "",
            "## Paired change versus RSSI+S17 no-alpha refreeze",
            "",
            "| Split | Stage | old/new correct | delta pp | W2R/R2W | changed | McNemar p |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['split']} | {row['stage']} | {row['reference_correct']}/{row['new_correct']} | "
            f"{float(row['accuracy_delta_pp']):+.2f} | {row['W2R']}/{row['R2W']} | "
            f"{row['changed']} | {float(row['mcnemar_exact_two_sided_p']):.6f} |"
        )
    lines.extend(
        [
            "",
            "The formal split is exploratory because it was inspected by earlier experiments.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distances = {
        row["position_key"]: float(row["distance_m"])
        for row in read_csv(weakness.LOCATION_CSV)
    }

    # Selection boundary: train/evaluate validation and freeze beta first.
    validation = weakness.load_context("validation", build_training_state=True)
    validation.models, validation.probabilities, validation_classes = fit_rssi_only_lda(
        validation.train_rows, validation.eval_rows
    )
    validation_search = no_alpha.evaluate_search(validation)
    beta_rows = []
    for beta in no_alpha.BETA_GRID:
        beta_predictions = no_alpha.apply_beta(validation_search, beta)
        metric = no_alpha.metric_row(beta_predictions, "full_aco", distances)
        beta_rows.append(
            {
                "beta": beta,
                "final_correct": metric["final_correct"],
                "final_accuracy": metric["final_accuracy"],
                "changes_vs_lda": metric["changes_vs_lda"],
                "beneficial_corrections": metric["beneficial_corrections"],
                "harmful_corrections": metric["harmful_corrections"],
                "correction_precision": metric["correction_precision"],
            }
        )
    selected = max(
        beta_rows,
        key=lambda row: (int(row["final_correct"]), -float(row["beta"])),
    )
    frozen_beta = float(selected["beta"])
    validation_predictions = no_alpha.apply_beta(validation_search, frozen_beta)

    # Formal training/evaluation begins only after beta is frozen.
    formal = weakness.load_context("formal_test", build_training_state=True)
    formal.models, formal.probabilities, formal_classes = fit_rssi_only_lda(
        formal.train_rows, formal.eval_rows
    )
    formal_search = no_alpha.evaluate_search(formal)
    formal_predictions = no_alpha.apply_beta(formal_search, frozen_beta)

    for split_rows in (validation_predictions, formal_predictions):
        full = [row for row in split_rows if row["method"] == "full_aco"]
        if len(full) != 128:
            raise RuntimeError("Expected 128 full-ACO packet predictions")
        if any(row["candidate_labels"].split(";")[0] != row["lda_label"] for row in full):
            raise RuntimeError("Candidate order is not direct RSSI-only LDA order")

    write_csv(args.output_dir / "validation_beta_selection.csv", beta_rows)
    write_csv(args.output_dir / "validation_predictions.csv", validation_predictions)
    write_csv(args.output_dir / "formal_predictions.csv", formal_predictions)
    joblib.dump(validation.models, args.output_dir / "validation_rssi_only_lda_model.joblib")
    joblib.dump(formal.models, args.output_dir / "formal_rssi_only_lda_model.joblib")

    metrics = [
        no_alpha.metric_row(split_rows, method, distances)
        for split_rows in (validation_predictions, formal_predictions)
        for method in no_alpha.METHOD_ORDER
    ]
    write_csv(args.output_dir / "metrics.csv", metrics)
    pairwise = no_alpha.pairwise_rows(validation_predictions) + no_alpha.pairwise_rows(
        formal_predictions
    )
    write_csv(args.output_dir / "full_aco_pairwise.csv", pairwise)

    recall_rows = [
        reference_topk(REFERENCE_ROOT / "validation_predictions.csv"),
        topk_summary(validation_predictions, "RSSI+ only"),
        reference_topk(REFERENCE_ROOT / "formal_predictions.csv"),
        topk_summary(formal_predictions, "RSSI+ only"),
    ]
    write_csv(args.output_dir / "lda_candidate_recall_comparison.csv", recall_rows)
    comparisons = compare_stages(
        validation_predictions, REFERENCE_ROOT / "validation_predictions.csv"
    ) + compare_stages(formal_predictions, REFERENCE_ROOT / "formal_predictions.csv")
    write_csv(args.output_dir / "comparison_vs_rssi_s17.csv", comparisons)

    frozen_config = {
        "candidate_policy": "direct RSSI-only LDA Top-5",
        "lda_feature_columns": supervised.RSSI_COLUMNS,
        "excluded_from_lda": supervised.RAW_COLUMNS,
        "alpha_fusion": False,
        "beta": frozen_beta,
        "validation_t_seg": float(validation.args.t_seg_resolved),
        "formal_t_seg": float(formal.args.t_seg_resolved),
        "aco_parameters": {
            key: getattr(validation.args, key)
            for key in [
                "top_k",
                "rssi_class_k",
                "segment_count",
                "ants",
                "iterations",
                "elite_ants",
                "pheromone_power",
                "heuristic_power",
                "evaporation",
                "min_pheromone",
                "tau_stay",
                "tau_switch",
                "lambda_c",
                "lambda_div",
                "lambda_g",
                "max_garbage",
                "garbage_overuse_penalty",
                "lambda_q_switch",
                "lambda_rssi_prior",
                "lambda_raw_prior",
                "lambda_veto",
                "lambda_score_vote",
                "lambda_score_cost",
                "rssi_weight",
                "raw_weight",
                "energy_weight",
                "bin_weight",
                "q4_weight",
                "seed",
            ]
        },
    }
    (args.output_dir / "FROZEN_CONFIG.json").write_text(
        json.dumps(frozen_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    render_report(
        args.output_dir,
        frozen_beta,
        beta_rows,
        recall_rows,
        metrics,
        comparisons,
    )

    payload = {
        "experiment": "EXPANDED_RSSI_ONLY_LDA_ACO_NO_ALPHA_REFREEZE_20260717",
        "status": "CONTROLLED_FEATURE_ABLATION",
        "protocol": {
            "changed_factor": "LDA inputs reduced from RSSI+S17 (27) to RSSI+ only (6)",
            "unchanged_factor": "ACO observations, templates, parameters, seed, and no-alpha candidate policy",
            "beta_selection": "validation maximum accuracy; smaller beta on ties",
            "frozen_beta": frozen_beta,
            "formal_evaluated_after_freeze": True,
            "seed": SEED,
            "test_status": "exploratory because Expanded formal test was previously inspected",
        },
        "validation_beta_selection": beta_rows,
        "frozen_configuration": frozen_config,
        "candidate_recall": recall_rows,
        "metrics": metrics,
        "comparison_vs_rssi_s17": comparisons,
        "checks": {
            "validation_train_rows": len(validation.train_rows),
            "validation_eval_rows": len(validation.eval_rows),
            "formal_train_rows": len(formal.train_rows),
            "formal_eval_rows": len(formal.eval_rows),
            "validation_class_count": len(validation_classes),
            "formal_class_count": len(formal_classes),
            "validation_prediction_rows": len(validation_predictions),
            "formal_prediction_rows": len(formal_predictions),
            "validation_model_feature_count": int(
                validation.models["lda_svd"].n_features_in_
            ),
            "formal_model_feature_count": int(
                formal.models["lda_svd"].n_features_in_
            ),
        },
        "runtime": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "inputs": {
            "reference_config_sha256": sha256(REFERENCE_ROOT / "FROZEN_CONFIG.json"),
            "reference_validation_predictions_sha256": sha256(
                REFERENCE_ROOT / "validation_predictions.csv"
            ),
            "reference_formal_predictions_sha256": sha256(
                REFERENCE_ROOT / "formal_predictions.csv"
            ),
        },
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    checksum_path = args.output_dir / "CHECKSUMS.sha256"
    files = sorted(
        path for path in args.output_dir.iterdir() if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
