#!/usr/bin/env python3
"""Expanded-649 ACO with a validation-weighted discriminative candidate prior.

The LDA+RF model is not the final classifier.  Its posterior only adjusts the
RSSI candidate ranking and prior costs.  Segment observations, pheromone
updates, elite paths, and Score4 remain responsible for the final prediction.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_aco_v4_on_split as runner
import run_aco_v4_source_level_on_split as source
import run_expanded_supervised_ensemble as supervised


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "aco_ml_candidate_prior"
)
WEIGHT_GRID = [round(index / 10.0, 1) for index in range(11)]
EPS = 1e-12


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def default_runner_args() -> argparse.Namespace:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0]]
        return runner.parse_args()
    finally:
        sys.argv = original_argv


def make_prior_models(seed: int) -> dict:
    return {
        "lda_svd": make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="svd"),
        ),
        "rf_leaf2": RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def fit_prior(
    train_rows: Sequence[dict],
    eval_rows: Sequence[dict],
    seed: int,
    components: Sequence[str],
) -> tuple[dict[tuple[str, int], dict[str, float]], list[str], dict]:
    x_train = supervised.matrix(train_rows)
    y_train = supervised.labels(train_rows)
    x_eval = supervised.matrix(eval_rows)
    class_order = sorted(set(y_train), key=supervised.natural_label_key)
    probabilities = np.zeros((len(eval_rows), len(class_order)), dtype=float)
    available_models = make_prior_models(seed)
    unknown = set(components) - set(available_models)
    if not components or unknown:
        raise ValueError(f"Invalid prior components: {list(components)}; unknown={sorted(unknown)}")
    models = {name: available_models[name] for name in components}
    for name, model in models.items():
        print(f"fit candidate prior {name}", flush=True)
        model.fit(x_train, y_train)
        model_probabilities = supervised.common_probabilities(
            model,
            model.predict_proba(x_eval),
            class_order,
        )
        probabilities += model_probabilities / len(models)
    output = {}
    for row, row_probabilities in zip(eval_rows, probabilities):
        key = (row["file_stem"], int(float(row["packet_index"])))
        output[key] = {
            label: float(row_probabilities[index])
            for index, label in enumerate(class_order)
        }
    return output, class_order, models


def make_ml_source_class_rank(
    samples: Sequence,
    probability_by_source: dict[tuple[str, int], dict[str, float]],
    prior_weight: float,
):
    source_rank = source.make_source_level_class_rank(samples)

    def class_rank(
        rows: Sequence[Sequence[float]],
        labels: Sequence[str],
        train_indices: Sequence[int],
        test_index: int,
        class_neighbor_k: int,
        candidate_labels: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        ranked = source_rank(
            rows,
            labels,
            train_indices,
            test_index,
            class_neighbor_k,
            candidate_labels,
        )
        if prior_weight <= EPS:
            return ranked
        probabilities = probability_by_source.get(source.source_id(samples[test_index]))
        if not probabilities or len(ranked) < 2:
            return ranked
        rssi_scores = {label: float(score) for label, score in ranked}
        rssi_min = min(rssi_scores.values())
        rssi_max = max(rssi_scores.values())
        rssi_span = max(EPS, rssi_max - rssi_min)
        ml_costs = {
            label: -math.log(max(EPS, probabilities.get(label, EPS)))
            for label in rssi_scores
        }
        ml_min = min(ml_costs.values())
        ml_max = max(ml_costs.values())
        ml_span = max(EPS, ml_max - ml_min)
        combined = []
        for label in rssi_scores:
            ml_scaled = rssi_min + (ml_costs[label] - ml_min) / ml_span * rssi_span
            score = (1.0 - prior_weight) * rssi_scores[label] + prior_weight * ml_scaled
            combined.append((label, score))
        combined.sort(key=lambda item: (item[1], supervised.natural_label_key(item[0])))
        return combined

    return class_rank


def prepare_stage_args(
    base_args: argparse.Namespace,
    rssi_csv: Path,
    spectrum_csv: Path,
    split_csv: Path,
    output_dir: Path,
    split: str,
    t_seg: float,
) -> argparse.Namespace:
    args = copy.deepcopy(base_args)
    args.rssi_csv = rssi_csv
    args.spectrum_csv = spectrum_csv
    args.split_csv = split_csv
    args.output_dir = output_dir
    args.result_dir = output_dir.parent
    args.method_summary = output_dir / "method_summary_with_aco_v4.csv"
    args.aco_v2_dir = output_dir / "nonexistent_aco_v2"
    args.splits = split
    args.top_k = 5
    args.rssi_class_k = 3
    args.segment_count = 4
    args.seed = 20260626
    args.t_seg = t_seg
    return args


def configure_aco(samples: Sequence, probabilities: dict, weight: float) -> None:
    source.aco2.build_templates = source.source_level_build_templates
    source.aco2.build_segment_prototypes = source.source_level_build_prototypes
    source.base.class_rank = make_ml_source_class_rank(samples, probabilities, weight)


def run_aco_stage(
    args: argparse.Namespace,
    samples: Sequence,
    probabilities: dict,
    weight: float,
) -> dict:
    configure_aco(samples, probabilities, weight)
    metadata = runner.run(args)
    if len(metadata["summary"]) != 1:
        raise RuntimeError("Expected one requested split")
    return metadata["summary"][0]


def calibrate_source_t_seg(args: argparse.Namespace) -> tuple[float, dict, list]:
    (
        aco_args,
        samples,
        split_indices,
        q4_offsets,
        chirp_shapes,
        chirp_struct,
        _chirp_metadata,
    ) = source.load_samples(args)
    resolved, metadata = source.calibrate_t_seg(
        samples,
        split_indices["train"],
        q4_offsets,
        chirp_shapes,
        chirp_struct,
        aco_args,
        5,
    )
    return resolved, metadata, samples


def load_prediction_map(path: Path, column: str) -> dict[tuple[str, int], str]:
    return {
        (Path(row["file_name"]).stem, int(float(row["packet_index"]))): row[column]
        for row in read_csv(path)
    }


def augment_predictions(
    prediction_path: Path,
    probability_by_source: dict[tuple[str, int], dict[str, float]],
    baseline_path: Path,
) -> tuple[list[dict], dict]:
    baseline = load_prediction_map(baseline_path, "final_label")
    rows = read_csv(prediction_path)
    w2r = r2w = changed = 0
    ml_correct = aco_correct = 0
    aco_changes_from_ml = 0
    output = []
    for row in rows:
        key = (Path(row["file_name"]).stem, int(float(row["packet_index"])))
        probabilities = probability_by_source[key]
        ml_label = max(
            probabilities,
            key=lambda label: (probabilities[label], tuple(-value for value in supervised.natural_label_key(label))),
        )
        ordered = sorted(probabilities.values(), reverse=True)
        ml_margin = ordered[0] - ordered[1]
        true_label = row["true_label"]
        final_label = row["final_label"]
        baseline_label = baseline[key]
        baseline_ok = baseline_label == true_label
        final_ok = final_label == true_label
        changed += int(final_label != baseline_label)
        w2r += int(not baseline_ok and final_ok)
        r2w += int(baseline_ok and not final_ok)
        ml_correct += int(ml_label == true_label)
        aco_correct += int(final_ok)
        aco_changes_from_ml += int(final_label != ml_label)
        output.append(
            {
                **row,
                "ml_prior_top1_label": ml_label,
                "ml_prior_top1_probability": probabilities[ml_label],
                "ml_prior_margin": ml_margin,
                "ml_prior_top1_correct": int(ml_label == true_label),
                "source_level_aco_baseline_label": baseline_label,
                "aco_changed_from_ml_top1": int(final_label != ml_label),
            }
        )
    transition = {
        "packet_count": len(rows),
        "baseline_correct": sum(
            baseline[(Path(row["file_name"]).stem, int(float(row["packet_index"]))) ] == row["true_label"]
            for row in rows
        ),
        "final_correct": aco_correct,
        "changed_from_baseline": changed,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "ml_prior_top1_correct": ml_correct,
        "aco_changed_from_ml_top1": aco_changes_from_ml,
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(w2r, r2w),
    }
    return output, transition


def mcnemar_exact_p(w2r: int, r2w: int) -> float:
    discordant = w2r + r2w
    if discordant == 0:
        return 1.0
    lower = min(w2r, r2w)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def wilson_interval(correct: int, count: int, z: float = 1.96) -> tuple[float, float]:
    p = correct / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return center - radius, center + radius


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path) -> None:
    path = output_dir / "RESULT_CHECKSUMS.sha256"
    lines = []
    for item in sorted(output_dir.iterdir()):
        if item.is_file() and item != path:
            lines.append(f"{sha256(item)}  {item.name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_args = default_runner_args()

    validation_table = supervised.load_validation_table(args.validation_features)
    validation_train = [row for row in validation_table if row["split"] == "train"]
    validation_eval = [row for row in validation_table if row["split"] == "val"]
    val_probabilities, _class_order, val_models = fit_prior(
        validation_train,
        validation_eval,
        args.seed,
        args.prior_components,
    )
    validation_args = prepare_stage_args(
        base_args,
        args.validation_rssi,
        args.validation_spectrum,
        args.validation_split,
        args.output_dir / "validation" / "placeholder",
        "val",
        1.0,
    )
    validation_t_seg, validation_calibration, validation_samples = calibrate_source_t_seg(validation_args)
    validation_rows = []
    for weight in WEIGHT_GRID:
        print(f"validation ACO learned-prior weight={weight:.1f}", flush=True)
        trial_dir = args.output_dir / "validation" / f"weight_{weight:.1f}"
        trial_args = prepare_stage_args(
            base_args,
            args.validation_rssi,
            args.validation_spectrum,
            args.validation_split,
            trial_dir,
            "val",
            validation_t_seg,
        )
        metrics = run_aco_stage(trial_args, validation_samples, val_probabilities, weight)
        validation_rows.append(
            {
                "prior_weight": weight,
                "correct": metrics["final_correct"],
                "accuracy": metrics["final_accuracy"],
                "rssi_top1_correct": metrics["rssi_top1_correct"],
                "rssi_top5_recall": metrics["rssi_topk_recall"],
                "T_seg": metrics["T_seg"],
                "output_dir": str(trial_dir),
            }
        )
    selected = max(validation_rows, key=lambda row: (row["correct"], -row["prior_weight"]))
    selected_weight = float(selected["prior_weight"])
    write_csv(args.output_dir / "validation_weight_selection.csv", validation_rows)
    joblib.dump(val_models, args.output_dir / "validation_prior_models.joblib")
    print(f"selected learned-prior weight={selected_weight:.1f}: {selected['correct']}/128", flush=True)

    refit_table = supervised.load_refit_table(args.refit_rssi, args.refit_raw, args.refit_split)
    refit_train = [row for row in refit_table if row["split"] == "train"]
    formal_eval = [row for row in refit_table if row["split"] == "test"]
    test_probabilities, _formal_classes, formal_models = fit_prior(
        refit_train,
        formal_eval,
        args.seed,
        args.prior_components,
    )
    formal_args_for_calibration = prepare_stage_args(
        base_args,
        args.refit_rssi,
        args.refit_spectrum,
        args.refit_split,
        args.output_dir / "formal_test",
        "test",
        1.0,
    )
    formal_t_seg, formal_calibration, formal_samples = calibrate_source_t_seg(formal_args_for_calibration)
    formal_args = prepare_stage_args(
        base_args,
        args.refit_rssi,
        args.refit_spectrum,
        args.refit_split,
        args.output_dir / "formal_test",
        "test",
        formal_t_seg,
    )
    formal_metrics = run_aco_stage(
        formal_args,
        formal_samples,
        test_probabilities,
        selected_weight,
    )
    joblib.dump(formal_models, args.output_dir / "formal_prior_models.joblib")
    detailed_rows, transitions = augment_predictions(
        args.output_dir / "formal_test" / "test_predictions.csv",
        test_probabilities,
        args.aco_baseline / "test_predictions.csv",
    )
    write_csv(args.output_dir / "test_predictions_with_prior.csv", detailed_rows)
    test_correct = int(formal_metrics["final_correct"])
    payload = {
        "status": "PASS",
        "method": "ACO v4 with discriminative candidate prior: " + ";".join(args.prior_components),
        "aco_role": (
            "The learned posterior only modifies candidate ranking and RSSI prior costs. Final inference "
            "uses four-segment observation costs, pheromone updates, elite paths, and Score4."
        ),
        "protocol": {
            "dataset": "ExpandedReal-649-v1",
            "source_split_seed": 20260626,
            "validation_sources": {"train": 393, "validation": 128},
            "formal_sources": {"train_refit": 521, "test": 128},
            "validation_and_test_augmentation": False,
            "source_overlap": 0,
            "prior_architecture": "equal posterior average: " + ";".join(args.prior_components),
            "prior_weight_selection": "validation only; ties choose the smaller learned-prior weight",
            "test_status": "exploratory because this test split was already inspected",
        },
        "validation_selection": selected,
        "validation_grid": validation_rows,
        "T_seg_calibration": {
            "validation": validation_calibration,
            "formal_test": formal_calibration,
        },
        "formal_test": {
            "packet_count": 128,
            "correct": test_correct,
            "accuracy": test_correct / 128,
            "target_correct_for_at_least_90_54_percent": 116,
            "target_met": test_correct >= 116,
            "wilson95": wilson_interval(test_correct, 128),
            "transitions": transitions,
        },
        "ablation_reference": {
            "no_learned_prior_validation": next(row for row in validation_rows if row["prior_weight"] == 0.0),
            "no_learned_prior_formal_test_correct": transitions["baseline_correct"],
        },
    }
    (args.output_dir / "aco_ml_prior_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = (
        "# Expanded-649 ACO Learned Candidate Prior\n\n"
        f"- Validation-selected prior weight: {selected_weight:.1f}.\n"
        f"- Validation: {selected['correct']}/128 = {selected['accuracy']:.2%}.\n"
        f"- Formal test: {test_correct}/128 = {test_correct / 128:.2%}.\n"
        f"- No-prior source-level ACO: {transitions['baseline_correct']}/128.\n"
        f"- Versus no-prior ACO: W2R={transitions['W2R']}, R2W={transitions['R2W']}, "
        f"McNemar p={transitions['mcnemar_exact_two_sided_p']:.6f}.\n"
        f"- ACO predictions differing from the ML prior Top-1: {transitions['aco_changed_from_ml_top1']}.\n\n"
        "The ML posterior is an offline candidate prior; the final prediction remains the output of "
        "the segmented ACO path search and Score4. The result is exploratory because the test split "
        "had already been inspected.\n"
    )
    (args.output_dir / "aco_ml_prior_report.md").write_text(report, encoding="utf-8")
    write_checksums(args.output_dir)
    print(json.dumps(payload["formal_test"], indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-features", type=Path, default=supervised.DEFAULT_VALIDATION_FEATURES)
    parser.add_argument("--validation-rssi", type=Path, default=source.SOURCE_LEVEL_RSSI if hasattr(source, "SOURCE_LEVEL_RSSI") else supervised.SOURCE_SAFE_DIR / "data" / "noisy_rssi_plus_packet_level_32points_649.csv")
    parser.add_argument("--validation-spectrum", type=Path, default=supervised.SOURCE_SAFE_DIR / "data" / "noisy_subbin_spectrum_long_32points_649.csv")
    parser.add_argument("--validation-split", type=Path, default=supervised.SOURCE_SAFE_DIR / "data" / "split_assignments.csv")
    parser.add_argument("--refit-rssi", type=Path, default=supervised.DEFAULT_REFIT_RSSI)
    parser.add_argument("--refit-raw", type=Path, default=supervised.DEFAULT_REFIT_RAW)
    parser.add_argument("--refit-spectrum", type=Path, default=supervised.REFIT_DIR / "data" / "noisy_subbin_spectrum_long_32points_649.csv")
    parser.add_argument("--refit-split", type=Path, default=supervised.DEFAULT_REFIT_SPLIT)
    parser.add_argument("--aco-baseline", type=Path, default=supervised.DEFAULT_ACO_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--prior-components", default="lda_svd,rf_leaf2")
    args = parser.parse_args()
    args.prior_components = [part.strip() for part in args.prior_components.split(",") if part.strip()]
    return args


if __name__ == "__main__":
    run(parse_args())
