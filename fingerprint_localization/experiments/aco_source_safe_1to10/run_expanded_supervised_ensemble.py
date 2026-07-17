#!/usr/bin/env python3
"""Validation-select and evaluate a supervised Expanded-649 challenger.

The selected model is fitted to the 27 packet-level RSSI+S17 features. Model
selection uses the 128 untouched validation sources.  The selected model is
then refitted on the 5210 train+validation augmented rows and evaluated on the
128 untouched formal-test sources.

The Expanded test has already been inspected by earlier experiments, so this
run is exploratory even though its configuration is validation-selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
EXPANDED_DIR = PROJECT_DIR / "data" / "expanded_real_32points_20260716"
SOURCE_SAFE_DIR = (
    EXPANDED_DIR
    / "source_safe_1to10"
    / "ExpandedReal649_source_safe_1to10_seed20260626"
)
REFIT_DIR = (
    EXPANDED_DIR
    / "trainval_refit"
    / "ExpandedReal649_trainval_refit_seed20260626"
)
DEFAULT_VALIDATION_FEATURES = SOURCE_SAFE_DIR / "features" / "source_safe_1to10_ml_features.csv"
DEFAULT_REFIT_RSSI = REFIT_DIR / "data" / "noisy_rssi_plus_packet_level_32points_649.csv"
DEFAULT_REFIT_RAW = REFIT_DIR / "data" / "noisy_lora_frequency_s17_32points_649.csv"
DEFAULT_REFIT_SPLIT = REFIT_DIR / "data" / "split_assignments.csv"
DEFAULT_ACO_VAL = (
    PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "source_level_val" / "template_and_rssi"
)
DEFAULT_ACO_TEST = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "source_level_formal_refit"
    / "template_and_rssi"
)
DEFAULT_CANDIDATE_RERANK = (
    PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "candidate_rerank_corridor_first"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "supervised_ensemble_challenger"
)

METADATA_COLUMNS = {
    "row_index",
    "split",
    "source_key",
    "augmentation_id",
    "file_stem",
    "packet_index",
    "position_key",
    "label_id",
}
RSSI_COLUMNS = [
    "snr",
    "realtime_average_rssi",
    "median_rssi",
    "mode_rssi",
    "rssi_variance",
    "residual",
]
RAW_COLUMNS = [f"preamble_fft_mag_bin_{offset:+d}" for offset in range(-8, 9)] + [
    "preamble_peak_to_residual_db",
    "detect_score_db",
    "s17_c_s",
    "s17_j_s",
]
FEATURE_COLUMNS = RSSI_COLUMNS + RAW_COLUMNS


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    return output if math.isfinite(output) else 0.0


def packet_key(row: dict) -> tuple[str, int]:
    return Path(row["file_name"]).stem, int(float(row["packet_index"]))


def split_key(row: dict) -> tuple[str, int]:
    return row["file_stem"], int(float(row["packet_index"]))


def natural_label_key(label: str) -> tuple[int, int]:
    corridor, location = label.split("_", 1)
    return int(corridor), int(location)


def load_validation_table(path: Path) -> list[dict]:
    rows = read_csv(path)
    actual_features = [column for column in rows[0] if column not in METADATA_COLUMNS]
    if actual_features != FEATURE_COLUMNS:
        raise RuntimeError(f"Unexpected validation feature schema: {actual_features}")
    counts = {name: sum(row["split"] == name for row in rows) for name in ["train", "val", "test"]}
    if counts != {"train": 3930, "val": 128, "test": 128}:
        raise RuntimeError(f"Unexpected validation table split counts: {counts}")
    return rows


def load_refit_table(rssi_path: Path, raw_path: Path, split_path: Path) -> list[dict]:
    rssi = {packet_key(row): row for row in read_csv(rssi_path)}
    raw = {packet_key(row): row for row in read_csv(raw_path)}
    split = {split_key(row): row["split"] for row in read_csv(split_path)}
    rows = []
    for key in sorted(set(rssi) & set(raw)):
        if key not in split:
            continue
        rssi_row = rssi[key]
        raw_row = raw[key]
        label = rssi_row["position_key"]
        raw_label = f"{int(float(raw_row['corridor_id']))}_{int(float(raw_row['position_id']))}"
        if label != raw_label:
            raise RuntimeError(f"Label mismatch for {key}: {label} != {raw_label}")
        row = {
            "split": split[key],
            "file_stem": key[0],
            "packet_index": key[1],
            "position_key": label,
        }
        row.update({column: number(rssi_row[column]) for column in RSSI_COLUMNS})
        row.update({column: number(raw_row[column]) for column in RAW_COLUMNS})
        rows.append(row)
    counts = {name: sum(row["split"] == name for row in rows) for name in ["train", "val", "test"]}
    if counts != {"train": 5210, "val": 0, "test": 128}:
        raise RuntimeError(f"Unexpected formal refit split counts: {counts}")
    return rows


def matrix(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([[number(row[column]) for column in FEATURE_COLUMNS] for row in rows], dtype=float)


def labels(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([row["position_key"] for row in rows])


def make_models(seed: int) -> dict:
    return {
        "lda_svd": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd")),
        "lda_shrink_0.001": make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.001)
        ),
        "lda_shrink_0.01": make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.01)
        ),
        "lda_auto": make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        ),
        "rf_leaf2": RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "extra_leaf1": ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "svc_c10": make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", probability=True, random_state=seed),
        ),
    }


def common_probabilities(model, probabilities: np.ndarray, class_order: Sequence[str]) -> np.ndarray:
    output = np.zeros((len(probabilities), len(class_order)), dtype=float)
    column = {label: index for index, label in enumerate(class_order)}
    for index, label in enumerate(model.classes_):
        output[:, column[str(label)]] = probabilities[:, index]
    return output


def fit_models(train_rows: Sequence[dict], eval_rows: Sequence[dict], seed: int) -> tuple[dict, list[str]]:
    x_train = matrix(train_rows)
    y_train = labels(train_rows)
    x_eval = matrix(eval_rows)
    class_order = sorted(set(y_train), key=natural_label_key)
    fitted = {}
    for name, model in make_models(seed).items():
        print(f"fit {name}", flush=True)
        model.fit(x_train, y_train)
        probabilities = common_probabilities(model, model.predict_proba(x_eval), class_order)
        fitted[name] = {"model": model, "probabilities": probabilities}
    return fitted, class_order


def predicted_labels(probabilities: np.ndarray, class_order: Sequence[str]) -> list[str]:
    return [class_order[index] for index in probabilities.argmax(axis=1)]


def accuracy_rows(method: str, truth: Sequence[str], predicted: Sequence[str]) -> dict:
    correct = sum(left == right for left, right in zip(truth, predicted))
    return {
        "method": method,
        "packet_count": len(truth),
        "correct": correct,
        "accuracy": correct / len(truth),
    }


def validation_grid(fitted: dict, class_order: Sequence[str], truth: Sequence[str]) -> tuple[list[dict], dict]:
    rows = []
    probability_sets = {}
    for name, item in fitted.items():
        probability_sets[name] = item["probabilities"]
        rows.append({**accuracy_rows(name, truth, predicted_labels(item["probabilities"], class_order)), "components": name})
    ensembles = [
        ("lda_svd_rf_equal", ["lda_svd", "rf_leaf2"]),
        ("lda_svd_extra_equal", ["lda_svd", "extra_leaf1"]),
        ("lda_svd_svc_equal", ["lda_svd", "svc_c10"]),
        ("lda_svd_rf_extra_equal", ["lda_svd", "rf_leaf2", "extra_leaf1"]),
    ]
    for name, components in ensembles:
        probabilities = sum(fitted[component]["probabilities"] for component in components) / len(components)
        probability_sets[name] = probabilities
        rows.append(
            {
                **accuracy_rows(name, truth, predicted_labels(probabilities, class_order)),
                "components": ";".join(components),
            }
        )
    priority = {name: index for index, (name, _components) in enumerate(ensembles)}
    selected = max(
        rows,
        key=lambda row: (
            row["correct"],
            int(row["method"] in priority),
            -priority.get(row["method"], 999),
        ),
    )
    selected = {**selected, "probabilities": probability_sets[selected["method"]]}
    return rows, selected


def prediction_rows(
    split: str,
    eval_rows: Sequence[dict],
    truth: Sequence[str],
    predicted: Sequence[str],
    probabilities: np.ndarray,
    class_order: Sequence[str],
) -> list[dict]:
    output = []
    for row, true_label, predicted_label, probability in zip(eval_rows, truth, predicted, probabilities):
        ordered = np.argsort(-probability)
        top_label = class_order[int(ordered[0])]
        second_label = class_order[int(ordered[1])]
        output.append(
            {
                "split": split,
                "file_stem": row["file_stem"],
                "packet_index": int(float(row["packet_index"])),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": int(predicted_label == true_label),
                "top_probability": float(probability[ordered[0]]),
                "second_label": second_label,
                "second_probability": float(probability[ordered[1]]),
                "probability_margin": float(probability[ordered[0]] - probability[ordered[1]]),
                "prediction_consistent": int(predicted_label == top_label),
            }
        )
    return output


def load_external_predictions(path: Path, predicted_column: str) -> dict[tuple[str, int], str]:
    output = {}
    for row in read_csv(path):
        file_value = row.get("file_name") or row.get("file_stem")
        output[(Path(file_value).stem, int(float(row["packet_index"])))] = row[predicted_column]
    return output


def compare_predictions(rows: list[dict], external: dict[tuple[str, int], str], name: str) -> dict:
    w2r = r2w = changed = external_correct = 0
    for row in rows:
        key = (row["file_stem"], int(row["packet_index"]))
        external_label = external[key]
        true_label = row["true_label"]
        external_ok = external_label == true_label
        final_ok = bool(row["correct"])
        external_correct += int(external_ok)
        changed += int(external_label != row["predicted_label"])
        w2r += int(not external_ok and final_ok)
        r2w += int(external_ok and not final_ok)
    return {
        "baseline": name,
        "baseline_correct": external_correct,
        "challenger_correct": sum(row["correct"] for row in rows),
        "changed": changed,
        "W2R": w2r,
        "R2W": r2w,
        "net_gain": w2r - r2w,
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(w2r, r2w),
    }


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
    checksum_path = output_dir / "RESULT_CHECKSUMS.sha256"
    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path != checksum_path:
            lines.append(f"{sha256(path)}  {path.name}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_table = load_validation_table(args.validation_features)
    validation_train = [row for row in validation_table if row["split"] == "train"]
    validation_eval = [row for row in validation_table if row["split"] == "val"]
    val_fitted, class_order = fit_models(validation_train, validation_eval, args.seed)
    val_truth = list(labels(validation_eval))
    selection_rows, selected = validation_grid(val_fitted, class_order, val_truth)
    selected_name = selected["method"]
    selected_components = selected["components"].split(";")
    val_probabilities = selected.pop("probabilities")
    val_predicted = predicted_labels(val_probabilities, class_order)
    val_rows = prediction_rows(
        "val", validation_eval, val_truth, val_predicted, val_probabilities, class_order
    )
    write_csv(args.output_dir / "validation_model_selection.csv", selection_rows)
    write_csv(args.output_dir / "validation_predictions.csv", val_rows)
    print(f"selected {selected_name}: {selected['correct']}/128", flush=True)

    refit_table = load_refit_table(args.refit_rssi, args.refit_raw, args.refit_split)
    refit_train = [row for row in refit_table if row["split"] == "train"]
    formal_test = [row for row in refit_table if row["split"] == "test"]
    formal_fitted, formal_class_order = fit_models(refit_train, formal_test, args.seed)
    if formal_class_order != class_order:
        raise RuntimeError("Validation and formal class orders differ")
    formal_probabilities = (
        sum(formal_fitted[component]["probabilities"] for component in selected_components)
        / len(selected_components)
    )
    test_truth = list(labels(formal_test))
    test_predicted = predicted_labels(formal_probabilities, formal_class_order)
    test_rows = prediction_rows(
        "test", formal_test, test_truth, test_predicted, formal_probabilities, formal_class_order
    )
    write_csv(args.output_dir / "test_predictions.csv", test_rows)
    joblib.dump(
        {component: formal_fitted[component]["model"] for component in selected_components},
        args.output_dir / "selected_models.joblib",
    )

    test_correct = sum(row["correct"] for row in test_rows)
    aco = load_external_predictions(args.aco_test / "test_predictions.csv", "final_label")
    reranker = load_external_predictions(args.candidate_rerank / "selected_test_predictions.csv", "final_label")
    comparisons = [
        compare_predictions(test_rows, aco, "source_level_aco"),
        compare_predictions(test_rows, reranker, "candidate_rerank"),
    ]
    write_csv(args.output_dir / "test_comparisons.csv", comparisons)
    payload = {
        "status": "PASS",
        "protocol": {
            "dataset": "ExpandedReal-649-v1",
            "source_split_seed": 20260626,
            "validation_source_counts": {"train": 393, "validation": 128},
            "formal_source_counts": {"train_refit": 521, "test": 128},
            "training_rows": {"validation_stage": 3930, "formal_refit": 5210},
            "validation_and_test_augmentation": False,
            "source_overlap": 0,
            "selection": "model family and ensemble selected on validation only",
            "test_status": "exploratory because the same test split was inspected by earlier experiments",
        },
        "feature_columns": FEATURE_COLUMNS,
        "validation_selection": selected,
        "formal_test": {
            "method": selected_name,
            "components": selected_components,
            "packet_count": 128,
            "correct": test_correct,
            "accuracy": test_correct / 128,
            "target_correct_for_90_54_percent": 116,
            "target_met": test_correct >= 116,
            "wilson95": wilson_interval(test_correct, 128),
        },
        "comparisons": comparisons,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "input_sha256": {
            "validation_features": sha256(args.validation_features),
            "refit_rssi": sha256(args.refit_rssi),
            "refit_raw": sha256(args.refit_raw),
            "refit_split": sha256(args.refit_split),
        },
    }
    (args.output_dir / "supervised_ensemble_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    target_text = "met" if test_correct >= 116 else "not met"
    report = (
        "# Expanded-649 Supervised Ensemble Challenger\n\n"
        f"- Validation-selected model: `{selected_name}` ({selected['correct']}/128 = "
        f"{selected['accuracy']:.2%}).\n"
        f"- Formal test: {test_correct}/128 = {test_correct / 128:.2%}.\n"
        f"- Target corresponding to at least 90.54%: 116/128; target {target_text}.\n"
        f"- Versus source-level ACO: W2R={comparisons[0]['W2R']}, R2W={comparisons[0]['R2W']}, "
        f"McNemar p={comparisons[0]['mcnemar_exact_two_sided_p']:.6f}.\n\n"
        "The result is exploratory because the Expanded test split had already been inspected before "
        "this challenger was designed.\n"
    )
    (args.output_dir / "supervised_ensemble_report.md").write_text(report, encoding="utf-8")
    write_checksums(args.output_dir)
    print(json.dumps(payload["formal_test"], indent=2, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-features", type=Path, default=DEFAULT_VALIDATION_FEATURES)
    parser.add_argument("--refit-rssi", type=Path, default=DEFAULT_REFIT_RSSI)
    parser.add_argument("--refit-raw", type=Path, default=DEFAULT_REFIT_RAW)
    parser.add_argument("--refit-split", type=Path, default=DEFAULT_REFIT_SPLIT)
    parser.add_argument("--aco-val", type=Path, default=DEFAULT_ACO_VAL)
    parser.add_argument("--aco-test", type=Path, default=DEFAULT_ACO_TEST)
    parser.add_argument("--candidate-rerank", type=Path, default=DEFAULT_CANDIDATE_RERANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
