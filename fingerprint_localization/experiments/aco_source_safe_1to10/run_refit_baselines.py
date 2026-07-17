#!/usr/bin/env python3
"""Run fixed conventional baselines on the frozen train+val refit protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
MODEL_V3_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3"
if str(MODEL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_V3_DIR))

import pgar_heuristic as pgar  # noqa: E402


REFIT_DIR = EXPERIMENT_DIR / "group_safe_trainval_refit"
DEFAULT_RSSI_CSV = REFIT_DIR / "data" / "noisy_rssi_plus_packet_level_54points.csv"
DEFAULT_RAW_CSV = REFIT_DIR / "data" / "noisy_lora_frequency_s17_54points.csv"
DEFAULT_SPECTRUM_CSV = REFIT_DIR / "data" / "noisy_subbin_spectrum_long.csv"
DEFAULT_SPLIT_CSV = REFIT_DIR / "data" / "split_assignments.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "comparison_minimal_20260715" / "baselines"

RSSI_COLUMNS = list(pgar.RSSI_PLUS_COLUMNS)
RAW_COLUMNS = [f"preamble_fft_mag_bin_{offset:+d}" for offset in range(-8, 9)] + [
    "preamble_peak_to_residual_db",
    "detect_score_db",
    "s17_c_s",
    "s17_j_s",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0]) + sorted({key for row in rows for key in row} - set(rows[0]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def packet_key(row: dict) -> tuple[str, int]:
    return Path(row["file_name"]).stem, int(float(row["packet_index"]))


def split_key(row: dict) -> tuple[str, int]:
    return row["file_stem"], int(float(row["packet_index"]))


def number(row: dict, column: str) -> float:
    try:
        value = float(row.get(column, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_packet_table(args: argparse.Namespace) -> tuple[list[dict], list[str], list[str]]:
    rssi = {packet_key(row): row for row in read_csv(args.rssi_csv)}
    raw = {packet_key(row): row for row in read_csv(args.raw_csv)}
    assignments = {split_key(row): row["split"] for row in read_csv(args.split_csv)}
    rows = []
    for key in sorted(set(rssi) & set(raw)):
        if key not in assignments:
            continue
        rr, rw = rssi[key], raw[key]
        label = rr["position_key"]
        raw_label = rw.get("position_key") or f"{int(float(rw['corridor_id']))}_{int(float(rw['position_id']))}"
        if label != raw_label:
            raise ValueError(f"Label mismatch for {key}: {label} != {raw_label}")
        rows.append(
            {
                "key": key,
                "split": assignments[key],
                "label": label,
                "rssi": [number(rr, column) for column in RSSI_COLUMNS],
                "combined": [number(rr, column) for column in RSSI_COLUMNS]
                + [number(rw, column) for column in RAW_COLUMNS],
            }
        )
    counts = Counter(row["split"] for row in rows)
    if counts != Counter({"train": 2960, "val": 73, "test": 74}):
        raise RuntimeError(f"Unexpected aligned split counts: {dict(counts)}")
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    return rows, [row["label"] for row in train], [row["label"] for row in test]


def add_result(
    method: str,
    feature_set: str,
    truth: list[str],
    predicted: list[str],
    test_rows: list[dict],
    summaries: list[dict],
    predictions: list[dict],
) -> None:
    correct = sum(left == right for left, right in zip(truth, predicted))
    summaries.append(
        {
            "method": method,
            "feature_set": feature_set,
            "train_rows": 2960,
            "test_rows": len(truth),
            "correct": correct,
            "accuracy": accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        }
    )
    for row, true_label, predicted_label in zip(test_rows, truth, predicted):
        predictions.append(
            {
                "method": method,
                "file_stem": row["key"][0],
                "packet_index": row["key"][1],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": int(true_label == predicted_label),
            }
        )


def evaluate_pgar_refit(args: argparse.Namespace) -> tuple[list[str], list[dict]]:
    rssi_rows = read_csv(args.rssi_csv)
    raw_rows = read_csv(args.raw_csv)
    q4_packets, _offsets = pgar.read_q4_curves(args.spectrum_csv)
    samples = pgar.align_samples(
        pgar.rssi_rows_to_packets(rssi_rows),
        pgar.raw_rows_to_packets(raw_rows),
        q4_packets,
    )
    split_map = {split_key(row): row["split"] for row in read_csv(args.split_csv)}
    train_indices = [idx for idx, sample in enumerate(samples) if split_map.get((Path(sample.file_name).stem, sample.packet_index)) == "train"]
    test_indices = [idx for idx, sample in enumerate(samples) if split_map.get((Path(sample.file_name).stem, sample.packet_index)) == "test"]
    labels = [sample.label for sample in samples]
    rssi = [sample.rssi_plus for sample in samples]
    structures = [sample.structure for sample in samples]
    q4_curves = [sample.q4_curve for sample in samples]
    train_samples = [samples[idx] for idx in train_indices]
    cfg = SimpleNamespace(
        peak_threshold=None,
        peak_iqr_threshold=None,
        auto_peak_quantile=0.10,
        auto_peak_iqr_quantile=0.75,
        q4_stability_threshold=None,
        auto_q4_stability_quantile=0.75,
    )
    peak_threshold, peak_iqr_threshold = pgar.resolve_auto_thresholds(train_samples, cfg)
    q4_stability_threshold = pgar.resolve_q4_stability_threshold(train_samples, cfg)
    prototypes = pgar.build_prototypes(structures, labels, train_indices)
    q4_prototypes = pgar.build_q4_prototypes(q4_curves, labels, train_indices)
    predicted, detail = [], []
    for idx in test_indices:
        sample = samples[idx]
        ranked = pgar.class_rank(rssi, labels, train_indices, idx, 3)
        candidates = [label for label, _score in ranked[:3]]
        choice = candidates[0]
        margin = ranked[1][1] - ranked[0][1] if len(ranked) > 1 else float("inf")
        use_raw = margin <= 0.2 and sample.peak_mean > peak_threshold and sample.peak_iqr < peak_iqr_threshold
        q4_used = False
        if use_raw:
            weights = pgar.candidate_weights(prototypes, candidates)
            rssi_norm = pgar.normalize_candidate_scores(ranked, candidates)
            scored = sorted(
                (
                    label,
                    rssi_norm.get(label, 0.0) + pgar.structure_distance(sample.structure, prototypes[label], weights),
                )
                for label in candidates
                if label in prototypes
            )
            scored.sort(key=lambda item: (item[1], pgar.natural_label_key(item[0])))
            if scored:
                choice = scored[0][0]
                raw_margin = scored[1][1] - scored[0][1] if len(scored) > 1 else float("inf")
                if (
                    len(scored) > 1
                    and raw_margin <= 0.2
                    and sample.q4_curve is not None
                    and sample.q4_stability < q4_stability_threshold
                    and pgar.q4_discriminability(scored[0][0], scored[1][0], q4_prototypes) > 0.5
                ):
                    q4_scores = [
                        (label, pgar.q4_distance(sample.q4_curve, q4_prototypes[label]))
                        for label in candidates
                        if label in q4_prototypes
                    ]
                    q4_norm = pgar.normalize_candidate_scores(q4_scores, candidates)
                    rescored = sorted(
                        ((label, score + 0.25 * q4_norm.get(label, 0.0)) for label, score in scored),
                        key=lambda item: (item[1], pgar.natural_label_key(item[0])),
                    )
                    choice = rescored[0][0]
                    q4_used = True
        predicted.append(choice)
        detail.append({"raw_gate_used": int(use_raw), "q4_gate_used": int(q4_used)})
    return predicted, detail


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, y_train, y_test = load_packet_table(args)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    x_rssi_train = np.asarray([row["rssi"] for row in train_rows], dtype=float)
    x_rssi_test = np.asarray([row["rssi"] for row in test_rows], dtype=float)
    x_train = np.asarray([row["combined"] for row in train_rows], dtype=float)
    x_test = np.asarray([row["combined"] for row in test_rows], dtype=float)
    summaries: list[dict] = []
    predictions: list[dict] = []

    models = [
        ("1-NN", "RSSI+ (6)", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)), x_rssi_train, x_rssi_test),
        ("KNN (k=3)", "RSSI+ (6)", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)), x_rssi_train, x_rssi_test),
    ]
    for name, feature_set, model, train_x, test_x in models:
        model.fit(train_x, y_train)
        add_result(name, feature_set, y_test, list(model.predict(test_x)), test_rows, summaries, predictions)

    encoder = LabelEncoder().fit(y_train)
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=1,
    )
    rf_model.fit(x_train, encoder.transform(y_train))
    rf_pred = list(encoder.inverse_transform(rf_model.predict(x_test).astype(int)))
    add_result("Random Forest", "RSSI+raw packet (27)", y_test, rf_pred, test_rows, summaries, predictions)

    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-4, max_iter=500, random_state=args.seed),
    )
    mlp.fit(x_train, y_train)
    add_result("MLP (128,64)", "RSSI+raw packet (27)", y_test, list(mlp.predict(x_test)), test_rows, summaries, predictions)

    pgar_pred, pgar_detail = evaluate_pgar_refit(args)
    add_result("PGAR", "RSSI+raw+gated q4", y_test, pgar_pred, test_rows, summaries, predictions)
    for row, details in zip([row for row in predictions if row["method"] == "PGAR"], pgar_detail):
        row.update(details)

    write_csv(args.output_dir / "baseline_summary.csv", summaries)
    write_csv(args.output_dir / "baseline_test_predictions.csv", predictions)
    metadata = {
        "protocol": "source-safe train+val refit; frozen original test",
        "seed": args.seed,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "features": {"rssi": RSSI_COLUMNS, "combined": RSSI_COLUMNS + RAW_COLUMNS},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "tree_baseline": "RandomForestClassifier",
        },
        "input_sha256": {
            "rssi": sha256(args.rssi_csv),
            "raw": sha256(args.raw_csv),
            "spectrum": sha256(args.spectrum_csv),
            "split": sha256(args.split_csv),
        },
        "summary": summaries,
    }
    (args.output_dir / "baseline_metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
