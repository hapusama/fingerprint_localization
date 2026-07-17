#!/usr/bin/env python3
"""Audit candidate recall and stress the adopted Expanded LDA/ACO mainline.

The script deliberately keeps all model fitting and gate selection on the
existing train/validation protocol.  The formal test is used only for the
reported audit.  Raw IQ is unavailable in this repository, so the four active
stressors are deterministic feature-space proxies applied jointly to the LDA
packet features and the ACO segment representation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import joblib
import numpy as np

import finalize_expanded_aco_ml_score4 as finalizer
import run_aco_v4_source_level_on_split as source
import run_expanded_aco_ml_prior as prior
import run_expanded_lda_aco_mainline as mainline
import run_expanded_supervised_ensemble as supervised


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
MAINLINE_ROOT = (
    PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "aco_lda_only_mainline"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "candidate_recall_and_controlled_weakness_20260716"
)
LOCATION_CSV = PROJECT_DIR / "docs" / "location_distance_54points.csv"
ALPHA = mainline.ALPHA
BETA = mainline.BETA
SEVERE_ERROR_M = 10.0
SEED = 20260626
METHODS = ("LDA", "Fixed fusion", "Gated ACO")
PERTURBATIONS = {
    "preamble_missing": [1.0, 2.0, 4.0],
    "amplitude_noise": [0.05, 0.1, 0.25, 0.5, 1.0],
    "cfo_shift": [0.25, 0.5, 1.0],
    "segment_anomaly": [0.25, 0.5, 1.0],
}


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


def key_from_row(row: dict) -> tuple[str, int]:
    stem = row.get("file_stem") or Path(row["file_name"]).stem
    return stem, int(float(row["packet_index"]))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def natural_max(scores: dict[str, float]) -> str:
    return max(
        scores,
        key=lambda label: (
            scores[label],
            tuple(-value for value in supervised.natural_label_key(label)),
        ),
    )


def score_margin(scores: dict[str, float]) -> float:
    ordered = sorted(scores.values(), reverse=True)
    return float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0


@dataclass
class Context:
    name: str
    split_name: str
    train_rows: list[dict]
    eval_rows: list[dict]
    models: dict
    probabilities: dict
    args: object
    samples: list
    split_indices: dict
    q4_offsets: list[float]
    chirp_shapes: dict
    chirp_struct: dict
    templates: dict
    prototypes: dict
    rssi_ranker: Callable
    mainline_output: Path


def load_context(name: str, build_training_state: bool) -> Context:
    if name == "validation":
        table = supervised.load_validation_table(supervised.DEFAULT_VALIDATION_FEATURES)
        train_rows = [row for row in table if row["split"] == "train"]
        eval_rows = [row for row in table if row["split"] == "val"]
        models = joblib.load(MAINLINE_ROOT / "validation_lda_model.joblib")
        mainline_output = MAINLINE_ROOT / "validation"
        rssi_csv = supervised.SOURCE_SAFE_DIR / "data" / "noisy_rssi_plus_packet_level_32points_649.csv"
        spectrum_csv = supervised.SOURCE_SAFE_DIR / "data" / "noisy_subbin_spectrum_long_32points_649.csv"
        split_csv = supervised.SOURCE_SAFE_DIR / "data" / "split_assignments.csv"
        split_name = "val"
        t_seg = mainline.VALIDATION_T_SEG
    elif name == "formal_test":
        table = supervised.load_refit_table(
            supervised.DEFAULT_REFIT_RSSI,
            supervised.DEFAULT_REFIT_RAW,
            supervised.DEFAULT_REFIT_SPLIT,
        )
        train_rows = [row for row in table if row["split"] == "train"]
        eval_rows = [row for row in table if row["split"] == "test"]
        models = joblib.load(MAINLINE_ROOT / "formal_lda_model.joblib")
        mainline_output = MAINLINE_ROOT / "formal_test"
        rssi_csv = supervised.DEFAULT_REFIT_RSSI
        spectrum_csv = (
            supervised.REFIT_DIR / "data" / "noisy_subbin_spectrum_long_32points_649.csv"
        )
        split_csv = supervised.DEFAULT_REFIT_SPLIT
        split_name = "test"
        t_seg = mainline.FORMAL_T_SEG
    else:
        raise ValueError(name)

    stage_args = prior.prepare_stage_args(
        prior.default_runner_args(),
        rssi_csv,
        spectrum_csv,
        split_csv,
        mainline_output,
        split_name,
        t_seg,
    )
    (
        aco_args,
        samples,
        split_indices,
        q4_offsets,
        chirp_shapes,
        chirp_struct,
        _chirp_metadata,
    ) = source.load_samples(stage_args)
    labels = [sample.label for sample in samples]
    if build_training_state:
        templates = source.source_level_build_templates(
            samples,
            labels,
            split_indices["train"],
            chirp_shapes,
            chirp_struct,
            aco_args,
        )
        prototypes = source.source_level_build_prototypes(
            samples,
            labels,
            split_indices["train"],
        )
    else:
        templates = {}
        prototypes = {}
    probabilities, _classes = finalizer.common_probabilities(models, eval_rows)
    return Context(
        name=name,
        split_name=split_name,
        train_rows=train_rows,
        eval_rows=eval_rows,
        models=models,
        probabilities=probabilities,
        args=aco_args,
        samples=samples,
        split_indices=split_indices,
        q4_offsets=list(q4_offsets),
        chirp_shapes=chirp_shapes,
        chirp_struct=chirp_struct,
        templates=templates,
        prototypes=prototypes,
        rssi_ranker=source.make_source_level_class_rank(samples),
        mainline_output=mainline_output,
    )


def rank_maps(context: Context) -> tuple[dict, dict, dict]:
    rssi_rows = [sample.rssi_plus for sample in context.samples]
    labels = [sample.label for sample in context.samples]
    rssi = {}
    fusion = {}
    fused_ranker = prior.make_ml_source_class_rank(context.samples, context.probabilities, ALPHA)
    for sample_index in context.split_indices[context.split_name]:
        sample_key = source.source_id(context.samples[sample_index])
        rssi[sample_key] = [
            label
            for label, _score in context.rssi_ranker(
                rssi_rows,
                labels,
                context.split_indices["train"],
                sample_index,
                context.args.rssi_class_k,
            )
        ]
        fusion[sample_key] = [
            label
            for label, _score in fused_ranker(
                rssi_rows,
                labels,
                context.split_indices["train"],
                sample_index,
                context.args.rssi_class_k,
            )
        ]
    lda = {
        sample_key: sorted(values, key=values.get, reverse=True)
        for sample_key, values in context.probabilities.items()
    }
    return rssi, lda, fusion


def candidate_audit(context: Context) -> tuple[dict, list[dict]]:
    rssi, lda, fusion = rank_maps(context)
    rows = []
    for eval_row in context.eval_rows:
        sample_key = key_from_row(eval_row)
        true_label = eval_row["position_key"]
        rssi_top5 = rssi[sample_key][:5]
        lda_top5 = lda[sample_key][:5]
        fusion_top5 = fusion[sample_key][:5]
        union = lda_top5 + [label for label in rssi_top5 if label not in lda_top5]
        rows.append(
            {
                "split": context.name,
                "file_name": f"{sample_key[0]}.bin",
                "packet_index": sample_key[1],
                "true_label": true_label,
                "rssi_rank": rssi[sample_key].index(true_label) + 1,
                "lda_rank": lda[sample_key].index(true_label) + 1,
                "fusion_rank": fusion[sample_key].index(true_label) + 1,
                "rssi_top5": ";".join(rssi_top5),
                "lda_top5": ";".join(lda_top5),
                "fusion_top5": ";".join(fusion_top5),
                "union_candidates": ";".join(union),
                "union_length": len(union),
                "true_in_fusion_top5": int(true_label in fusion_top5),
                "true_in_union": int(true_label in union),
            }
        )
    n = len(rows)
    summary = {
        "split": context.name,
        "packets": n,
        "rssi_top1_recall": sum(row["rssi_rank"] <= 1 for row in rows) / n,
        "rssi_top3_recall": sum(row["rssi_rank"] <= 3 for row in rows) / n,
        "rssi_top5_recall": sum(row["rssi_rank"] <= 5 for row in rows) / n,
        "lda_top1_recall": sum(row["lda_rank"] <= 1 for row in rows) / n,
        "lda_top3_recall": sum(row["lda_rank"] <= 3 for row in rows) / n,
        "lda_top5_recall": sum(row["lda_rank"] <= 5 for row in rows) / n,
        "fusion_top5_recall": sum(row["true_in_fusion_top5"] for row in rows) / n,
        "union_variable_recall": sum(row["true_in_union"] for row in rows) / n,
        "union_length_mean": sum(row["union_length"] for row in rows) / n,
        "union_length_min": min(row["union_length"] for row in rows),
        "union_length_max": max(row["union_length"] for row in rows),
        "candidate_truncation_unrecoverable_errors": sum(
            not row["true_in_fusion_top5"] for row in rows
        ),
    }
    for name in ["rssi", "lda"]:
        for k in [1, 3, 5]:
            summary[f"{name}_top{k}_count"] = sum(row[f"{name}_rank"] <= k for row in rows)
    summary["fusion_top5_count"] = sum(row["true_in_fusion_top5"] for row in rows)
    summary["union_variable_count"] = sum(row["true_in_union"] for row in rows)
    return summary, rows


def load_mainline_groups(context: Context) -> tuple[list[dict], list[dict]]:
    prediction_path = context.mainline_output / f"{context.split_name}_predictions.csv"
    candidate_path = context.mainline_output / f"{context.split_name}_candidate_scores.csv"
    groups = finalizer.load_groups(prediction_path, candidate_path, context.probabilities)
    _metrics, final_rows = finalizer.evaluate(groups, BETA, context.split_name)
    return groups, final_rows


def calibrate_gate(validation: Context) -> tuple[float, list[dict], dict]:
    groups, final_rows = load_mainline_groups(validation)
    candidates = {1.0000001}
    records = []
    for group, final_row in zip(groups, final_rows):
        q_seg = float(group["prediction"]["Q_seg"])
        if final_row["final_label"] != final_row["ml_prior_top1_label"]:
            candidates.add(q_seg)
        records.append((group, final_row, q_seg))
    trials = []
    for threshold in sorted(candidates, reverse=True):
        correct = beneficial = harmful = changed_wrong = changed = 0
        for _group, row, q_seg in records:
            lda_label = row["ml_prior_top1_label"]
            fixed_label = row["final_label"]
            true_label = row["true_label"]
            gated_label = fixed_label if fixed_label != lda_label and q_seg >= threshold else lda_label
            correct += int(gated_label == true_label)
            if gated_label != lda_label:
                changed += 1
                beneficial += int(lda_label != true_label and gated_label == true_label)
                harmful += int(lda_label == true_label and gated_label != true_label)
                changed_wrong += int(lda_label != true_label and gated_label != true_label)
        precision = beneficial / changed if changed else 1.0
        trials.append(
            {
                "q_seg_threshold": threshold,
                "correct": correct,
                "accuracy": correct / len(records),
                "accepted_corrections": changed,
                "beneficial_corrections": beneficial,
                "harmful_corrections": harmful,
                "changed_but_still_wrong": changed_wrong,
                "correction_precision": precision,
            }
        )
    selected = max(
        trials,
        key=lambda row: (
            row["correct"],
            row["correction_precision"],
            -row["accepted_corrections"],
            row["q_seg_threshold"],
        ),
    )
    return float(selected["q_seg_threshold"]), trials, selected


def vector_stats(rows: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(rows, dtype=float)
    return np.median(values, axis=0), np.mean(values, axis=0), np.std(values, axis=0) + 1e-9


def training_stats(context: Context) -> dict:
    feature_values = np.asarray(
        [[float(row[column]) for column in supervised.FEATURE_COLUMNS] for row in context.train_rows],
        dtype=float,
    )
    train_samples = [context.samples[index] for index in context.split_indices["train"]]
    shape_rows = [shape for sample in train_samples for shape in sample.segment_shapes]
    zw_rows = [zw for sample in train_samples for zw in sample.segment_zw]
    q4_rows = [curve for sample in train_samples for curve in sample.segment_q4_curves]
    rssi_rows = [sample.rssi_plus for sample in train_samples]
    feature_median = np.median(feature_values, axis=0)
    feature_std = np.std(feature_values, axis=0) + 1e-9
    return {
        "feature_median": dict(zip(supervised.FEATURE_COLUMNS, feature_median)),
        "feature_std": dict(zip(supervised.FEATURE_COLUMNS, feature_std)),
        "shape": vector_stats(shape_rows),
        "zw": vector_stats(zw_rows),
        "q4": vector_stats(q4_rows),
        "rssi": vector_stats(rssi_rows),
    }


def blend(left: Sequence[float], right: Sequence[float], weight: float) -> list[float]:
    return [
        (1.0 - weight) * float(a) + weight * float(b)
        for a, b in zip(left, right)
    ]


def centered(values: Sequence[float]) -> list[float]:
    mean = sum(values) / len(values)
    return [float(value - mean) for value in values]


def shift_vector(values: Sequence[float], bins: float, fill: float) -> list[float]:
    array = np.asarray(values, dtype=float)
    x = np.arange(len(array), dtype=float)
    shifted = np.interp(x - bins, x, array, left=fill, right=fill)
    return [float(value) for value in shifted]


def perturb_missing(sample, row: dict, strength: float, stats: dict, _donor, _donor_row, _rng):
    missing = int(round(strength))
    out = copy.deepcopy(sample)
    changed = dict(row)
    fraction_segment = min(1.0, missing / 4.0)
    out.segment_shapes[0] = centered(
        blend(out.segment_shapes[0], stats["shape"][0], fraction_segment)
    )
    out.segment_zw[0] = blend(out.segment_zw[0], stats["zw"][0], fraction_segment)
    out.segment_q4_curves[0] = blend(
        out.segment_q4_curves[0], stats["q4"][0], fraction_segment
    )
    if missing >= 2:
        out.segment_q4_reliable[0] = False
    fraction_packet = min(1.0, missing / 8.0)
    for column in [f"preamble_fft_mag_bin_{offset:+d}" for offset in range(-8, 9)]:
        changed[column] = (
            (1.0 - fraction_packet) * float(changed[column])
            + fraction_packet * float(stats["feature_median"][column])
        )
    changed["preamble_peak_to_residual_db"] = float(changed["preamble_peak_to_residual_db"]) - missing
    changed["detect_score_db"] = float(changed["detect_score_db"]) - 0.75 * missing
    changed["s17_c_s"] = max(0.0, float(changed["s17_c_s"]) * (1.0 - 0.05 * missing))
    return out, changed


def perturb_noise(sample, row: dict, strength: float, stats: dict, _donor, _donor_row, rng):
    out = copy.deepcopy(sample)
    changed = dict(row)
    for column in supervised.FEATURE_COLUMNS:
        scale = float(stats["feature_std"][column])
        changed[column] = float(changed[column]) + float(rng.normal(0.0, strength * scale))
    for column in [f"preamble_fft_mag_bin_{offset:+d}" for offset in range(-8, 9)]:
        changed[column] = max(1e-12, float(changed[column]))
    changed["snr"] = float(changed["snr"]) - 4.0 * strength
    changed["detect_score_db"] = float(changed["detect_score_db"]) - 2.0 * strength
    changed["preamble_peak_to_residual_db"] = float(changed["preamble_peak_to_residual_db"]) - 2.0 * strength
    rssi_std = stats["rssi"][2]
    out.rssi_plus = [
        float(value + rng.normal(0.0, strength * rssi_std[index]))
        for index, value in enumerate(out.rssi_plus)
    ]
    for segment in range(len(out.segment_shapes)):
        out.segment_shapes[segment] = centered(
            [
                float(value + rng.normal(0.0, strength * stats["shape"][2][index]))
                for index, value in enumerate(out.segment_shapes[segment])
            ]
        )
        out.segment_zw[segment] = [
            float(value + rng.normal(0.0, strength * stats["zw"][2][index]))
            for index, value in enumerate(out.segment_zw[segment])
        ]
        out.segment_q4_curves[segment] = [
            float(value + rng.normal(0.0, strength * stats["q4"][2][index]))
            for index, value in enumerate(out.segment_q4_curves[segment])
        ]
    return out, changed


def perturb_cfo(sample, row: dict, strength: float, _stats: dict, _donor, _donor_row, _rng):
    out = copy.deepcopy(sample)
    changed = dict(row)
    mag_columns = [f"preamble_fft_mag_bin_{offset:+d}" for offset in range(-8, 9)]
    shifted = shift_vector([float(changed[column]) for column in mag_columns], strength, 1e-12)
    for column, value in zip(mag_columns, shifted):
        changed[column] = max(1e-12, value)
    q4_step = 0.25
    for segment in range(len(out.segment_shapes)):
        shape = out.segment_shapes[segment]
        out.segment_shapes[segment] = centered(
            shift_vector(shape, strength, min(shape))
        )
        out.segment_q4_curves[segment] = shift_vector(
            out.segment_q4_curves[segment], strength / q4_step, 0.0
        )
        out.segment_q4_peak_offsets[segment] += strength
    return out, changed


def perturb_segment(sample, row: dict, strength: float, _stats: dict, donor, donor_row, _rng):
    out = copy.deepcopy(sample)
    changed = dict(row)
    segment = stable_seed(sample.file_name, sample.packet_index, "segment") % len(out.segment_shapes)
    out.segment_shapes[segment] = centered(
        blend(out.segment_shapes[segment], donor.segment_shapes[segment], strength)
    )
    out.segment_zw[segment] = blend(out.segment_zw[segment], donor.segment_zw[segment], strength)
    out.segment_q4_curves[segment] = blend(
        out.segment_q4_curves[segment], donor.segment_q4_curves[segment], strength
    )
    out.segment_q4_peak_offsets[segment] = (
        (1.0 - strength) * out.segment_q4_peak_offsets[segment]
        + strength * donor.segment_q4_peak_offsets[segment]
    )
    out.segment_q4_peak_to_side_db[segment] = (
        (1.0 - strength) * out.segment_q4_peak_to_side_db[segment]
        + strength * donor.segment_q4_peak_to_side_db[segment]
    )
    if strength >= 0.5:
        out.segment_q4_reliable[segment] = donor.segment_q4_reliable[segment]
    packet_weight = strength / len(out.segment_shapes)
    for column in supervised.RAW_COLUMNS:
        changed[column] = (
            (1.0 - packet_weight) * float(changed[column])
            + packet_weight * float(donor_row[column])
        )
    return out, changed


PERTURB_FUNCTIONS = {
    "preamble_missing": perturb_missing,
    "amplitude_noise": perturb_noise,
    "cfo_shift": perturb_cfo,
    "segment_anomaly": perturb_segment,
}


def donor_maps(context: Context, distances: dict[str, float]) -> tuple[dict, dict]:
    rows = {key_from_row(row): row for row in context.eval_rows}
    samples = {
        source.source_id(context.samples[index]): context.samples[index]
        for index in context.split_indices[context.split_name]
    }
    donor_sample = {}
    donor_row = {}
    for sample_key, sample in samples.items():
        candidates = [
            (abs(distances[other.label] - distances[sample.label]), other_key, other)
            for other_key, other in samples.items()
            if other.label != sample.label
        ]
        _distance, selected_key, selected = max(candidates, key=lambda item: (item[0], item[1]))
        donor_sample[sample_key] = selected
        donor_row[sample_key] = rows[selected_key]
    return donor_sample, donor_row


def evaluate_scenario(
    context: Context,
    gate_threshold: float,
    stats: dict,
    distances: dict[str, float],
    condition: str,
    strength: float,
) -> list[dict]:
    eval_by_key = {key_from_row(row): dict(row) for row in context.eval_rows}
    perturbed_samples = list(context.samples)
    donor_sample, donor_row = donor_maps(context, distances)
    if condition != "clean":
        perturb = PERTURB_FUNCTIONS[condition]
        for sample_index in context.split_indices[context.split_name]:
            sample = context.samples[sample_index]
            sample_key = source.source_id(sample)
            rng = np.random.default_rng(stable_seed(SEED, condition, strength, *sample_key))
            changed_sample, changed_row = perturb(
                sample,
                eval_by_key[sample_key],
                strength,
                stats,
                donor_sample[sample_key],
                donor_row[sample_key],
                rng,
            )
            perturbed_samples[sample_index] = changed_sample
            eval_by_key[sample_key] = changed_row
    eval_rows = [eval_by_key[key_from_row(row)] for row in context.eval_rows]
    probabilities, _classes = finalizer.common_probabilities(context.models, eval_rows)
    ranker = prior.make_ml_source_class_rank(perturbed_samples, probabilities, ALPHA)
    labels = [sample.label for sample in perturbed_samples]
    rssi_rows = [sample.rssi_plus for sample in perturbed_samples]
    aco_rng = random.Random(context.args.seed)
    outputs = []
    diagnostics = {key_from_row(row): row for row in eval_rows}
    for sample_index in context.split_indices[context.split_name]:
        sample = perturbed_samples[sample_index]
        sample_key = source.source_id(sample)
        ranked = ranker(
            rssi_rows,
            labels,
            context.split_indices["train"],
            sample_index,
            context.args.rssi_class_k,
        )
        candidates = [label for label, _score in ranked[: context.args.top_k]]
        rssi_costs = {label: score for label, score in ranked if label in candidates}
        obs_costs, _segment_rows, meta = source.aco4.build_observation_costs_v4(
            sample,
            candidates,
            rssi_costs,
            context.templates,
            context.prototypes,
            context.q4_offsets,
            context.args,
        )
        result = source.aco4.run_aco_v4_for_packet(
            obs_costs,
            candidates,
            context.templates,
            meta,
            context.args,
            aco_rng,
        )
        score4_norm = finalizer.minmax(result["score4"])
        lda_candidate_norm = finalizer.minmax(
            {label: probabilities[sample_key][label] for label in candidates}
        )
        combined = {
            label: (1.0 - BETA) * score4_norm[label] + BETA * lda_candidate_norm[label]
            for label in candidates
        }
        fixed_label = natural_max(combined)
        lda_label = natural_max(probabilities[sample_key])
        gated_label = (
            fixed_label
            if fixed_label != lda_label and float(meta["q_seg"]) >= gate_threshold
            else lda_label
        )
        gated_confidence = (
            score_margin(combined) if gated_label == fixed_label and fixed_label != lda_label
            else score_margin(probabilities[sample_key])
        )
        row = diagnostics[sample_key]
        outputs.append(
            {
                "scenario_type": "artificial_perturbation",
                "condition": condition,
                "strength": strength,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "candidate_labels": ";".join(candidates),
                "true_in_candidates": int(sample.label in candidates),
                "lda_label": lda_label,
                "fixed_label": fixed_label,
                "gated_label": gated_label,
                "lda_confidence": score_margin(probabilities[sample_key]),
                "fixed_confidence": score_margin(combined),
                "gated_confidence": gated_confidence,
                "lda_top1_probability": probabilities[sample_key][lda_label],
                "Q_seg": float(meta["q_seg"]),
                "segment_cost_std": float(meta["segment_cost_std"]),
                "detect_score_db": float(row["detect_score_db"]),
                "snr": float(row["snr"]),
            }
        )
    return outputs


def metric_row(
    rows: Sequence[dict],
    method: str,
    distances: dict[str, float],
    scenario_type: str,
    condition: str,
    severity: str,
    strength: object,
) -> dict:
    label_column = {
        "LDA": "lda_label",
        "Fixed fusion": "fixed_label",
        "Gated ACO": "gated_label",
    }[method]
    errors = [
        abs(distances[row[label_column]] - distances[row["true_label"]])
        for row in rows
    ]
    correct = sum(row[label_column] == row["true_label"] for row in rows)
    changed = beneficial = harmful = changed_wrong = 0
    if method != "LDA":
        for row in rows:
            if row[label_column] == row["lda_label"]:
                continue
            changed += 1
            lda_ok = row["lda_label"] == row["true_label"]
            method_ok = row[label_column] == row["true_label"]
            beneficial += int(not lda_ok and method_ok)
            harmful += int(lda_ok and not method_ok)
            changed_wrong += int(not lda_ok and not method_ok)
    return {
        "scenario_type": scenario_type,
        "condition": condition,
        "severity": severity,
        "strength": strength,
        "method": method,
        "packets": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "topology_mae_m": sum(errors) / len(errors),
        "topology_p95_m": percentile(errors, 0.95),
        "severe_error_threshold_m": SEVERE_ERROR_M,
        "severe_error_count": sum(error > SEVERE_ERROR_M for error in errors),
        "severe_error_rate": sum(error > SEVERE_ERROR_M for error in errors) / len(errors),
        "prediction_changes_vs_lda": changed,
        "correction_coverage": changed / len(rows),
        "beneficial_corrections": beneficial,
        "harmful_corrections": harmful,
        "changed_but_still_wrong": changed_wrong,
        "correction_precision": beneficial / changed if changed else "",
    }


def coverage_risk_rows(
    rows: Sequence[dict],
    method: str,
    distances: dict[str, float],
    scenario_type: str,
    condition: str,
    severity: str,
    strength: object,
) -> list[dict]:
    label_column = {
        "LDA": "lda_label",
        "Fixed fusion": "fixed_label",
        "Gated ACO": "gated_label",
    }[method]
    confidence_column = {
        "LDA": "lda_confidence",
        "Fixed fusion": "fixed_confidence",
        "Gated ACO": "gated_confidence",
    }[method]
    ordered = sorted(rows, key=lambda row: float(row[confidence_column]), reverse=True)
    output = []
    for coverage in [index / 10.0 for index in range(1, 11)]:
        count = max(1, int(math.ceil(coverage * len(ordered))))
        selected = ordered[:count]
        errors = [
            abs(distances[row[label_column]] - distances[row["true_label"]])
            for row in selected
        ]
        output.append(
            {
                "scenario_type": scenario_type,
                "condition": condition,
                "severity": severity,
                "strength": strength,
                "method": method,
                "target_coverage": coverage,
                "selected_packets": count,
                "actual_coverage": count / len(ordered),
                "classification_risk": sum(
                    row[label_column] != row["true_label"] for row in selected
                )
                / count,
                "topology_mae_m": sum(errors) / count,
                "severe_error_rate": sum(error > SEVERE_ERROR_M for error in errors) / count,
            }
        )
    return output


def diagnostic_thresholds(validation: Context) -> dict:
    prediction_rows = read_csv(validation.mainline_output / "val_predictions.csv")
    segment_by_key = {
        key_from_row(row): float(row["segment_cost_std"])
        for row in prediction_rows
    }
    eval_by_key = {key_from_row(row): row for row in validation.eval_rows}
    values = {
        "detect_score_db": [float(row["detect_score_db"]) for row in validation.eval_rows],
        "snr": [float(row["snr"]) for row in validation.eval_rows],
        "segment_cost_std": [segment_by_key[key] for key in eval_by_key],
    }
    return {
        name: {"q25": percentile(items, 0.25), "q50": percentile(items, 0.5), "q75": percentile(items, 0.75)}
        for name, items in values.items()
    }


def diagnostic_level(value: float, thresholds: dict, high_is_bad: bool) -> str:
    if high_is_bad:
        if value <= thresholds["q25"]:
            return "0_cleanest"
        if value <= thresholds["q50"]:
            return "1_mild"
        if value <= thresholds["q75"]:
            return "2_moderate"
        return "3_severe"
    if value > thresholds["q75"]:
        return "0_cleanest"
    if value > thresholds["q50"]:
        return "1_mild"
    if value > thresholds["q25"]:
        return "2_moderate"
    return "3_severe"


def add_diagnostic_results(
    clean_rows: Sequence[dict],
    thresholds: dict,
    distances: dict[str, float],
    metrics: list[dict],
    curves: list[dict],
) -> None:
    for condition in ["detect_score_db", "snr", "segment_cost_std"]:
        high_is_bad = condition == "segment_cost_std"
        groups = {level: [] for level in ["0_cleanest", "1_mild", "2_moderate", "3_severe"]}
        for row in clean_rows:
            level = diagnostic_level(float(row[condition]), thresholds[condition], high_is_bad)
            groups[level].append(row)
        for level, rows in groups.items():
            if not rows:
                continue
            for method in METHODS:
                metrics.append(
                    metric_row(
                        rows,
                        method,
                        distances,
                        "diagnostic_group",
                        condition,
                        level,
                        "validation_quartile",
                    )
                )
                curves.extend(
                    coverage_risk_rows(
                        rows,
                        method,
                        distances,
                        "diagnostic_group",
                        condition,
                        level,
                        "validation_quartile",
                    )
                )


def plot_results(metrics: Sequence[dict], curves: Sequence[dict], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    artificial = [row for row in metrics if row["scenario_type"] == "artificial_perturbation"]
    generated = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, condition in zip(axes.ravel(), PERTURBATIONS):
        rows = [row for row in artificial if row["condition"] == condition]
        strengths = sorted({float(row["strength"]) for row in rows})
        for method in METHODS:
            values = [
                next(float(row["accuracy"]) for row in rows if row["method"] == method and float(row["strength"]) == strength)
                for strength in strengths
            ]
            axis.plot(strengths, values, marker="o", label=method)
        axis.set_title(condition.replace("_", " "))
        axis.set_xlabel("degradation strength")
        axis.set_ylabel("accuracy")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    path = output_dir / "artificial_perturbation_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    selected_curves = []
    for condition, strengths in PERTURBATIONS.items():
        strongest = max(strengths)
        selected_curves.extend(
            row
            for row in curves
            if row["scenario_type"] == "artificial_perturbation"
            and row["condition"] == condition
            and float(row["strength"]) == strongest
        )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, condition in zip(axes.ravel(), PERTURBATIONS):
        rows = [row for row in selected_curves if row["condition"] == condition]
        for method in METHODS:
            method_rows = sorted(
                [row for row in rows if row["method"] == method],
                key=lambda row: float(row["actual_coverage"]),
            )
            axis.plot(
                [float(row["actual_coverage"]) for row in method_rows],
                [float(row["classification_risk"]) for row in method_rows],
                marker="o",
                label=method,
            )
        axis.set_title(f"{condition.replace('_', ' ')} (strongest)")
        axis.set_xlabel("coverage")
        axis.set_ylabel("classification risk")
        axis.set_xlim(0.0, 1.02)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    path = output_dir / "coverage_risk_strongest_perturbations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)

    diagnostic = [row for row in metrics if row["scenario_type"] == "diagnostic_group"]
    levels = ["0_cleanest", "1_mild", "2_moderate", "3_severe"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.28, wspace=0.25)
    for axis, condition in zip(axes, ["detect_score_db", "snr", "segment_cost_std"]):
        rows = [row for row in diagnostic if row["condition"] == condition]
        for method in METHODS:
            by_level = {
                row["severity"]: float(row["accuracy"])
                for row in rows
                if row["method"] == method
            }
            axis.plot(
                range(len(levels)),
                [by_level.get(level, float("nan")) for level in levels],
                marker="o",
                label=method,
            )
        axis.set_title(condition.replace("_", " "))
        axis.set_xticks(range(len(levels)), ["clean", "mild", "moderate", "severe"], rotation=20)
        axis.set_ylabel("accuracy")
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.03), ncol=3, frameon=False)
    path = output_dir / "diagnostic_group_accuracy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    generated.append(path.name)
    return generated


def render_report(
    output_dir: Path,
    candidate_summaries: Sequence[dict],
    gate_selected: dict,
    metrics: Sequence[dict],
    diagnostic_cuts: dict,
) -> None:
    formal = next(row for row in candidate_summaries if row["split"] == "formal_test")
    validation = next(row for row in candidate_summaries if row["split"] == "validation")
    strongest_rows = [
        row
        for row in metrics
        if row["scenario_type"] == "artificial_perturbation"
        and row["condition"] in PERTURBATIONS
        and float(row["strength"]) == max(PERTURBATIONS[row["condition"]])
    ]
    lines = [
        "# Expanded LDA/ACO candidate recall and controlled weakness audit",
        "",
        "Date: 2026-07-16",
        "",
        "## Candidate recall",
        "",
        "| Split | RSSI R@1/3/5 | LDA R@1/3/5 | fused R@5 | variable union recall | mean union length | unrecoverable truncations |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in [validation, formal]:
        lines.append(
            f"| {row['split']} | {row['rssi_top1_count']}/{row['rssi_top3_count']}/{row['rssi_top5_count']} "
            f"| {row['lda_top1_count']}/{row['lda_top3_count']}/{row['lda_top5_count']} "
            f"| {row['fusion_top5_count']}/{row['packets']} "
            f"| {row['union_variable_count']}/{row['packets']} "
            f"| {row['union_length_mean']:.2f} "
            f"| {row['candidate_truncation_unrecoverable_errors']} |"
        )
    lines.extend(
        [
            "",
            "The variable union is `LDA Top-5 union RSSI Top-5` (deduplicated, length 5--10).",
            "An unrecoverable truncation is a packet whose true label is absent from the fused Top-5.",
            "",
            "## Gated ACO calibration",
            "",
            f"The gate accepts a fixed-fusion correction over LDA only when `Q_seg >= {gate_selected['q_seg_threshold']:.6f}`. "
            f"This threshold was selected only on validation: {gate_selected['correct']}/128 correct, "
            f"{gate_selected['accepted_corrections']} accepted corrections, correction precision "
            f"{gate_selected['correction_precision']:.2%}.",
            "",
            "## Strongest artificial degradation",
            "",
            "| Condition | Method | Accuracy | topology MAE / P95 (m) | severe >10 m | correction precision |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for condition in PERTURBATIONS:
        for method in METHODS:
            row = next(
                item
                for item in strongest_rows
                if item["condition"] == condition and item["method"] == method
            )
            precision = row["correction_precision"]
            precision_text = "n/a" if precision == "" else f"{float(precision):.2%}"
            lines.append(
                f"| {condition} ({row['strength']}) | {method} | {float(row['accuracy']):.2%} "
                f"| {float(row['topology_mae_m']):.2f} / {float(row['topology_p95_m']):.2f} "
                f"| {float(row['severe_error_rate']):.2%} | {precision_text} |"
            )
    lines.extend(
        [
            "",
            "## Definitions and limitations",
            "",
            "- Topology error is `abs(distance_m(prediction) - distance_m(truth))`; it is not Euclidean error.",
            "- Severe error is fixed at topology error >10 m (about three sampling intervals).",
            "- Correction precision is beneficial LDA corrections divided by all predictions changed from LDA.",
            "- Coverage-risk selects the most confident 10%--100% of packets using each method's native margin.",
            "- Low detect score, low SNR and high segment-cost standard deviation use validation-quartile thresholds.",
            "- Preamble loss, amplitude noise, CFO shift and a single-segment anomaly are feature-space proxies. "
            "They do not replace a raw-IQ channel/noise injection study.",
            "",
            "Validation diagnostic cut points:",
            "",
            "```json",
            json.dumps(diagnostic_cuts, indent=2, ensure_ascii=False),
            "```",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python fingerprint_localization/experiments/aco_source_safe_1to10/run_candidate_recall_and_controlled_weakness.py",
            "```",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = load_context("validation", build_training_state=False)
    formal = load_context("formal_test", build_training_state=True)

    candidate_summaries = []
    candidate_rows = []
    for context in [validation, formal]:
        summary, rows = candidate_audit(context)
        candidate_summaries.append(summary)
        candidate_rows.extend(rows)
    write_csv(args.output_dir / "candidate_recall_summary.csv", candidate_summaries)
    write_csv(args.output_dir / "candidate_recall_per_packet.csv", candidate_rows)

    gate_threshold, gate_trials, gate_selected = calibrate_gate(validation)
    write_csv(args.output_dir / "gate_calibration_validation.csv", gate_trials)

    distances = {
        row["position_key"]: float(row["distance_m"])
        for row in read_csv(LOCATION_CSV)
    }
    stats = training_stats(formal)
    all_predictions = []
    clean_rows = evaluate_scenario(
        formal, gate_threshold, stats, distances, "clean", 0.0
    )
    all_predictions.extend(clean_rows)
    stored = {
        key_from_row(row): row["final_label"]
        for row in read_csv(formal.mainline_output / "fixed_beta_final_test_predictions.csv")
    }
    mismatch = [
        row for row in clean_rows if row["fixed_label"] != stored[key_from_row(row)]
    ]
    if mismatch:
        raise RuntimeError(f"Clean fixed-fusion reproduction mismatch: {len(mismatch)} packets")
    clean_counts = {
        "lda": sum(row["lda_label"] == row["true_label"] for row in clean_rows),
        "fixed": sum(row["fixed_label"] == row["true_label"] for row in clean_rows),
        "gated": sum(row["gated_label"] == row["true_label"] for row in clean_rows),
    }
    if clean_counts != {"lda": 120, "fixed": 120, "gated": 120}:
        raise RuntimeError(f"Unexpected clean result: {clean_counts}")

    for condition, strengths in PERTURBATIONS.items():
        for strength in strengths:
            all_predictions.extend(
                evaluate_scenario(
                    formal,
                    gate_threshold,
                    stats,
                    distances,
                    condition,
                    strength,
                )
            )
    write_csv(args.output_dir / "weakness_predictions.csv", all_predictions)

    metrics = []
    curves = []
    for condition in PERTURBATIONS:
        for strength in [0.0] + PERTURBATIONS[condition]:
            rows = clean_rows if strength == 0.0 else [
                row
                for row in all_predictions
                if row["condition"] == condition and float(row["strength"]) == strength
            ]
            severity = "clean" if strength == 0.0 else f"level_{PERTURBATIONS[condition].index(strength) + 1}"
            for method in METHODS:
                metrics.append(
                    metric_row(
                        rows,
                        method,
                        distances,
                        "artificial_perturbation",
                        condition,
                        severity,
                        strength,
                    )
                )
                curves.extend(
                    coverage_risk_rows(
                        rows,
                        method,
                        distances,
                        "artificial_perturbation",
                        condition,
                        severity,
                        strength,
                    )
                )
    thresholds = diagnostic_thresholds(validation)
    add_diagnostic_results(clean_rows, thresholds, distances, metrics, curves)
    write_csv(args.output_dir / "weakness_metrics.csv", metrics)
    write_csv(args.output_dir / "coverage_risk.csv", curves)
    plot_files = plot_results(metrics, curves, args.output_dir)

    render_report(args.output_dir, candidate_summaries, gate_selected, metrics, thresholds)
    manifest = {
        "experiment": "EXPANDED_LDA_ACO_CANDIDATE_RECALL_AND_CONTROLLED_WEAKNESS_20260716",
        "status": "complete",
        "seed": SEED,
        "protocol": {
            "gate_selected_on": "validation only",
            "formal_test_packets": 128,
            "alpha": ALPHA,
            "beta": BETA,
            "severe_error_threshold_m": SEVERE_ERROR_M,
            "test_status": "exploratory because Expanded formal test was previously inspected",
            "active_perturbations": PERTURBATIONS,
            "active_perturbation_scope": "deterministic feature-space proxy; raw IQ unavailable",
        },
        "candidate_recall": candidate_summaries,
        "gate": gate_selected,
        "clean_reproduction": clean_counts,
        "diagnostic_validation_quartiles": thresholds,
        "plots": plot_files,
        "input_sha256": {
            "formal_model": sha256(MAINLINE_ROOT / "formal_lda_model.joblib"),
            "validation_model": sha256(MAINLINE_ROOT / "validation_lda_model.joblib"),
            "formal_predictions": sha256(
                formal.mainline_output / "fixed_beta_final_test_predictions.csv"
            ),
            "location_distance": sha256(LOCATION_CSV),
        },
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    checksum_path = args.output_dir / "CHECKSUMS.sha256"
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path != checksum_path)
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
