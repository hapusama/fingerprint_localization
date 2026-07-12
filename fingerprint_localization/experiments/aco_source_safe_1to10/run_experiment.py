#!/usr/bin/env python3
"""Gaussian-noise 1:10 split experiment for LoRa fingerprint localization.

The experiment rebuilds noisy inputs from the trusted processed CSV files,
creates a stratified train/validation/test split, and evaluates four methods:

1. LOOCV / 1-NN
2. KNN
3. model/v3 heuristic PGAR
4. model/v3 packet-internal ant colony optimization
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
PKG_ROOT = PROJECT_ROOT / "fingerprint_localization"
DATA_ROOT = PKG_ROOT / "data" / "mainline_202607"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from model.v3 import aco_packet_path as aco  # noqa: E402
from model.v3 import pgar_heuristic as pgar  # noqa: E402


DEFAULT_INPUTS = {
    "rssi": DATA_ROOT / "inputs" / "rssi_plus_packet_level_54points.csv",
    "raw": DATA_ROOT / "inputs" / "lora_frequency_s17_54points.csv",
    "spectrum": DATA_ROOT / "external" / "subbin_spectrum_long.csv",
    "chirp": DATA_ROOT / "features" / "chirp_point_multipath_structure_features.csv",
}

RSSI_NOISE_COLUMNS = list(pgar.RSSI_PLUS_COLUMNS)
RAW_NOISE_COLUMNS = [f"preamble_fft_mag_bin_{offset:+d}" for offset in range(-8, 9)]
RAW_NOISE_COLUMNS += ["preamble_peak_to_residual_db", "detect_score_db", "s17_c_s", "s17_j_s"]
SPECTRUM_NOISE_COLUMNS = ["mag_raw", "mag_norm", "mag_db_rel_peak"]
POSITIVE_COLUMNS = {
    "rssi_variance",
    "preamble_fft_mag_bin_-8",
    "preamble_fft_mag_bin_-7",
    "preamble_fft_mag_bin_-6",
    "preamble_fft_mag_bin_-5",
    "preamble_fft_mag_bin_-4",
    "preamble_fft_mag_bin_-3",
    "preamble_fft_mag_bin_-2",
    "preamble_fft_mag_bin_-1",
    "preamble_fft_mag_bin_+0",
    "preamble_fft_mag_bin_+1",
    "preamble_fft_mag_bin_+2",
    "preamble_fft_mag_bin_+3",
    "preamble_fft_mag_bin_+4",
    "preamble_fft_mag_bin_+5",
    "preamble_fft_mag_bin_+6",
    "preamble_fft_mag_bin_+7",
    "preamble_fft_mag_bin_+8",
    "s17_c_s",
    "s17_j_s",
    "mag_raw",
    "mag_norm",
}
EPS = 1e-12


def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def file_stem(file_name: str) -> str:
    return Path(file_name).stem


def packet_key(row: dict) -> tuple[str, int]:
    return file_stem(row["file_name"]), int(float(row["packet_index"]))


def column_noise_stats(
    rows: list[dict],
    columns: Sequence[str],
    noise_ratio: float,
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        values = [parse_float(row.get(column)) for row in rows]
        clean = [value for value in values if value is not None]
        if not clean:
            continue
        mean = sum(clean) / len(clean)
        variance = sum((value - mean) ** 2 for value in clean) / len(clean)
        source_std = math.sqrt(variance)
        noise_std = source_std / noise_ratio if source_std > EPS else 0.0
        stats[column] = {
            "source_std": source_std,
            "noise_std": noise_std,
            "noise_ratio": noise_ratio,
        }
    return stats


def augmented_file_name(file_name: str, aug_id: int) -> str:
    path = Path(file_name)
    return f"{path.stem}_aug{aug_id:02d}{path.suffix}"


def augment_rows_with_noise(
    rows: list[dict],
    columns: Sequence[str],
    rng: random.Random,
    noise_ratio: float,
    augment_factor: int,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    stats = column_noise_stats(rows, columns, noise_ratio)
    augmented_rows: list[dict] = []
    for row in rows:
        for aug_id in range(augment_factor):
            out = dict(row)
            out["source_file_name"] = row.get("file_name", "")
            out["source_packet_index"] = row.get("packet_index", "")
            out["augmentation_id"] = aug_id
            if "file_name" in out:
                out["file_name"] = augmented_file_name(str(out["file_name"]), aug_id)
            for column in columns:
                value = parse_float(row.get(column))
                if value is None:
                    continue
                noise_std = stats.get(column, {}).get("noise_std", 0.0)
                noised = value + rng.gauss(0.0, noise_std)
                if column in POSITIVE_COLUMNS:
                    noised = max(noised, EPS)
                out[column] = f"{noised:.12g}"
            augmented_rows.append(out)
    return augmented_rows, stats


def add_metadata_fields(fieldnames: Sequence[str]) -> list[str]:
    fields = list(fieldnames)
    for field in ["source_file_name", "source_packet_index", "augmentation_id"]:
        if field not in fields:
            fields.append(field)
    return fields


def add_column_noise(
    rows: list[dict],
    columns: Sequence[str],
    rng: random.Random,
    noise_ratio: float,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    noisy_rows = [dict(row) for row in rows]
    stats = column_noise_stats(rows, columns, noise_ratio)
    for column in columns:
        for row in noisy_rows:
            value = parse_float(row.get(column))
            if value is None:
                continue
            noised = value + rng.gauss(0.0, stats.get(column, {}).get("noise_std", 0.0))
            if column in POSITIVE_COLUMNS:
                noised = max(noised, EPS)
            row[column] = f"{noised:.12g}"
    return noisy_rows, stats


def point_label_from_raw(row: dict) -> str:
    return f"{int(float(row['corridor_id']))}_{int(float(row['position_id']))}"


def build_noisy_inputs(args: argparse.Namespace) -> dict:
    data_dir = args.output_dir / "data"
    rng = random.Random(args.seed)

    rssi_rows, rssi_fields = read_csv(args.rssi_csv)
    raw_rows, raw_fields = read_csv(args.raw_csv)
    spectrum_rows, spectrum_fields = read_csv(args.spectrum_csv)

    noisy_rssi, rssi_noise = augment_rows_with_noise(
        rssi_rows, RSSI_NOISE_COLUMNS, rng, args.noise_ratio, args.augment_factor
    )
    noisy_raw, raw_noise = augment_rows_with_noise(
        raw_rows, RAW_NOISE_COLUMNS, rng, args.noise_ratio, args.augment_factor
    )
    noisy_spectrum, spectrum_noise = augment_rows_with_noise(
        spectrum_rows, SPECTRUM_NOISE_COLUMNS, rng, args.noise_ratio, args.augment_factor
    )

    paths = {
        "rssi": data_dir / "noisy_rssi_plus_packet_level_54points.csv",
        "raw": data_dir / "noisy_lora_frequency_s17_54points.csv",
        "spectrum": data_dir / "noisy_subbin_spectrum_long.csv",
    }
    write_csv(paths["rssi"], noisy_rssi, add_metadata_fields(rssi_fields))
    write_csv(paths["raw"], noisy_raw, add_metadata_fields(raw_fields))
    write_csv(paths["spectrum"], noisy_spectrum, add_metadata_fields(spectrum_fields))

    metadata = {
        "noise_definition": "For each selected numeric feature column, Gaussian noise N(0, (column_std / noise_ratio)^2) is added independently to each augmented copy.",
        "noise_ratio": args.noise_ratio,
        "augment_factor": args.augment_factor,
        "augmentation_definition": "Each original packet is expanded into augment_factor noisy packet copies before train/validation/test splitting.",
        "seed": args.seed,
        "inputs": {name: str(path) for name, path in DEFAULT_INPUTS.items()},
        "noisy_inputs": {name: str(path) for name, path in paths.items()},
        "noise_stats": {
            "rssi": rssi_noise,
            "raw": raw_noise,
            "spectrum": spectrum_noise,
        },
    }
    with (data_dir / "noise_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return {"paths": paths, "metadata": metadata}


def build_pgar_samples(paths: dict, output_dir: Path) -> list[pgar.Sample]:
    pgar_args = SimpleNamespace(
        rssi_csv=paths["rssi"],
        raw_feature_csv=paths["raw"],
        q4_spectrum_csv=paths["spectrum"],
        output_dir=output_dir / "features",
        disable_q4=False,
    )
    rssi_rows, raw_rows, q4_packets, _q4_offsets = pgar.build_or_load_features(pgar_args)
    return pgar.align_samples(
        pgar.rssi_rows_to_packets(rssi_rows),
        pgar.raw_rows_to_packets(raw_rows),
        q4_packets,
    )


def split_keys(samples: Sequence[pgar.Sample], seed: int) -> tuple[dict[str, list[tuple[str, int]]], list[dict]]:
    by_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample.key)

    rng = random.Random(seed)
    split: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    rows = []
    for label in sorted(by_label, key=pgar.natural_label_key):
        keys = list(by_label[label])
        rng.shuffle(keys)
        n = len(keys)
        if n == 1:
            n_train = 1
            n_val = 0
        elif n == 2:
            n_train = 1
            n_val = 0
        elif n == 3:
            n_train = 1
            n_val = 1
        else:
            n_train = max(2, int(round(n * 0.6)))
            n_val = max(1, int(round(n * 0.2)))
            if n_train + n_val >= n:
                n_val = 1
                n_train = n - 2
        n_test = n - n_train - n_val
        parts = {
            "train": keys[:n_train],
            "val": keys[n_train : n_train + n_val],
            "test": keys[n_train + n_val :],
        }
        for split_name, split_keys_for_label in parts.items():
            split[split_name].extend(split_keys_for_label)
            for key in split_keys_for_label:
                rows.append(
                    {
                        "split": split_name,
                        "position_key": label,
                        "file_stem": key[0],
                        "packet_index": key[1],
                    }
                )
    return split, rows


def indices_for_split(samples: Sequence, split: dict[str, list[tuple[str, int]]]) -> dict[str, list[int]]:
    key_to_index = {sample.key: idx for idx, sample in enumerate(samples)}
    return {
        name: sorted(key_to_index[key] for key in keys if key in key_to_index)
        for name, keys in split.items()
    }


def standardize_stats(rows: Sequence[Sequence[float]], indices: Sequence[int]) -> tuple[list[float], list[float]]:
    dim = len(rows[0])
    means = [sum(rows[idx][j] for idx in indices) / len(indices) for j in range(dim)]
    stds = []
    for j in range(dim):
        variance = sum((rows[idx][j] - means[j]) ** 2 for idx in indices) / len(indices)
        std = math.sqrt(variance)
        stds.append(std if std > EPS else 1.0)
    return means, stds


def zdist(a: Sequence[float], b: Sequence[float], means: Sequence[float], stds: Sequence[float]) -> float:
    return math.sqrt(sum((((a[j] - means[j]) / stds[j]) - ((b[j] - means[j]) / stds[j])) ** 2 for j in range(len(a))))


def majority_vote(labels: Sequence[str], distances: Sequence[tuple[float, str]], k: int) -> str:
    nearest = sorted(distances, key=lambda item: (item[0], pgar.natural_label_key(item[1])))[:k]
    counts = Counter(label for _distance, label in nearest)
    best_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == best_count]
    if len(tied) == 1:
        return tied[0]
    mean_dist = {
        label: sum(distance for distance, item_label in nearest if item_label == label) / counts[label]
        for label in tied
    }
    return min(tied, key=lambda label: (mean_dist[label], pgar.natural_label_key(label)))


def evaluate_1nn_knn(
    samples: Sequence[pgar.Sample],
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    mode: str,
    k: int,
) -> tuple[dict, list[dict]]:
    rows = [sample.rssi_plus for sample in samples]
    labels = [sample.label for sample in samples]
    predictions = []
    correct = 0
    for eval_index in eval_indices:
        effective_train = [idx for idx in train_indices if idx != eval_index]
        if not effective_train:
            continue
        means, stds = standardize_stats(rows, effective_train)
        distances = [
            (zdist(rows[idx], rows[eval_index], means, stds), labels[idx])
            for idx in effective_train
        ]
        if mode == "1nn":
            pred = min(distances, key=lambda item: (item[0], pgar.natural_label_key(item[1])))[1]
        elif mode == "knn":
            pred = majority_vote(labels, distances, min(k, len(distances)))
        else:
            raise ValueError(mode)
        ok = int(pred == labels[eval_index])
        correct += ok
        predictions.append(
            {
                "sample_index": eval_index,
                "file_name": samples[eval_index].file_name,
                "packet_index": samples[eval_index].packet_index,
                "true_label": labels[eval_index],
                "pred_label": pred,
                "correct": ok,
                "train_reference_count": len(effective_train),
            }
        )
    n = len(predictions)
    return {"packet_count": n, "correct": correct, "accuracy": correct / n if n else 0.0, "k": k}, predictions


def pgar_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        top_k=args.top_k,
        rssi_class_k=args.rssi_class_k,
        rssi_margin_threshold=args.rssi_margin_threshold,
        alpha=args.pgar_alpha,
        beta=args.pgar_beta,
        peak_threshold=None,
        peak_iqr_threshold=None,
        auto_peak_quantile=0.10,
        auto_peak_iqr_quantile=0.75,
        gamma=args.pgar_gamma,
        q4_raw_margin_threshold=args.q4_raw_margin_threshold,
        q4_stability_threshold=None,
        auto_q4_stability_quantile=0.75,
        q4_disc_threshold=args.q4_disc_threshold,
    )


def evaluate_pgar_fixed_train(
    samples: Sequence[pgar.Sample],
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    args: SimpleNamespace,
) -> tuple[dict, list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    structures = [sample.structure for sample in samples]
    q4_curves = [sample.q4_curve for sample in samples]
    peak_threshold, peak_iqr_threshold = pgar.resolve_auto_thresholds([samples[idx] for idx in train_indices], args)
    q4_stability_threshold = pgar.resolve_q4_stability_threshold([samples[idx] for idx in train_indices], args)
    predictions = []
    counts = Counter()
    for test_index in eval_indices:
        sample = samples[test_index]
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        rssi_ranked = pgar.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        rssi_pred = candidates[0]
        margin = rssi_ranked[1][1] - rssi_ranked[0][1] if len(rssi_ranked) > 1 else float("inf")
        peak_reliable = sample.peak_mean > peak_threshold and sample.peak_iqr < peak_iqr_threshold
        use_raw = margin <= args.rssi_margin_threshold and peak_reliable
        raw_pred = rssi_pred
        pgar_pred = rssi_pred
        raw_margin = float("inf")
        q4_gate_reason = "raw_gate_closed"
        q4_disc = 0.0
        if use_raw:
            counts["raw_gate_used"] += 1
            prototypes = pgar.build_prototypes(structures, labels, effective_train)
            weights = pgar.candidate_weights(prototypes, candidates)
            rssi_norm = pgar.normalize_candidate_scores(rssi_ranked, candidates)
            scored = []
            for candidate in candidates:
                if candidate not in prototypes:
                    continue
                d_e = pgar.structure_distance(sample.structure, prototypes[candidate], weights)
                d_r = rssi_norm.get(candidate, 0.0)
                scored.append((candidate, args.alpha * d_r + args.beta * d_e, d_r, d_e))
            scored.sort(key=lambda item: (item[1], pgar.natural_label_key(item[0])))
            if scored:
                raw_pred = scored[0][0]
                pgar_pred = raw_pred
                raw_margin = scored[1][1] - scored[0][1] if len(scored) > 1 else float("inf")
                q4_gate_reason = "raw_top_margin_not_close"
                if len(scored) >= 2 and raw_margin <= args.q4_raw_margin_threshold:
                    q4_gate_reason = "q4_packet_missing"
                    if sample.q4_curve is not None:
                        q4_gate_reason = "q4_packet_unstable"
                        if sample.q4_stability < q4_stability_threshold:
                            q4_prototypes = pgar.build_q4_prototypes(q4_curves, labels, effective_train)
                            q4_disc = pgar.q4_discriminability(scored[0][0], scored[1][0], q4_prototypes)
                            q4_gate_reason = "q4_disc_too_low"
                            if q4_disc > args.q4_disc_threshold and args.gamma > 0.0:
                                q4_scores = [
                                    (candidate, pgar.q4_distance(sample.q4_curve, q4_prototypes[candidate]))
                                    for candidate in candidates
                                    if candidate in q4_prototypes
                                ]
                                q4_norm = pgar.normalize_candidate_scores(q4_scores, candidates)
                                final_scored = [
                                    (candidate, raw_score + args.gamma * q4_norm.get(candidate, 0.0))
                                    for candidate, raw_score, _d_r, _d_e in scored
                                ]
                                final_scored.sort(key=lambda item: (item[1], pgar.natural_label_key(item[0])))
                                if final_scored:
                                    pgar_pred = final_scored[0][0]
                                    q4_gate_reason = "q4_used"
                                    counts["q4_gate_used"] += 1
        rssi_ok = int(rssi_pred == sample.label)
        raw_ok = int(raw_pred == sample.label)
        pgar_ok = int(pgar_pred == sample.label)
        counts["rssi_correct"] += rssi_ok
        counts["raw_correct"] += raw_ok
        counts["pgar_correct"] += pgar_ok
        counts["topk_contains"] += int(sample.label in candidates)
        predictions.append(
            {
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": rssi_ok,
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "raw_rerank_label": raw_pred,
                "raw_rerank_correct": raw_ok,
                "pgar_label": pgar_pred,
                "pgar_correct": pgar_ok,
                "raw_gate_used": int(use_raw),
                "q4_gate_reason": q4_gate_reason,
                "q4_gate_used": int(q4_gate_reason == "q4_used"),
                "rssi_margin": margin,
                "raw_margin": raw_margin,
                "q4_disc_top12": q4_disc,
            }
        )
    n = len(predictions)
    metrics = {
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "rssi_class_k": args.rssi_class_k,
        "rssi_top1_correct": counts["rssi_correct"],
        "rssi_top1_accuracy": counts["rssi_correct"] / n if n else 0.0,
        "rssi_topk_contains_true": counts["topk_contains"],
        "rssi_topk_recall": counts["topk_contains"] / n if n else 0.0,
        "raw_rerank_correct": counts["raw_correct"],
        "raw_rerank_accuracy": counts["raw_correct"] / n if n else 0.0,
        "pgar_correct": counts["pgar_correct"],
        "pgar_accuracy": counts["pgar_correct"] / n if n else 0.0,
        "raw_gate_used": counts["raw_gate_used"],
        "raw_gate_rate": counts["raw_gate_used"] / n if n else 0.0,
        "q4_gate_used": counts["q4_gate_used"],
        "q4_gate_rate": counts["q4_gate_used"] / n if n else 0.0,
        "peak_threshold": peak_threshold,
        "peak_iqr_threshold": peak_iqr_threshold,
        "q4_stability_threshold": q4_stability_threshold,
    }
    return metrics, predictions


def build_aco_samples(paths: dict, args: argparse.Namespace) -> tuple[list[aco.PacketSample], list[float], dict]:
    aco_args = aco_args_from_cli(args)
    rssi_packets = aco.read_rssi_packets(paths["rssi"])
    symbol_packets, q4_offsets, thresholds = aco.read_symbol_packets(paths["spectrum"], aco_args)
    samples = aco.align_samples(rssi_packets, symbol_packets)
    return samples, q4_offsets, thresholds


def aco_args_from_cli(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        top_k=args.top_k,
        rssi_class_k=args.rssi_class_k,
        ants=args.ants,
        iterations=args.aco_iterations,
        elite_ants=args.elite_ants,
        seed=args.seed,
        rssi_weight=0.45,
        energy_weight=0.20,
        raw_weight=0.55,
        q4_weight=0.15,
        chirp_q4_boost=0.15,
        chirp_self_loop_boost=0.20,
        switch_penalty=0.70,
        diversity_penalty=0.20,
        pheromone_power=1.0,
        heuristic_power=1.4,
        evaporation=0.25,
        tau_stay=1.4,
        tau_switch=0.35,
        min_pheromone=1e-4,
        q4_shift_grid="-0.25,0,0.25",
        peak_threshold=None,
        auto_peak_quantile=0.10,
        q4_dev_threshold=None,
        auto_q4_dev_quantile=0.75,
        q4_peak_offset_max=0.50,
        q4_peak_to_side_threshold=6.0,
    )


def evaluate_aco_fixed_train(
    samples: Sequence[aco.PacketSample],
    q4_offsets: Sequence[float],
    chirp_priors: dict,
    thresholds: dict,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    args: SimpleNamespace,
    leave_one_out_prototypes: bool = False,
) -> tuple[dict, list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    rng = random.Random(args.seed)
    predictions = []
    correct = Counter()
    topk_contains = 0
    q4_reliable_total = 0
    symbol_total = 0
    prototype_cache: dict[tuple[int, ...], dict] = {}
    for test_index in eval_indices:
        sample = samples[test_index]
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        rssi_ranked = aco.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        rssi_pred = candidates[0]
        prototype_indices = effective_train if leave_one_out_prototypes else list(train_indices)
        prototype_key = tuple(prototype_indices)
        if prototype_key not in prototype_cache:
            prototype_cache[prototype_key] = aco.build_symbol_prototypes(samples, labels, prototype_indices)
        prototypes = prototype_cache[prototype_key]
        rssi_costs = {label: score for label, score in rssi_ranked if label in candidates}
        chirp_weight = aco.chirp_separability(candidates, chirp_priors)
        obs_costs, _rows = aco.build_observation_costs(
            sample,
            candidates,
            rssi_costs,
            prototypes,
            q4_offsets,
            chirp_weight,
            args,
        )
        result = aco.run_aco_for_packet(obs_costs, candidates, chirp_priors, args, rng)
        path_labels = [candidates[idx] for idx in result["best_path"]]
        topk_contains += int(sample.label in candidates)
        q4_reliable_total += sum(1 for symbol in sample.symbols if symbol.q4_reliable)
        symbol_total += len(sample.symbols)
        correct["rssi"] += int(rssi_pred == sample.label)
        correct["path_mode"] += int(result["path_mode_label"] == sample.label)
        correct["pheromone"] += int(result["pheromone_label"] == sample.label)
        correct["vote"] += int(result["vote_label"] == sample.label)
        predictions.append(
            {
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": int(rssi_pred == sample.label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "aco_path_mode_label": result["path_mode_label"],
                "aco_path_mode_correct": int(result["path_mode_label"] == sample.label),
                "aco_pheromone_label": result["pheromone_label"],
                "aco_pheromone_correct": int(result["pheromone_label"] == sample.label),
                "aco_vote_label": result["vote_label"],
                "aco_vote_correct": int(result["vote_label"] == sample.label),
                "best_path_cost": result["best_cost"],
                "best_path_labels": ";".join(path_labels),
                "q4_reliable_symbols": sum(1 for symbol in sample.symbols if symbol.q4_reliable),
                "symbol_count": len(sample.symbols),
            }
        )
    n = len(predictions)
    metrics = {
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "rssi_top1_correct": correct["rssi"],
        "rssi_top1_accuracy": correct["rssi"] / n if n else 0.0,
        "rssi_topk_contains_true": topk_contains,
        "rssi_topk_recall": topk_contains / n if n else 0.0,
        "aco_path_mode_correct": correct["path_mode"],
        "aco_path_mode_accuracy": correct["path_mode"] / n if n else 0.0,
        "aco_pheromone_correct": correct["pheromone"],
        "aco_pheromone_accuracy": correct["pheromone"] / n if n else 0.0,
        "aco_vote_correct": correct["vote"],
        "aco_vote_accuracy": correct["vote"] / n if n else 0.0,
        "q4_reliable_symbol_count": q4_reliable_total,
        "symbol_count": symbol_total,
        "q4_reliable_symbol_rate": q4_reliable_total / symbol_total if symbol_total else 0.0,
        **thresholds,
    }
    return metrics, predictions


def write_method_outputs(base_dir: Path, method: str, split_name: str, metrics: dict, predictions: list[dict]) -> dict:
    out_dir = base_dir / method
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{split_name}_metrics.json"
    pred_path = out_dir / f"{split_name}_predictions.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if predictions:
        fields = list(predictions[0].keys())
        write_csv(pred_path, predictions, fields)
    else:
        pred_path.write_text("", encoding="utf-8")
    return {"metrics": str(metrics_path), "predictions": str(pred_path)}


def run_all(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    noisy = build_noisy_inputs(args)
    pgar_samples = build_pgar_samples(noisy["paths"], args.output_dir)
    split, split_rows = split_keys(pgar_samples, args.seed)
    write_csv(args.output_dir / "data" / "split_assignments.csv", split_rows, ["split", "position_key", "file_stem", "packet_index"])

    pgar_indices = indices_for_split(pgar_samples, split)
    aco_samples, q4_offsets, aco_thresholds = build_aco_samples(noisy["paths"], args)
    aco_indices = indices_for_split(aco_samples, split)
    chirp_priors = aco.read_chirp_priors(args.chirp_csv)

    result_dir = args.output_dir / "results"
    summary_rows = []
    outputs = {}
    p_args = pgar_args(args)
    ac_args = aco_args_from_cli(args)

    eval_plan = [
        ("train_loocv", pgar_indices["train"], pgar_indices["train"]),
        ("val", pgar_indices["train"], pgar_indices["val"]),
        ("test", pgar_indices["train"], pgar_indices["test"]),
    ]

    for split_name, train_idx, eval_idx in eval_plan:
        for method, mode, k in [("loocv_1nn", "1nn", 1), ("knn", "knn", args.knn_k)]:
            metrics, predictions = evaluate_1nn_knn(pgar_samples, train_idx, eval_idx, mode, k)
            metrics = {"method": method, "split": split_name, **metrics}
            outputs[f"{method}_{split_name}"] = write_method_outputs(result_dir, method, split_name, metrics, predictions)
            summary_rows.append(metrics)

        metrics, predictions = evaluate_pgar_fixed_train(pgar_samples, train_idx, eval_idx, p_args)
        metrics = {"method": "pgar", "split": split_name, **metrics}
        outputs[f"pgar_{split_name}"] = write_method_outputs(result_dir, "pgar", split_name, metrics, predictions)
        summary_rows.append(metrics)

    aco_eval_plan = [
        ("train_loocv", aco_indices["train"], aco_indices["train"]),
        ("val", aco_indices["train"], aco_indices["val"]),
        ("test", aco_indices["train"], aco_indices["test"]),
    ]
    for split_name, train_idx, eval_idx in aco_eval_plan:
        metrics, predictions = evaluate_aco_fixed_train(
            aco_samples,
            q4_offsets,
            chirp_priors,
            aco_thresholds,
            train_idx,
            eval_idx,
            ac_args,
        )
        metrics = {"method": "aco", "split": split_name, **metrics}
        outputs[f"aco_{split_name}"] = write_method_outputs(result_dir, "aco", split_name, metrics, predictions)
        summary_rows.append(metrics)

    preferred_summary_fields = [
        "method",
        "split",
        "packet_count",
        "location_count",
        "correct",
        "accuracy",
        "k",
        "rssi_top1_correct",
        "rssi_top1_accuracy",
        "rssi_topk_contains_true",
        "rssi_topk_recall",
        "raw_rerank_correct",
        "raw_rerank_accuracy",
        "pgar_correct",
        "pgar_accuracy",
        "raw_gate_used",
        "raw_gate_rate",
        "q4_gate_used",
        "q4_gate_rate",
        "aco_path_mode_correct",
        "aco_path_mode_accuracy",
        "aco_pheromone_correct",
        "aco_pheromone_accuracy",
        "aco_vote_correct",
        "aco_vote_accuracy",
        "q4_reliable_symbol_count",
        "q4_reliable_symbol_rate",
    ]
    extra_fields = sorted({key for row in summary_rows for key in row} - set(preferred_summary_fields))
    summary_fields = preferred_summary_fields + extra_fields
    write_csv(args.output_dir / "results" / "method_summary.csv", summary_rows, summary_fields)
    payload = {
        "output_dir": str(args.output_dir),
        "config": {
            "noise_ratio": args.noise_ratio,
            "augment_factor": args.augment_factor,
            "seed": args.seed,
            "split": "after augmentation, stratified by position_key, approximately train/val/test = 6/2/2",
            "knn_k": args.knn_k,
            "top_k": args.top_k,
            "rssi_class_k": args.rssi_class_k,
            "ants": args.ants,
            "aco_iterations": args.aco_iterations,
        },
        "sample_counts": {
            "pgar_aligned": len(pgar_samples),
            "aco_aligned": len(aco_samples),
            "train": len(pgar_indices["train"]),
            "val": len(pgar_indices["val"]),
            "test": len(pgar_indices["test"]),
            "locations": len({sample.label for sample in pgar_samples}),
        },
        "outputs": outputs,
        "summary": summary_rows,
    }
    with (args.output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_INPUTS["rssi"])
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_INPUTS["raw"])
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_INPUTS["spectrum"])
    parser.add_argument("--chirp-csv", type=Path, default=DEFAULT_INPUTS["chirp"])
    parser.add_argument("--noise-ratio", type=float, default=10.0)
    parser.add_argument("--augment-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--knn-k", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--rssi-margin-threshold", type=float, default=0.2)
    parser.add_argument("--pgar-alpha", type=float, default=1.0)
    parser.add_argument("--pgar-beta", type=float, default=1.0)
    parser.add_argument("--pgar-gamma", type=float, default=0.25)
    parser.add_argument("--q4-raw-margin-threshold", type=float, default=0.2)
    parser.add_argument("--q4-disc-threshold", type=float, default=0.5)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--aco-iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    payload = run_all(parse_args())
    print(json.dumps(payload["sample_counts"], indent=2, ensure_ascii=False))
    print(f"Wrote {payload['output_dir']}")


if __name__ == "__main__":
    main()
