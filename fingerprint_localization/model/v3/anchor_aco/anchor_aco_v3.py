#!/usr/bin/env python3
"""Anchor-ACO 3.0 on the fixed 1:10 Gaussian-noise split.

This implements the state redesign from `external_design_notes/蚁群算法3.0.md`:
an ant first chooses a packet-level anchor location L from RSSI+ Top-K, then
each segment chooses whether it supports L or should be treated as abnormal.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


ANCHOR_DIR = Path(__file__).resolve().parent
MODEL_V3_DIR = ANCHOR_DIR.parent
PROJECT_ROOT = ANCHOR_DIR.parents[3]
if str(MODEL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_V3_DIR))

import aco_packet_path_v2 as aco2  # noqa: E402


GAUSSIAN_DIR = (
    PROJECT_ROOT / "fingerprint_localization" / "experiments" / "aco_source_safe_1to10"
)
DEFAULT_DATA_DIR = GAUSSIAN_DIR / "data"
DEFAULT_RSSI_CSV = DEFAULT_DATA_DIR / "noisy_rssi_plus_packet_level_54points.csv"
DEFAULT_SPECTRUM_CSV = DEFAULT_DATA_DIR / "noisy_subbin_spectrum_long.csv"
DEFAULT_SPLIT_CSV = DEFAULT_DATA_DIR / "split_assignments.csv"
DEFAULT_OUTPUT_DIR = ANCHOR_DIR / "output_gaussian_noise_1to10"
EPS = 1e-12


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def key_from_split_row(row: dict) -> tuple[str, int]:
    return row["file_stem"], int(float(row["packet_index"]))


def load_split_indices(samples: Sequence[aco2.SegmentPacket], split_csv: Path) -> dict[str, list[int]]:
    split_keys: dict[str, list[tuple[str, int]]] = {"train": [], "val": [], "test": []}
    for row in read_csv(split_csv):
        split_keys[row["split"]].append(key_from_split_row(row))
    key_to_index = {sample.key: idx for idx, sample in enumerate(samples)}
    missing = {name: [key for key in keys if key not in key_to_index] for name, keys in split_keys.items()}
    missing_count = sum(len(keys) for keys in missing.values())
    if missing_count:
        details = ", ".join(f"{name}={len(keys)}" for name, keys in missing.items() if keys)
        raise ValueError(f"{missing_count} split keys were not found in Anchor-ACO samples: {details}")
    return {name: sorted(key_to_index[key] for key in keys) for name, keys in split_keys.items()}


def weighted_choice(weights: Sequence[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= EPS:
        return rng.randrange(len(weights))
    pick = rng.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += weight
        if pick <= acc:
            return idx
    return len(weights) - 1


def segment_support_cost(row: dict, args: argparse.Namespace) -> float:
    q_weight = args.q4_weight if int(float(row.get("q4_reliable", 0))) else 0.0
    return (
        args.energy_weight * parse_float(row["C_E"])
        + args.raw_weight * parse_float(row["C_W"])
        + args.bin_weight * parse_float(row["C_bin"])
        + q_weight * parse_float(row["C_Q"])
    )


def build_anchor_inputs(
    packet: aco2.SegmentPacket,
    candidates: Sequence[str],
    rssi_costs: dict[str, float],
    templates: dict[str, aco2.BinTemplate],
    segment_prototypes: dict,
    q4_offsets: Sequence[float],
    args: argparse.Namespace,
) -> tuple[dict[str, dict], list[dict]]:
    _obs_costs, rows = aco2.build_observation_costs_v2(
        packet,
        candidates,
        rssi_costs,
        templates,
        segment_prototypes,
        q4_offsets,
        args,
    )
    by_label = {
        label: {
            "label": label,
            "C_R": 0.0,
            "segment_costs": [],
            "template_reliability": templates[label].reliability,
        }
        for label in candidates
    }
    segment_rows = []
    for row in rows:
        label = row["candidate_label"]
        if label == aco2.GARBAGE_LABEL:
            continue
        c_seg = segment_support_cost(row, args)
        by_label[label]["C_R"] = parse_float(row["C_R"])
        by_label[label]["segment_costs"].append(c_seg)
        row = dict(row)
        row["C_anchor_seg"] = c_seg
        segment_rows.append(row)
    return by_label, segment_rows


def cost_for_mask(c_r: float, segment_costs: Sequence[float], mask: Sequence[int], args: argparse.Namespace) -> float:
    normal_costs = [cost for cost, is_garbage in zip(segment_costs, mask) if not is_garbage]
    if not normal_costs:
        return float("inf")
    if args.average_segment_cost:
        segment_term = sum(normal_costs) / len(normal_costs)
    else:
        segment_term = sum(normal_costs)
    return (
        args.anchor_rssi_weight * c_r
        + args.segment_weight * segment_term
        + args.anchor_garbage_penalty * sum(mask)
    )


def best_anchor_mask(c_r: float, segment_costs: Sequence[float], args: argparse.Namespace) -> tuple[float, tuple[int, ...]]:
    best_cost = float("inf")
    best_mask: tuple[int, ...] = tuple(0 for _ in segment_costs)
    for mask in itertools.product([0, 1], repeat=len(segment_costs)):
        if sum(mask) > args.max_anchor_garbage:
            continue
        cost = cost_for_mask(c_r, segment_costs, mask, args)
        if cost < best_cost:
            best_cost = cost
            best_mask = tuple(mask)
    return best_cost, best_mask


def robust_best3_cost(c_r: float, segment_costs: Sequence[float], args: argparse.Namespace) -> tuple[float, tuple[int, ...]]:
    drop_count = min(args.max_anchor_garbage, max(0, len(segment_costs) - 1))
    keep_count = len(segment_costs) - drop_count
    ranked = sorted(range(len(segment_costs)), key=lambda idx: (segment_costs[idx], idx))
    keep = set(ranked[:keep_count])
    mask = tuple(0 if idx in keep else 1 for idx in range(len(segment_costs)))
    normal_costs = [segment_costs[idx] for idx in range(len(segment_costs)) if idx in keep]
    robust_mean = sum(normal_costs) / len(normal_costs)
    cost = (
        args.anchor_rssi_weight * c_r
        + args.segment_weight * robust_mean
        + args.anchor_garbage_penalty * sum(mask)
    )
    return cost, mask


def run_anchor_aco_for_packet(anchor_inputs: dict[str, dict], args: argparse.Namespace, rng: random.Random) -> dict:
    candidates = list(anchor_inputs)
    tau = {label: 1.0 for label in candidates}
    best_path = {"cost": float("inf"), "anchor": "", "mask": tuple()}
    elite_vote = {label: 0.0 for label in candidates}

    for _iteration in range(args.iterations):
        ant_paths = []
        anchor_weights = [
            (tau[label] ** args.pheromone_power)
            * math.exp(-args.anchor_rssi_beta * anchor_inputs[label]["C_R"])
            for label in candidates
        ]
        for _ant in range(args.ants):
            anchor = candidates[weighted_choice(anchor_weights, rng)]
            segment_costs = anchor_inputs[anchor]["segment_costs"]
            mask = []
            for c_seg in segment_costs:
                if sum(mask) >= args.max_anchor_garbage:
                    mask.append(0)
                    continue
                p_normal = math.exp(-args.segment_support_beta * c_seg)
                p_garbage = math.exp(-args.segment_support_beta * args.garbage_cost)
                mask.append(weighted_choice([p_normal, p_garbage], rng))
            mask_tuple = tuple(mask)
            cost = cost_for_mask(anchor_inputs[anchor]["C_R"], segment_costs, mask_tuple, args)
            ant_paths.append((cost, anchor, mask_tuple))
            if cost < best_path["cost"]:
                best_path = {"cost": cost, "anchor": anchor, "mask": mask_tuple}

        ant_paths.sort(key=lambda item: item[0])
        elite = ant_paths[: max(1, min(args.elite_ants, len(ant_paths)))]
        temp = args.aco_temperature
        if temp is None or temp <= EPS:
            temp = aco2.median([cost for cost, _anchor, _mask in elite])
            temp = temp if temp > EPS else 1.0
        weights = [math.exp(-cost / (temp + EPS)) for cost, _anchor, _mask in elite]
        total = sum(weights) or 1.0
        weights = [value / total for value in weights]

        for label in candidates:
            tau[label] = max(args.min_pheromone, tau[label] * (1.0 - args.evaporation))
        for (_cost, anchor, mask), weight in zip(elite, weights):
            normal_rate = 1.0 - (sum(mask) / max(1, len(mask)))
            tau[anchor] += weight * (1.0 + args.reliability_bonus * normal_rate)
            elite_vote[anchor] += weight

    anchor_costs = {}
    robust3_costs = {}
    best_masks = {}
    robust3_masks = {}
    for label, info in anchor_inputs.items():
        cost, mask = best_anchor_mask(info["C_R"], info["segment_costs"], args)
        robust_cost, robust_mask = robust_best3_cost(info["C_R"], info["segment_costs"], args)
        anchor_costs[label] = cost
        best_masks[label] = mask
        robust3_costs[label] = robust_cost
        robust3_masks[label] = robust_mask

    cost_label = min(candidates, key=lambda label: (anchor_costs[label], aco2.natural_label_key(label)))
    robust3_label = min(candidates, key=lambda label: (robust3_costs[label], aco2.natural_label_key(label)))
    pheromone_label = max(candidates, key=lambda label: (tau[label], tuple(-x for x in aco2.natural_label_key(label))))
    vote_label = max(candidates, key=lambda label: (elite_vote[label], tuple(-x for x in aco2.natural_label_key(label))))
    return {
        "anchor_cost_label": cost_label,
        "anchor_cost": anchor_costs[cost_label],
        "anchor_cost_mask": best_masks[cost_label],
        "robust3_label": robust3_label,
        "robust3_cost": robust3_costs[robust3_label],
        "robust3_mask": robust3_masks[robust3_label],
        "anchor_pheromone_label": pheromone_label,
        "anchor_vote_label": vote_label,
        "best_ant_anchor": best_path["anchor"],
        "best_ant_cost": best_path["cost"],
        "best_ant_mask": best_path["mask"],
        "tau": tau,
        "elite_vote": elite_vote,
        "anchor_costs": anchor_costs,
        "robust3_costs": robust3_costs,
        "best_masks": best_masks,
        "robust3_masks": robust3_masks,
    }


def evaluate_split(
    samples: Sequence[aco2.SegmentPacket],
    q4_offsets: Sequence[float],
    chirp_shapes,
    chirp_struct,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    split_name: str,
    args: argparse.Namespace,
    leave_one_out_prototypes: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    rng = random.Random(args.seed + {"train_loocv": 0, "val": 10000, "test": 20000}.get(split_name, 0))
    correct = Counter()
    topk_contains = 0
    predictions = []
    candidate_rows = []
    segment_rows = []
    template_cache = {}
    prototype_cache = {}

    def training_state(indices: Sequence[int]):
        key = tuple(indices)
        if key not in template_cache:
            template_cache[key] = aco2.build_templates(samples, labels, indices, chirp_shapes, chirp_struct, args)
            prototype_cache[key] = aco2.build_segment_prototypes(samples, labels, indices)
        return template_cache[key], prototype_cache[key]

    fixed_templates, fixed_prototypes = training_state(train_indices)
    for test_index in eval_indices:
        sample = samples[test_index]
        effective_train = [idx for idx in train_indices if idx != test_index]
        if not effective_train:
            continue
        rssi_ranked = aco2.base.class_rank(rssi_rows, labels, effective_train, test_index, args.rssi_class_k)
        candidates = [label for label, _score in rssi_ranked[: args.top_k]]
        if not candidates:
            continue
        if leave_one_out_prototypes and test_index in train_indices:
            templates, prototypes = training_state(effective_train)
        else:
            templates, prototypes = fixed_templates, fixed_prototypes
        rssi_pred = candidates[0]
        rssi_costs = {label: score for label, score in rssi_ranked if label in candidates}
        anchor_inputs, seg_rows = build_anchor_inputs(
            sample,
            candidates,
            rssi_costs,
            templates,
            prototypes,
            q4_offsets,
            args,
        )
        result = run_anchor_aco_for_packet(anchor_inputs, args, rng)

        topk_contains += int(sample.label in candidates)
        correct["rssi"] += int(rssi_pred == sample.label)
        for key in ["anchor_cost", "robust3", "anchor_pheromone", "anchor_vote"]:
            correct[key] += int(result[f"{key}_label"] == sample.label)

        for row in seg_rows:
            row.update(
                {
                    "split": split_name,
                    "sample_index": test_index,
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                }
            )
            segment_rows.append(row)

        for label in candidates:
            info = anchor_inputs[label]
            candidate_rows.append(
                {
                    "split": split_name,
                    "sample_index": test_index,
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                    "candidate_label": label,
                    "C_R": info["C_R"],
                    "anchor_cost": result["anchor_costs"][label],
                    "anchor_mask": ";".join(str(item) for item in result["best_masks"][label]),
                    "robust3_cost": result["robust3_costs"][label],
                    "robust3_mask": ";".join(str(item) for item in result["robust3_masks"][label]),
                    "anchor_pheromone": result["tau"][label],
                    "elite_anchor_vote": result["elite_vote"][label],
                    "template_reliability": info["template_reliability"],
                    "segment_costs": ";".join(f"{value:.10g}" for value in info["segment_costs"]),
                }
            )

        predictions.append(
            {
                "split": split_name,
                "sample_index": test_index,
                "file_name": sample.file_name,
                "packet_index": sample.packet_index,
                "true_label": sample.label,
                "true_display": aco2.point_display(sample.label),
                "rssi_top1_label": rssi_pred,
                "rssi_top1_correct": int(rssi_pred == sample.label),
                "rssi_topk_candidates": ";".join(candidates),
                "true_in_rssi_topk": int(sample.label in candidates),
                "anchor_cost_label": result["anchor_cost_label"],
                "anchor_cost_correct": int(result["anchor_cost_label"] == sample.label),
                "anchor_cost": result["anchor_cost"],
                "anchor_cost_mask": ";".join(str(item) for item in result["anchor_cost_mask"]),
                "anchor_cost_garbage_count": sum(result["anchor_cost_mask"]),
                "robust3_label": result["robust3_label"],
                "robust3_correct": int(result["robust3_label"] == sample.label),
                "robust3_cost": result["robust3_cost"],
                "robust3_mask": ";".join(str(item) for item in result["robust3_mask"]),
                "anchor_pheromone_label": result["anchor_pheromone_label"],
                "anchor_pheromone_correct": int(result["anchor_pheromone_label"] == sample.label),
                "anchor_vote_label": result["anchor_vote_label"],
                "anchor_vote_correct": int(result["anchor_vote_label"] == sample.label),
                "best_ant_anchor": result["best_ant_anchor"],
                "best_ant_correct": int(result["best_ant_anchor"] == sample.label),
                "best_ant_cost": result["best_ant_cost"],
                "best_ant_mask": ";".join(str(item) for item in result["best_ant_mask"]),
                "best_ant_garbage_count": sum(result["best_ant_mask"]),
            }
        )

    n = len(predictions)
    metrics = {
        "method": "anchor_aco_v3",
        "split": split_name,
        "packet_count": n,
        "location_count": len({labels[idx] for idx in eval_indices}),
        "top_k": args.top_k,
        "segment_count": args.segment_count,
        "max_anchor_garbage": args.max_anchor_garbage,
        "rssi_class_k": args.rssi_class_k,
        "rssi_top1_correct": correct["rssi"],
        "rssi_top1_accuracy": correct["rssi"] / n if n else 0.0,
        "rssi_topk_contains_true": topk_contains,
        "rssi_topk_recall": topk_contains / n if n else 0.0,
        "anchor_cost_correct": correct["anchor_cost"],
        "anchor_cost_accuracy": correct["anchor_cost"] / n if n else 0.0,
        "robust3_correct": correct["robust3"],
        "robust3_accuracy": correct["robust3"] / n if n else 0.0,
        "anchor_pheromone_correct": correct["anchor_pheromone"],
        "anchor_pheromone_accuracy": correct["anchor_pheromone"] / n if n else 0.0,
        "anchor_vote_correct": correct["anchor_vote"],
        "anchor_vote_accuracy": correct["anchor_vote"] / n if n else 0.0,
        "anchor_cost_garbage_mean": sum(row["anchor_cost_garbage_count"] for row in predictions) / n if n else 0.0,
        "best_ant_garbage_mean": sum(row["best_ant_garbage_count"] for row in predictions) / n if n else 0.0,
    }
    return metrics, predictions, candidate_rows, segment_rows


def write_split_outputs(
    output_dir: Path,
    split_name: str,
    metrics: dict,
    predictions: list[dict],
    candidate_rows: list[dict],
    segment_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{split_name}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_csv(output_dir / f"{split_name}_predictions.csv", predictions, list(predictions[0].keys()))
    write_csv(output_dir / f"{split_name}_candidate_scores.csv", candidate_rows, list(candidate_rows[0].keys()))
    preferred = [
        "split",
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "segment_index",
        "candidate_label",
        "C_anchor_seg",
        "C_R",
        "C_bin",
        "C_bin_raw",
        "C_E",
        "C_W",
        "C_Q",
        "q4_reliable",
    ]
    fields = preferred + sorted({key for row in segment_rows for key in row} - set(preferred))
    write_csv(output_dir / f"{split_name}_segment_costs.csv", segment_rows, fields)


def run(args: argparse.Namespace) -> dict:
    rssi_packets = aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, thresholds = aco2.base.read_symbol_packets(args.spectrum_csv, args)
    base_samples = aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = aco2.prepare_chirp_fields(args, labels)

    eval_plan = [
        ("train_loocv", split_indices["train"], split_indices["train"]),
        ("val", split_indices["train"], split_indices["val"]),
        ("test", split_indices["train"], split_indices["test"]),
    ]
    summary_rows = []
    for split_name, train_indices, eval_indices in eval_plan:
        metrics, predictions, candidate_rows, segment_rows = evaluate_split(
            samples,
            q4_offsets,
            chirp_shapes,
            chirp_struct,
            train_indices,
            eval_indices,
            split_name,
            args,
            leave_one_out_prototypes=args.leave_one_out_prototypes,
        )
        metrics.update(thresholds)
        write_split_outputs(args.output_dir, split_name, metrics, predictions, candidate_rows, segment_rows)
        summary_rows.append(metrics)

    write_csv(args.output_dir / "anchor_aco_v3_summary.csv", summary_rows, list(summary_rows[0].keys()))
    metadata = {
        "method": "Anchor-ACO 3.0 on gaussian_noise_1to10_split",
        "source": "external_design_notes/蚁群算法3.0.md",
        "data_policy": "Consumes existing 1:10 Gaussian-noise augmented CSVs and split_assignments.csv; does not regenerate noise or split.",
        "inputs": {
            "rssi_csv": str(args.rssi_csv),
            "spectrum_csv": str(args.spectrum_csv),
            "split_csv": str(args.split_csv),
            "chirp_template_csv": str(args.chirp_template_csv),
            "chirp_structure_csv": str(args.chirp_structure_csv),
            "location_csv": str(args.location_csv),
        },
        "sample_counts": {
            "aligned": len(samples),
            "train": len(split_indices["train"]),
            "val": len(split_indices["val"]),
            "test": len(split_indices["test"]),
            "locations": len(set(labels)),
        },
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "symbol_thresholds": thresholds,
        "chirp_template_field": chirp_metadata,
        "summary": summary_rows,
    }
    with (args.output_dir / "anchor_aco_v3_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rssi-csv", type=Path, default=DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--chirp-template-csv", type=Path, default=aco2.DEFAULT_CHIRP_TEMPLATE_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=aco2.DEFAULT_CHIRP_STRUCT_CSV)
    parser.add_argument("--location-csv", type=Path, default=aco2.DEFAULT_LOCATION_CSV)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--segment-count", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--anchor-rssi-weight", type=float, default=0.45)
    parser.add_argument("--rssi-weight", type=float, default=0.45)
    parser.add_argument("--segment-weight", type=float, default=1.0)
    parser.add_argument("--anchor-garbage-penalty", type=float, default=0.10)
    parser.add_argument("--max-anchor-garbage", type=int, default=1)
    parser.add_argument("--anchor-rssi-beta", type=float, default=1.4)
    parser.add_argument("--segment-support-beta", type=float, default=1.4)
    parser.add_argument("--reliability-bonus", type=float, default=0.15)
    parser.set_defaults(average_segment_cost=True)
    parser.add_argument("--average-segment-cost", dest="average_segment_cost", action="store_true")
    parser.add_argument("--sum-segment-cost", dest="average_segment_cost", action="store_false")
    parser.add_argument("--energy-weight", type=float, default=0.20)
    parser.add_argument("--raw-weight", type=float, default=0.55)
    parser.add_argument("--bin-weight", type=float, default=0.02)
    parser.add_argument("--q4-weight", type=float, default=0.0)
    parser.add_argument("--shrinkage-lambda", type=float, default=8.0)
    parser.add_argument("--phy-var-c0", type=float, default=0.05)
    parser.add_argument("--phy-var-c1", type=float, default=0.50)
    parser.add_argument("--phy-var-c2", type=float, default=1.0)
    parser.add_argument("--sigma0-sq", type=float, default=0.02)
    parser.add_argument("--min-variance", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--logdet-weight", type=float, default=0.05)
    parser.set_defaults(normalize_bin_cost=True)
    parser.add_argument("--normalize-bin-cost", dest="normalize_bin_cost", action="store_true")
    parser.add_argument("--raw-bin-cost", dest="normalize_bin_cost", action="store_false")
    parser.add_argument("--garbage-cost", type=float, default=0.60)
    parser.add_argument("--pheromone-power", type=float, default=1.0)
    parser.add_argument("--evaporation", type=float, default=0.25)
    parser.add_argument("--min-pheromone", type=float, default=1e-4)
    parser.add_argument("--aco-temperature", type=float, default=None)
    parser.add_argument("--q4-shift-grid", default="-0.25,0,0.25")
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    parser.add_argument("--leave-one-out-prototypes", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = run(parse_args())
    print(json.dumps(metadata["sample_counts"], indent=2, ensure_ascii=False))
    for row in metadata["summary"]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"Wrote {metadata['args']['output_dir']}")


if __name__ == "__main__":
    main()
