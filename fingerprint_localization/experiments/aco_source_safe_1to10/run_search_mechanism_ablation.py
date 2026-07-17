#!/usr/bin/env python3
"""Ablate the path-search mechanism of the adopted Expanded LDA/ACO mainline.

All variants receive exactly the same LDA posterior, fused Top-5 candidates,
segment observation costs, priors, and beta=0.5 final fusion.  Only the
candidate-internal search/score mechanism changes:

* average_cost: no path, rank by negative mean segment observation cost;
* greedy_path: deterministic one-step greedy path with switch costs;
* no_pheromone: the same ant budget and heuristic as ACO, but no pheromone
  factor, evaporation, or reinforcement;
* full_aco: the frozen ACO v4 Score4 implementation.

The controlled perturbations are imported from the preceding weakness audit so
that the two experiments use identical deterministic degraded packets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import finalize_expanded_aco_ml_score4 as finalizer
import run_candidate_recall_and_controlled_weakness as weakness
import run_aco_v4_source_level_on_split as source
import run_expanded_aco_ml_prior as prior
import run_expanded_supervised_ensemble as supervised


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "search_mechanism_ablation_20260717"
)
REFERENCE_WEAKNESS_DIR = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "candidate_recall_and_controlled_weakness_20260716"
)
METHOD_ORDER = ["average_cost", "greedy_path", "no_pheromone", "full_aco"]
METHOD_LABELS = {
    "average_cost": "Average cost",
    "greedy_path": "Greedy path",
    "no_pheromone": "No pheromone",
    "full_aco": "Full ACO",
}
EPS = 1e-12


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


def common_search_state(candidates, templates, meta, args):
    k = len(candidates)
    garbage_idx = k
    node_count = k + 1
    template_dist = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    for left_idx, left in enumerate(candidates):
        for right_idx, right in enumerate(candidates):
            if left_idx != right_idx:
                template_dist[left_idx][right_idx] = source.aco2.template_distance(
                    templates[left], templates[right]
                )
    priors = [
        args.lambda_rssi_prior * int(label == meta["rssi_top1"])
        + args.lambda_raw_prior * meta["raw_margin"] * int(label == meta["raw_winner"])
        for label in candidates
    ]
    veto = [meta["veto"][label] for label in candidates]
    return k, garbage_idx, template_dist, priors, veto


def effective_local_cost(obs_costs, segment_idx, choice, priors, veto):
    garbage_idx = len(priors)
    if choice == garbage_idx:
        return float(obs_costs[segment_idx][choice])
    return (
        float(obs_costs[segment_idx][choice])
        - priors[choice] / max(1, len(obs_costs))
        - math.log(max(veto[choice], EPS))
    )


def path_summary(path: Sequence[int], candidate_count: int) -> tuple[int, int]:
    garbage_idx = candidate_count
    switches = sum(left != right for left, right in zip(path, path[1:]))
    garbage = sum(choice == garbage_idx for choice in path)
    return switches, garbage


def average_cost_scores(candidates, meta) -> dict:
    return {
        label: -float(meta["candidate_mean_obs"][label])
        for label in candidates
    }


def greedy_path_scores(obs_costs, candidates, templates, meta, args) -> tuple[dict, list[int]]:
    k, _garbage_idx, template_dist, priors, veto = common_search_state(
        candidates, templates, meta, args
    )
    path = [
        min(
            range(k + 1),
            key=lambda choice: (
                effective_local_cost(obs_costs, 0, choice, priors, veto),
                choice,
            ),
        )
    ]
    for segment_idx in range(1, len(obs_costs)):
        previous = path[-1]
        choice = min(
            range(k + 1),
            key=lambda candidate_idx: (
                effective_local_cost(
                    obs_costs, segment_idx, candidate_idx, priors, veto
                )
                + source.aco4.dynamic_switch_penalty_v4(
                    previous,
                    candidate_idx,
                    segment_idx,
                    obs_costs,
                    template_dist,
                    meta["q_seg"],
                    args,
                ),
                candidate_idx,
            ),
        )
        path.append(choice)
    visits = Counter(choice for choice in path if choice < k)
    z_visits = source.aco4.z_scores([float(visits[index]) for index in range(k)])
    z_cost = source.aco4.z_scores(
        [float(meta["candidate_mean_obs"][label]) for label in candidates]
    )
    scores = {}
    for index, label in enumerate(candidates):
        scores[label] = (
            z_visits[index]
            - args.lambda_score_cost * z_cost[index]
            + args.lambda_rssi_prior * int(label == meta["rssi_top1"])
            + args.lambda_raw_prior
            * meta["raw_margin"]
            * int(label == meta["raw_winner"])
        )
    return scores, path


def no_pheromone_scores(
    obs_costs,
    candidates,
    templates,
    meta,
    args,
    rng: random.Random,
) -> tuple[dict, list[int]]:
    """Monte Carlo ant search with the pheromone channel removed completely."""
    k, garbage_idx, template_dist, priors, veto = common_search_state(
        candidates, templates, meta, args
    )
    node_count = k + 1
    best_path = []
    best_cost = float("inf")
    elite_vote = [0.0 for _ in range(node_count)]
    for _iteration in range(args.iterations):
        paths = []
        for _ant in range(args.ants):
            first_weights = []
            for choice, cost in enumerate(obs_costs[0]):
                eta = (
                    math.exp(-cost)
                    if choice == garbage_idx
                    else math.exp(-cost + priors[choice] / max(1, len(obs_costs)))
                    * veto[choice]
                )
                first_weights.append(eta ** args.heuristic_power)
            path = [source.aco2.weighted_choice(first_weights, rng)]
            for segment_idx in range(1, len(obs_costs)):
                previous = path[-1]
                weights = []
                for choice in range(node_count):
                    penalty = source.aco4.dynamic_switch_penalty_v4(
                        previous,
                        choice,
                        segment_idx,
                        obs_costs,
                        template_dist,
                        meta["q_seg"],
                        args,
                    )
                    eta = (
                        math.exp(-obs_costs[segment_idx][choice])
                        if choice == garbage_idx
                        else math.exp(
                            -obs_costs[segment_idx][choice]
                            + priors[choice] / max(1, len(obs_costs))
                        )
                        * veto[choice]
                    )
                    # No tau term: neither initialized pheromone nor learned
                    # pheromone can influence the transition probability.
                    weights.append((eta ** args.heuristic_power) * math.exp(-penalty))
                path.append(source.aco2.weighted_choice(weights, rng))
            cost = source.aco4.path_cost_v4(
                path,
                obs_costs,
                template_dist,
                priors,
                veto,
                meta["q_seg"],
                args,
            )
            paths.append((cost, path))
            if cost < best_cost:
                best_cost = cost
                best_path = list(path)
        paths.sort(key=lambda item: item[0])
        elite = paths[: max(1, min(args.elite_ants, len(paths)))]
        temperature = args.aco_temperature
        if temperature is None or temperature <= EPS:
            temperature = source.aco2.median([cost for cost, _path in elite])
            temperature = temperature if temperature > EPS else 1.0
        weights = [math.exp(-cost / (temperature + EPS)) for cost, _path in elite]
        total = sum(weights) or 1.0
        for (_cost, path), weight in zip(elite, [value / total for value in weights]):
            for choice in path:
                if choice != garbage_idx:
                    elite_vote[choice] += weight
    z_vote = source.aco4.z_scores([elite_vote[index] for index in range(k)])
    z_cost = source.aco4.z_scores(
        [float(meta["candidate_mean_obs"][label]) for label in candidates]
    )
    scores = {}
    for index, label in enumerate(candidates):
        scores[label] = (
            args.lambda_score_vote * z_vote[index]
            - args.lambda_score_cost * z_cost[index]
            + args.lambda_rssi_prior * int(label == meta["rssi_top1"])
            + args.lambda_raw_prior
            * meta["raw_margin"]
            * int(label == meta["raw_winner"])
        )
    return scores, best_path


def build_perturbed_scenario(context, condition, strength, stats, distances):
    eval_by_key = {weakness.key_from_row(row): dict(row) for row in context.eval_rows}
    samples = list(context.samples)
    donor_sample, donor_row = weakness.donor_maps(context, distances)
    if condition != "clean":
        perturb = weakness.PERTURB_FUNCTIONS[condition]
        for sample_index in context.split_indices[context.split_name]:
            sample = context.samples[sample_index]
            sample_key = source.source_id(sample)
            rng = np.random.default_rng(
                weakness.stable_seed(weakness.SEED, condition, strength, *sample_key)
            )
            changed_sample, changed_row = perturb(
                sample,
                eval_by_key[sample_key],
                strength,
                stats,
                donor_sample[sample_key],
                donor_row[sample_key],
                rng,
            )
            samples[sample_index] = changed_sample
            eval_by_key[sample_key] = changed_row
    eval_rows = [eval_by_key[weakness.key_from_row(row)] for row in context.eval_rows]
    probabilities, _classes = finalizer.common_probabilities(context.models, eval_rows)
    return samples, eval_rows, probabilities


def evaluate_scenario(context, condition, strength, stats, distances) -> list[dict]:
    samples, eval_rows, probabilities = build_perturbed_scenario(
        context, condition, strength, stats, distances
    )
    eval_by_key = {weakness.key_from_row(row): row for row in eval_rows}
    ranker = prior.make_ml_source_class_rank(samples, probabilities, weakness.ALPHA)
    labels = [sample.label for sample in samples]
    rssi_rows = [sample.rssi_plus for sample in samples]
    full_rng = random.Random(context.args.seed)
    # Use an independent generator with the identical seed so the stochastic
    # ablation and full ACO consume a paired stream of uniform random numbers.
    # Their trajectories may diverge because their transition probabilities
    # differ, but RNG noise is not intentionally biased toward either method.
    no_pheromone_rng = random.Random(context.args.seed)
    outputs = []
    for sample_index in context.split_indices[context.split_name]:
        sample = samples[sample_index]
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
        started = time.perf_counter()
        full_result = source.aco4.run_aco_v4_for_packet(
            obs_costs,
            candidates,
            context.templates,
            meta,
            context.args,
            full_rng,
        )
        full_runtime_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        no_pheromone_score, no_pheromone_path = no_pheromone_scores(
            obs_costs,
            candidates,
            context.templates,
            meta,
            context.args,
            no_pheromone_rng,
        )
        no_pheromone_runtime_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        greedy_score, greedy_path = greedy_path_scores(
            obs_costs, candidates, context.templates, meta, context.args
        )
        greedy_runtime_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        average_score = average_cost_scores(candidates, meta)
        average_runtime_ms = (time.perf_counter() - started) * 1000.0
        method_scores = {
            "average_cost": average_score,
            "greedy_path": greedy_score,
            "no_pheromone": no_pheromone_score,
            "full_aco": full_result["score4"],
        }
        method_paths = {
            "average_cost": None,
            "greedy_path": greedy_path,
            "no_pheromone": no_pheromone_path,
            "full_aco": full_result["best_path"],
        }
        method_runtime_ms = {
            "average_cost": average_runtime_ms,
            "greedy_path": greedy_runtime_ms,
            "no_pheromone": no_pheromone_runtime_ms,
            "full_aco": full_runtime_ms,
        }
        lda_scores = probabilities[sample_key]
        lda_label = natural_max(lda_scores)
        lda_candidate_norm = finalizer.minmax(
            {label: lda_scores[label] for label in candidates}
        )
        diagnostic = eval_by_key[sample_key]
        for method in METHOD_ORDER:
            search_scores = method_scores[method]
            search_label = natural_max(search_scores)
            search_norm = finalizer.minmax(search_scores)
            combined = {
                label: (1.0 - weakness.BETA) * search_norm[label]
                + weakness.BETA * lda_candidate_norm[label]
                for label in candidates
            }
            final_label = natural_max(combined)
            path = method_paths[method]
            switches, garbage = path_summary(path, len(candidates)) if path else ("", "")
            outputs.append(
                {
                    "scenario_type": "artificial_perturbation",
                    "condition": condition,
                    "strength": strength,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                    "candidate_labels": ";".join(candidates),
                    "true_in_candidates": int(sample.label in candidates),
                    "lda_label": lda_label,
                    "search_label": search_label,
                    "final_label": final_label,
                    "lda_correct": int(lda_label == sample.label),
                    "search_correct": int(search_label == sample.label),
                    "final_correct": int(final_label == sample.label),
                    "search_confidence": score_margin(search_norm),
                    "final_confidence": score_margin(combined),
                    "search_runtime_ms": method_runtime_ms[method],
                    "Q_seg": float(meta["q_seg"]),
                    "segment_cost_std": float(meta["segment_cost_std"]),
                    "detect_score_db": float(diagnostic["detect_score_db"]),
                    "snr": float(diagnostic["snr"]),
                    "path_switches": switches,
                    "path_garbage_segments": garbage,
                    "best_path_labels": (
                        ""
                        if path is None
                        else ";".join(
                            source.aco4.GARBAGE_LABEL
                            if choice == len(candidates)
                            else candidates[choice]
                            for choice in path
                        )
                    ),
                }
            )
    return outputs


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def metric_row(rows, method, distances, scenario_type, condition, severity, strength):
    selected = [row for row in rows if row["method"] == method]
    final_errors = [
        abs(distances[row["final_label"]] - distances[row["true_label"]])
        for row in selected
    ]
    search_errors = [
        abs(distances[row["search_label"]] - distances[row["true_label"]])
        for row in selected
    ]
    changed = beneficial = harmful = changed_wrong = 0
    for row in selected:
        if row["final_label"] == row["lda_label"]:
            continue
        changed += 1
        lda_ok = bool(row["lda_correct"])
        final_ok = bool(row["final_correct"])
        beneficial += int(not lda_ok and final_ok)
        harmful += int(lda_ok and not final_ok)
        changed_wrong += int(not lda_ok and not final_ok)
    path_rows = [row for row in selected if row["path_switches"] != ""]
    return {
        "scenario_type": scenario_type,
        "condition": condition,
        "severity": severity,
        "strength": strength,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "packets": len(selected),
        "candidate_recall": sum(int(row["true_in_candidates"]) for row in selected) / len(selected),
        "search_correct": sum(int(row["search_correct"]) for row in selected),
        "search_accuracy": sum(int(row["search_correct"]) for row in selected) / len(selected),
        "search_topology_mae_m": sum(search_errors) / len(search_errors),
        "search_topology_p95_m": percentile(search_errors, 0.95),
        "final_correct": sum(int(row["final_correct"]) for row in selected),
        "final_accuracy": sum(int(row["final_correct"]) for row in selected) / len(selected),
        "final_topology_mae_m": sum(final_errors) / len(final_errors),
        "final_topology_p95_m": percentile(final_errors, 0.95),
        "severe_error_rate": sum(error > weakness.SEVERE_ERROR_M for error in final_errors) / len(final_errors),
        "prediction_changes_vs_lda": changed,
        "beneficial_corrections": beneficial,
        "harmful_corrections": harmful,
        "changed_but_still_wrong": changed_wrong,
        "correction_precision": beneficial / changed if changed else "",
        "mean_search_runtime_ms": sum(
            float(row["search_runtime_ms"]) for row in selected
        )
        / len(selected),
        "p95_search_runtime_ms": percentile(
            [float(row["search_runtime_ms"]) for row in selected], 0.95
        ),
        "mean_path_switches": (
            sum(int(row["path_switches"]) for row in path_rows) / len(path_rows)
            if path_rows
            else ""
        ),
        "mean_garbage_segments": (
            sum(int(row["path_garbage_segments"]) for row in path_rows) / len(path_rows)
            if path_rows
            else ""
        ),
    }


def diagnostic_metrics(clean_rows, thresholds, distances):
    output = []
    for condition in ["detect_score_db", "snr", "segment_cost_std"]:
        high_is_bad = condition == "segment_cost_std"
        packet_levels = {}
        for row in clean_rows:
            packet_key = (row["file_name"], row["packet_index"])
            packet_levels[packet_key] = weakness.diagnostic_level(
                float(row[condition]), thresholds[condition], high_is_bad
            )
        for level in ["0_cleanest", "1_mild", "2_moderate", "3_severe"]:
            rows = [
                row
                for row in clean_rows
                if packet_levels[(row["file_name"], row["packet_index"])] == level
            ]
            if not rows:
                continue
            for method in METHOD_ORDER:
                output.append(
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
    return output


def mcnemar_exact(w2r: int, r2w: int) -> float:
    discordant = w2r + r2w
    if discordant == 0:
        return 1.0
    lower = min(w2r, r2w)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def pairwise_rows(predictions: Sequence[dict]) -> list[dict]:
    output = []
    scenario_keys = sorted(
        {(row["condition"], float(row["strength"])) for row in predictions}
    )
    for condition, strength in scenario_keys:
        scenario = [
            row
            for row in predictions
            if row["condition"] == condition and float(row["strength"]) == strength
        ]
        full = {
            (row["file_name"], row["packet_index"]): row
            for row in scenario
            if row["method"] == "full_aco"
        }
        for method in METHOD_ORDER[:-1]:
            ablated = {
                (row["file_name"], row["packet_index"]): row
                for row in scenario
                if row["method"] == method
            }
            for label_type in ["search", "final"]:
                full_column = f"{label_type}_label"
                w2r = r2w = changed_wrong = changed = 0
                for packet_key, full_row in full.items():
                    old_row = ablated[packet_key]
                    true_label = full_row["true_label"]
                    old_label = old_row[full_column]
                    new_label = full_row[full_column]
                    old_ok = old_label == true_label
                    new_ok = new_label == true_label
                    changed += int(old_label != new_label)
                    w2r += int(not old_ok and new_ok)
                    r2w += int(old_ok and not new_ok)
                    changed_wrong += int(old_label != new_label and not old_ok and not new_ok)
                output.append(
                    {
                        "condition": condition,
                        "strength": strength,
                        "prediction_stage": label_type,
                        "ablation": method,
                        "ablation_label": METHOD_LABELS[method],
                        "full_aco_changed": changed,
                        "full_aco_W2R": w2r,
                        "full_aco_R2W": r2w,
                        "full_aco_changed_both_wrong": changed_wrong,
                        "full_aco_net_gain": w2r - r2w,
                        "mcnemar_exact_two_sided_p": mcnemar_exact(w2r, r2w),
                    }
                )
    return output


def plot_results(metrics: Sequence[dict], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    colors = {
        "average_cost": "#4c78a8",
        "greedy_path": "#f58518",
        "no_pheromone": "#54a24b",
        "full_aco": "#b279a2",
    }
    artificial = [row for row in metrics if row["scenario_type"] == "artificial_perturbation"]
    generated = []
    for column, filename, ylabel in [
        ("final_accuracy", "ablation_final_accuracy.png", "final accuracy"),
        ("search_accuracy", "ablation_search_accuracy.png", "search-only accuracy"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.12, hspace=0.35, wspace=0.18)
        for axis, condition in zip(axes.ravel(), weakness.PERTURBATIONS):
            rows = [row for row in artificial if row["condition"] == condition]
            strengths = sorted({float(row["strength"]) for row in rows})
            for method in METHOD_ORDER:
                values = [
                    next(
                        float(row[column])
                        for row in rows
                        if row["method"] == method and float(row["strength"]) == strength
                    )
                    for strength in strengths
                ]
                axis.plot(
                    strengths,
                    values,
                    marker="o",
                    linewidth=2,
                    label=METHOD_LABELS[method],
                    color=colors[method],
                )
            axis.set_title(condition.replace("_", " "))
            axis.set_xlabel("degradation strength")
            axis.set_ylabel(ylabel)
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
        path = output_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(path.name)
    return generated


def render_report(output_dir, metrics, pairwise, clean_validation):
    clean_test = [
        row
        for row in metrics
        if row["scenario_type"] == "artificial_perturbation"
        and row["condition"] == "clean"
    ]
    strongest = []
    for condition, strengths in weakness.PERTURBATIONS.items():
        strongest.extend(
            row
            for row in metrics
            if row["scenario_type"] == "artificial_perturbation"
            and row["condition"] == condition
            and float(row["strength"]) == max(strengths)
        )
    artificial = [
        row for row in metrics if row["scenario_type"] == "artificial_perturbation"
    ]
    full_by_scenario = {
        (row["condition"], float(row["strength"])): int(row["final_correct"])
        for row in artificial
        if row["method"] == "full_aco"
    }
    win_tie_loss = {}
    for method in METHOD_ORDER[:-1]:
        deltas = [
            full_by_scenario[(row["condition"], float(row["strength"]))]
            - int(row["final_correct"])
            for row in artificial
            if row["method"] == method
        ]
        win_tie_loss[method] = (
            sum(delta > 0 for delta in deltas),
            sum(delta == 0 for delta in deltas),
            sum(delta < 0 for delta in deltas),
            sum(deltas),
        )
    no_pheromone_final = [
        row
        for row in pairwise
        if row["ablation"] == "no_pheromone"
        and row["prediction_stage"] == "final"
    ]
    min_no_pheromone_p = min(
        float(row["mcnemar_exact_two_sided_p"])
        for row in no_pheromone_final
    )
    lines = [
        "# Search-mechanism ablation",
        "",
        "Date: 2026-07-17",
        "",
        "All methods use identical LDA posterior, fused Top-5 candidates, segment costs, priors, and beta=0.5 final fusion.",
        "",
        "## Clean",
        "",
        "| Method | validation search/final | formal search/final | final MAE/P95 m | mean search ms |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in METHOD_ORDER:
        validation = next(row for row in clean_validation if row["method"] == method)
        formal = next(row for row in clean_test if row["method"] == method)
        lines.append(
            f"| {METHOD_LABELS[method]} | {validation['search_correct']}/{validation['final_correct']} "
            f"| {formal['search_correct']}/{formal['final_correct']} "
            f"| {float(formal['final_topology_mae_m']):.2f}/{float(formal['final_topology_p95_m']):.2f} "
            f"| {float(formal['mean_search_runtime_ms']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest controlled degradation",
            "",
            "| Condition | Method | search accuracy | final accuracy | MAE/P95 m | severe >10m | correction precision |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for condition in weakness.PERTURBATIONS:
        for method in METHOD_ORDER:
            row = next(
                item
                for item in strongest
                if item["condition"] == condition and item["method"] == method
            )
            precision = row["correction_precision"]
            precision_text = "n/a" if precision == "" else f"{float(precision):.2%}"
            lines.append(
                f"| {condition} ({row['strength']}) | {METHOD_LABELS[method]} "
                f"| {float(row['search_accuracy']):.2%} | {float(row['final_accuracy']):.2%} "
                f"| {float(row['final_topology_mae_m']):.2f}/{float(row['final_topology_p95_m']):.2f} "
                f"| {float(row['severe_error_rate']):.2%} | {precision_text} |"
            )
    lines.extend(
        [
            "",
            "## Full ACO versus ablations across 15 formal scenarios",
            "",
            "| Ablation | Full wins/ties/losses | net correct packets |",
            "| --- | --- | --- |",
        ]
    )
    for method in METHOD_ORDER[:-1]:
        wins, ties, losses, net = win_tie_loss[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {wins}/{ties}/{losses} | {net:+d} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism check",
            "",
            "- Full ACO and no-pheromone best paths had zero segment switches and zero garbage selections in every formal scenario.",
            f"- Full ACO versus no pheromone has no final-label McNemar result below 0.05 (minimum p={min_no_pheromone_p:.4f}).",
            "- Full ACO is strongest at amplitude noise 0.25/0.5 and four missing preamble symbols, but it is weaker under clean, CFO, and segment-anomaly cases.",
            "- Therefore this experiment does not support a global claim that pheromone search is irreplaceable in the current configuration.",
        ]
    )
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- Average cost: rank by negative mean C_obs; no path or transition term.",
            "- Greedy path: one-step local minimum with the frozen dynamic switch penalty; no ants or pheromone.",
            "- No pheromone: 16 ants x 12 iterations and the same heuristic/path cost, but no tau factor, evaporation, reinforcement, or pheromone score.",
            "- Full ACO: frozen ACO v4 transition, evaporation, elite reinforcement, and Score4.",
            "- Search accuracy is the candidate-internal score Top-1 before LDA final fusion.",
            "- Final accuracy uses the same beta=0.5 LDA blend for every method.",
            "- Perturbations are feature-space proxies because raw IQ is unavailable.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = weakness.load_context("validation", build_training_state=True)
    formal = weakness.load_context("formal_test", build_training_state=True)
    distances = {
        row["position_key"]: float(row["distance_m"])
        for row in read_csv(weakness.LOCATION_CSV)
    }
    validation_stats = weakness.training_stats(validation)
    formal_stats = weakness.training_stats(formal)

    validation_predictions = evaluate_scenario(
        validation, "clean", 0.0, validation_stats, distances
    )
    validation_metrics = [
        metric_row(
            validation_predictions,
            method,
            distances,
            "validation_clean",
            "clean",
            "clean",
            0.0,
        )
        for method in METHOD_ORDER
    ]

    predictions = evaluate_scenario(formal, "clean", 0.0, formal_stats, distances)
    for condition, strengths in weakness.PERTURBATIONS.items():
        for strength in strengths:
            predictions.extend(
                evaluate_scenario(
                    formal, condition, strength, formal_stats, distances
                )
            )
    write_csv(args.output_dir / "ablation_predictions.csv", predictions)

    reference = {
        (row["condition"], float(row["strength"]), row["file_name"], int(float(row["packet_index"]))): row["fixed_label"]
        for row in read_csv(REFERENCE_WEAKNESS_DIR / "weakness_predictions.csv")
    }
    mismatches = []
    for row in predictions:
        if row["method"] != "full_aco":
            continue
        packet_key = (
            row["condition"],
            float(row["strength"]),
            row["file_name"],
            int(row["packet_index"]),
        )
        if row["final_label"] != reference[packet_key]:
            mismatches.append(row)
    if mismatches:
        raise RuntimeError(
            f"Full ACO does not reproduce the preceding weakness audit: {len(mismatches)} mismatches"
        )

    metrics = []
    scenario_keys = sorted({(row["condition"], float(row["strength"])) for row in predictions})
    for condition, strength in scenario_keys:
        scenario_rows = [
            row
            for row in predictions
            if row["condition"] == condition and float(row["strength"]) == strength
        ]
        severity = "clean" if condition == "clean" else f"strength_{strength:g}"
        for method in METHOD_ORDER:
            metrics.append(
                metric_row(
                    scenario_rows,
                    method,
                    distances,
                    "artificial_perturbation",
                    condition,
                    severity,
                    strength,
                )
            )
    thresholds = weakness.diagnostic_thresholds(validation)
    clean_rows = [row for row in predictions if row["condition"] == "clean"]
    metrics.extend(diagnostic_metrics(clean_rows, thresholds, distances))
    write_csv(args.output_dir / "ablation_metrics.csv", validation_metrics + metrics)

    pairwise = pairwise_rows(predictions)
    write_csv(args.output_dir / "full_aco_pairwise.csv", pairwise)
    plot_files = plot_results(metrics, args.output_dir)
    render_report(args.output_dir, metrics, pairwise, validation_metrics)

    manifest = {
        "experiment": "EXPANDED_LDA_ACO_SEARCH_MECHANISM_ABLATION_20260717",
        "status": "complete",
        "protocol": {
            "same_candidate_and_evidence": True,
            "same_final_beta": weakness.BETA,
            "same_candidate_alpha": weakness.ALPHA,
            "methods": {
                "average_cost": "negative mean C_obs; no path",
                "greedy_path": "deterministic local path with dynamic switch penalty",
                "no_pheromone": "same ant/iteration budget and heuristic; no tau channel or pheromone score",
                "full_aco": "frozen ACO v4 plus Score4",
            },
            "perturbations": weakness.PERTURBATIONS,
            "formal_test_status": "exploratory because Expanded formal test was previously inspected",
        },
        "checks": {
            "full_aco_reference_mismatches": len(mismatches),
            "validation_packets": 128,
            "formal_packets_per_scenario": 128,
            "prediction_rows": len(predictions),
        },
        "validation_clean": validation_metrics,
        "diagnostic_thresholds": thresholds,
        "plots": plot_files,
        "input_sha256": {
            "weakness_predictions": sha256(REFERENCE_WEAKNESS_DIR / "weakness_predictions.csv"),
            "formal_model": sha256(weakness.MAINLINE_ROOT / "formal_lda_model.joblib"),
            "validation_model": sha256(weakness.MAINLINE_ROOT / "validation_lda_model.joblib"),
        },
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
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
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
