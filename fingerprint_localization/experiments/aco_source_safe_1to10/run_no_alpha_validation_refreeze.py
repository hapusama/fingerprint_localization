#!/usr/bin/env python3
"""Remove alpha candidate fusion, re-freeze beta on validation, then run formal.

The candidate set is the direct LDA Top-5. RSSI is retained only as an ACO
observation cost and weak prior; it is never blended with LDA to rank
candidates. The full-ACO beta is selected on validation (maximum accuracy,
smaller beta on ties), frozen, and then applied unchanged to formal test and to
the search-mechanism ablations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import finalize_expanded_aco_ml_score4 as finalizer
import run_candidate_recall_and_controlled_weakness as weakness
import run_search_mechanism_ablation as search_ablation
import run_aco_v4_source_level_on_split as source
import run_expanded_supervised_ensemble as supervised


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "aco_lda_only_no_alpha_refrozen_20260717"
)
METHOD_ORDER = search_ablation.METHOD_ORDER
METHOD_LABELS = search_ablation.METHOD_LABELS
BETA_GRID = [round(index / 10.0, 1) for index in range(11)]
SEED = weakness.SEED
EPS = 1e-12


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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lda_rank(probabilities: dict[str, float]) -> list[str]:
    return sorted(
        probabilities,
        key=lambda label: (
            -float(probabilities[label]),
            supervised.natural_label_key(label),
        ),
    )


def margin(scores: dict[str, float]) -> float:
    values = sorted(scores.values(), reverse=True)
    return float(values[0] - values[1]) if len(values) > 1 else 1.0


def path_counts(path, candidate_count: int) -> tuple[int | str, int | str]:
    if path is None:
        return "", ""
    return search_ablation.path_summary(path, candidate_count)


def evaluate_search(context) -> list[dict]:
    """Evaluate all search mechanisms with direct LDA Top-5 candidates."""
    labels = [sample.label for sample in context.samples]
    rssi_rows = [sample.rssi_plus for sample in context.samples]
    full_rng = random.Random(context.args.seed)
    no_pheromone_rng = random.Random(context.args.seed)
    rows = []
    for sample_index in context.split_indices[context.split_name]:
        sample = context.samples[sample_index]
        sample_key = source.source_id(sample)
        lda_probabilities = context.probabilities[sample_key]
        ranked_lda = lda_rank(lda_probabilities)
        candidates = ranked_lda[: context.args.top_k]
        ranked_rssi = context.rssi_ranker(
            rssi_rows,
            labels,
            context.split_indices["train"],
            sample_index,
            context.args.rssi_class_k,
        )
        rssi_cost_by_label = {label: float(cost) for label, cost in ranked_rssi}
        if any(label not in rssi_cost_by_label for label in candidates):
            raise RuntimeError("An LDA candidate is absent from the RSSI class ranking")
        rssi_costs = {label: rssi_cost_by_label[label] for label in candidates}
        obs_costs, _segment_rows, meta = source.aco4.build_observation_costs_v4(
            sample,
            candidates,
            rssi_costs,
            context.templates,
            context.prototypes,
            context.q4_offsets,
            context.args,
        )
        # build_observation_costs_v4 historically assumes candidates[0] is RSSI
        # Top-1. With LDA candidates, restore the actual RSSI Top-1 explicitly.
        actual_rssi_top1 = ranked_rssi[0][0]
        meta["rssi_top1"] = actual_rssi_top1

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
        no_pheromone_scores, no_pheromone_path = (
            search_ablation.no_pheromone_scores(
                obs_costs,
                candidates,
                context.templates,
                meta,
                context.args,
                no_pheromone_rng,
            )
        )
        no_pheromone_runtime_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        greedy_scores, greedy_path = search_ablation.greedy_path_scores(
            obs_costs,
            candidates,
            context.templates,
            meta,
            context.args,
        )
        greedy_runtime_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        average_scores = search_ablation.average_cost_scores(candidates, meta)
        average_runtime_ms = (time.perf_counter() - started) * 1000.0

        method_scores = {
            "average_cost": average_scores,
            "greedy_path": greedy_scores,
            "no_pheromone": no_pheromone_scores,
            "full_aco": full_result["score4"],
        }
        method_paths = {
            "average_cost": None,
            "greedy_path": greedy_path,
            "no_pheromone": no_pheromone_path,
            "full_aco": full_result["best_path"],
        }
        method_runtime = {
            "average_cost": average_runtime_ms,
            "greedy_path": greedy_runtime_ms,
            "no_pheromone": no_pheromone_runtime_ms,
            "full_aco": full_runtime_ms,
        }
        lda_candidate_norm = finalizer.minmax(
            {label: float(lda_probabilities[label]) for label in candidates}
        )
        lda_label = ranked_lda[0]
        for method in METHOD_ORDER:
            scores = method_scores[method]
            score_norm = finalizer.minmax(scores)
            path = method_paths[method]
            switches, garbage = path_counts(path, len(candidates))
            rows.append(
                {
                    "split": context.name,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "file_name": sample.file_name,
                    "packet_index": sample.packet_index,
                    "true_label": sample.label,
                    "candidate_policy": "direct_lda_top5",
                    "candidate_labels": ";".join(candidates),
                    "true_in_candidates": int(sample.label in candidates),
                    "lda_label": lda_label,
                    "lda_correct": int(lda_label == sample.label),
                    "rssi_top1_label": actual_rssi_top1,
                    "search_label": search_ablation.natural_max(scores),
                    "search_correct": int(
                        search_ablation.natural_max(scores) == sample.label
                    ),
                    "search_margin": margin(score_norm),
                    "search_runtime_ms": method_runtime[method],
                    "path_switches": switches,
                    "path_garbage_segments": garbage,
                    "search_scores_json": json.dumps(
                        score_norm, ensure_ascii=False, sort_keys=True
                    ),
                    "lda_candidate_scores_json": json.dumps(
                        lda_candidate_norm, ensure_ascii=False, sort_keys=True
                    ),
                    "_search_scores": score_norm,
                    "_lda_candidate_scores": lda_candidate_norm,
                }
            )
    return rows


def apply_beta(rows: Sequence[dict], beta: float) -> list[dict]:
    output = []
    for row in rows:
        combined = {
            label: (1.0 - beta) * row["_search_scores"][label]
            + beta * row["_lda_candidate_scores"][label]
            for label in row["_search_scores"]
        }
        final_label = search_ablation.natural_max(combined)
        rendered = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        rendered.update(
            {
                "frozen_beta": beta,
                "final_label": final_label,
                "final_correct": int(final_label == row["true_label"]),
                "final_margin": margin(combined),
                "final_scores_json": json.dumps(
                    combined, ensure_ascii=False, sort_keys=True
                ),
            }
        )
        output.append(rendered)
    return output


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def metric_row(rows: Sequence[dict], method: str, distances: dict[str, float]) -> dict:
    selected = [row for row in rows if row["method"] == method]
    search_errors = [
        abs(distances[row["search_label"]] - distances[row["true_label"]])
        for row in selected
    ]
    final_errors = [
        abs(distances[row["final_label"]] - distances[row["true_label"]])
        for row in selected
    ]
    changed = [row for row in selected if row["final_label"] != row["lda_label"]]
    beneficial = sum(
        row["lda_label"] != row["true_label"]
        and row["final_label"] == row["true_label"]
        for row in changed
    )
    harmful = sum(
        row["lda_label"] == row["true_label"]
        and row["final_label"] != row["true_label"]
        for row in changed
    )
    path_rows = [row for row in selected if row["path_switches"] != ""]
    return {
        "split": selected[0]["split"],
        "method": method,
        "method_label": METHOD_LABELS[method],
        "packets": len(selected),
        "candidate_recall": sum(int(row["true_in_candidates"]) for row in selected)
        / len(selected),
        "lda_correct": sum(int(row["lda_correct"]) for row in selected),
        "search_correct": sum(int(row["search_correct"]) for row in selected),
        "search_accuracy": sum(int(row["search_correct"]) for row in selected)
        / len(selected),
        "search_mae_m": sum(search_errors) / len(search_errors),
        "search_p95_m": percentile(search_errors, 0.95),
        "final_correct": sum(int(row["final_correct"]) for row in selected),
        "final_accuracy": sum(int(row["final_correct"]) for row in selected)
        / len(selected),
        "final_mae_m": sum(final_errors) / len(final_errors),
        "final_p95_m": percentile(final_errors, 0.95),
        "severe_error_rate": sum(error > weakness.SEVERE_ERROR_M for error in final_errors)
        / len(final_errors),
        "changes_vs_lda": len(changed),
        "beneficial_corrections": beneficial,
        "harmful_corrections": harmful,
        "correction_precision": beneficial / len(changed) if changed else "",
        "mean_search_runtime_ms": sum(
            float(row["search_runtime_ms"]) for row in selected
        )
        / len(selected),
        "mean_path_switches": (
            sum(int(row["path_switches"]) for row in path_rows) / len(path_rows)
            if path_rows
            else ""
        ),
        "mean_path_garbage": (
            sum(int(row["path_garbage_segments"]) for row in path_rows)
            / len(path_rows)
            if path_rows
            else ""
        ),
    }


def mcnemar_exact(w2r: int, r2w: int) -> float:
    discordant = w2r + r2w
    if discordant == 0:
        return 1.0
    lower = min(w2r, r2w)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * tail)


def compare_predictions(
    new_rows: Sequence[dict],
    old_path: Path,
    old_column: str,
    reference: str,
) -> dict:
    old = {
        (Path(row["file_name"]).stem, int(float(row["packet_index"]))): row
        for row in read_csv(old_path)
    }
    full = [row for row in new_rows if row["method"] == "full_aco"]
    changed = w2r = r2w = changed_both_wrong = 0
    for row in full:
        key = (Path(row["file_name"]).stem, int(row["packet_index"]))
        old_row = old[key]
        old_label = old_row[old_column]
        new_label = row["final_label"]
        true_label = row["true_label"]
        old_ok = old_label == true_label
        new_ok = new_label == true_label
        changed += int(old_label != new_label)
        w2r += int(not old_ok and new_ok)
        r2w += int(old_ok and not new_ok)
        changed_both_wrong += int(
            old_label != new_label and not old_ok and not new_ok
        )
    return {
        "split": full[0]["split"],
        "reference": reference,
        "reference_correct": sum(
            old_row[old_column] == old_row["true_label"] for old_row in old.values()
        ),
        "no_alpha_correct": sum(int(row["final_correct"]) for row in full),
        "changed": changed,
        "W2R": w2r,
        "R2W": r2w,
        "changed_both_wrong": changed_both_wrong,
        "net_gain": w2r - r2w,
        "mcnemar_exact_two_sided_p": mcnemar_exact(w2r, r2w),
    }


def pairwise_rows(rows: Sequence[dict]) -> list[dict]:
    output = []
    full = {
        (row["file_name"], int(row["packet_index"])): row
        for row in rows
        if row["method"] == "full_aco"
    }
    for method in METHOD_ORDER[:-1]:
        other = {
            (row["file_name"], int(row["packet_index"])): row
            for row in rows
            if row["method"] == method
        }
        for stage in ["search", "final"]:
            w2r = r2w = changed = changed_both_wrong = 0
            for key, full_row in full.items():
                old = other[key]
                true_label = full_row["true_label"]
                full_label = full_row[f"{stage}_label"]
                old_label = old[f"{stage}_label"]
                full_ok = full_label == true_label
                old_ok = old_label == true_label
                changed += int(full_label != old_label)
                w2r += int(not old_ok and full_ok)
                r2w += int(old_ok and not full_ok)
                changed_both_wrong += int(
                    full_label != old_label and not full_ok and not old_ok
                )
            output.append(
                {
                    "split": next(iter(full.values()))["split"],
                    "stage": stage,
                    "ablation": method,
                    "ablation_label": METHOD_LABELS[method],
                    "changed": changed,
                    "full_W2R": w2r,
                    "full_R2W": r2w,
                    "full_net_gain": w2r - r2w,
                    "changed_both_wrong": changed_both_wrong,
                    "mcnemar_exact_two_sided_p": mcnemar_exact(w2r, r2w),
                }
            )
    return output


def render_report(
    output_dir: Path,
    beta_rows: Sequence[dict],
    beta: float,
    metrics: Sequence[dict],
    comparisons: Sequence[dict],
) -> None:
    lines = [
        "# LDA/ACO without alpha candidate fusion",
        "",
        "Date: 2026-07-17",
        "",
        "Candidate policy: direct LDA Top-5. RSSI remains only inside ACO observation costs and weak priors.",
        "",
        "## Validation beta freeze",
        "",
        "| beta | Full ACO correct | accuracy | changes vs LDA |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in beta_rows:
        mark = " **(selected)**" if float(row["beta"]) == beta else ""
        lines.append(
            f"| {float(row['beta']):.1f}{mark} | {row['final_correct']}/128 "
            f"| {float(row['final_accuracy']):.2%} | {row['changes_vs_lda']} |"
        )
    lines.extend(
        [
            "",
            "Selection rule: maximize validation accuracy; choose the smaller beta on ties.",
            "",
            "## Frozen-beta results",
            "",
            "| Split | Method | search/final correct | final accuracy | MAE/P95 m | severe >10m | correction precision |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ["validation", "formal_test"]:
        for method in METHOD_ORDER:
            row = next(
                item for item in metrics if item["split"] == split and item["method"] == method
            )
            precision = row["correction_precision"]
            precision_text = "n/a" if precision == "" else f"{float(precision):.2%}"
            lines.append(
                f"| {split} | {METHOD_LABELS[method]} | {row['search_correct']}/{row['final_correct']} "
                f"| {float(row['final_accuracy']):.2%} "
                f"| {float(row['final_mae_m']):.3f}/{float(row['final_p95_m']):.3f} "
                f"| {float(row['severe_error_rate']):.2%} | {precision_text} |"
            )
    lines.extend(
        [
            "",
            "## Full ACO comparison with old alpha=0.3 mainline",
            "",
            "| Split | old/new correct | changed | W2R/R2W | net | McNemar p |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['split']} | {row['reference_correct']}/{row['no_alpha_correct']} "
            f"| {row['changed']} | {row['W2R']}/{row['R2W']} "
            f"| {int(row['net_gain']):+d} | {float(row['mcnemar_exact_two_sided_p']):.6f} |"
        )
    lines.extend(
        [
            "",
            "The formal split remains exploratory because it was inspected by prior experiments.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distances = {
        row["position_key"]: float(row["distance_m"])
        for row in read_csv(weakness.LOCATION_CSV)
    }

    # All selection happens here, before formal context is loaded/evaluated.
    validation = weakness.load_context("validation", build_training_state=True)
    validation_search = evaluate_search(validation)
    beta_rows = []
    for beta in BETA_GRID:
        beta_predictions = apply_beta(validation_search, beta)
        metric = metric_row(beta_predictions, "full_aco", distances)
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
    validation_predictions = apply_beta(validation_search, frozen_beta)

    # Formal is evaluated only after the validation choice is frozen.
    formal = weakness.load_context("formal_test", build_training_state=True)
    formal_search = evaluate_search(formal)
    formal_predictions = apply_beta(formal_search, frozen_beta)

    for split_rows in [validation_predictions, formal_predictions]:
        candidate_sets = {
            (row["file_name"], int(row["packet_index"])): row["candidate_labels"]
            for row in split_rows
            if row["method"] == "full_aco"
        }
        if len(candidate_sets) != 128:
            raise RuntimeError("Candidate coverage is not 128 packets")
        if sum(
            int(row["true_in_candidates"])
            for row in split_rows
            if row["method"] == "full_aco"
        ) != 128:
            raise RuntimeError("Direct LDA Top-5 unexpectedly truncates a true label")
        for row in split_rows:
            if row["candidate_labels"].split(";")[0] != row["lda_label"]:
                raise RuntimeError("Candidate order is not direct LDA order")

    write_csv(args.output_dir / "validation_beta_selection.csv", beta_rows)
    write_csv(args.output_dir / "validation_predictions.csv", validation_predictions)
    write_csv(args.output_dir / "formal_predictions.csv", formal_predictions)

    metrics = [
        metric_row(split_rows, method, distances)
        for split_rows in [validation_predictions, formal_predictions]
        for method in METHOD_ORDER
    ]
    write_csv(args.output_dir / "metrics.csv", metrics)
    pairwise = pairwise_rows(validation_predictions) + pairwise_rows(formal_predictions)
    write_csv(args.output_dir / "full_aco_pairwise.csv", pairwise)

    comparisons = [
        compare_predictions(
            validation_predictions,
            weakness.MAINLINE_ROOT
            / "validation"
            / "fixed_beta_final_val_predictions.csv",
            "final_label",
            "alpha_0.3_mainline",
        ),
        compare_predictions(
            formal_predictions,
            weakness.MAINLINE_ROOT
            / "formal_test"
            / "fixed_beta_final_test_predictions.csv",
            "final_label",
            "alpha_0.3_mainline",
        ),
    ]
    write_csv(args.output_dir / "comparison_vs_alpha_0.3.csv", comparisons)
    render_report(args.output_dir, beta_rows, frozen_beta, metrics, comparisons)

    frozen_config = {
        "candidate_policy": "direct LDA Top-5",
        "alpha_fusion": False,
        "beta": frozen_beta,
        "validation_t_seg": float(validation.args.t_seg_resolved),
        "formal_t_seg": float(formal.args.t_seg_resolved),
        "aco": {
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
    payload = {
        "experiment": "EXPANDED_LDA_ACO_NO_ALPHA_VALIDATION_REFREEZE_20260717",
        "status": "REFROZEN_MAINLINE",
        "protocol": {
            "candidate_policy": "direct LDA Top-5",
            "alpha_fusion": False,
            "rssi_role": "ACO observation cost and weak prior only",
            "beta_grid": BETA_GRID,
            "beta_selection": "validation maximum accuracy; smaller beta on ties",
            "frozen_beta": frozen_beta,
            "formal_evaluated_after_freeze": True,
            "other_aco_parameters": "unchanged from adopted 2026-07-16 mainline",
            "seed": SEED,
            "test_status": "exploratory because Expanded formal test was previously inspected",
        },
        "validation_beta_selection": beta_rows,
        "frozen_configuration": frozen_config,
        "metrics": metrics,
        "comparison_vs_alpha_0.3": comparisons,
        "checks": {
            "validation_candidate_recall_count": 128,
            "formal_candidate_recall_count": 128,
            "validation_prediction_rows": len(validation_predictions),
            "formal_prediction_rows": len(formal_predictions),
        },
        "inputs": {
            "validation_model_sha256": sha256(
                weakness.MAINLINE_ROOT / "validation_lda_model.joblib"
            ),
            "formal_model_sha256": sha256(
                weakness.MAINLINE_ROOT / "formal_lda_model.joblib"
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
