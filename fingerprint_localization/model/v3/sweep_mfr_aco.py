#!/usr/bin/env python3
"""Parameter sweep and V0-V5 ablation for MFR-ACO."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Sequence

import aco_packet_path as base
import mfr_aco_global_multipath as mfr


DEFAULT_OUTPUT_DIR = mfr.PROJECT_ROOT / "fingerprint_localization" / "model" / "v3" / "output_mfr_aco_sweep"


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_grid(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def build_args(args: argparse.Namespace, **overrides) -> SimpleNamespace:
    values = {
        "rssi_csv": args.rssi_csv,
        "spectrum_csv": args.spectrum_csv,
        "chirp_csv": args.chirp_csv,
        "location_csv": args.location_csv,
        "output_dir": args.output_dir,
        "top_k": args.top_k,
        "rssi_class_k": args.rssi_class_k,
        "search_depth": args.search_depth,
        "raw_segments": args.raw_segments,
        "ants": args.ants,
        "iterations": args.iterations,
        "elite_ants": args.elite_ants,
        "seed": args.seed,
        "tau_stay": args.tau_stay,
        "tau_switch": args.tau_switch,
        "pheromone_power": args.pheromone_power,
        "evaporation": args.evaporation,
        "min_pheromone": args.min_pheromone,
        "aco_temperature": args.aco_temperature,
        "multipath_temperature": args.multipath_temperature,
        "kappa_r": args.kappa_r,
        "kappa_w": args.kappa_w,
        "beta": args.beta,
        "gamma": args.gamma,
        "lambda_m": args.lambda_m,
        "lambda_div": args.lambda_div,
        "ablation_stage": args.ablation_stage,
        "peak_threshold": args.peak_threshold,
        "auto_peak_quantile": args.auto_peak_quantile,
        "q4_dev_threshold": args.q4_dev_threshold,
        "auto_q4_dev_quantile": args.auto_q4_dev_quantile,
        "q4_peak_offset_max": args.q4_peak_offset_max,
        "q4_peak_to_side_threshold": args.q4_peak_to_side_threshold,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def prepare_data(args: argparse.Namespace):
    read_args = build_args(args)
    rssi_packets = base.read_rssi_packets(args.rssi_csv)
    symbol_packets, _q4_offsets, symbol_thresholds = base.read_symbol_packets(args.spectrum_csv, read_args)
    samples = base.align_samples(rssi_packets, symbol_packets)
    evidence = mfr.build_all_packet_evidence(samples, args.raw_segments)
    field, field_metadata = mfr.build_multipath_field(args.location_csv, args.chirp_csv)
    labels = [sample.label for sample in samples]
    base_t_m = mfr.compute_multipath_temperature(field, labels)
    return samples, evidence, field, symbol_thresholds, field_metadata, base_t_m


def metric_row(prefix: dict, metrics: dict) -> dict:
    return {**prefix, **metrics}


def run_sweep(args: argparse.Namespace, samples, evidence, field, base_t_m: float) -> tuple[list[dict], dict]:
    rows = []
    kappa_w_grid = parse_grid(args.kappa_w_grid)
    beta_grid = parse_grid(args.beta_grid)
    gamma_grid = parse_grid(args.gamma_grid)
    tm_grid = parse_grid(args.tm_multiplier_grid)
    total = len(kappa_w_grid) * len(beta_grid) * len(gamma_grid) * len(tm_grid)
    count = 0
    best = None
    for kappa_w in kappa_w_grid:
        for beta in beta_grid:
            for gamma in gamma_grid:
                for tm_mult in tm_grid:
                    count += 1
                    run_args = build_args(
                        args,
                        kappa_w=kappa_w,
                        beta=beta,
                        gamma=gamma,
                        multipath_temperature=base_t_m * tm_mult,
                        ablation_stage=5,
                    )
                    metrics, _pred, _cand, _temps = mfr.evaluate_mfr_aco(samples, evidence, field, run_args)
                    row = metric_row(
                        {
                            "combo_index": count,
                            "kappa_w": kappa_w,
                            "beta": beta,
                            "gamma": gamma,
                            "tm_multiplier": tm_mult,
                            "t_m_swept": base_t_m * tm_mult,
                        },
                        metrics,
                    )
                    rows.append(row)
                    if best is None or (
                        row["mfr_physical_accuracy"],
                        row["aco_vote_accuracy"],
                        row["aco_pheromone_accuracy"],
                    ) > (
                        best["mfr_physical_accuracy"],
                        best["aco_vote_accuracy"],
                        best["aco_pheromone_accuracy"],
                    ):
                        best = row
                    if count % args.progress_every == 0 or count == total:
                        print(
                            f"{count}/{total} best physical={best['mfr_physical_accuracy']:.6f} "
                            f"vote={best['aco_vote_accuracy']:.6f} kappa_w={best['kappa_w']} "
                            f"beta={best['beta']} gamma={best['gamma']} tm_mult={best['tm_multiplier']}",
                            flush=True,
                        )
    return rows, best or {}


def run_ablation(args: argparse.Namespace, samples, evidence, field, best: dict | None = None) -> list[dict]:
    rows = []
    stage_names = {
        0: "V0_RSSI_consistency_only",
        1: "V1_plus_packet_raw_etaW",
        2: "V2_plus_raw_stability_QW",
        3: "V3_plus_multipath_gate_GM",
        4: "V4_plus_KM_pheromone_init",
        5: "V5_full_MFR_ACO",
    }
    for stage in range(6):
        overrides = {"ablation_stage": stage}
        if best:
            overrides.update(
                {
                    "kappa_w": float(best["kappa_w"]),
                    "beta": float(best["beta"]),
                    "gamma": float(best["gamma"]),
                    "multipath_temperature": float(best["t_m_swept"]),
                }
            )
        run_args = build_args(args, **overrides)
        metrics, _pred, _cand, _temps = mfr.evaluate_mfr_aco(samples, evidence, field, run_args)
        rows.append(metric_row({"version": f"V{stage}", "version_name": stage_names[stage]}, metrics))
        print(
            f"ablation V{stage}: physical={metrics['mfr_physical_accuracy']:.6f} "
            f"vote={metrics['aco_vote_accuracy']:.6f} pher={metrics['aco_pheromone_accuracy']:.6f}",
            flush=True,
        )
    return rows


def save_best_outputs(args: argparse.Namespace, samples, evidence, field, best: dict) -> dict:
    best_args = build_args(
        args,
        kappa_w=float(best["kappa_w"]),
        beta=float(best["beta"]),
        gamma=float(best["gamma"]),
        multipath_temperature=float(best["t_m_swept"]),
        ablation_stage=5,
    )
    metrics, predictions, candidate_rows, temperatures = mfr.evaluate_mfr_aco(samples, evidence, field, best_args)
    out_dir = args.output_dir
    prediction_fields = [
        "sample_index",
        "file_name",
        "packet_index",
        "true_label",
        "true_display",
        "rssi_top1_label",
        "rssi_top1_correct",
        "rssi_topk_candidates",
        "true_in_rssi_topk",
        "q_w",
        "aco_path_mode_label",
        "aco_path_mode_correct",
        "aco_pheromone_label",
        "aco_pheromone_correct",
        "aco_vote_label",
        "aco_vote_correct",
        "mfr_physical_label",
        "mfr_physical_correct",
        "best_cost",
        "best_path_labels",
    ]
    candidate_fields = [
        "sample_index",
        "true_label",
        "candidate_label",
        "eta_r",
        "eta_w",
        "q_w",
        "rel_m",
        "sep_m",
        "g_m",
        "raw_weight",
        "candidate_boost",
        "self_pheromone",
        "elite_vote",
        "physical_score",
        "multipath_confidence",
        "multipath_source",
    ]
    write_csv(out_dir / "best_mfr_aco_predictions.csv", predictions, prediction_fields)
    write_csv(out_dir / "best_mfr_aco_candidate_scores.csv", candidate_rows, candidate_fields)
    write_csv(out_dir / "best_mfr_aco_summary.csv", [metrics], list(metrics.keys()))
    return {"metrics": metrics, "temperatures": temperatures}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rssi-csv", type=Path, default=mfr.DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=mfr.DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--chirp-csv", type=Path, default=mfr.DEFAULT_CHIRP_CSV)
    parser.add_argument("--location-csv", type=Path, default=mfr.DEFAULT_LOCATION_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--search-depth", type=int, default=4)
    parser.add_argument("--raw-segments", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tau-stay", type=float, default=1.4)
    parser.add_argument("--tau-switch", type=float, default=0.35)
    parser.add_argument("--pheromone-power", type=float, default=1.0)
    parser.add_argument("--evaporation", type=float, default=0.25)
    parser.add_argument("--min-pheromone", type=float, default=1e-4)
    parser.add_argument("--aco-temperature", type=float, default=None)
    parser.add_argument("--multipath-temperature", type=float, default=None)
    parser.add_argument("--kappa-r", type=float, default=1.0)
    parser.add_argument("--kappa-w", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-m", type=float, default=1.0)
    parser.add_argument("--lambda-div", type=float, default=0.2)
    parser.add_argument("--ablation-stage", type=int, default=5)
    parser.add_argument("--peak-threshold", type=float, default=None)
    parser.add_argument("--auto-peak-quantile", type=float, default=0.10)
    parser.add_argument("--q4-dev-threshold", type=float, default=None)
    parser.add_argument("--auto-q4-dev-quantile", type=float, default=0.75)
    parser.add_argument("--q4-peak-offset-max", type=float, default=0.50)
    parser.add_argument("--q4-peak-to-side-threshold", type=float, default=6.0)
    parser.add_argument("--kappa-w-grid", default="0.5,1.0,2.0")
    parser.add_argument("--beta-grid", default="0.5,1.0")
    parser.add_argument("--gamma-grid", default="0.5,1.0,2.0")
    parser.add_argument("--tm-multiplier-grid", default="0.5,1.0,2.0")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, evidence, field, symbol_thresholds, field_metadata, base_t_m = prepare_data(args)
    sweep_rows, best = run_sweep(args, samples, evidence, field, base_t_m)
    sweep_fields = sorted({key for row in sweep_rows for key in row})
    preferred_sweep = [
        "combo_index",
        "kappa_w",
        "beta",
        "gamma",
        "tm_multiplier",
        "t_m_swept",
        "packet_count",
        "rssi_top1_accuracy",
        "rssi_topk_recall",
        "aco_path_mode_accuracy",
        "aco_pheromone_accuracy",
        "aco_vote_accuracy",
        "mfr_physical_accuracy",
    ]
    sweep_fields = preferred_sweep + [key for key in sweep_fields if key not in preferred_sweep]
    write_csv(args.output_dir / "mfr_aco_parameter_sweep.csv", sweep_rows, sweep_fields)

    ablation_rows = run_ablation(args, samples, evidence, field, best)
    ablation_fields = sorted({key for row in ablation_rows for key in row})
    preferred_ablation = [
        "version",
        "version_name",
        "packet_count",
        "rssi_top1_accuracy",
        "rssi_topk_recall",
        "aco_path_mode_accuracy",
        "aco_pheromone_accuracy",
        "aco_vote_accuracy",
        "mfr_physical_accuracy",
    ]
    ablation_fields = preferred_ablation + [key for key in ablation_fields if key not in preferred_ablation]
    write_csv(args.output_dir / "mfr_aco_ablation_v0_v5.csv", ablation_rows, ablation_fields)

    best_outputs = save_best_outputs(args, samples, evidence, field, best)
    payload = {
        "selection_metric": "max mfr_physical_accuracy, ties by aco_vote_accuracy then pheromone",
        "grid": {
            "kappa_w": parse_grid(args.kappa_w_grid),
            "beta": parse_grid(args.beta_grid),
            "gamma": parse_grid(args.gamma_grid),
            "tm_multiplier": parse_grid(args.tm_multiplier_grid),
            "base_t_m": base_t_m,
        },
        "best": best,
        "best_outputs": best_outputs,
        "symbol_thresholds": symbol_thresholds,
        "multipath_field": field_metadata,
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
    }
    with (args.output_dir / "mfr_aco_sweep_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps({"best": best, "best_outputs": best_outputs["metrics"], "output_dir": str(args.output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
