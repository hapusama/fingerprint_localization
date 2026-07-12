#!/usr/bin/env python3
"""Validation-selected ACO 4.0 parameter sweep on the 1:10 6:2:2 split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
MODEL_V3_DIR = PROJECT_ROOT / "fingerprint_localization" / "model" / "v3"
if str(MODEL_V3_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_V3_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import aco_packet_path_v4 as aco4  # noqa: E402
import run_aco_v4_on_split as split_v4  # noqa: E402


DEFAULT_OUTPUT_DIR = split_v4.split_runner.DEFAULT_RESULT_DIR / "aco_v4_sweep"
DEFAULT_FINAL_OUTPUT_DIR = split_v4.split_runner.DEFAULT_RESULT_DIR / "aco_v4_best"


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_float_list(value: str, default: Sequence[float]) -> list[float]:
    if not value.strip():
        return list(default)
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def metric_key(row: dict) -> tuple:
    return (
        row["aco_score4_accuracy"],
        row["aco_vote_accuracy"],
        row.get("aco_v4_net_vs_aco_v2", 0),
        -row.get("aco_v4_R2W_vs_aco_v2", 0),
        -row["score4_R2W_from_vote"],
    )


def config_key(config: dict) -> tuple:
    return tuple(round(config[key], 12) for key in ["T_seg", "lambda_R", "lambda_W", "lambda_V", "lambda_Q"])


def run(args: argparse.Namespace) -> dict:
    aco_args = split_v4.build_args(args)
    rssi_packets = aco4.aco2.base.read_rssi_packets(args.rssi_csv)
    symbol_packets, q4_offsets, thresholds = aco4.aco2.base.read_symbol_packets(args.spectrum_csv, aco_args)
    base_samples = aco4.aco2.base.align_samples(rssi_packets, symbol_packets)
    samples = aco4.aco2.build_segment_packets(base_samples, args.segment_count)
    split_indices = split_v4.split_runner.load_split_indices(samples, args.split_csv)
    labels = [sample.label for sample in samples]
    chirp_shapes, chirp_struct, chirp_metadata = aco4.aco2.prepare_chirp_fields(aco_args, labels)
    if aco_args.t_seg_resolved is None:
        aco_args.t_seg_resolved = split_v4.estimate_t_seg(
            samples, q4_offsets, chirp_shapes, chirp_struct, split_indices["train"], aco_args
        )

    base_config = {
        "T_seg": aco_args.t_seg_resolved,
        "lambda_R": aco_args.lambda_rssi_prior,
        "lambda_W": aco_args.lambda_raw_prior,
        "lambda_V": aco_args.lambda_veto,
        "lambda_Q": aco_args.lambda_q_switch,
    }
    candidates = []
    seen = set()

    def add(stage: str, config: dict) -> None:
        key = config_key(config)
        if key not in seen:
            seen.add(key)
            candidates.append({"stage": stage, **config})

    for value in parse_float_list(args.sweep_t_seg, [base_config["T_seg"], 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20]):
        add("T_seg", {**base_config, "T_seg": value})
    for value in parse_float_list(args.sweep_lambda_r, [0.1, 0.2, 0.3]):
        add("lambda_R", {**base_config, "lambda_R": value})
    for value in parse_float_list(args.sweep_lambda_w, [0.05, 0.1, 0.2]):
        add("lambda_W", {**base_config, "lambda_W": value})
    for value in parse_float_list(args.sweep_lambda_v, [0.2, 0.5, 1.0]):
        add("lambda_V", {**base_config, "lambda_V": value})
    for value in parse_float_list(args.sweep_lambda_q, [0.5, 1.0, 1.5]):
        add("lambda_Q", {**base_config, "lambda_Q": value})

    sweep_rows = []
    best_row = None
    for sweep_index, config in enumerate(candidates, start=1):
        aco_args.t_seg_resolved = config["T_seg"]
        aco_args.lambda_rssi_prior = config["lambda_R"]
        aco_args.lambda_raw_prior = config["lambda_W"]
        aco_args.lambda_veto = config["lambda_V"]
        aco_args.lambda_q_switch = config["lambda_Q"]
        metrics, _predictions, _candidate_rows, _segment_rows = split_v4.evaluate_split(
            samples,
            q4_offsets,
            chirp_shapes,
            chirp_struct,
            split_indices["train"],
            split_indices["val"],
            "val",
            aco_args,
            leave_one_out_prototypes=args.leave_one_out_prototypes,
        )
        metrics.update(thresholds)
        metrics.update(
            {
                "sweep_index": sweep_index,
                "stage": config["stage"],
                "T_seg": config["T_seg"],
                "lambda_R": config["lambda_R"],
                "lambda_W": config["lambda_W"],
                "lambda_V": config["lambda_V"],
                "lambda_Q": config["lambda_Q"],
            }
        )
        sweep_rows.append(metrics)
        if best_row is None or metric_key(metrics) > metric_key(best_row):
            best_row = metrics
        print(
            f"[{sweep_index}/{len(candidates)}] {config['stage']} "
            f"T={config['T_seg']:.6g} R={config['lambda_R']:.3g} "
            f"W={config['lambda_W']:.3g} V={config['lambda_V']:.3g} "
            f"Q={config['lambda_Q']:.3g} val_score4={metrics['aco_score4_accuracy']:.6f}",
            flush=True,
        )

    assert best_row is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sweep_index", "stage", "split", "packet_count", "T_seg", "lambda_R", "lambda_W", "lambda_V", "lambda_Q",
        "rssi_topk_recall", "aco_vote_accuracy", "aco_score4_accuracy", "aco_score4_correct",
        "score4_change_count_from_vote", "score4_W2R_from_vote", "score4_R2W_from_vote", "score4_net_from_vote",
        "Q_seg_mean", "segment_cost_std_mean",
    ]
    fields += sorted({key for row in sweep_rows for key in row} - set(fields))
    write_csv(args.output_dir / "aco_v4_val_sweep_summary.csv", sweep_rows, fields)
    with (args.output_dir / "aco_v4_val_sweep_best.json").open("w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2, ensure_ascii=False)

    final_args = argparse.Namespace(**vars(args))
    final_args.output_dir = args.final_output_dir
    final_args.t_seg = best_row["T_seg"]
    final_args.lambda_rssi_prior = best_row["lambda_R"]
    final_args.lambda_raw_prior = best_row["lambda_W"]
    final_args.lambda_veto = best_row["lambda_V"]
    final_args.lambda_q_switch = best_row["lambda_Q"]
    final_metadata = split_v4.run(final_args)
    payload = {
        "selection_policy": "ACO 4.0 parameters selected by validation split; test split is evaluated once with the selected config.",
        "best_val": best_row,
        "final_run": final_metadata,
        "chirp_template_field": chirp_metadata,
    }
    with (args.output_dir / "aco_v4_sweep_and_final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-output-dir", type=Path, default=DEFAULT_FINAL_OUTPUT_DIR)
    parser.add_argument("--result-dir", type=Path, default=split_v4.split_runner.DEFAULT_RESULT_DIR)
    parser.add_argument("--method-summary", type=Path, default=split_v4.DEFAULT_METHOD_SUMMARY)
    parser.add_argument("--aco-v2-dir", type=Path, default=split_v4.DEFAULT_ACO_V2_DIR)
    parser.add_argument("--rssi-csv", type=Path, default=split_v4.split_runner.DEFAULT_RSSI_CSV)
    parser.add_argument("--spectrum-csv", type=Path, default=split_v4.split_runner.DEFAULT_SPECTRUM_CSV)
    parser.add_argument("--split-csv", type=Path, default=split_v4.split_runner.DEFAULT_SPLIT_CSV)
    parser.add_argument("--chirp-template-csv", type=Path, default=aco4.aco2.DEFAULT_CHIRP_TEMPLATE_CSV)
    parser.add_argument("--chirp-structure-csv", type=Path, default=aco4.aco2.DEFAULT_CHIRP_STRUCT_CSV)
    parser.add_argument("--location-csv", type=Path, default=aco4.aco2.DEFAULT_LOCATION_CSV)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rssi-class-k", type=int, default=3)
    parser.add_argument("--segment-count", type=int, default=4)
    parser.add_argument("--ants", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--elite-ants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--rssi-weight", type=float, default=0.45)
    parser.add_argument("--bin-weight", type=float, default=0.02)
    parser.add_argument("--energy-weight", type=float, default=0.20)
    parser.add_argument("--raw-weight", type=float, default=0.55)
    parser.add_argument("--q4-weight", type=float, default=0.0)
    parser.add_argument("--shrinkage-lambda", type=float, default=8.0)
    parser.add_argument("--phy-var-c0", type=float, default=0.05)
    parser.add_argument("--phy-var-c1", type=float, default=0.50)
    parser.add_argument("--phy-var-c2", type=float, default=1.0)
    parser.add_argument("--sigma0-sq", type=float, default=0.02)
    parser.add_argument("--min-variance", type=float, default=1e-3)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--logdet-weight", type=float, default=0.05)
    parser.add_argument("--normalize-bin-cost", action="store_true", default=True)
    parser.add_argument("--garbage-cost", type=float, default=1.0)
    parser.add_argument("--garbage-cost-min", type=float, default=0.35)
    parser.add_argument("--lambda-garbage-stability", type=float, default=0.35)
    parser.add_argument("--lambda0-switch", type=float, default=0.70)
    parser.add_argument("--switch-eta", type=float, default=0.20)
    parser.add_argument("--lambda-div", type=float, default=0.20)
    parser.add_argument("--lambda-g", type=float, default=0.50)
    parser.add_argument("--max-garbage", type=int, default=2)
    parser.add_argument("--garbage-overuse-penalty", type=float, default=4.0)
    parser.add_argument("--lambda-c", type=float, default=0.15)
    parser.add_argument("--tau-stay", type=float, default=1.4)
    parser.add_argument("--tau-switch", type=float, default=0.35)
    parser.add_argument("--pheromone-power", type=float, default=1.0)
    parser.add_argument("--heuristic-power", type=float, default=1.4)
    parser.add_argument("--evaporation", type=float, default=0.25)
    parser.add_argument("--min-pheromone", type=float, default=1e-4)
    parser.add_argument("--aco-temperature", type=float, default=None)
    parser.add_argument("--lambda-rssi-prior", type=float, default=0.2)
    parser.add_argument("--lambda-raw-prior", type=float, default=0.1)
    parser.add_argument("--lambda-veto", type=float, default=0.5)
    parser.add_argument("--lambda-q-switch", type=float, default=1.0)
    parser.add_argument("--t-seg", type=float, default=None)
    parser.add_argument("--t-seg-quantile", type=float, default=0.95)
    parser.add_argument("--lambda-score-vote", type=float, default=1.0)
    parser.add_argument("--lambda-score-cost", type=float, default=0.15)
    parser.add_argument("--sweep-t-seg", default="")
    parser.add_argument("--sweep-lambda-r", default="")
    parser.add_argument("--sweep-lambda-w", default="")
    parser.add_argument("--sweep-lambda-v", default="")
    parser.add_argument("--sweep-lambda-q", default="")
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
    payload = run(parse_args())
    print(json.dumps({"best_val": payload["best_val"], "final_summary": payload["final_run"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
