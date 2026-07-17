#!/usr/bin/env python3
"""Back-to-back latency benchmark for alpha=0.3 and no-alpha refrozen ACO."""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import sklearn

import benchmark_expanded_aco_ml_prior_latency as bench
import run_expanded_supervised_ensemble as supervised


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
OLD_ROOT = PROJECT_DIR / "results" / "expanded_source_safe_1to10" / "aco_lda_only_mainline"
NEW_ROOT = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "aco_lda_only_no_alpha_refrozen_20260717"
)
DEFAULT_OUTPUT_DIR = NEW_ROOT / "latency"


def no_alpha_frozen_predictions(path: Path) -> dict:
    rows = [row for row in bench.read_csv(path) if row["method"] == "full_aco"]
    return {
        bench.packet_key(row["file_name"], row["packet_index"]): row
        for row in rows
    }


def infer_packet_no_alpha(
    state: dict,
    test_index: int,
    packet,
    ml_row: dict,
    rng: random.Random,
):
    total_start = bench.now_ns()
    probabilities, timings = bench.infer_posterior(state, ml_row)

    start = bench.now_ns()
    segmented = bench.source.aco2.packet_to_segments(
        packet, state["aco_args"].segment_count
    )
    end = bench.now_ns()
    timings["packet_segmentation"] = bench.elapsed(start, end)

    start = bench.now_ns()
    ranked_rssi = state["class_rank"](
        state["rssi_rows"],
        state["labels"],
        state["split_indices"]["train"],
        test_index,
        state["aco_args"].rssi_class_k,
    )
    candidates = sorted(
        probabilities,
        key=lambda label: (
            -float(probabilities[label]),
            supervised.natural_label_key(label),
        ),
    )[: state["aco_args"].top_k]
    rssi_cost_by_label = {label: float(cost) for label, cost in ranked_rssi}
    rssi_costs = {label: rssi_cost_by_label[label] for label in candidates}
    end = bench.now_ns()
    timings["source_candidate_ranking_no_alpha"] = bench.elapsed(start, end)

    start = bench.now_ns()
    obs_costs, _rows, meta = bench.prior.runner.aco4.build_observation_costs_v4(
        segmented,
        candidates,
        rssi_costs,
        state["templates"],
        state["prototypes"],
        state["q4_offsets"],
        state["aco_args"],
    )
    meta["rssi_top1"] = ranked_rssi[0][0]
    end = bench.now_ns()
    timings["observation_costs"] = bench.elapsed(start, end)

    start = bench.now_ns()
    result = bench.prior.runner.aco4.run_aco_v4_for_packet(
        obs_costs,
        candidates,
        state["templates"],
        meta,
        state["aco_args"],
        rng,
    )
    end = bench.now_ns()
    timings["aco_search_and_score4"] = bench.elapsed(start, end)

    start = bench.now_ns()
    final_label = bench.final_fusion(
        result, candidates, probabilities, state["beta"]
    )
    end = bench.now_ns()
    timings["beta_score4_posterior_fusion"] = bench.elapsed(start, end)
    timings["feature_ready_online_total"] = bench.elapsed(total_start, end)
    return final_label, timings


def prepare_states() -> tuple[dict, dict]:
    alpha_state = bench.prepare_state(optimized_inference=True)
    alpha_state["models"] = {"lda_svd": alpha_state["models"]["lda_svd"]}
    alpha_state["rf_tree_count"] = 0
    alpha_state["frozen_predictions"] = bench.load_final_predictions(
        OLD_ROOT / "formal_test" / "fixed_beta_final_test_predictions.csv"
    )

    no_alpha_state = dict(alpha_state)
    no_alpha_state["probabilities_by_source"] = {}
    no_alpha_state["class_rank"] = bench.make_cached_ml_source_class_rank(
        no_alpha_state["samples"],
        no_alpha_state["rssi_rows"],
        no_alpha_state["labels"],
        no_alpha_state["split_indices"]["train"],
        no_alpha_state["probabilities_by_source"],
        0.0,
    )
    no_alpha_state["alpha"] = None
    no_alpha_state["beta"] = 0.6
    no_alpha_state["frozen_predictions"] = no_alpha_frozen_predictions(
        NEW_ROOT / "formal_predictions.csv"
    )
    return alpha_state, no_alpha_state


def comparison(alpha: dict, no_alpha: dict) -> dict:
    output = {}
    for boundary, total_key in [
        ("feature_ready_inference", "feature_ready_online_total"),
        ("packet_to_location", "packet_to_location_total"),
    ]:
        old = alpha[boundary]["per_packet_latency"][total_key]
        new = no_alpha[boundary]["per_packet_latency"][total_key]
        output[boundary] = {
            metric: {
                "alpha_0.3_us": old[metric],
                "no_alpha_us": new[metric],
                "delta_us": new[metric] - old[metric],
                "delta_percent": (new[metric] / old[metric] - 1.0) * 100.0,
            }
            for metric in ["mean_us", "median_us", "p95_us"]
        }
    old_rank = alpha["feature_ready_inference"]["per_packet_latency"][
        "source_candidate_ranking_alpha_fusion"
    ]
    new_rank = no_alpha["feature_ready_inference"]["per_packet_latency"][
        "source_candidate_ranking_no_alpha"
    ]
    output["candidate_ranking"] = {
        metric: {
            "alpha_0.3_us": old_rank[metric],
            "no_alpha_us": new_rank[metric],
            "delta_us": new_rank[metric] - old_rank[metric],
            "delta_percent": (new_rank[metric] / old_rank[metric] - 1.0) * 100.0,
        }
        for metric in ["mean_us", "median_us", "p95_us"]
    }
    return output


def write_report(path: Path, payload: dict) -> None:
    compare = payload["comparison"]
    feature = compare["feature_ready_inference"]
    packet = compare["packet_to_location"]
    rank = compare["candidate_ranking"]
    path.write_text(
        "# No-alpha refrozen latency comparison\n\n"
        "Both variants were benchmarked back-to-back in the same process.\n\n"
        "| Boundary | alpha=0.3 median/P95 | no-alpha median/P95 | median delta |\n"
        "| --- | ---: | ---: | ---: |\n"
        f"| Feature-ready | {feature['median_us']['alpha_0.3_us']/1000:.3f}/"
        f"{feature['p95_us']['alpha_0.3_us']/1000:.3f} ms | "
        f"{feature['median_us']['no_alpha_us']/1000:.3f}/"
        f"{feature['p95_us']['no_alpha_us']/1000:.3f} ms | "
        f"{feature['median_us']['delta_percent']:+.2f}% |\n"
        f"| Packet-to-location | {packet['median_us']['alpha_0.3_us']/1000:.3f}/"
        f"{packet['p95_us']['alpha_0.3_us']/1000:.3f} ms | "
        f"{packet['median_us']['no_alpha_us']/1000:.3f}/"
        f"{packet['p95_us']['no_alpha_us']/1000:.3f} ms | "
        f"{packet['median_us']['delta_percent']:+.2f}% |\n\n"
        f"Candidate ranking median changed from {rank['median_us']['alpha_0.3_us']/1000:.3f} "
        f"to {rank['median_us']['no_alpha_us']/1000:.3f} ms "
        f"({rank['median_us']['delta_percent']:+.2f}%).\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict:
    alpha_state, no_alpha_state = prepare_states()
    original_infer = bench.infer_packet
    try:
        bench.infer_packet = original_infer
        alpha_feature = bench.run_feature_benchmark(
            alpha_state,
            args.feature_warmups,
            args.feature_repeats,
            expected_correct=120,
            require_frozen_match=True,
        )
        alpha_raw = bench.run_raw_benchmark(
            alpha_state,
            args.accepted_starts,
            args.raw_warmups,
            args.raw_repeats,
            expected_correct=120,
            require_frozen_match=True,
        )

        bench.infer_packet = infer_packet_no_alpha
        no_alpha_feature = bench.run_feature_benchmark(
            no_alpha_state,
            args.feature_warmups,
            args.feature_repeats,
            expected_correct=120,
            require_frozen_match=True,
        )
        no_alpha_raw = bench.run_raw_benchmark(
            no_alpha_state,
            args.accepted_starts,
            args.raw_warmups,
            args.raw_repeats,
            expected_correct=120,
            require_frozen_match=True,
        )
    finally:
        bench.infer_packet = original_infer

    alpha = {
        "feature_ready_inference": alpha_feature,
        "packet_to_location": alpha_raw,
    }
    no_alpha = {
        "feature_ready_inference": no_alpha_feature,
        "packet_to_location": no_alpha_raw,
    }
    payload = {
        "experiment": "NO_ALPHA_REFROZEN_LATENCY_20260717",
        "status": "complete",
        "methodology": "back-to-back same-process benchmark; alpha first, no-alpha second; each has warmups",
        "configuration": {
            "alpha_reference": 0.3,
            "alpha_reference_beta": 0.5,
            "no_alpha": True,
            "no_alpha_beta": 0.6,
            "feature_warmups": args.feature_warmups,
            "feature_repeats": args.feature_repeats,
            "raw_warmups": args.raw_warmups,
            "raw_repeats": args.raw_repeats,
            "test_packets": 128,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "alpha_0.3": alpha,
        "no_alpha": no_alpha,
        "comparison": comparison(alpha, no_alpha),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "no_alpha_latency_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "no_alpha_latency_comparison.md", payload)
    print(json.dumps(payload["comparison"], indent=2, ensure_ascii=False))
    return payload


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-starts", type=Path, default=bench.DEFAULT_ACCEPTED_STARTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-warmups", type=int, default=2)
    parser.add_argument("--feature-repeats", type=int, default=10)
    parser.add_argument("--raw-warmups", type=int, default=1)
    parser.add_argument("--raw-repeats", type=int, default=5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
