#!/usr/bin/env python3
"""Matched-vs-mismatched chirp/LoRa bin-window similarity control.

The original chirp-to-LoRa projection experiment compared each LoRa point with
the chirp projection from the same location.  This control checks whether that
similarity is location-specific by comparing:

    sim(a_l^L, a_l^C)  vs.  sim(a_l^L, a_j^C), j != l

Only two shape metrics are reported here:
* magnitude cosine over bin[-2,+2]
* Pearson correlation over relative-dB bin[-2,+2]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = ROOT / "v2_output/20260626_chirp_lora_bin_projection"
DEFAULT_OUTPUT_DIR = ROOT / "v2_output/20260626_chirp_lora_bin_projection_mismatch_control"
BIN_OFFSETS = [-2, -1, 0, 1, 2]
EPS = 1e-12


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def point_key(row: dict) -> tuple[int, int]:
    return int(float(row["corridor_id"])), int(float(row["location_id"]))


def point_label(key: tuple[int, int]) -> str:
    return f"{key[0]}_{key[1]}"


def vector(row: dict, prefix: str, suffix: str) -> list[float]:
    return [float(row[f"{prefix}_{off:+d}{suffix}"]) for off in BIN_OFFSETS]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(EPS, na * nb)


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    denom = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return sum(x * y for x, y in zip(da, db)) / max(EPS, denom)


def percentile(values: Sequence[float], x: float) -> float:
    if not values:
        return 1.0
    return sum(1 for value in values if value <= x) / len(values)


def rank_desc(values_by_label: dict[str, float], label: str) -> int:
    ordered = sorted(values_by_label.items(), key=lambda item: (-item[1], item[0]))
    for idx, (item_label, _value) in enumerate(ordered, start=1):
        if item_label == label:
            return idx
    return len(ordered) + 1


def describe(values: Sequence[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def bootstrap_ci(
    values: Sequence[float],
    seed: int,
    rounds: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    # Tiny deterministic LCG, avoids importing random for complete reproducibility
    # across Python minor versions.
    state = seed & 0x7FFFFFFF

    def randrange(n: int) -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state % n

    n = len(values)
    means = []
    for _ in range(rounds):
        means.append(sum(values[randrange(n)] for _i in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * (rounds - 1))]
    hi = means[int((1 - alpha / 2) * (rounds - 1))]
    return lo, hi


def sign_test_p_one_sided(positive: int, total: int) -> float:
    """P[X >= positive], X~Binomial(total, 0.5)."""
    if total <= 0:
        return math.nan
    return sum(math.comb(total, k) for k in range(positive, total + 1)) / (2**total)


def run(args: argparse.Namespace) -> dict:
    synth_rows = read_csv(args.input_dir / "02_chirp_synth_point_bins.csv")
    meas_rows = read_csv(args.input_dir / "03_lora_measured_point_bins.csv")
    synth = {point_key(row): row for row in synth_rows}
    meas = {point_key(row): row for row in meas_rows}
    matched_keys = sorted(set(synth) & set(meas))

    pair_rows = []
    per_location_rows = []
    for lora_key in matched_keys:
        lora = meas[lora_key]
        lora_mag = vector(lora, "meas_mag_bin", "_mean")
        lora_rel = vector(lora, "meas_rel_db_bin", "_mean")
        metric_by_chirp: dict[str, dict[str, float]] = {}
        for chirp_key, chirp in sorted(synth.items()):
            chirp_mag = vector(chirp, "synth_mag_bin", "_mean")
            chirp_rel = vector(chirp, "synth_rel_db_bin", "_mean")
            relation = "matched" if chirp_key == lora_key else "mismatched"
            row = {
                "lora_label": point_label(lora_key),
                "chirp_label": point_label(chirp_key),
                "relation": relation,
                "cosine": cosine(lora_mag, chirp_mag),
                "relative_db_pearson": pearson(lora_rel, chirp_rel),
                "lora_packet_count": int(float(lora["lora_packet_count"])),
                "chirp_segment_count": int(float(chirp["chirp_segment_count"])),
                "chirp_corr_score_mean": float(chirp["chirp_corr_score_mean"]),
            }
            pair_rows.append(row)
            metric_by_chirp[point_label(chirp_key)] = {
                "cosine": row["cosine"],
                "relative_db_pearson": row["relative_db_pearson"],
                "relation": relation,
            }

        self_label = point_label(lora_key)
        mismatches = [
            item
            for label, item in metric_by_chirp.items()
            if label != self_label
        ]
        for metric in ["cosine", "relative_db_pearson"]:
            matched_value = metric_by_chirp[self_label][metric]
            mismatch_values = [item[metric] for item in mismatches]
            per_location_rows.append(
                {
                    "lora_label": self_label,
                    "metric": metric,
                    "matched": matched_value,
                    "mismatch_mean": mean(mismatch_values),
                    "mismatch_median": median(mismatch_values),
                    "mismatch_max": max(mismatch_values),
                    "matched_minus_mismatch_mean": matched_value - mean(mismatch_values),
                    "matched_minus_mismatch_median": matched_value - median(mismatch_values),
                    "matched_minus_mismatch_max": matched_value - max(mismatch_values),
                    "matched_percentile_vs_mismatch": percentile(mismatch_values, matched_value),
                    "matched_rank_among_chirp_points": rank_desc(
                        {label: item[metric] for label, item in metric_by_chirp.items()},
                        self_label,
                    ),
                    "chirp_candidate_count": len(metric_by_chirp),
                }
            )

    summary_rows = []
    summary_payload: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "matched_location_count": len(matched_keys),
        "chirp_location_count": len(synth),
        "measured_lora_location_count": len(meas),
        "mismatched_pair_count": sum(1 for row in pair_rows if row["relation"] == "mismatched"),
        "metrics": {},
    }
    for metric in ["cosine", "relative_db_pearson"]:
        matched_values = [row[metric] for row in pair_rows if row["relation"] == "matched"]
        mismatch_values = [row[metric] for row in pair_rows if row["relation"] == "mismatched"]
        per_metric = [row for row in per_location_rows if row["metric"] == metric]
        diff_mean = [row["matched_minus_mismatch_mean"] for row in per_metric]
        diff_median = [row["matched_minus_mismatch_median"] for row in per_metric]
        top1_count = sum(1 for row in per_metric if row["matched_rank_among_chirp_points"] == 1)
        positive_mean = sum(1 for value in diff_mean if value > 0)
        ci_lo, ci_hi = bootstrap_ci(diff_mean, args.seed, args.bootstrap_rounds)
        summary = {
            "metric": metric,
            "matched": describe(matched_values),
            "mismatched_all_pairs": describe(mismatch_values),
            "per_location_matched_minus_mismatch_mean": describe(diff_mean),
            "per_location_matched_minus_mismatch_median": describe(diff_median),
            "bootstrap95_mean_diff_vs_mismatch_mean": [ci_lo, ci_hi],
            "locations_matched_gt_mismatch_mean": positive_mean,
            "locations_total": len(per_metric),
            "one_sided_sign_p_matched_gt_mismatch_mean": sign_test_p_one_sided(
                positive_mean, len(per_metric)
            ),
            "matched_top1_count": top1_count,
            "matched_top1_rate": top1_count / len(per_metric) if per_metric else math.nan,
            "median_matched_rank": median(
                [row["matched_rank_among_chirp_points"] for row in per_metric]
            )
            if per_metric
            else math.nan,
        }
        summary_payload["metrics"][metric] = summary
        flat = {
            "metric": metric,
            "matched_count": summary["matched"]["count"],
            "matched_mean": summary["matched"]["mean"],
            "matched_median": summary["matched"]["median"],
            "mismatch_pair_count": summary["mismatched_all_pairs"]["count"],
            "mismatch_mean": summary["mismatched_all_pairs"]["mean"],
            "mismatch_median": summary["mismatched_all_pairs"]["median"],
            "mean_diff_vs_mismatch_mean": summary[
                "per_location_matched_minus_mismatch_mean"
            ]["mean"],
            "median_diff_vs_mismatch_mean": summary[
                "per_location_matched_minus_mismatch_mean"
            ]["median"],
            "bootstrap95_diff_low": ci_lo,
            "bootstrap95_diff_high": ci_hi,
            "locations_matched_gt_mismatch_mean": positive_mean,
            "locations_total": len(per_metric),
            "sign_test_p": summary["one_sided_sign_p_matched_gt_mismatch_mean"],
            "matched_top1_count": top1_count,
            "matched_top1_rate": summary["matched_top1_rate"],
            "median_matched_rank": summary["median_matched_rank"],
        }
        summary_rows.append(flat)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "01_matched_mismatched_pairs.csv",
        pair_rows,
        [
            "lora_label",
            "chirp_label",
            "relation",
            "cosine",
            "relative_db_pearson",
            "lora_packet_count",
            "chirp_segment_count",
            "chirp_corr_score_mean",
        ],
    )
    write_csv(
        args.output_dir / "02_per_location_rank_summary.csv",
        per_location_rows,
        [
            "lora_label",
            "metric",
            "matched",
            "mismatch_mean",
            "mismatch_median",
            "mismatch_max",
            "matched_minus_mismatch_mean",
            "matched_minus_mismatch_median",
            "matched_minus_mismatch_max",
            "matched_percentile_vs_mismatch",
            "matched_rank_among_chirp_points",
            "chirp_candidate_count",
        ],
    )
    write_csv(
        args.output_dir / "03_metric_summary.csv",
        summary_rows,
        [
            "metric",
            "matched_count",
            "matched_mean",
            "matched_median",
            "mismatch_pair_count",
            "mismatch_mean",
            "mismatch_median",
            "mean_diff_vs_mismatch_mean",
            "median_diff_vs_mismatch_mean",
            "bootstrap95_diff_low",
            "bootstrap95_diff_high",
            "locations_matched_gt_mismatch_mean",
            "locations_total",
            "sign_test_p",
            "matched_top1_count",
            "matched_top1_rate",
            "median_matched_rank",
        ],
    )
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Chirp-LoRa matched/mismatched control",
                "",
                "This control compares same-location chirp projection vs LoRa "
                "shape similarity with all other-location chirp projections.",
                "",
                "Metrics: magnitude cosine and relative-dB Pearson over bin[-2,+2].",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
