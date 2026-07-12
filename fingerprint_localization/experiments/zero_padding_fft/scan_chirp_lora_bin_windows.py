#!/usr/bin/env python3
"""Scan chirp/LoRa matched-vs-mismatched similarity over bin windows.

This script intentionally uses only the Python standard library so it can run
even when numpy/pandas are unavailable.  It expects a projection directory with
wide-bin files produced by project_cached_chirp_to_lora_bins.mjs:

  02_chirp_synth_point_bins.csv
  03_lora_measured_point_bins.csv

For each requested K, it evaluates both:

  full: [-K, ..., -1, 0, +1, ..., +K]
  side: [-K, ..., -1, +1, ..., +K]

Only common chirp/LoRa points are used as candidate locations, so mismatches are
strictly other points within the common set.
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
DEFAULT_INPUT_DIR = ROOT / "v2_output/20260710_chirp_lora_bin_window_scan_projection"
DEFAULT_OUTPUT_DIR = ROOT / "v2_output/20260710_chirp_lora_bin_window_scan"
EPS = 1e-12


class Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0x7FFFFFFF

    def randrange(self, n: int) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state % n

    def random_sign(self) -> int:
        return 1 if self.randrange(2) else -1


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def point_key(row: dict) -> tuple[int, int]:
    return int(float(row["corridor_id"])), int(float(row["location_id"]))


def point_label(key: tuple[int, int]) -> str:
    return f"{key[0]}_{key[1]}"


def offset_label(offset: int) -> str:
    return f"{offset:+d}"


def offsets_for_k(k: int, scope: str) -> list[int]:
    if scope == "full":
        return list(range(-k, k + 1))
    if scope == "side":
        return [offset for offset in range(-k, k + 1) if offset != 0]
    raise ValueError(scope)


def vector(row: dict, prefix: str, suffix: str, offsets: Sequence[int]) -> list[float]:
    return [float(row[f"{prefix}_{offset_label(offset)}{suffix}"]) for offset in offsets]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(EPS, na * nb)


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return math.nan
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    denom = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return sum(x * y for x, y in zip(da, db)) / max(EPS, denom)


def describe(values: Sequence[float]) -> dict[str, float]:
    values = [value for value in values if math.isfinite(value)]
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


def percentile(values: Sequence[float], x: float) -> float:
    values = [value for value in values if math.isfinite(value)]
    if not values or not math.isfinite(x):
        return math.nan
    return sum(1 for value in values if value <= x) / len(values)


def rank_desc(values_by_label: dict[str, float], label: str) -> int:
    finite = {
        item_label: value
        for item_label, value in values_by_label.items()
        if math.isfinite(value)
    }
    ordered = sorted(finite.items(), key=lambda item: (-item[1], item[0]))
    for idx, (item_label, _value) in enumerate(ordered, start=1):
        if item_label == label:
            return idx
    return len(ordered) + 1


def bootstrap_mean_ci(
    values: Sequence[float],
    seed: int,
    rounds: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan, math.nan
    rng = Lcg(seed)
    n = len(values)
    means = []
    for _ in range(rounds):
        means.append(sum(values[rng.randrange(n)] for _i in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2.0) * (rounds - 1))]
    hi = means[int((1.0 - alpha / 2.0) * (rounds - 1))]
    return lo, hi


def sign_test_p_one_sided(positive: int, total: int) -> float:
    if total <= 0:
        return math.nan
    return sum(math.comb(total, k) for k in range(positive, total + 1)) / (2**total)


def sign_flip_p_one_sided(values: Sequence[float], seed: int, rounds: int) -> float:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return math.nan
    observed = mean(values)
    rng = Lcg(seed)
    ge = 0
    for _ in range(rounds):
        candidate = mean(value * rng.random_sign() for value in values)
        if candidate >= observed:
            ge += 1
    return (ge + 1) / (rounds + 1)


def scan_metric(
    synth: dict[tuple[int, int], dict],
    meas: dict[tuple[int, int], dict],
    common_keys: Sequence[tuple[int, int]],
    k: int,
    scope: str,
    metric: str,
) -> tuple[list[dict], list[dict]]:
    offsets = offsets_for_k(k, scope)
    pair_rows = []
    per_location_rows = []

    for lora_key in common_keys:
        lora = meas[lora_key]
        if metric == "mag_cosine":
            lora_vec = vector(lora, "meas_mag_bin", "_mean", offsets)
            chirp_prefix = "synth_mag_bin"
            fn = cosine
        elif metric == "rel_db_pearson":
            lora_vec = vector(lora, "meas_rel_db_bin", "_mean", offsets)
            chirp_prefix = "synth_rel_db_bin"
            fn = pearson
        else:
            raise ValueError(metric)

        values_by_label = {}
        self_label = point_label(lora_key)
        for chirp_key in common_keys:
            chirp = synth[chirp_key]
            chirp_vec = vector(chirp, chirp_prefix, "_mean", offsets)
            value = fn(lora_vec, chirp_vec)
            chirp_label = point_label(chirp_key)
            relation = "matched" if chirp_key == lora_key else "mismatched"
            values_by_label[chirp_label] = value
            pair_rows.append(
                {
                    "k": k,
                    "scope": scope,
                    "metric": metric,
                    "lora_label": self_label,
                    "chirp_label": chirp_label,
                    "relation": relation,
                    "similarity": value,
                }
            )

        matched_value = values_by_label[self_label]
        mismatch_values = [
            value
            for label, value in values_by_label.items()
            if label != self_label and math.isfinite(value)
        ]
        per_location_rows.append(
            {
                "k": k,
                "scope": scope,
                "metric": metric,
                "lora_label": self_label,
                "matched": matched_value,
                "mismatch_mean": mean(mismatch_values),
                "mismatch_median": median(mismatch_values),
                "mismatch_max": max(mismatch_values),
                "matched_minus_mismatch_mean": matched_value - mean(mismatch_values),
                "matched_minus_mismatch_median": matched_value - median(mismatch_values),
                "matched_minus_mismatch_max": matched_value - max(mismatch_values),
                "matched_percentile_vs_mismatch": percentile(mismatch_values, matched_value),
                "matched_rank_among_common_chirp_points": rank_desc(values_by_label, self_label),
                "common_chirp_candidate_count": len(common_keys),
                "bin_count": len(offsets),
            }
        )
    return pair_rows, per_location_rows


def run(args: argparse.Namespace) -> dict:
    synth_rows = read_csv(args.input_dir / "02_chirp_synth_point_bins.csv")
    meas_rows = read_csv(args.input_dir / "03_lora_measured_point_bins.csv")
    synth = {point_key(row): row for row in synth_rows}
    meas = {point_key(row): row for row in meas_rows}
    common_keys = sorted(set(synth) & set(meas))

    pair_rows = []
    per_location_rows = []
    for k in args.k_values:
        for scope in ("full", "side"):
            for metric in ("mag_cosine", "rel_db_pearson"):
                current_pairs, current_locations = scan_metric(
                    synth,
                    meas,
                    common_keys,
                    k,
                    scope,
                    metric,
                )
                pair_rows.extend(current_pairs)
                per_location_rows.extend(current_locations)

    summary_rows = []
    summary_payload: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "common_location_count": len(common_keys),
        "common_locations": [point_label(key) for key in common_keys],
        "k_values": args.k_values,
        "metrics": [],
    }

    for k in args.k_values:
        for scope in ("full", "side"):
            for metric in ("mag_cosine", "rel_db_pearson"):
                pairs = [
                    row
                    for row in pair_rows
                    if row["k"] == k and row["scope"] == scope and row["metric"] == metric
                ]
                locs = [
                    row
                    for row in per_location_rows
                    if row["k"] == k and row["scope"] == scope and row["metric"] == metric
                ]
                matched_values = [
                    row["similarity"]
                    for row in pairs
                    if row["relation"] == "matched" and math.isfinite(row["similarity"])
                ]
                mismatch_values = [
                    row["similarity"]
                    for row in pairs
                    if row["relation"] == "mismatched" and math.isfinite(row["similarity"])
                ]
                diff_mean = [row["matched_minus_mismatch_mean"] for row in locs]
                positive = sum(1 for value in diff_mean if value > 0)
                ci_lo, ci_hi = bootstrap_mean_ci(
                    diff_mean,
                    args.seed + k * 1009 + (0 if scope == "full" else 97),
                    args.bootstrap_rounds,
                )
                sign_flip_p = sign_flip_p_one_sided(
                    diff_mean,
                    args.seed + k * 2003 + (0 if metric == "mag_cosine" else 193),
                    args.permutation_rounds,
                )
                ranks = [row["matched_rank_among_common_chirp_points"] for row in locs]
                top1 = sum(1 for rank in ranks if rank == 1)
                top3 = sum(1 for rank in ranks if rank <= 3)
                summary = {
                    "k": k,
                    "scope": scope,
                    "metric": metric,
                    "bin_count": len(offsets_for_k(k, scope)),
                    "matched_count": len(matched_values),
                    "mismatch_pair_count": len(mismatch_values),
                    "matched_mean": describe(matched_values)["mean"],
                    "matched_median": describe(matched_values)["median"],
                    "mismatch_mean": describe(mismatch_values)["mean"],
                    "mismatch_median": describe(mismatch_values)["median"],
                    "mean_diff_vs_mismatch_mean": describe(diff_mean)["mean"],
                    "median_diff_vs_mismatch_mean": describe(diff_mean)["median"],
                    "bootstrap95_diff_low": ci_lo,
                    "bootstrap95_diff_high": ci_hi,
                    "locations_matched_gt_mismatch_mean": positive,
                    "locations_total": len(locs),
                    "sign_test_p": sign_test_p_one_sided(positive, len(locs)),
                    "sign_flip_p": sign_flip_p,
                    "matched_top1_count": top1,
                    "matched_top1_rate": top1 / len(locs) if locs else math.nan,
                    "matched_top3_count": top3,
                    "matched_top3_rate": top3 / len(locs) if locs else math.nan,
                    "median_matched_rank": median(ranks) if ranks else math.nan,
                    "significant_positive": bool(ci_lo > 0 and sign_flip_p < 0.05),
                }
                summary_rows.append(summary)
                summary_payload["metrics"].append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "01_window_metric_summary.csv",
        summary_rows,
        [
            "k",
            "scope",
            "metric",
            "bin_count",
            "matched_count",
            "mismatch_pair_count",
            "matched_mean",
            "matched_median",
            "mismatch_mean",
            "mismatch_median",
            "mean_diff_vs_mismatch_mean",
            "median_diff_vs_mismatch_mean",
            "bootstrap95_diff_low",
            "bootstrap95_diff_high",
            "locations_matched_gt_mismatch_mean",
            "locations_total",
            "sign_test_p",
            "sign_flip_p",
            "matched_top1_count",
            "matched_top1_rate",
            "matched_top3_count",
            "matched_top3_rate",
            "median_matched_rank",
            "significant_positive",
        ],
    )
    write_csv(
        args.output_dir / "02_window_per_location_summary.csv",
        per_location_rows,
        [
            "k",
            "scope",
            "metric",
            "lora_label",
            "matched",
            "mismatch_mean",
            "mismatch_median",
            "mismatch_max",
            "matched_minus_mismatch_mean",
            "matched_minus_mismatch_median",
            "matched_minus_mismatch_max",
            "matched_percentile_vs_mismatch",
            "matched_rank_among_common_chirp_points",
            "common_chirp_candidate_count",
            "bin_count",
        ],
    )
    write_csv(
        args.output_dir / "03_window_pair_scores.csv",
        pair_rows,
        ["k", "scope", "metric", "lora_label", "chirp_label", "relation", "similarity"],
    )
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Chirp-LoRa bin window scan",
                "",
                "This control scans matched-vs-mismatched similarity over LoRa bin windows.",
                "",
                "- `full`: uses `[-K,+K]` including center bin.",
                "- `side`: uses `[-K,-1] U [+1,+K]`, excluding center bin.",
                "- Candidate chirp points are restricted to common chirp/LoRa locations.",
                "- `significant_positive` requires bootstrap lower bound > 0 and sign-flip p < 0.05.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", default="2,3,4,6,8")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--bootstrap-rounds", type=int, default=10000)
    parser.add_argument("--permutation-rounds", type=int, default=10000)
    args = parser.parse_args()
    args.k_values = [int(part.strip()) for part in args.k_values.split(",") if part.strip()]
    if not args.k_values:
        raise ValueError("At least one K value is required")
    if min(args.k_values) < 1:
        raise ValueError("K values must be positive")
    return args


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
