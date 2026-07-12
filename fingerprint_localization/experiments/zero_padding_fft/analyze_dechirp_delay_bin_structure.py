#!/usr/bin/env python3
"""Build an interpretable chirp-to-LoRa dechirp delay-bin structure.

For a LoRa upchirp, an echo delayed by tau is shifted after dechirping by roughly

    delta_bin = -BW * tau

where tau is in seconds.  The measured chirp paths here are mostly sub-us, so
their LoRa-equivalent shifts are much smaller than one FFT bin.  This script
therefore compares low-order structure, not full spectral templates:

  * chirp-predicted fractional-bin centroid,
  * chirp-predicted fractional-bin spread,
  * chirp-predicted left/right sign,
  * LoRa q=4 measured centroid, spread, and left/right asymmetry.

The goal is a more physically explainable motivation result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHIRP_PATHS = (
    ROOT
    / "v2_output/20260623_from_raw/step6c_chirp_structure_original_minus25"
    / "02_stable_equivalent_paths_with_reference_overlap.csv"
)
DEFAULT_CHIRP_POINTS = (
    ROOT
    / "v2_output/20260623_from_raw/step6c_chirp_structure_original_minus25"
    / "01_point_multipath_structure_features.csv"
)
DEFAULT_LORA_Q4_BINS = (
    ROOT
    / "v2_output/20260710_chirp_lora_q4_subbin_projection"
    / "03_lora_measured_point_bins.csv"
)
DEFAULT_LORA_Q4_POINT = (
    ROOT
    / "v2_output/20260624_zero_padding_fft_q4_from_trusted_starts"
    / "point_q_summary.csv"
)
DEFAULT_OUTPUT = ROOT / "v2_output/20260711_dechirp_delay_bin_structure"

BW_HZ = 125_000.0
EPS = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fnum(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        out = float(str(value).strip())
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def inum(value: object, default: int = 0) -> int:
    value = fnum(value, math.nan)
    return int(value) if math.isfinite(value) else default


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def point_key(row: dict[str, object]) -> tuple[int, int]:
    loc = row.get("location_id", row.get("position_id"))
    return inum(row.get("corridor_id")), inum(loc)


def point_label(key: tuple[int, int]) -> str:
    return f"{key[0]}_{key[1]}"


def format_offset(offset: float) -> str:
    rounded = round(offset, 6)
    if rounded == 0:
        rounded = 0.0
    text = str(int(rounded)) if rounded.is_integer() else f"{rounded:.6f}".rstrip("0").rstrip(".")
    return f"{'+' if rounded >= 0 else ''}{text}"


def ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    idx = 0
    while idx < len(ordered):
        end = idx + 1
        while end < len(ordered) and ordered[end][1] == ordered[idx][1]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for original_idx, _value in ordered[idx:end]:
            out[original_idx] = rank
        idx = end
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3 or len(x) != len(y):
        return math.nan
    mx = mean(x)
    my = mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx <= EPS or sy <= EPS:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(ranks(x), ranks(y))


def permutation_p(x: Sequence[float], y: Sequence[float], observed: float, seed: int, rounds: int) -> float:
    if not math.isfinite(observed):
        return math.nan
    rng = random.Random(seed)
    rx = ranks(x)
    ry = ranks(y)
    extreme = 0
    for _ in range(rounds):
        rng.shuffle(ry)
        candidate = pearson(rx, ry)
        if math.isfinite(candidate) and abs(candidate) >= abs(observed) - 1e-15:
            extreme += 1
    return (extreme + 1) / (rounds + 1)


def finite_pairs(rows: Sequence[dict], x_name: str, y_name: str) -> tuple[list[float], list[float], list[str]]:
    xs = []
    ys = []
    labels = []
    for row in rows:
        x = fnum(row.get(x_name))
        y = fnum(row.get(y_name))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
            labels.append(str(row.get("position_key")))
    return xs, ys, labels


def leave_one_out_range(x: Sequence[float], y: Sequence[float], labels: Sequence[str]) -> dict:
    values = []
    for idx, label in enumerate(labels):
        xx = list(x[:idx]) + list(x[idx + 1 :])
        yy = list(y[:idx]) + list(y[idx + 1 :])
        values.append({"removed": label, "rho": spearman(xx, yy)})
    low = min(values, key=lambda row: row["rho"])
    high = max(values, key=lambda row: row["rho"])
    return {
        "loo_rho_min": low["rho"],
        "loo_rho_min_removed": low["removed"],
        "loo_rho_max": high["rho"],
        "loo_rho_max_removed": high["removed"],
    }


def summarize(values: Sequence[float]) -> dict[str, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {"mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def q4_measured_structure(row: dict[str, str]) -> dict[str, float]:
    offsets = [idx / 4.0 for idx in range(-8, 9)]
    powers = []
    for offset in offsets:
        tag = format_offset(offset)
        mag = fnum(row.get(f"meas_mag_norm_bin_{tag}_mean"), 0.0)
        powers.append(mag * mag)
    total = sum(powers) + EPS
    left = sum(power for offset, power in zip(offsets, powers) if offset < 0)
    right = sum(power for offset, power in zip(offsets, powers) if offset > 0)
    center = powers[offsets.index(0.0)] + EPS
    centroid = sum(offset * power for offset, power in zip(offsets, powers)) / total
    rms = math.sqrt(sum((offset - centroid) ** 2 * power for offset, power in zip(offsets, powers)) / total)
    inner_left = sum(power for offset, power in zip(offsets, powers) if -0.75 <= offset < 0)
    inner_right = sum(power for offset, power in zip(offsets, powers) if 0 < offset <= 0.75)
    return {
        "lora_q4_centroid_bin": centroid,
        "lora_q4_abs_centroid_bin": abs(centroid),
        "lora_q4_rms_width_bin": rms,
        "lora_q4_lr_asymmetry": (right - left) / (right + left + EPS),
        "lora_q4_inner_lr_asymmetry": (inner_right - inner_left) / (inner_right + inner_left + EPS),
        "lora_q4_side_to_center_power": (left + right) / center,
    }


def build_chirp_delay_bin_structure(
    path_rows: Sequence[dict[str, str]],
    point_rows: dict[tuple[int, int], dict[str, str]],
) -> tuple[dict[tuple[int, int], dict], list[dict]]:
    by_key: dict[tuple[int, int], list[dict]] = {}
    path_out = []
    for row in path_rows:
        if abs(fnum(row.get("threshold_db")) + 25.0) > 1e-6:
            continue
        if str(row.get("stable_20pct", "")).strip().lower() not in {"true", "1"}:
            continue
        key = point_key(row)
        delay_us = fnum(row.get("delay_center_us"), 0.0)
        relative_amp_db = fnum(row.get("amplitude_db_median"), -300.0)
        recurrence = fnum(row.get("recurrence_fraction"), 0.0)
        amp_weight = math.sqrt(max(recurrence, 0.0)) * 10.0 ** (relative_amp_db / 20.0)
        power_weight = max(recurrence, 0.0) * 10.0 ** (relative_amp_db / 10.0)
        delay_bin = -BW_HZ * delay_us * 1e-6
        item = {
            "corridor_id": key[0],
            "location_id": key[1],
            "position_key": point_label(key),
            "delay_us": delay_us,
            "lora_equivalent_delay_bin": delay_bin,
            "relative_amp_db": relative_amp_db,
            "recurrence_fraction": recurrence,
            "amp_weight": amp_weight,
            "power_weight": power_weight,
            "reference_like_delay": row.get("reference_like_delay", ""),
        }
        by_key.setdefault(key, []).append(item)
        path_out.append(item)

    out = {}
    for key, point in point_rows.items():
        paths = by_key.get(key, [])
        # Include the main path at 0 bin with unit weight.  This is the
        # noncoherent low-order envelope predicted by the delay-bin model.
        bins_amp = [0.0] + [fnum(path["lora_equivalent_delay_bin"], 0.0) for path in paths]
        amp_weights = [1.0] + [fnum(path["amp_weight"], 0.0) for path in paths]
        power_weights = [1.0] + [fnum(path["power_weight"], 0.0) for path in paths]
        amp_total = sum(amp_weights) + EPS
        power_total = sum(power_weights) + EPS

        amp_centroid = sum(b * w for b, w in zip(bins_amp, amp_weights)) / amp_total
        power_centroid = sum(b * w for b, w in zip(bins_amp, power_weights)) / power_total
        amp_spread = math.sqrt(sum((b - amp_centroid) ** 2 * w for b, w in zip(bins_amp, amp_weights)) / amp_total)
        power_spread = math.sqrt(sum((b - power_centroid) ** 2 * w for b, w in zip(bins_amp, power_weights)) / power_total)
        secondary_amp_total = sum(amp_weights[1:])
        secondary_power_total = sum(power_weights[1:])
        neg_amp = sum(w for b, w in zip(bins_amp[1:], amp_weights[1:]) if b < 0)
        pos_amp = sum(w for b, w in zip(bins_amp[1:], amp_weights[1:]) if b > 0)
        neg_power = sum(w for b, w in zip(bins_amp[1:], power_weights[1:]) if b < 0)
        pos_power = sum(w for b, w in zip(bins_amp[1:], power_weights[1:]) if b > 0)

        out[key] = {
            "corridor_id": key[0],
            "location_id": key[1],
            "position_key": point_label(key),
            "state": point.get("state", ""),
            "distance_m": fnum(point.get("distance_m")),
            "is_reference_point_32": str(point.get("is_reference_point_32", "")).strip().lower() == "true",
            "stable_path_count": len(paths),
            "chirp_delay_bin_amp_centroid": amp_centroid,
            "chirp_delay_bin_power_centroid": power_centroid,
            "chirp_delay_bin_amp_abs_centroid": abs(amp_centroid),
            "chirp_delay_bin_power_abs_centroid": abs(power_centroid),
            "chirp_delay_bin_amp_spread": amp_spread,
            "chirp_delay_bin_power_spread": power_spread,
            "chirp_delay_bin_max_abs": max([abs(b) for b in bins_amp] or [0.0]),
            "chirp_delay_bin_secondary_amp_sum": secondary_amp_total,
            "chirp_delay_bin_secondary_power_sum": secondary_power_total,
            "chirp_delay_bin_amp_lr_asymmetry": (pos_amp - neg_amp) / (pos_amp + neg_amp + EPS),
            "chirp_delay_bin_power_lr_asymmetry": (pos_power - neg_power) / (pos_power + neg_power + EPS),
            "chirp_delay_bin_expected_unresolved": max([abs(b) for b in bins_amp] or [0.0]) < 0.25,
            "stable_equivalent_paths": point.get("stable_equivalent_paths", ""),
        }
    return out, path_out


def make_joined_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    point_rows = {point_key(row): row for row in read_csv(args.chirp_point_csv)}
    chirp, path_rows = build_chirp_delay_bin_structure(read_csv(args.chirp_path_csv), point_rows)
    q4_bins = {point_key(row): row for row in read_csv(args.lora_q4_bins_csv)}
    q4_point = {point_key(row): row for row in read_csv(args.lora_q4_point_csv)}
    joined = []
    for key in sorted(set(chirp) & set(q4_bins)):
        row = dict(chirp[key])
        row.update(q4_measured_structure(q4_bins[key]))
        point = q4_point.get(key, {})
        row["lora_interpolated_peak_offset_bin"] = fnum(point.get("interpolated_peak_offset_bins_mean_point_mean"))
        row["lora_asymmetry_mean"] = fnum(point.get("asymmetry_mean_point_mean"))
        row["lora_side_power_fraction"] = fnum(point.get("side_power_fraction_mean_point_mean"))
        row["lora_secondary_peak_rel_db"] = fnum(point.get("secondary_peak_rel_db_mean_point_mean"))
        joined.append(row)
    return joined, path_rows


def correlation_table(rows: Sequence[dict], scope: str, seed: int, rounds: int) -> list[dict]:
    pairs = [
        ("chirp_delay_bin_amp_centroid", "lora_q4_centroid_bin", "centroid_sign"),
        ("chirp_delay_bin_power_centroid", "lora_q4_centroid_bin", "centroid_sign_power"),
        ("chirp_delay_bin_amp_abs_centroid", "lora_q4_abs_centroid_bin", "centroid_magnitude"),
        ("chirp_delay_bin_amp_spread", "lora_q4_rms_width_bin", "spread"),
        ("chirp_delay_bin_power_spread", "lora_q4_rms_width_bin", "spread_power"),
        ("chirp_delay_bin_amp_lr_asymmetry", "lora_q4_lr_asymmetry", "left_right_asymmetry"),
        ("chirp_delay_bin_amp_lr_asymmetry", "lora_q4_inner_lr_asymmetry", "inner_left_right_asymmetry"),
        ("chirp_delay_bin_secondary_amp_sum", "lora_q4_side_to_center_power", "secondary_strength_to_side_power"),
        ("chirp_delay_bin_secondary_amp_sum", "lora_secondary_peak_rel_db", "secondary_strength_to_lora_secondary_peak"),
        ("chirp_delay_bin_max_abs", "lora_q4_abs_centroid_bin", "max_delay_to_centroid_magnitude"),
        ("stable_path_count", "lora_q4_centroid_bin", "path_count_to_centroid"),
        ("stable_path_count", "lora_q4_lr_asymmetry", "path_count_to_lr_asymmetry"),
    ]
    out = []
    for idx, (x_name, y_name, hypothesis) in enumerate(pairs):
        xs, ys, labels = finite_pairs(rows, x_name, y_name)
        if len(xs) < 6:
            continue
        rho = spearman(xs, ys)
        pear = pearson(xs, ys)
        item = {
            "scope": scope,
            "hypothesis": hypothesis,
            "chirp_structure_feature": x_name,
            "lora_structure_feature": y_name,
            "n": len(xs),
            "spearman_rho": rho,
            "pearson_r": pear,
            "perm_p_two_sided": permutation_p(xs, ys, rho, seed + idx * 17, rounds),
        }
        item.update(leave_one_out_range(xs, ys, labels))
        out.append(item)
    return out


def run(args: argparse.Namespace) -> dict:
    joined, path_rows = make_joined_rows(args)
    no_ref = [
        row for row in joined
        if not bool(row.get("is_reference_point_32")) and inum(row.get("location_id")) != 32
    ]
    all_corr = correlation_table(joined, "all_overlap_points", args.seed, args.permutation_rounds)
    no_ref_corr = correlation_table(no_ref, "exclude_reference_point32", args.seed + 1000, args.permutation_rounds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined_fields = [
        "corridor_id",
        "location_id",
        "position_key",
        "state",
        "distance_m",
        "is_reference_point_32",
        "stable_path_count",
        "chirp_delay_bin_amp_centroid",
        "chirp_delay_bin_power_centroid",
        "chirp_delay_bin_amp_abs_centroid",
        "chirp_delay_bin_power_abs_centroid",
        "chirp_delay_bin_amp_spread",
        "chirp_delay_bin_power_spread",
        "chirp_delay_bin_max_abs",
        "chirp_delay_bin_secondary_amp_sum",
        "chirp_delay_bin_secondary_power_sum",
        "chirp_delay_bin_amp_lr_asymmetry",
        "chirp_delay_bin_power_lr_asymmetry",
        "chirp_delay_bin_expected_unresolved",
        "lora_q4_centroid_bin",
        "lora_q4_abs_centroid_bin",
        "lora_q4_rms_width_bin",
        "lora_q4_lr_asymmetry",
        "lora_q4_inner_lr_asymmetry",
        "lora_q4_side_to_center_power",
        "lora_interpolated_peak_offset_bin",
        "lora_asymmetry_mean",
        "lora_side_power_fraction",
        "lora_secondary_peak_rel_db",
        "stable_equivalent_paths",
    ]
    path_fields = [
        "corridor_id",
        "location_id",
        "position_key",
        "delay_us",
        "lora_equivalent_delay_bin",
        "relative_amp_db",
        "recurrence_fraction",
        "amp_weight",
        "power_weight",
        "reference_like_delay",
    ]
    corr_fields = [
        "scope",
        "hypothesis",
        "chirp_structure_feature",
        "lora_structure_feature",
        "n",
        "spearman_rho",
        "pearson_r",
        "perm_p_two_sided",
        "loo_rho_min",
        "loo_rho_min_removed",
        "loo_rho_max",
        "loo_rho_max_removed",
    ]
    write_csv(args.output_dir / "01_delay_bin_joined_structure.csv", joined, joined_fields)
    write_csv(args.output_dir / "02_path_delay_bin_mapping.csv", path_rows, path_fields)
    write_csv(args.output_dir / "03_delay_bin_structure_correlations.csv", all_corr + no_ref_corr, corr_fields)

    delay_bins = [fnum(row.get("lora_equivalent_delay_bin")) for row in path_rows]
    max_bins = [fnum(row.get("chirp_delay_bin_max_abs")) for row in joined]
    unresolved_count = sum(1 for row in joined if row.get("chirp_delay_bin_expected_unresolved"))
    summary = {
        "overlap_point_count": len(joined),
        "exclude_reference_point32_count": len(no_ref),
        "lora_bandwidth_hz": BW_HZ,
        "delay_to_bin_mapping": "delta_bin = -BW * tau; tau in seconds",
        "path_delay_bin_summary": summarize(delay_bins),
        "point_max_abs_delay_bin_summary": summarize(max_bins),
        "points_with_all_stable_paths_within_q4_subbin_0p25": unresolved_count,
        "all_overlap_correlations": all_corr,
        "exclude_reference_point32_correlations": no_ref_corr,
        "interpretation": [
            "Most wideband stable paths map to much less than one LoRa FFT bin, so LoRa should not resolve them as separate peaks.",
            "A physically interpretable link should appear, if at all, through low-order q4 main-peak deformation: centroid, width, and left/right asymmetry.",
            "This output is intentionally small-hypothesis and structure-based rather than feature-mining-based.",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Dechirp Delay-Bin Structure Analysis",
                "",
                "This run maps chirp path delays into LoRa dechirped FFT-bin shifts using `delta_bin = -BW*tau`.",
                "",
                "The analysis compares a few low-order, physically interpretable structure features:",
                "",
                "- chirp-predicted fractional-bin centroid vs LoRa q=4 centroid",
                "- chirp-predicted spread vs LoRa q=4 main-peak width",
                "- chirp-predicted left/right delay-bin asymmetry vs LoRa q=4 left/right spectral asymmetry",
                "",
                "Because path phases are not available in the stable-path table, this is a noncoherent structural analysis rather than a deterministic spectral prediction.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chirp-path-csv", type=Path, default=DEFAULT_CHIRP_PATHS)
    parser.add_argument("--chirp-point-csv", type=Path, default=DEFAULT_CHIRP_POINTS)
    parser.add_argument("--lora-q4-bins-csv", type=Path, default=DEFAULT_LORA_Q4_BINS)
    parser.add_argument("--lora-q4-point-csv", type=Path, default=DEFAULT_LORA_Q4_POINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutation-rounds", type=int, default=49999)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
