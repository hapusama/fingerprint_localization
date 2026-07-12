#!/usr/bin/env python3
"""Explore point-wise links between wideband chirp multipath and LoRa spectra.

This analysis deliberately avoids chirp-to-LoRa template matching.  Instead it
asks whether wideband physical summaries, such as path count, secondary energy,
and delay spread, move together with LoRa point-level narrowband observables.
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
DEFAULT_CHIRP_POINT = (
    ROOT
    / "v2_output/20260623_from_raw/step6c_chirp_structure_original_minus25"
    / "01_point_multipath_structure_features.csv"
)
DEFAULT_CHIRP_PATHS = (
    ROOT
    / "v2_output/20260623_from_raw/step6c_chirp_structure_original_minus25"
    / "02_stable_equivalent_paths_with_reference_overlap.csv"
)
DEFAULT_LORA_Q4_POINT = (
    ROOT
    / "v2_output/20260624_zero_padding_fft_q4_from_trusted_starts"
    / "point_q_summary.csv"
)
DEFAULT_LORA_Q1_SHAPE = (
    ROOT
    / "v2_output/20260624_chirp_multipath_data_processing"
    / "05_lora_main_peak_shape_point_summary.csv"
)
DEFAULT_LORA_Q4_BINS = (
    ROOT
    / "v2_output/20260710_chirp_lora_q4_subbin_projection"
    / "03_lora_measured_point_bins.csv"
)
DEFAULT_OUTPUT = ROOT / "v2_output/20260711_chirp_lora_point_physics"

BW_HZ = 125_000.0
EPS = 1e-12


CHIRP_FEATURES = [
    "mean_peak_count",
    "fraction_segments_2plus",
    "stable_secondary_path_count",
    "total_stable_path_count_including_main",
    "strong_secondary_path_count_40pct",
    "stable_path_occupancy_sum",
    "raw_secondary_peak_mean",
    "unstable_secondary_peak_load",
    "stable_detection_explained_fraction",
    "secondary_effective_power_sum",
    "main_effective_power_fraction",
    "effective_path_number",
    "entropy_effective_path_number",
    "equivalent_mean_delay_us",
    "equivalent_rms_delay_us",
    "mean_abs_stable_delay_us",
    "max_abs_stable_delay_us",
    "stable_delay_span_us",
    "precursor_path_count",
    "postcursor_path_count",
    "precursor_effective_power",
    "postcursor_effective_power",
    "post_to_precursor_power_ratio",
    "reference_like_stable_path_count",
    "nonreference_delay_path_count",
    "reference_like_path_fraction",
    "strongest_secondary_delay_us",
    "strongest_secondary_amplitude_db",
    "strongest_secondary_recurrence",
    "nb_path_count",
    "nb_nonreference_path_count",
    "nb_sum_effective_amp",
    "nb_sum_effective_power",
    "nb_max_phase_span_rad",
    "nb_rms_phase_span_rad",
    "nb_amp_phase_score",
    "nb_power_phase_score",
    "nb_signed_amp_delay_moment_us",
    "nb_signed_power_delay_moment_us",
    "nb_post_minus_pre_amp",
    "nb_post_minus_pre_power",
    "nb_max_effective_amp",
]

LORA_FEATURES = [
    "detect_score_db_point_mean",
    "score_db_mean_point_mean",
    "score_db_std_point_mean",
    "peak_offset_bins_mean_point_mean",
    "peak_offset_bins_std_point_mean",
    "interpolated_peak_offset_bins_mean_point_mean",
    "interpolated_peak_offset_bins_std_point_mean",
    "local_peak_width_3db_bins_mean_point_mean",
    "side_power_fraction_mean_point_mean",
    "left_power_fraction_mean_point_mean",
    "right_power_fraction_mean_point_mean",
    "asymmetry_mean_point_mean",
    "asymmetry_std_point_mean",
    "curvature_db_per_zp_bin_mean_point_mean",
    "secondary_peak_offset_bins_mean_point_mean",
    "secondary_peak_rel_db_mean_point_mean",
    "q1_side_main_mean",
    "q1_asymmetry_mean",
    "q1_curvature_mean",
    "q1_fractional_peak_offset_mean",
    "q1_main_peak_ratio_mean",
    "q1_side_power_fraction_mean",
    "q4_bin_power_side_fraction",
    "q4_bin_power_left_fraction",
    "q4_bin_power_right_fraction",
    "q4_bin_lr_asymmetry",
    "q4_bin_centroid",
    "q4_bin_abs_centroid",
    "q4_bin_rms_width",
    "q4_bin_skew",
    "q4_bin_peak_offset",
    "q4_bin_side_to_center_power",
    "q4_bin_inner_lr_asymmetry",
    "q4_bin_outer_lr_asymmetry",
    "q4_complex_phase_slope",
    "q4_complex_phase_rms_residual",
    "q4_complex_phase_lr_asymmetry",
]


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


def to_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def to_int(value: object, default: int = 0) -> int:
    num = to_float(value, math.nan)
    return int(num) if math.isfinite(num) else default


def fast_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def point_key(row: dict[str, object], location_field: str = "location_id") -> tuple[int, int]:
    loc = row.get(location_field)
    if loc is None:
        loc = row.get("position_id")
    return to_int(row.get("corridor_id")), to_int(loc)


def point_label(key: tuple[int, int]) -> str:
    return f"{key[0]}_{key[1]}"


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
    mx = fast_mean(x)
    my = fast_mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx <= EPS or sy <= EPS:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(ranks(x), ranks(y))


def permutation_p_two_sided(
    x: Sequence[float],
    y: Sequence[float],
    observed: float,
    rng: random.Random,
    rounds: int,
) -> float:
    if not math.isfinite(observed) or rounds <= 0:
        return math.nan
    x_rank = ranks(x)
    y_rank = ranks(y)
    extreme = 0
    y_perm = list(y_rank)
    for _ in range(rounds):
        rng.shuffle(y_perm)
        candidate = pearson(x_rank, y_perm)
        if math.isfinite(candidate) and abs(candidate) >= abs(observed) - 1e-15:
            extreme += 1
    return (extreme + 1) / (rounds + 1)


def bh_q_values(rows: list[dict], p_field: str = "perm_p_two_sided") -> None:
    valid = [
        (idx, to_float(row.get(p_field)))
        for idx, row in enumerate(rows)
        if math.isfinite(to_float(row.get(p_field)))
    ]
    valid.sort(key=lambda item: item[1])
    m = len(valid)
    adjusted = [math.nan] * len(rows)
    running = 1.0
    for rank_idx in range(m, 0, -1):
        row_idx, p_value = valid[rank_idx - 1]
        running = min(running, p_value * m / rank_idx)
        adjusted[row_idx] = running
    for idx, value in enumerate(adjusted):
        rows[idx]["bh_q"] = value


def finite_pairs(rows: Sequence[dict], x_name: str, y_name: str) -> tuple[list[float], list[float]]:
    xs = []
    ys = []
    for row in rows:
        x = to_float(row.get(x_name))
        y = to_float(row.get(y_name))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {"mean": math.nan, "median": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": fast_mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def format_offset(offset: float) -> str:
    rounded = round(offset, 6)
    if rounded == 0:
        rounded = 0.0
    text = str(int(rounded)) if rounded.is_integer() else f"{rounded:.6f}".rstrip("0").rstrip(".")
    return f"{'+' if rounded >= 0 else ''}{text}"


def unwrap_phase(phases: Sequence[float]) -> list[float]:
    if not phases:
        return []
    out = [phases[0]]
    for phase in phases[1:]:
        candidate = phase
        while candidate - out[-1] > math.pi:
            candidate -= 2.0 * math.pi
        while candidate - out[-1] < -math.pi:
            candidate += 2.0 * math.pi
        out.append(candidate)
    return out


def weighted_linear_fit(
    x: Sequence[float],
    y: Sequence[float],
    w: Sequence[float],
) -> tuple[float, float, float]:
    total_w = sum(w)
    if len(x) < 3 or total_w <= EPS:
        return math.nan, math.nan, math.nan
    mx = sum(weight * value for weight, value in zip(w, x)) / total_w
    my = sum(weight * value for weight, value in zip(w, y)) / total_w
    denom = sum(weight * (value - mx) ** 2 for weight, value in zip(w, x))
    if denom <= EPS:
        return math.nan, math.nan, math.nan
    slope = sum(weight * (xx - mx) * (yy - my) for xx, yy, weight in zip(x, y, w)) / denom
    intercept = my - slope * mx
    rms = math.sqrt(
        sum(weight * (yy - (intercept + slope * xx)) ** 2 for xx, yy, weight in zip(x, y, w))
        / total_w
    )
    return slope, intercept, rms


def build_path_observability(path_rows: Sequence[dict[str, str]]) -> tuple[dict[tuple[int, int], dict], list[dict]]:
    by_key: dict[tuple[int, int], list[dict]] = {}
    detail_rows = []
    for row in path_rows:
        if abs(to_float(row.get("threshold_db")) - (-25.0)) > 1e-6:
            continue
        if str(row.get("stable_20pct", "")).strip().lower() not in {"true", "1"}:
            continue
        key = point_key(row)
        delay_us = to_float(row.get("delay_center_us"), 0.0)
        recurrence = to_float(row.get("recurrence_fraction"), 0.0)
        amp_db = to_float(row.get("amplitude_db_median"), -300.0)
        amp_ratio = 10.0 ** (amp_db / 20.0)
        power_ratio = 10.0 ** (amp_db / 10.0)
        effective_amp = math.sqrt(max(recurrence, 0.0)) * amp_ratio
        effective_power = max(recurrence, 0.0) * power_ratio
        phase_span = 2.0 * math.pi * BW_HZ * abs(delay_us) * 1e-6
        detail = {
            "corridor_id": key[0],
            "location_id": key[1],
            "position_key": point_label(key),
            "delay_us": delay_us,
            "amplitude_db_median": amp_db,
            "recurrence_fraction": recurrence,
            "reference_like_delay": row.get("reference_like_delay"),
            "effective_amp_sqrt_recurrence": effective_amp,
            "effective_power_recurrence": effective_power,
            "lora_full_bw_phase_span_rad": phase_span,
            "amp_phase_score": effective_amp * phase_span,
            "power_phase_score": effective_power * phase_span,
        }
        by_key.setdefault(key, []).append(detail)
        detail_rows.append(detail)

    out = {}
    for key, paths in by_key.items():
        amp_values = [to_float(path["effective_amp_sqrt_recurrence"], 0.0) for path in paths]
        power_values = [to_float(path["effective_power_recurrence"], 0.0) for path in paths]
        phase_values = [to_float(path["lora_full_bw_phase_span_rad"], 0.0) for path in paths]
        delays = [to_float(path["delay_us"], 0.0) for path in paths]
        nonref = [
            path for path in paths
            if str(path.get("reference_like_delay", "")).strip().lower() not in {"true", "1"}
        ]
        total_amp = sum(amp_values)
        total_power = sum(power_values)
        rms_phase = math.sqrt(sum((amp * phase) ** 2 for amp, phase in zip(amp_values, phase_values)))
        post_amp = sum(amp for amp, delay in zip(amp_values, delays) if delay > 0)
        pre_amp = sum(amp for amp, delay in zip(amp_values, delays) if delay < 0)
        post_power = sum(power for power, delay in zip(power_values, delays) if delay > 0)
        pre_power = sum(power for power, delay in zip(power_values, delays) if delay < 0)
        out[key] = {
            "nb_path_count": len(paths),
            "nb_nonreference_path_count": len(nonref),
            "nb_sum_effective_amp": total_amp,
            "nb_sum_effective_power": total_power,
            "nb_max_phase_span_rad": max(phase_values) if phase_values else 0.0,
            "nb_rms_phase_span_rad": rms_phase,
            "nb_amp_phase_score": sum(
                amp * phase for amp, phase in zip(amp_values, phase_values)
            ),
            "nb_power_phase_score": sum(
                power * phase for power, phase in zip(power_values, phase_values)
            ),
            "nb_signed_amp_delay_moment_us": sum(
                amp * delay for amp, delay in zip(amp_values, delays)
            ),
            "nb_signed_power_delay_moment_us": sum(
                power * delay for power, delay in zip(power_values, delays)
            ),
            "nb_post_minus_pre_amp": post_amp - pre_amp,
            "nb_post_minus_pre_power": post_power - pre_power,
            "nb_max_effective_amp": max(amp_values) if amp_values else 0.0,
        }
    return out, detail_rows


def q4_bin_features(row: dict[str, str]) -> dict[str, float]:
    offsets = [idx / 4.0 for idx in range(-8, 9)]
    mags = []
    complex_values = []
    for offset in offsets:
        tag = format_offset(offset)
        mag = to_float(row.get(f"meas_mag_norm_bin_{tag}_mean"))
        real = to_float(row.get(f"meas_real_norm_bin_{tag}_mean"))
        imag = to_float(row.get(f"meas_imag_norm_bin_{tag}_mean"))
        mags.append(mag)
        complex_values.append(complex(real, imag) if math.isfinite(real) and math.isfinite(imag) else 0j)
    powers = [mag * mag if math.isfinite(mag) else 0.0 for mag in mags]
    total = sum(powers) + EPS
    center_power = powers[offsets.index(0.0)] + EPS
    side_power = total - center_power
    left_power = sum(power for offset, power in zip(offsets, powers) if offset < 0)
    right_power = sum(power for offset, power in zip(offsets, powers) if offset > 0)
    centroid = sum(offset * power for offset, power in zip(offsets, powers)) / total
    rms_width = math.sqrt(sum((offset - centroid) ** 2 * power for offset, power in zip(offsets, powers)) / total)
    skew = (
        sum((offset - centroid) ** 3 * power for offset, power in zip(offsets, powers))
        / total
        / ((rms_width + EPS) ** 3)
    )
    peak_offset = offsets[max(range(len(powers)), key=lambda idx: powers[idx])]
    inner_left = sum(power for offset, power in zip(offsets, powers) if -1.0 <= offset < 0)
    inner_right = sum(power for offset, power in zip(offsets, powers) if 0 < offset <= 1.0)
    outer_left = sum(power for offset, power in zip(offsets, powers) if offset < -1.0)
    outer_right = sum(power for offset, power in zip(offsets, powers) if offset > 1.0)

    phase_offsets = [offset for offset in offsets if offset != 0.0]
    phases = [
        math.atan2(value.imag, value.real)
        for offset, value in zip(offsets, complex_values)
        if offset != 0.0
    ]
    phase_weights = [
        max(abs(value), EPS) ** 2
        for offset, value in zip(offsets, complex_values)
        if offset != 0.0
    ]
    unwrapped = unwrap_phase(phases)
    slope, _intercept, phase_rms = weighted_linear_fit(phase_offsets, unwrapped, phase_weights)
    left_phase = [
        math.atan2(value.imag, value.real)
        for offset, value in zip(offsets, complex_values)
        if offset < 0
    ]
    right_phase = [
        math.atan2(value.imag, value.real)
        for offset, value in zip(offsets, complex_values)
        if offset > 0
    ]
    phase_lr = fast_mean(right_phase) - fast_mean(left_phase) if left_phase and right_phase else math.nan
    while math.isfinite(phase_lr) and phase_lr > math.pi:
        phase_lr -= 2.0 * math.pi
    while math.isfinite(phase_lr) and phase_lr < -math.pi:
        phase_lr += 2.0 * math.pi

    return {
        "q4_bin_power_side_fraction": side_power / total,
        "q4_bin_power_left_fraction": left_power / total,
        "q4_bin_power_right_fraction": right_power / total,
        "q4_bin_lr_asymmetry": (right_power - left_power) / (right_power + left_power + EPS),
        "q4_bin_centroid": centroid,
        "q4_bin_abs_centroid": abs(centroid),
        "q4_bin_rms_width": rms_width,
        "q4_bin_skew": skew,
        "q4_bin_peak_offset": peak_offset,
        "q4_bin_side_to_center_power": side_power / center_power,
        "q4_bin_inner_lr_asymmetry": (inner_right - inner_left) / (inner_right + inner_left + EPS),
        "q4_bin_outer_lr_asymmetry": (outer_right - outer_left) / (outer_right + outer_left + EPS),
        "q4_complex_phase_slope": slope,
        "q4_complex_phase_rms_residual": phase_rms,
        "q4_complex_phase_lr_asymmetry": phase_lr,
    }


def build_joined_rows(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    chirp_rows = {point_key(row): row for row in read_csv(args.chirp_point_csv)}
    q4_rows = {point_key(row, "position_id"): row for row in read_csv(args.lora_q4_point_csv)}
    q1_rows = {point_key(row): row for row in read_csv(args.lora_q1_shape_csv)}
    bin_rows = {point_key(row): row for row in read_csv(args.lora_q4_bins_csv)}
    path_features, path_detail_rows = build_path_observability(read_csv(args.chirp_path_csv))

    joined = []
    for key in sorted(set(chirp_rows) & set(q4_rows)):
        chirp = chirp_rows[key]
        q4 = q4_rows.get(key, {})
        q1 = q1_rows.get(key, {})
        bins = bin_rows.get(key, {})
        row: dict[str, object] = {
            "corridor_id": key[0],
            "location_id": key[1],
            "position_key": point_label(key),
            "state": chirp.get("state", ""),
            "distance_m": to_float(chirp.get("distance_m")),
            "is_reference_point_32": str(chirp.get("is_reference_point_32", "")).lower() == "true",
            "structure_type": chirp.get("structure_type", ""),
            "stable_equivalent_paths": chirp.get("stable_equivalent_paths", ""),
            "lora_packet_count": to_int(q4.get("packet_count")),
        }
        for name in CHIRP_FEATURES:
            if name.startswith("nb_"):
                row[name] = path_features.get(key, {}).get(name, 0.0)
            else:
                row[name] = to_float(chirp.get(name))
        for name in LORA_FEATURES:
            if name.startswith("q1_"):
                source_name = name[3:]
                row[name] = to_float(q1.get(source_name))
            elif name.startswith("q4_bin_") or name.startswith("q4_complex_"):
                continue
            else:
                row[name] = to_float(q4.get(name))
        row.update(q4_bin_features(bins) if bins else {})
        joined.append(row)
    return joined, path_detail_rows


def correlation_rows(
    joined: Sequence[dict],
    scope: str,
    seed: int,
    permutation_rounds: int,
) -> list[dict]:
    rows = []
    rng = random.Random(seed)
    for chirp_feature in CHIRP_FEATURES:
        for lora_feature in LORA_FEATURES:
            xs, ys = finite_pairs(joined, chirp_feature, lora_feature)
            if len(xs) < 6:
                continue
            rho = spearman(xs, ys)
            pear = pearson(xs, ys)
            p_value = permutation_p_two_sided(xs, ys, rho, rng, permutation_rounds)
            rows.append(
                {
                    "scope": scope,
                    "chirp_feature": chirp_feature,
                    "lora_feature": lora_feature,
                    "n": len(xs),
                    "spearman_rho": rho,
                    "pearson_r": pear,
                    "perm_p_two_sided": p_value,
                    "abs_spearman": abs(rho) if math.isfinite(rho) else math.nan,
                }
            )
    bh_q_values(rows)
    return rows


def top_rows(rows: Sequence[dict], limit: int = 40) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            -to_float(row.get("abs_spearman"), -1.0),
            to_float(row.get("perm_p_two_sided"), 1.0),
            str(row.get("chirp_feature")),
            str(row.get("lora_feature")),
        ),
    )[:limit]


def point_diagnostics(joined: Sequence[dict]) -> list[dict]:
    rows = []
    for row in sorted(joined, key=lambda item: -to_float(item.get("stable_secondary_path_count"), 0.0)):
        rows.append(
            {
                "position_key": row["position_key"],
                "state": row["state"],
                "distance_m": row["distance_m"],
                "is_reference_point_32": row["is_reference_point_32"],
                "stable_secondary_path_count": row.get("stable_secondary_path_count"),
                "secondary_effective_power_sum": row.get("secondary_effective_power_sum"),
                "main_effective_power_fraction": row.get("main_effective_power_fraction"),
                "equivalent_rms_delay_us": row.get("equivalent_rms_delay_us"),
                "max_abs_stable_delay_us": row.get("max_abs_stable_delay_us"),
                "nb_max_phase_span_rad": row.get("nb_max_phase_span_rad"),
                "nb_amp_phase_score": row.get("nb_amp_phase_score"),
                "q4_bin_lr_asymmetry": row.get("q4_bin_lr_asymmetry"),
                "q4_bin_centroid": row.get("q4_bin_centroid"),
                "q4_bin_side_to_center_power": row.get("q4_bin_side_to_center_power"),
                "asymmetry_mean_point_mean": row.get("asymmetry_mean_point_mean"),
                "interpolated_peak_offset_bins_mean_point_mean": row.get("interpolated_peak_offset_bins_mean_point_mean"),
                "stable_equivalent_paths": row.get("stable_equivalent_paths"),
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict:
    joined, path_detail_rows = build_joined_rows(args)
    joined_no_ref = [
        row for row in joined
        if not bool(row.get("is_reference_point_32")) and to_int(row.get("location_id")) != 32
    ]

    all_corr = correlation_rows(
        joined,
        "all_overlap_points",
        args.seed,
        args.permutation_rounds,
    )
    no_ref_corr = correlation_rows(
        joined_no_ref,
        "exclude_reference_point32",
        args.seed + 101,
        args.permutation_rounds,
    )
    combined_top = top_rows(all_corr, args.top_limit) + top_rows(no_ref_corr, args.top_limit)
    combined_top = sorted(
        combined_top,
        key=lambda row: (
            row["scope"],
            -to_float(row.get("abs_spearman"), -1.0),
            to_float(row.get("perm_p_two_sided"), 1.0),
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "01_joined_point_features.csv",
        joined,
        [
            "corridor_id",
            "location_id",
            "position_key",
            "state",
            "distance_m",
            "is_reference_point_32",
            "structure_type",
            "lora_packet_count",
            *CHIRP_FEATURES,
            *LORA_FEATURES,
            "stable_equivalent_paths",
        ],
    )
    write_csv(
        args.output_dir / "02_stable_path_lora_observability.csv",
        path_detail_rows,
        [
            "corridor_id",
            "location_id",
            "position_key",
            "delay_us",
            "amplitude_db_median",
            "recurrence_fraction",
            "reference_like_delay",
            "effective_amp_sqrt_recurrence",
            "effective_power_recurrence",
            "lora_full_bw_phase_span_rad",
            "amp_phase_score",
            "power_phase_score",
        ],
    )
    corr_fields = [
        "scope",
        "chirp_feature",
        "lora_feature",
        "n",
        "spearman_rho",
        "pearson_r",
        "perm_p_two_sided",
        "bh_q",
        "abs_spearman",
    ]
    write_csv(args.output_dir / "03_correlations_all_overlap_points.csv", all_corr, corr_fields)
    write_csv(args.output_dir / "04_correlations_exclude_reference_point32.csv", no_ref_corr, corr_fields)
    write_csv(args.output_dir / "05_top_correlations.csv", combined_top, corr_fields)
    write_csv(
        args.output_dir / "06_point_diagnostics.csv",
        point_diagnostics(joined),
        [
            "position_key",
            "state",
            "distance_m",
            "is_reference_point_32",
            "stable_secondary_path_count",
            "secondary_effective_power_sum",
            "main_effective_power_fraction",
            "equivalent_rms_delay_us",
            "max_abs_stable_delay_us",
            "nb_max_phase_span_rad",
            "nb_amp_phase_score",
            "q4_bin_lr_asymmetry",
            "q4_bin_centroid",
            "q4_bin_side_to_center_power",
            "asymmetry_mean_point_mean",
            "interpolated_peak_offset_bins_mean_point_mean",
            "stable_equivalent_paths",
        ],
    )

    key_lora_features = [
        "q4_bin_lr_asymmetry",
        "q4_bin_centroid",
        "q4_bin_side_to_center_power",
        "q4_bin_rms_width",
        "asymmetry_mean_point_mean",
        "interpolated_peak_offset_bins_mean_point_mean",
        "side_power_fraction_mean_point_mean",
    ]
    key_chirp_features = [
        "stable_secondary_path_count",
        "secondary_effective_power_sum",
        "main_effective_power_fraction",
        "equivalent_rms_delay_us",
        "max_abs_stable_delay_us",
        "nb_amp_phase_score",
        "nb_signed_amp_delay_moment_us",
        "nb_post_minus_pre_amp",
    ]
    key_pairs = []
    for scope, rows in [("all_overlap_points", all_corr), ("exclude_reference_point32", no_ref_corr)]:
        lookup = {
            (row["chirp_feature"], row["lora_feature"]): row
            for row in rows
        }
        for chirp_feature in key_chirp_features:
            for lora_feature in key_lora_features:
                item = lookup.get((chirp_feature, lora_feature))
                if item:
                    key_pairs.append(item)

    phase_spans = [to_float(row.get("nb_max_phase_span_rad")) for row in joined]
    amp_scores = [to_float(row.get("nb_amp_phase_score")) for row in joined]
    summary = {
        "input_files": {
            "chirp_point_csv": str(args.chirp_point_csv),
            "chirp_path_csv": str(args.chirp_path_csv),
            "lora_q4_point_csv": str(args.lora_q4_point_csv),
            "lora_q1_shape_csv": str(args.lora_q1_shape_csv),
            "lora_q4_bins_csv": str(args.lora_q4_bins_csv),
        },
        "overlap_point_count": len(joined),
        "overlap_points": [row["position_key"] for row in joined],
        "exclude_reference_point32_count": len(joined_no_ref),
        "lora_bandwidth_hz": BW_HZ,
        "phase_span_definition": "2*pi*125kHz*abs(delay_us); edge-to-edge phase change across LoRa bandwidth",
        "max_lora_phase_span_rad_summary": summarize_values(phase_spans),
        "amp_phase_score_summary": summarize_values(amp_scores),
        "top_all_overlap": top_rows(all_corr, 10),
        "top_exclude_reference_point32": top_rows(no_ref_corr, 10),
        "selected_key_pairs": key_pairs,
        "interpretation": [
            "Point-wise chirp multipath features can show moderate associations with LoRa narrowband shape features.",
            "The associations are evidence of an aggregate multipath influence, not evidence that LoRa resolves individual paths.",
            "The physical phase-span scores are small because observed sub-microsecond delays occupy only a small fraction of a 125 kHz LoRa bandwidth.",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Chirp-LoRa point-wise physical link exploration",
                "",
                "This analysis joins wideband chirp multipath structure with LoRa q=4 point-level spectral shape.",
                "",
                "- It does not perform chirp-to-LoRa template matching.",
                "- It reports Spearman/Pearson associations for all overlap points and for a conservative subset excluding reference point 32.",
                "- `nb_*` features estimate how much the measured wideband delays can perturb a 125 kHz LoRa-band observation.",
                "- Treat significant-looking correlations as exploratory because the overlap set is small.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chirp-point-csv", type=Path, default=DEFAULT_CHIRP_POINT)
    parser.add_argument("--chirp-path-csv", type=Path, default=DEFAULT_CHIRP_PATHS)
    parser.add_argument("--lora-q4-point-csv", type=Path, default=DEFAULT_LORA_Q4_POINT)
    parser.add_argument("--lora-q1-shape-csv", type=Path, default=DEFAULT_LORA_Q1_SHAPE)
    parser.add_argument("--lora-q4-bins-csv", type=Path, default=DEFAULT_LORA_Q4_BINS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutation-rounds", type=int, default=999)
    parser.add_argument("--top-limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
