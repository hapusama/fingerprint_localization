#!/usr/bin/env python3
"""Compare LoRa main-lobe similarity and secondary-shoulder metrics.

The previous chirp-LoRa curve matching was dominated by the very similar LoRa
main lobe.  This script puts that main-lobe similarity on the same point-level
axis as the q=4 secondary-peak/shoulder metric, then tests which dimension is
actually associated with wideband chirp secondary strength.
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
DEFAULT_DELAY_BIN_JOINED = (
    ROOT
    / "v2_output/20260711_dechirp_delay_bin_structure"
    / "01_delay_bin_joined_structure.csv"
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
DEFAULT_OUTPUT = ROOT / "v2_output/20260711_lora_main_secondary_dimensions"

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
    ge = 0
    for _ in range(rounds):
        rng.shuffle(ry)
        candidate = pearson(rx, ry)
        if math.isfinite(candidate) and abs(candidate) >= abs(observed) - 1e-15:
            ge += 1
    return (ge + 1) / (rounds + 1)


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


def leave_one_out(x: Sequence[float], y: Sequence[float], labels: Sequence[str]) -> dict[str, object]:
    rows = []
    for idx, label in enumerate(labels):
        rows.append(
            {
                "removed": label,
                "rho": spearman(list(x[:idx]) + list(x[idx + 1 :]), list(y[:idx]) + list(y[idx + 1 :])),
            }
        )
    low = min(rows, key=lambda row: row["rho"])
    high = max(rows, key=lambda row: row["rho"])
    return {
        "loo_rho_min": low["rho"],
        "loo_rho_min_removed": low["removed"],
        "loo_rho_max": high["rho"],
        "loo_rho_max_removed": high["removed"],
    }


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    nl = math.sqrt(sum(value * value for value in left))
    nr = math.sqrt(sum(value * value for value in right))
    if nl <= EPS or nr <= EPS:
        return math.nan
    return dot / (nl * nr)


def residual_l2(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def vector_from_bins(row: dict[str, str], offsets: Sequence[float]) -> list[float]:
    out = []
    for offset in offsets:
        tag = format_offset(offset)
        out.append(fnum(row.get(f"meas_mag_norm_bin_{tag}_mean"), 0.0))
    return out


def loo_template(vectors: dict[tuple[int, int], list[float]], key: tuple[int, int]) -> list[float]:
    others = [vec for other_key, vec in vectors.items() if other_key != key]
    if not others:
        return list(vectors[key])
    return [mean([vec[idx] for vec in others]) for idx in range(len(others[0]))]


def build_rows(args: argparse.Namespace) -> list[dict]:
    joined = {point_key(row): row for row in read_csv(args.delay_bin_joined_csv)}
    bins = {point_key(row): row for row in read_csv(args.lora_q4_bins_csv)}
    points = {point_key(row): row for row in read_csv(args.lora_q4_point_csv)}
    common_keys = sorted(set(joined) & set(bins) & set(points))

    main_offsets = [-0.5, -0.25, 0.0, 0.25, 0.5]
    near_offsets = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    full_offsets = [idx / 4.0 for idx in range(-8, 9)]
    main_vectors = {key: vector_from_bins(bins[key], main_offsets) for key in common_keys}
    near_vectors = {key: vector_from_bins(bins[key], near_offsets) for key in common_keys}
    full_vectors = {key: vector_from_bins(bins[key], full_offsets) for key in common_keys}

    rows = []
    for key in common_keys:
        source = joined[key]
        q4 = points[key]
        main_template = loo_template(main_vectors, key)
        near_template = loo_template(near_vectors, key)
        full_template = loo_template(full_vectors, key)
        main_vec = main_vectors[key]
        near_vec = near_vectors[key]
        full_vec = full_vectors[key]

        point_mean_secondary_db = fnum(q4.get("secondary_peak_rel_db_mean_point_mean"))
        point_mean_secondary_ratio = 10.0 ** (point_mean_secondary_db / 20.0) if math.isfinite(point_mean_secondary_db) else math.nan

        # Same definition as the q=4 summary, but applied to the point-mean
        # curve: remove +/-0.25 bin around the point-mean peak and take the
        # strongest remaining q=4 sample in [-2,+2].
        peak_idx = max(range(len(full_vec)), key=lambda idx: full_vec[idx])
        peak_offset = full_offsets[peak_idx]
        peak_mag = full_vec[peak_idx] + EPS
        shoulder_candidates = [
            idx for idx, offset in enumerate(full_offsets)
            if abs(offset - peak_offset) > 0.25
        ]
        shoulder_idx = max(shoulder_candidates, key=lambda idx: full_vec[idx])
        shoulder_db_from_point_mean_curve = 20.0 * math.log10((full_vec[shoulder_idx] + EPS) / peak_mag)

        rows.append(
            {
                "corridor_id": key[0],
                "location_id": key[1],
                "position_key": point_label(key),
                "is_reference_point32": source.get("is_reference_point_32"),
                "chirp_secondary_amp_sum": source.get("chirp_delay_bin_secondary_amp_sum"),
                "chirp_secondary_power_sum": source.get("chirp_delay_bin_secondary_power_sum"),
                "chirp_stable_path_count": source.get("stable_path_count"),
                "lora_main_lobe_loo_similarity": cosine(main_vec, main_template),
                "lora_main_lobe_loo_residual": residual_l2(main_vec, main_template),
                "lora_near_lobe_loo_similarity": cosine(near_vec, near_template),
                "lora_near_lobe_loo_residual": residual_l2(near_vec, near_template),
                "lora_full_window_loo_similarity": cosine(full_vec, full_template),
                "lora_full_window_loo_residual": residual_l2(full_vec, full_template),
                "lora_secondary_peak_rel_db_symbol_mean": point_mean_secondary_db,
                "lora_secondary_peak_linear_ratio_symbol_mean": point_mean_secondary_ratio,
                "lora_secondary_peak_offset_symbol_mean": q4.get("secondary_peak_offset_bins_mean_point_mean"),
                "lora_point_mean_shoulder_rel_db": shoulder_db_from_point_mean_curve,
                "lora_point_mean_shoulder_offset": full_offsets[shoulder_idx],
                "lora_q4_centroid_bin": source.get("lora_q4_centroid_bin"),
                "lora_q4_lr_asymmetry": source.get("lora_q4_lr_asymmetry"),
                "lora_q4_side_to_center_power": source.get("lora_q4_side_to_center_power"),
                "stable_equivalent_paths": source.get("stable_equivalent_paths"),
            }
        )
    return rows


def correlation_rows(rows: Sequence[dict], scope: str, seed: int, rounds: int) -> list[dict]:
    pairs = [
        ("chirp_secondary_amp_sum", "lora_main_lobe_loo_similarity", "main_lobe_similarity"),
        ("chirp_secondary_amp_sum", "lora_main_lobe_loo_residual", "main_lobe_residual"),
        ("chirp_secondary_amp_sum", "lora_near_lobe_loo_similarity", "near_lobe_similarity"),
        ("chirp_secondary_amp_sum", "lora_near_lobe_loo_residual", "near_lobe_residual"),
        ("chirp_secondary_amp_sum", "lora_full_window_loo_similarity", "full_window_similarity"),
        ("chirp_secondary_amp_sum", "lora_full_window_loo_residual", "full_window_residual"),
        ("chirp_secondary_amp_sum", "lora_secondary_peak_rel_db_symbol_mean", "secondary_peak_rel_db_symbol_mean"),
        ("chirp_secondary_amp_sum", "lora_secondary_peak_linear_ratio_symbol_mean", "secondary_peak_linear_ratio_symbol_mean"),
        ("chirp_secondary_amp_sum", "lora_point_mean_shoulder_rel_db", "secondary_shoulder_rel_db_point_mean_curve"),
        ("chirp_secondary_amp_sum", "lora_q4_centroid_bin", "q4_centroid"),
        ("chirp_secondary_amp_sum", "lora_q4_lr_asymmetry", "q4_left_right_asymmetry"),
        ("chirp_stable_path_count", "lora_main_lobe_loo_similarity", "path_count_main_lobe_similarity"),
        ("chirp_stable_path_count", "lora_secondary_peak_rel_db_symbol_mean", "path_count_secondary_peak_rel_db"),
        ("chirp_stable_path_count", "lora_q4_centroid_bin", "path_count_q4_centroid"),
    ]
    out = []
    for idx, (x_name, y_name, hypothesis) in enumerate(pairs):
        xs, ys, labels = finite_pairs(rows, x_name, y_name)
        if len(xs) < 6:
            continue
        rho = spearman(xs, ys)
        item = {
            "scope": scope,
            "hypothesis": hypothesis,
            "x_feature": x_name,
            "y_feature": y_name,
            "n": len(xs),
            "spearman_rho": rho,
            "pearson_r": pearson(xs, ys),
            "perm_p_two_sided": permutation_p(xs, ys, rho, seed + 41 * idx, rounds),
        }
        item.update(leave_one_out(xs, ys, labels))
        out.append(item)
    return out


def run(args: argparse.Namespace) -> dict:
    rows = build_rows(args)
    no_ref = [
        row for row in rows
        if str(row.get("is_reference_point32", "")).strip().lower() not in {"true", "1"}
        and inum(row.get("location_id")) != 32
    ]
    all_corr = correlation_rows(rows, "all_overlap_points", args.seed, args.permutation_rounds)
    no_ref_corr = correlation_rows(no_ref, "exclude_reference_point32", args.seed + 1000, args.permutation_rounds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_fields = [
        "corridor_id",
        "location_id",
        "position_key",
        "is_reference_point32",
        "chirp_secondary_amp_sum",
        "chirp_secondary_power_sum",
        "chirp_stable_path_count",
        "lora_main_lobe_loo_similarity",
        "lora_main_lobe_loo_residual",
        "lora_near_lobe_loo_similarity",
        "lora_near_lobe_loo_residual",
        "lora_full_window_loo_similarity",
        "lora_full_window_loo_residual",
        "lora_secondary_peak_rel_db_symbol_mean",
        "lora_secondary_peak_linear_ratio_symbol_mean",
        "lora_secondary_peak_offset_symbol_mean",
        "lora_point_mean_shoulder_rel_db",
        "lora_point_mean_shoulder_offset",
        "lora_q4_centroid_bin",
        "lora_q4_lr_asymmetry",
        "lora_q4_side_to_center_power",
        "stable_equivalent_paths",
    ]
    corr_fields = [
        "scope",
        "hypothesis",
        "x_feature",
        "y_feature",
        "n",
        "spearman_rho",
        "pearson_r",
        "perm_p_two_sided",
        "loo_rho_min",
        "loo_rho_min_removed",
        "loo_rho_max",
        "loo_rho_max_removed",
    ]
    write_csv(args.output_dir / "01_main_secondary_point_dimensions.csv", rows, point_fields)
    write_csv(args.output_dir / "02_main_secondary_dimension_correlations.csv", all_corr + no_ref_corr, corr_fields)

    summary = {
        "overlap_point_count": len(rows),
        "exclude_reference_point32_count": len(no_ref),
        "q": 4,
        "main_lobe_window_bins": [-0.5, 0.5],
        "near_lobe_window_bins": [-0.75, 0.75],
        "full_window_bins": [-2.0, 2.0],
        "secondary_peak_definition": "q=4 zero-padded dechirped FFT; per symbol, remove +/-0.25 bin around interpolated local peak and take the strongest remaining sample in [-2,+2].",
        "main_lobe_similarity_summary_all": summarize([fnum(row["lora_main_lobe_loo_similarity"]) for row in rows]),
        "near_lobe_similarity_summary_all": summarize([fnum(row["lora_near_lobe_loo_similarity"]) for row in rows]),
        "full_window_similarity_summary_all": summarize([fnum(row["lora_full_window_loo_similarity"]) for row in rows]),
        "secondary_peak_rel_db_summary_all": summarize([fnum(row["lora_secondary_peak_rel_db_symbol_mean"]) for row in rows]),
        "all_overlap_correlations": all_corr,
        "exclude_reference_point32_correlations": no_ref_corr,
        "interpretation": [
            "The main lobe is nearly universal across locations, so cosine similarity is very high and weakly informative.",
            "The secondary peak metric is a q=4 spectral shoulder after excluding the main-peak neighborhood, not a resolved physical second path.",
            "If chirp secondary strength correlates with the secondary shoulder but not with main-lobe similarity, the evidence supports unresolved multipath deformation rather than resolved-path matching.",
        ],
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# LoRa Main-Lobe vs Secondary-Shoulder Dimensions",
                "",
                "This run compares q=4 main-lobe template similarity and q=4 secondary-shoulder features on the same point-level correlation axis.",
                "",
                "- Main lobe: `[-0.5,+0.5]` bin leave-one-out cosine similarity to the other-location mean template.",
                "- Near lobe: `[-0.75,+0.75]` bin leave-one-out cosine similarity.",
                "- Full window: `[-2,+2]` bin leave-one-out cosine similarity.",
                "- Secondary peak: q=4 symbol-level strongest sample after excluding `+-0.25` bin around the interpolated main peak.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-bin-joined-csv", type=Path, default=DEFAULT_DELAY_BIN_JOINED)
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
