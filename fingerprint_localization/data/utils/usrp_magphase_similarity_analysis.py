from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-8
MAG_PREFIX = "preamble_fft_mag_bin_"
PHASE_PREFIX = "preamble_fft_phase_bin_"


def offset_suffix(offset: int) -> str:
    return f"{offset:+d}"


def make_offsets(bin_count: int) -> list[int]:
    if bin_count <= 0:
        raise ValueError("--bin-count must be positive.")
    left = bin_count // 2
    return list(range(-left, bin_count - left))


def feature_columns(offsets: Sequence[int]) -> tuple[list[str], list[str]]:
    mag_columns = [f"{MAG_PREFIX}{offset_suffix(offset)}" for offset in offsets]
    phase_columns = [f"{PHASE_PREFIX}{offset_suffix(offset)}" for offset in offsets]
    return mag_columns, phase_columns


def zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < EPS, 1.0, std)
    return (values - mean) / std


def circular_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.angle(np.mean(np.exp(1j * values), axis=axis))


def circular_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    resultant = np.abs(np.mean(np.exp(1j * values), axis=axis))
    resultant = np.clip(resultant, EPS, 1.0)
    return np.sqrt(-2.0 * np.log(resultant))


def distance_matrix(features: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def circular_phase_distance_matrix(phase: np.ndarray) -> np.ndarray:
    diff = np.angle(np.exp(1j * (phase[:, None, :] - phase[None, :, :])))
    return np.sqrt(np.sum(diff * diff, axis=-1))


def split_pairwise(distances: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    upper = np.triu_indices(len(labels), k=1)
    same = labels[upper[0]] == labels[upper[1]]
    values = distances[upper]
    return values[same], values[~same]


def separation_probability(intra: np.ndarray, inter: np.ndarray) -> float:
    if len(intra) == 0 or len(inter) == 0:
        return float("nan")
    inter_sorted = np.sort(inter)
    less_equal = np.searchsorted(inter_sorted, intra, side="right")
    less = np.searchsorted(inter_sorted, intra, side="left")
    greater = len(inter_sorted) - less_equal
    ties = less_equal - less
    return float(np.mean((greater + 0.5 * ties) / len(inter_sorted)))


def overlap_coefficient(intra: np.ndarray, inter: np.ndarray, bins: int = 120) -> float:
    if len(intra) == 0 or len(inter) == 0:
        return float("nan")
    lo = float(min(intra.min(), inter.min()))
    hi = float(max(intra.max(), inter.max()))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return float("nan")
    hist_intra, edges = np.histogram(intra, bins=bins, range=(lo, hi), density=True)
    hist_inter, _ = np.histogram(inter, bins=edges, density=True)
    return float(np.sum(np.minimum(hist_intra, hist_inter) * np.diff(edges)))


def describe_distribution(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def metric_summary(intra: np.ndarray, inter: np.ndarray) -> dict:
    return {
        "intra": describe_distribution(intra),
        "inter": describe_distribution(inter),
        "separation_probability_inter_gt_intra": separation_probability(intra, inter),
        "overlap_coefficient_histogram": overlap_coefficient(intra, inter),
        "inter_fraction_below_intra_median": float(np.mean(inter < np.median(intra))) if len(intra) and len(inter) else float("nan"),
        "intra_fraction_above_inter_median": float(np.mean(intra > np.median(inter))) if len(intra) and len(inter) else float("nan"),
    }


def plot_loc_curves(
    output_path: Path,
    x: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    value_kind: str,
) -> None:
    unique_labels = np.array(sorted(np.unique(labels)))
    ncols = 4
    nrows = int(math.ceil(len(unique_labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.35 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, label in zip(axes, unique_labels):
        loc_values = values[labels == label]
        if value_kind == "phase":
            mean = np.unwrap(circular_mean(loc_values, axis=0))
            spread = circular_std(loc_values, axis=0)
            ax.set_ylim(float(np.min(mean - spread)) - 0.35, float(np.max(mean + spread)) + 0.35)
        else:
            mean = loc_values.mean(axis=0)
            spread = loc_values.std(axis=0)
            ax.set_ylim(0.0, max(1.15, float(np.max(mean + spread)) * 1.08))

        ax.plot(x, mean, color="#1f77b4", linewidth=1.5)
        ax.fill_between(x, mean - spread, mean + spread, color="#1f77b4", alpha=0.22, linewidth=0)
        ax.axvline(0, color="#444444", linewidth=0.7, alpha=0.45)
        ax.set_title(f"loc {int(label)}  n={len(loc_values)}", fontsize=9)
        ax.grid(True, alpha=0.22, linewidth=0.5)

    for ax in axes[len(unique_labels) :]:
        ax.axis("off")

    title = "USRP 16-bin magnitude mean +/- std by location"
    ylabel = "peak-normalized magnitude"
    if value_kind == "phase":
        title = "USRP 16-bin relative phase circular mean +/- circular std by location"
        ylabel = "phase, unwrapped per location (rad)"

    fig.suptitle(title, fontsize=14)
    fig.supxlabel("FFT bin offset after peak alignment")
    fig.supylabel(ylabel)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.975))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / max(len(x), 1)
    return x, y


def plot_distance_distributions(output_path: Path, metrics: dict[str, tuple[np.ndarray, np.ndarray, dict]]) -> None:
    fig, axes = plt.subplots(len(metrics), 2, figsize=(12, 3.3 * len(metrics)))
    if len(metrics) == 1:
        axes = np.asarray([axes])

    for row_idx, (name, (intra, inter, summary)) in enumerate(metrics.items()):
        ax_hist = axes[row_idx, 0]
        ax_cdf = axes[row_idx, 1]
        hi = float(np.percentile(np.concatenate([intra, inter]), 99.0))
        bins = np.linspace(0.0, hi, 80)

        ax_hist.hist(intra, bins=bins, density=True, histtype="stepfilled", alpha=0.34, color="#d62728", label="intra")
        ax_hist.hist(inter, bins=bins, density=True, histtype="step", linewidth=1.7, color="#1f77b4", label="inter")
        ax_hist.set_xlim(0.0, hi)
        ax_hist.set_title(
            f"{name}: overlap={summary['overlap_coefficient_histogram']:.3f}, "
            f"P(inter>intra)={summary['separation_probability_inter_gt_intra']:.3f}"
        )
        ax_hist.set_xlabel("pairwise distance")
        ax_hist.set_ylabel("density")
        ax_hist.grid(True, alpha=0.25)
        ax_hist.legend()

        intra_x, intra_y = ecdf(intra)
        inter_x, inter_y = ecdf(inter)
        ax_cdf.plot(intra_x, intra_y, color="#d62728", label="intra")
        ax_cdf.plot(inter_x, inter_y, color="#1f77b4", label="inter")
        ax_cdf.set_xlim(0.0, hi)
        ax_cdf.set_ylim(0.0, 1.0)
        ax_cdf.set_xlabel("pairwise distance")
        ax_cdf.set_ylabel("empirical CDF")
        ax_cdf.grid(True, alpha=0.25)
        ax_cdf.legend()

    fig.suptitle("Intra-location vs inter-location pairwise distance distributions", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot USRP mag/phase feature shape similarity diagnostics.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processedData/usrp_similarity_analysis"))
    parser.add_argument("--bin-count", type=int, default=16)
    parser.add_argument("--label-column", default="position_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offsets = make_offsets(args.bin_count)
    mag_columns, phase_columns = feature_columns(offsets)
    required = [args.label_column, *mag_columns, *phase_columns]

    df = pd.read_csv(args.input_csv)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[args.label_column] = pd.to_numeric(df[args.label_column], errors="coerce")
    for column in [*mag_columns, *phase_columns]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required).copy()
    df[args.label_column] = df[args.label_column].astype(int)
    df = df.sort_values([args.label_column, "packet_index" if "packet_index" in df.columns else args.label_column])

    labels = df[args.label_column].to_numpy(dtype=np.int64)
    mag = df[mag_columns].to_numpy(dtype=np.float64)
    phase = df[phase_columns].to_numpy(dtype=np.float64)
    x = np.asarray(offsets, dtype=np.float64)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_loc_curves(args.output_dir / "loc_mag_mean_std.png", x, mag, labels, "mag")
    plot_loc_curves(args.output_dir / "loc_phase_mean_std.png", x, phase, labels, "phase")

    mag_z = zscore(mag)
    phase_sincos_z = zscore(np.concatenate([np.sin(phase), np.cos(phase)], axis=1))
    combined_z = zscore(np.concatenate([mag, np.sin(phase), np.cos(phase)], axis=1))

    distance_metrics: dict[str, tuple[np.ndarray, np.ndarray, dict]] = {}
    for name, distances in {
        "magnitude_zscore": distance_matrix(mag_z),
        "phase_circular_radian": circular_phase_distance_matrix(phase),
        "mag_plus_phase_sincos_zscore": distance_matrix(combined_z),
    }.items():
        intra, inter = split_pairwise(distances, labels)
        summary = metric_summary(intra, inter)
        distance_metrics[name] = (intra, inter, summary)

    plot_distance_distributions(args.output_dir / "pairwise_distance_distributions.png", distance_metrics)

    loc_rows = []
    for label in sorted(np.unique(labels)):
        mask = labels == label
        row = {
            "loc_id": int(label),
            "sample_count": int(mask.sum()),
            "mean_score_db": float(df.loc[mask, "preamble_peak_to_residual_db"].mean()) if "preamble_peak_to_residual_db" in df else float("nan"),
            "mean_detect_score_db": float(df.loc[mask, "detect_score_db"].mean()) if "detect_score_db" in df else float("nan"),
        }
        loc_mag_z = mag_z[mask]
        loc_phase = phase[mask]
        if mask.sum() >= 2:
            loc_intra_mag, _ = split_pairwise(distance_matrix(loc_mag_z), labels[mask])
            loc_intra_phase, _ = split_pairwise(circular_phase_distance_matrix(loc_phase), labels[mask])
            row["intra_mag_zscore_mean"] = float(np.mean(loc_intra_mag))
            row["intra_phase_circular_mean"] = float(np.mean(loc_intra_phase))
        else:
            row["intra_mag_zscore_mean"] = float("nan")
            row["intra_phase_circular_mean"] = float("nan")
        loc_rows.append(row)

    loc_stats = pd.DataFrame(loc_rows)
    loc_stats.to_csv(args.output_dir / "loc_shape_stats.csv", index=False)

    summary = {
        "input_csv": str(args.input_csv),
        "rows": int(len(df)),
        "present_labels": [int(x) for x in sorted(np.unique(labels))],
        "label_counts": {str(int(k)): int(v) for k, v in df[args.label_column].value_counts().sort_index().items()},
        "bin_offsets": offsets,
        "distance_metrics": {name: item[2] for name, item in distance_metrics.items()},
        "outputs": {
            "magnitude_plot": str(args.output_dir / "loc_mag_mean_std.png"),
            "phase_plot": str(args.output_dir / "loc_phase_mean_std.png"),
            "distance_plot": str(args.output_dir / "pairwise_distance_distributions.png"),
            "loc_stats_csv": str(args.output_dir / "loc_shape_stats.csv"),
        },
    }
    (args.output_dir / "similarity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name, (_, _, metric) in distance_metrics.items():
        print(
            f"{name}: intra median={metric['intra']['median']:.4f}, "
            f"inter median={metric['inter']['median']:.4f}, "
            f"overlap={metric['overlap_coefficient_histogram']:.4f}, "
            f"P(inter>intra)={metric['separation_probability_inter_gt_intra']:.4f}"
        )
    print(f"Wrote diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
