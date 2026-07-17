#!/usr/bin/env python3
"""Plot and register the search-mechanism ablation CSV outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERIMENT_DIR.parent.parent
DEFAULT_RESULTS_DIR = (
    PROJECT_DIR
    / "results"
    / "expanded_source_safe_1to10"
    / "search_mechanism_ablation_20260717"
)
METHOD_ORDER = ["average_cost", "greedy_path", "no_pheromone", "full_aco"]
METHOD_LABELS = {
    "average_cost": "Average cost",
    "greedy_path": "Greedy path",
    "no_pheromone": "No pheromone",
    "full_aco": "Full ACO",
}
COLORS = {
    "average_cost": "#4C78A8",
    "greedy_path": "#F58518",
    "no_pheromone": "#54A24B",
    "full_aco": "#B279A2",
}
CONDITION_ORDER = [
    "preamble_missing",
    "amplitude_noise",
    "cfo_shift",
    "segment_anomaly",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_accuracy(rows: list[dict], column: str, output: Path, ylabel: str) -> None:
    formal = [
        row
        for row in rows
        if row["scenario_type"] == "artificial_perturbation"
        and row["condition"] != "clean"
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.95, bottom=0.12, hspace=0.35, wspace=0.20
    )
    for axis, condition in zip(axes.ravel(), CONDITION_ORDER):
        condition_rows = [row for row in formal if row["condition"] == condition]
        strengths = sorted({float(row["strength"]) for row in condition_rows})
        for method in METHOD_ORDER:
            values = [
                float(
                    next(
                        row[column]
                        for row in condition_rows
                        if row["method"] == method
                        and float(row["strength"]) == strength
                    )
                )
                for strength in strengths
            ]
            axis.plot(
                strengths,
                values,
                marker="o",
                linewidth=2,
                label=METHOD_LABELS[method],
                color=COLORS[method],
            )
        axis.set_title(condition.replace("_", " "))
        axis.set_xlabel("degradation strength")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 1.02)
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_full_delta(rows: list[dict], output: Path) -> None:
    formal = [
        row
        for row in rows
        if row["scenario_type"] == "artificial_perturbation"
    ]
    labels = []
    deltas = []
    ordered = ["clean", *CONDITION_ORDER]
    short = {
        "clean": "clean",
        "preamble_missing": "missing",
        "amplitude_noise": "amp",
        "cfo_shift": "CFO",
        "segment_anomaly": "segment",
    }
    for condition in ordered:
        condition_rows = [row for row in formal if row["condition"] == condition]
        strengths = sorted({float(row["strength"]) for row in condition_rows})
        for strength in strengths:
            full = next(
                int(row["final_correct"])
                for row in condition_rows
                if row["method"] == "full_aco" and float(row["strength"]) == strength
            )
            no_pheromone = next(
                int(row["final_correct"])
                for row in condition_rows
                if row["method"] == "no_pheromone"
                and float(row["strength"]) == strength
            )
            labels.append(
                short[condition]
                if condition == "clean"
                else f"{short[condition]} {strength:g}"
            )
            deltas.append(full - no_pheromone)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    colors = [COLORS["full_aco"] if value > 0 else COLORS["no_pheromone"] for value in deltas]
    bars = axis.bar(range(len(labels)), deltas, color=colors)
    axis.axhline(0, color="#666666", linewidth=1)
    axis.set_ylabel("Full ACO minus no-pheromone correct packets")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, deltas):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.15 if value >= 0 else -0.15),
            f"{value:+d}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_runtime(rows: list[dict], output: Path) -> None:
    clean = [
        row
        for row in rows
        if row["scenario_type"] == "artificial_perturbation"
        and row["condition"] == "clean"
    ]
    means = [
        float(next(row["mean_search_runtime_ms"] for row in clean if row["method"] == method))
        for method in METHOD_ORDER
    ]
    p95s = [
        float(next(row["p95_search_runtime_ms"] for row in clean if row["method"] == method))
        for method in METHOD_ORDER
    ]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    bars = axis.barh(
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        means,
        color=[COLORS[method] for method in METHOD_ORDER],
    )
    axis.set_xscale("log")
    axis.set_xlabel("mean search time per packet (ms, log scale)")
    axis.grid(axis="x", alpha=0.25)
    for bar, mean, p95 in zip(bars, means, p95s):
        axis.text(
            mean * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.3f} ms (P95 {p95:.3f})",
            va="center",
        )
    axis.set_xlim(left=max(min(means) * 0.55, 1e-5), right=max(p95s) * 3.0)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(results_dir: Path) -> list[str]:
    rows = read_csv(results_dir / "ablation_metrics.csv")
    generated = [
        "ablation_final_accuracy.png",
        "ablation_search_accuracy.png",
        "full_vs_no_pheromone_delta.png",
        "ablation_clean_runtime.png",
    ]
    plot_accuracy(
        rows,
        "final_accuracy",
        results_dir / generated[0],
        "final accuracy",
    )
    plot_accuracy(
        rows,
        "search_accuracy",
        results_dir / generated[1],
        "search-only accuracy",
    )
    plot_full_delta(rows, results_dir / generated[2])
    plot_runtime(rows, results_dir / generated[3])

    manifest_path = results_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plots"] = generated
    manifest["plotter"] = Path(__file__).name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    checksum_path = results_dir / "CHECKSUMS.sha256"
    files = sorted(
        path for path in results_dir.iterdir() if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n",
        encoding="utf-8",
    )
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    print("\n".join(run(parse_args().results_dir)))
