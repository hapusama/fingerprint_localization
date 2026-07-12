#!/usr/bin/env python3
"""Compare spatial distance matrices from LoRa and chirp-projected shapes.

This checks a weaker but more appropriate claim than pointwise absolute
matching: even if a_l^L is not closest to a_l^C, do LoRa and chirp projection
agree on which locations are mutually similar or dissimilar?

For common chirp/LoRa locations, build pairwise distance matrices:

    D_L(l, m): distance between LoRa measured shapes at l and m
    D_C(l, m): distance between chirp-projected shapes at l and m

Then correlate their upper triangles and use a Mantel-style permutation test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = ROOT / "v2_output/20260626_chirp_lora_bin_projection"
DEFAULT_OUTPUT_DIR = ROOT / "v2_output/20260626_chirp_lora_distance_matrix_consistency"
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


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rank_average(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return pearson(rank_average(a), rank_average(b))


def upper_triangle(matrix: Sequence[Sequence[float]]) -> list[float]:
    n = len(matrix)
    return [matrix[i][j] for i in range(n) for j in range(i + 1, n)]


def permute_matrix(matrix: Sequence[Sequence[float]], perm: Sequence[int]) -> list[list[float]]:
    n = len(matrix)
    return [[matrix[perm[i]][perm[j]] for j in range(n)] for i in range(n)]


class Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed & 0x7FFFFFFF

    def randrange(self, n: int) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state % n

    def shuffle(self, values: list[int]) -> None:
        for i in range(len(values) - 1, 0, -1):
            j = self.randrange(i + 1)
            values[i], values[j] = values[j], values[i]


def mantel_correlation(
    lora_matrix: Sequence[Sequence[float]],
    chirp_matrix: Sequence[Sequence[float]],
    permutations: int,
    seed: int,
    correlation,
) -> dict:
    lora_vec = upper_triangle(lora_matrix)
    chirp_vec = upper_triangle(chirp_matrix)
    observed = correlation(lora_vec, chirp_vec)
    rng = Lcg(seed)
    ge = 0
    abs_ge = 0
    null_values = []
    n = len(lora_matrix)
    base_perm = list(range(n))
    for _ in range(permutations):
        perm = list(base_perm)
        rng.shuffle(perm)
        value = correlation(lora_vec, upper_triangle(permute_matrix(chirp_matrix, perm)))
        null_values.append(value)
        ge += int(value >= observed)
        abs_ge += int(abs(value) >= abs(observed))
    null_values.sort()
    return {
        "correlation": observed,
        "mantel_p_one_sided_positive": (ge + 1) / (permutations + 1),
        "mantel_p_two_sided_abs": (abs_ge + 1) / (permutations + 1),
        "null_mean": sum(null_values) / len(null_values) if null_values else math.nan,
        "null_q025": null_values[round((len(null_values) - 1) * 0.025)] if null_values else math.nan,
        "null_q975": null_values[round((len(null_values) - 1) * 0.975)] if null_values else math.nan,
    }


def distance_matrix(vectors: Sequence[Sequence[float]], metric: str) -> list[list[float]]:
    n = len(vectors)
    out = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "mag_cosine_distance":
                value = 1.0 - cosine(vectors[i], vectors[j])
            elif metric == "rel_db_pearson_distance":
                value = 1.0 - pearson(vectors[i], vectors[j])
            elif metric == "rel_db_euclidean_distance":
                value = euclidean(vectors[i], vectors[j])
            else:
                raise ValueError(metric)
            out[i][j] = value
            out[j][i] = value
    return out


def run(args: argparse.Namespace) -> dict:
    synth_rows = read_csv(args.input_dir / "02_chirp_synth_point_bins.csv")
    meas_rows = read_csv(args.input_dir / "03_lora_measured_point_bins.csv")
    synth = {point_key(row): row for row in synth_rows}
    meas = {point_key(row): row for row in meas_rows}
    keys = sorted(set(synth) & set(meas))
    labels = [point_label(key) for key in keys]

    lora_mag = [vector(meas[key], "meas_mag_bin", "_mean") for key in keys]
    chirp_mag = [vector(synth[key], "synth_mag_bin", "_mean") for key in keys]
    lora_rel = [vector(meas[key], "meas_rel_db_bin", "_mean") for key in keys]
    chirp_rel = [vector(synth[key], "synth_rel_db_bin", "_mean") for key in keys]

    metric_specs = [
        ("mag_cosine_distance", lora_mag, chirp_mag),
        ("rel_db_pearson_distance", lora_rel, chirp_rel),
        ("rel_db_euclidean_distance", lora_rel, chirp_rel),
    ]

    summary_rows = []
    pair_rows = []
    matrices: dict[str, dict[str, list[list[float]]]] = {}
    for metric_name, lora_vectors, chirp_vectors in metric_specs:
        lora_matrix = distance_matrix(lora_vectors, metric_name)
        chirp_matrix = distance_matrix(chirp_vectors, metric_name)
        matrices[metric_name] = {"lora": lora_matrix, "chirp": chirp_matrix}
        for method_name, correlation in [
            ("spearman", spearman),
            ("pearson", pearson),
        ]:
            stats = mantel_correlation(
                lora_matrix,
                chirp_matrix,
                permutations=args.permutations,
                seed=args.seed + len(summary_rows) * 1009,
                correlation=correlation,
            )
            row = {
                "distance_metric": metric_name,
                "correlation_method": method_name,
                "point_count": len(keys),
                "pair_count": len(keys) * (len(keys) - 1) // 2,
                **stats,
            }
            summary_rows.append(row)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair_rows.append(
                    {
                        "distance_metric": metric_name,
                        "point_i": labels[i],
                        "point_j": labels[j],
                        "lora_distance": lora_matrix[i][j],
                        "chirp_distance": chirp_matrix[i][j],
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "01_common_points.csv",
        [{"index": idx, "label": label} for idx, label in enumerate(labels)],
        ["index", "label"],
    )
    write_csv(
        args.output_dir / "02_pairwise_distances_long.csv",
        pair_rows,
        ["distance_metric", "point_i", "point_j", "lora_distance", "chirp_distance"],
    )
    write_csv(
        args.output_dir / "03_distance_matrix_correlation_summary.csv",
        summary_rows,
        [
            "distance_metric",
            "correlation_method",
            "point_count",
            "pair_count",
            "correlation",
            "mantel_p_one_sided_positive",
            "mantel_p_two_sided_abs",
            "null_mean",
            "null_q025",
            "null_q975",
        ],
    )
    payload = {
        "input_dir": str(args.input_dir),
        "point_count": len(keys),
        "pair_count": len(keys) * (len(keys) - 1) // 2,
        "points": labels,
        "permutations": args.permutations,
        "summary": summary_rows,
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
