from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


GROUP_FIELDS = [
    "experiment_id",
    "corridor_id",
    "position_id",
    "filename_sf",
    "filename_tx_power_dbm",
    "filename_preamble_len",
]


@dataclass
class RunningStats:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            return
        self.n += 1
        self.total += value
        self.total_sq += value * value

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else float("nan")

    @property
    def std(self) -> float:
        if not self.n:
            return float("nan")
        variance = max(0.0, self.total_sq / self.n - self.mean * self.mean)
        return math.sqrt(variance)


def key_from_row(row: dict) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in GROUP_FIELDS)


def safe_float(text: object) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def safe_int(text: object) -> Optional[int]:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def load_packet_counts(point_summary_csv: Path) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if not point_summary_csv.exists():
        return counts
    with point_summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("q") != "4":
                continue
            packet_count = safe_int(row.get("packet_count"))
            if packet_count is not None:
                counts[key_from_row(row)] = packet_count
    return counts


def aggregate_profiles(long_csv: Path) -> dict[tuple[str, ...], dict[int, dict[float, RunningStats]]]:
    profiles: dict[tuple[str, ...], dict[int, dict[float, RunningStats]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(RunningStats))
    )
    with long_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = safe_int(row.get("q"))
            offset = safe_float(row.get("subbin_offset"))
            mag_db = safe_float(row.get("mag_db_rel_peak"))
            if q not in (1, 4) or offset is None or mag_db is None:
                continue
            profiles[key_from_row(row)][q][offset].add(mag_db)
    return profiles


def write_profile_csv(path: Path, profiles: dict[tuple[str, ...], dict[int, dict[float, RunningStats]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = GROUP_FIELDS + ["q", "subbin_offset", "sample_count", "mag_db_mean", "mag_db_std"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(profiles):
            for q in sorted(profiles[key]):
                for offset in sorted(profiles[key][q]):
                    stats = profiles[key][q][offset]
                    row = {field: value for field, value in zip(GROUP_FIELDS, key)}
                    row.update(
                        {
                            "q": q,
                            "subbin_offset": offset,
                            "sample_count": stats.n,
                            "mag_db_mean": stats.mean,
                            "mag_db_std": stats.std,
                        }
                    )
                    writer.writerow(row)


def point_label(key: tuple[str, ...]) -> str:
    experiment, corridor, position, sf, tx_power, preamble = key
    return f"exp{experiment}_corr{corridor}_pos{position}_sf{sf}_tp{tx_power}_pre{preamble}"


def display_label(key: tuple[str, ...]) -> str:
    experiment, corridor, position, sf, tx_power, preamble = key
    return (
        f"Experiment {experiment}, corridor {corridor}, position {position} | "
        f"SF{sf}, TP{tx_power}, preamble {preamble}"
    )


def make_polyline(points: list[tuple[float, float]], sx, sy) -> str:
    return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)


def write_svg(
    path: Path,
    key: tuple[str, ...],
    q1_points: list[tuple[float, float]],
    q4_points: list[tuple[float, float]],
    packet_count: Optional[int],
) -> None:
    width = 980
    height = 560
    left = 82
    right = 34
    top = 68
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = -2.0, 2.0

    all_y = [y for _, y in q1_points + q4_points if math.isfinite(y)]
    y_min = min(-35.0, math.floor((min(all_y) if all_y else -35.0) / 5.0) * 5.0)
    y_max = 2.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        value = min(max(value, y_min), y_max)
        return top + (y_max - value) / (y_max - y_min) * plot_h

    packet_text = "unknown packets" if packet_count is None else f"{packet_count} packets"
    title = display_label(key)
    subtitle = f"Point-level mean spectrum over {packet_text}; raw q=1 bins [-2,2] vs q=4 zero-padding"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="30" font-family="Arial" font-size="18" fill="#222">{html.escape(title)}</text>',
        f'<text x="{left}" y="52" font-family="Arial" font-size="13" fill="#555">{html.escape(subtitle)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#b8b8b8"/>',
    ]

    for tick in [-2, -1, 0, 1, 2]:
        x = sx(float(tick))
        stroke = "#cfcfcf" if tick == 0 else "#e8e8e8"
        width_attr = "1.4" if tick == 0 else "1"
        dash = ' stroke-dasharray="5 5"' if tick == 0 else ""
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="{stroke}" stroke-width="{width_attr}"{dash}/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{height - 42}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#444">{tick}</text>'
        )

    y_tick = math.ceil(y_min / 5.0) * 5.0
    while y_tick <= y_max:
        y = sy(y_tick)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e8e8e8"/>')
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#444">{y_tick:g}</text>'
        )
        y_tick += 5.0

    q4_line = make_polyline(q4_points, sx, sy)
    q1_line = make_polyline(q1_points, sx, sy)
    lines.append(f'<polyline points="{q4_line}" fill="none" stroke="#1f77b4" stroke-width="2.4"/>')
    lines.append(
        f'<polyline points="{q1_line}" fill="none" stroke="#d62728" stroke-width="1.8" '
        f'stroke-dasharray="7 5"/>'
    )

    for x, y in q1_points:
        lines.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="5.2" '
            f'fill="white" stroke="#d62728" stroke-width="2"/>'
        )

    legend_x = left + 20
    legend_y = top + 20
    lines.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="#1f77b4" stroke-width="3"/>',
            f'<text x="{legend_x + 36}" y="{legend_y + 5}" font-family="Arial" font-size="13" fill="#222">q=4 zero-padded mean curve</text>',
            f'<line x1="{legend_x + 260}" y1="{legend_y}" x2="{legend_x + 288}" y2="{legend_y}" stroke="#d62728" stroke-width="2" stroke-dasharray="7 5"/>',
            f'<circle cx="{legend_x + 274}" cy="{legend_y}" r="4.5" fill="white" stroke="#d62728" stroke-width="2"/>',
            f'<text x="{legend_x + 300}" y="{legend_y + 5}" font-family="Arial" font-size="13" fill="#222">raw q=1 FFT bins</text>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13" fill="#222">bin offset from aligned main LoRa bin</text>',
            f'<text x="22" y="{top + plot_h / 2:.2f}" transform="rotate(-90 22 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial" font-size="13" fill="#222">magnitude relative to local peak (dB)</text>',
        ]
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_index(path: Path, svg_paths: list[Path], output_dir: Path) -> None:
    items = []
    for svg_path in svg_paths:
        rel = svg_path.relative_to(output_dir)
        items.append(
            f'<section><h2>{html.escape(svg_path.stem)}</h2>'
            f'<img src="{html.escape(str(rel))}" alt="{html.escape(svg_path.stem)}"/></section>'
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Point Spectrum Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    section {{ margin: 0 0 32px; border-bottom: 1px solid #ddd; padding-bottom: 24px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    h1 {{ font-size: 22px; }}
    h2 {{ font-size: 15px; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>LoRa Raw Bin [-2,2] vs q=4 Zero-Padded Spectrum</h1>
  {''.join(items)}
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot point-level raw q=1 bins versus q=4 zero-padded LoRa spectra.")
    parser.add_argument("--input-long-csv", type=Path, required=True)
    parser.add_argument("--point-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "point_spectrum_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    profiles = aggregate_profiles(args.input_long_csv)
    packet_counts = load_packet_counts(args.point_summary_csv)
    write_profile_csv(args.output_dir / "point_spectrum_profile.csv", profiles)

    svg_paths: list[Path] = []
    for key in sorted(profiles):
        q1 = profiles[key].get(1, {})
        q4 = profiles[key].get(4, {})
        if not q1 or not q4:
            continue
        q1_points = [(offset, q1[offset].mean) for offset in sorted(q1)]
        q4_points = [(offset, q4[offset].mean) for offset in sorted(q4)]
        svg_path = plot_dir / f"{point_label(key)}_raw_vs_q4.svg"
        write_svg(svg_path, key, q1_points, q4_points, packet_counts.get(key))
        svg_paths.append(svg_path)

    write_index(args.output_dir / "point_spectrum_comparison_index.html", svg_paths, args.output_dir)
    print(f"Wrote {len(svg_paths)} point spectrum plots to {plot_dir}")
    print(f"Wrote aggregate profile CSV to {args.output_dir / 'point_spectrum_profile.csv'}")
    print(f"Wrote index HTML to {args.output_dir / 'point_spectrum_comparison_index.html'}")


if __name__ == "__main__":
    main()
