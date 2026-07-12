from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Sequence


GROUP_FIELDS = [
    "experiment_id",
    "corridor_id",
    "position_id",
    "filename_sf",
    "filename_tx_power_dbm",
    "filename_preamble_len",
]


def safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def point_key(row: dict) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in GROUP_FIELDS)


def point_label(key: tuple[str, ...]) -> str:
    exp, corridor, position, sf, tx_power, preamble = key
    return f"exp{exp}_corr{corridor}_pos{position}_sf{sf}_tp{tx_power}_pre{preamble}"


def point_title(key: tuple[str, ...]) -> str:
    exp, corridor, position, sf, tx_power, preamble = key
    return f"corr {corridor} / pos {position}"


def point_subtitle(key: tuple[str, ...], packet_count: int) -> str:
    exp, corridor, position, sf, tx_power, preamble = key
    return f"exp {exp}, SF{sf}, TP{tx_power}, preamble {preamble}, packets {packet_count}"


def percentile(values: Sequence[float], pct: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def iqr(values: Sequence[float]) -> float:
    return percentile(values, 75.0) - percentile(values, 25.0)


def esc(value: object) -> str:
    return html.escape(str(value))


def load_packet_curves(long_csv: Path) -> tuple[
    dict[tuple[str, ...], dict[tuple[str, str], dict[float, float]]],
    list[float],
]:
    # First average q=4 sub-bin values over preamble symbols inside each packet.
    accum: dict[tuple[tuple[str, ...], tuple[str, str], float], list[float]] = defaultdict(list)
    with long_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("q") != "4":
                continue
            offset = safe_float(row.get("subbin_offset"))
            mag_db = safe_float(row.get("mag_db_rel_peak"))
            if offset is None or mag_db is None:
                continue
            key = point_key(row)
            packet_id = (row.get("file_name", ""), row.get("packet_index", ""))
            accum[(key, packet_id, offset)].append(mag_db)

    curves: dict[tuple[str, ...], dict[tuple[str, str], dict[float, float]]] = defaultdict(dict)
    offsets = sorted({offset for _, _, offset in accum})
    for (key, packet_id, offset), values in accum.items():
        curves[key].setdefault(packet_id, {})[offset] = sum(values) / len(values)

    return dict(curves), offsets


def write_packet_curves_csv(
    path: Path,
    curves: dict[tuple[str, ...], dict[tuple[str, str], dict[float, float]]],
    offsets: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = GROUP_FIELDS + ["file_name", "packet_index"] + [f"bin_{offset:+.2f}" for offset in offsets]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(curves, key=lambda k: (int(k[1]), int(k[2]))):
            for packet_id in sorted(curves[key], key=lambda item: (item[0], int(float(item[1])) if item[1] else -1)):
                row = {field: value for field, value in zip(GROUP_FIELDS, key)}
                row["file_name"], row["packet_index"] = packet_id
                for offset in offsets:
                    row[f"bin_{offset:+.2f}"] = curves[key][packet_id].get(offset, float("nan"))
                writer.writerow(row)


def compute_point_stats(
    curves: dict[tuple[str, ...], dict[tuple[str, str], dict[float, float]]],
    offsets: Sequence[float],
) -> dict[tuple[str, ...], dict[str, dict[float, float]]]:
    stats: dict[tuple[str, ...], dict[str, dict[float, float]]] = {}
    for key, packets in curves.items():
        stats[key] = {"median": {}, "iqr": {}, "q25": {}, "q75": {}}
        for offset in offsets:
            values = [curve[offset] for curve in packets.values() if offset in curve and math.isfinite(curve[offset])]
            stats[key]["median"][offset] = median(values) if values else float("nan")
            stats[key]["q25"][offset] = percentile(values, 25.0)
            stats[key]["q75"][offset] = percentile(values, 75.0)
            stats[key]["iqr"][offset] = iqr(values)
    return stats


def write_iqr_csv(path: Path, stats: dict[tuple[str, ...], dict[str, dict[float, float]]], offsets: Sequence[float]) -> None:
    fields = GROUP_FIELDS + [f"iqr_bin_{offset:+.2f}" for offset in offsets]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(stats, key=lambda k: (int(k[1]), int(k[2]))):
            row = {field: value for field, value in zip(GROUP_FIELDS, key)}
            for offset in offsets:
                row[f"iqr_bin_{offset:+.2f}"] = stats[key]["iqr"][offset]
            writer.writerow(row)


def color_from_iqr(value: float, vmax: float) -> str:
    if not math.isfinite(value):
        return "#eeeeee"
    t = max(0.0, min(1.0, value / max(vmax, 1e-9)))
    # Light yellow to orange-red.
    stops = [
        (255, 255, 229),
        (254, 204, 92),
        (253, 141, 60),
        (227, 26, 28),
        (128, 0, 38),
    ]
    pos = t * (len(stops) - 1)
    idx = min(len(stops) - 2, int(math.floor(pos)))
    frac = pos - idx
    c0 = stops[idx]
    c1 = stops[idx + 1]
    rgb = tuple(round(c0[i] * (1 - frac) + c1[i] * frac) for i in range(3))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def polyline(points: Sequence[tuple[float, float]], sx, sy) -> str:
    return " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)


def packet_overlay_svg_fragment(
    key: tuple[str, ...],
    packets: dict[tuple[str, str], dict[float, float]],
    point_stats: dict[str, dict[float, float]],
    offsets: Sequence[float],
    x0: float,
    y0: float,
    panel_w: float,
    panel_h: float,
    compact: bool,
) -> list[str]:
    plot_l = x0 + (48 if compact else 62)
    plot_t = y0 + (42 if compact else 52)
    plot_w = panel_w - (70 if compact else 90)
    plot_h = panel_h - (70 if compact else 94)
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -35.0, 2.0

    def sx(value: float) -> float:
        return plot_l + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        value = min(max(value, y_min), y_max)
        return plot_t + (y_max - value) / (y_max - y_min) * plot_h

    lines = [
        f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="#fff" stroke="#d4d4d4"/>',
        f'<text x="{x0 + 12}" y="{y0 + 21}" font-family="Arial" font-size="{13 if compact else 16}" font-weight="700" fill="#222">{esc(point_title(key))}</text>',
        f'<text x="{x0 + 12}" y="{y0 + (37 if compact else 42)}" font-family="Arial" font-size="{9 if compact else 11}" fill="#666">{esc(point_subtitle(key, len(packets)))}</text>',
        f'<rect x="{plot_l}" y="{plot_t}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#c6c6c6"/>',
    ]

    for tick in [-2, -1, 0, 1, 2]:
        x = sx(tick)
        dash = ' stroke-dasharray="4 4"' if tick == 0 else ""
        stroke = "#cfcfcf" if tick == 0 else "#e8e8e8"
        lines.append(f'<line x1="{x:.2f}" y1="{plot_t}" x2="{x:.2f}" y2="{plot_t + plot_h}" stroke="{stroke}"{dash}/>')
        lines.append(f'<text x="{x:.2f}" y="{plot_t + plot_h + 15}" text-anchor="middle" font-family="Arial" font-size="{8 if compact else 10}" fill="#555">{tick}</text>')

    for tick in [-30, -20, -10, 0]:
        y = sy(tick)
        lines.append(f'<line x1="{plot_l}" y1="{y:.2f}" x2="{plot_l + plot_w}" y2="{y:.2f}" stroke="#e8e8e8"/>')
        lines.append(f'<text x="{plot_l - 7}" y="{y + 3:.2f}" text-anchor="end" font-family="Arial" font-size="{8 if compact else 10}" fill="#555">{tick}</text>')

    for packet_id, curve in sorted(packets.items(), key=lambda item: (item[0][0], int(float(item[0][1])) if item[0][1] else -1)):
        pts = [(offset, curve[offset]) for offset in offsets if offset in curve and math.isfinite(curve[offset])]
        lines.append(f'<polyline points="{polyline(pts, sx, sy)}" fill="none" stroke="#7aa6d8" stroke-width="{0.7 if compact else 0.9}" opacity="0.25"/>')

    med_points = [(offset, point_stats["median"][offset]) for offset in offsets if math.isfinite(point_stats["median"][offset])]
    lines.append(f'<polyline points="{polyline(med_points, sx, sy)}" fill="none" stroke="#d62728" stroke-width="{2.2 if compact else 2.8}"/>')
    lines.append(f'<text x="{plot_l + plot_w / 2:.2f}" y="{y0 + panel_h - 9}" text-anchor="middle" font-family="Arial" font-size="{8 if compact else 10}" fill="#555">sub-bin offset</text>')
    lines.append(f'<text x="{x0 + 16}" y="{plot_t + plot_h / 2:.2f}" transform="rotate(-90 {x0 + 16} {plot_t + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial" font-size="{8 if compact else 10}" fill="#555">rel. dB</text>')
    return lines


def write_individual_overlays(
    output_dir: Path,
    curves: dict[tuple[str, ...], dict[tuple[str, str], dict[float, float]]],
    stats: dict[tuple[str, ...], dict[str, dict[float, float]]],
    offsets: Sequence[float],
) -> list[Path]:
    plot_dir = output_dir / "packet_overlay_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key in sorted(curves, key=lambda k: (int(k[1]), int(k[2]))):
        width, height = 980, 560
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]
        lines.extend(packet_overlay_svg_fragment(key, curves[key], stats[key], offsets, 24, 24, width - 48, height - 48, compact=False))
        lines.extend(
            [
                '<line x1="92" y1="53" x2="126" y2="53" stroke="#7aa6d8" stroke-width="2" opacity="0.5"/>',
                '<text x="134" y="58" font-family="Arial" font-size="12" fill="#555">individual packets</text>',
                '<line x1="272" y1="53" x2="306" y2="53" stroke="#d62728" stroke-width="3"/>',
                '<text x="314" y="58" font-family="Arial" font-size="12" fill="#555">packet median</text>',
                "</svg>",
            ]
        )
        path = plot_dir / f"{point_label(key)}_packet_overlay.svg"
        path.write_text("\n".join(lines), encoding="utf-8")
        paths.append(path)
    return paths


def write_overlay_grid_svg(
    path: Path,
    curves: dict[tuple[str, ...], dict[tuple[str, str], dict[float, float]]],
    stats: dict[tuple[str, ...], dict[str, dict[float, float]]],
    offsets: Sequence[float],
) -> None:
    keys = sorted(curves, key=lambda k: (int(k[1]), int(k[2])))
    cols = 4
    rows = math.ceil(len(keys) / cols)
    panel_w, panel_h = 760, 320
    gap_x, gap_y = 18, 18
    margin_l, margin_r, margin_b = 42, 34, 40
    header_h = 94
    width = margin_l + margin_r + cols * panel_w + (cols - 1) * gap_x
    height = header_h + rows * panel_h + (rows - 1) * gap_y + margin_b
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_l}" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#222">Q=4 Packet Stability Overlay by Point</text>',
        f'<text x="{margin_l}" y="61" font-family="Arial" font-size="15" fill="#555">Thin blue lines are packet-level curves; red line is packet median. Unified preamble_len=16, skip=0, feature_symbols=16.</text>',
        f'<line x1="{margin_l}" y1="80" x2="{width - margin_r}" y2="80" stroke="#d8d8d8"/>',
    ]
    for idx, key in enumerate(keys):
        col = idx % cols
        row = idx // cols
        x0 = margin_l + col * (panel_w + gap_x)
        y0 = header_h + row * (panel_h + gap_y)
        lines.extend(packet_overlay_svg_fragment(key, curves[key], stats[key], offsets, x0, y0, panel_w, panel_h, compact=True))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_iqr_heatmap_svg(
    path: Path,
    stats: dict[tuple[str, ...], dict[str, dict[float, float]]],
    offsets: Sequence[float],
) -> None:
    keys = sorted(stats, key=lambda k: (int(k[1]), int(k[2])))
    cell_w, cell_h = 52, 27
    left = 172
    top = 94
    right = 150
    bottom = 58
    width = left + len(offsets) * cell_w + right
    height = top + len(keys) * cell_h + bottom
    all_iqr = [stats[key]["iqr"][offset] for key in keys for offset in offsets]
    finite_iqr = [value for value in all_iqr if math.isfinite(value)]
    vmax = percentile(finite_iqr, 95.0) if finite_iqr else 1.0
    vmax = max(vmax, 1.0)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="32" y="34" font-family="Arial" font-size="24" font-weight="700" fill="#222">Q=4 Sub-bin IQR Heatmap</text>',
        '<text x="32" y="61" font-family="Arial" font-size="14" fill="#555">Rows are points, columns are 17 q=4 sub-bins. Cell value is packet-to-packet IQR in relative dB.</text>',
    ]
    for col, offset in enumerate(offsets):
        x = left + col * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x:.2f}" y="{top - 10}" transform="rotate(-45 {x:.2f} {top - 10})" '
            f'text-anchor="start" font-family="Arial" font-size="10" fill="#444">{offset:+.2f}</text>'
        )

    for row, key in enumerate(keys):
        y = top + row * cell_h
        label = f"c{key[1]} p{key[2]}"
        lines.append(f'<text x="{left - 10}" y="{y + cell_h * 0.68:.2f}" text-anchor="end" font-family="Arial" font-size="11" fill="#333">{esc(label)}</text>')
        for col, offset in enumerate(offsets):
            value = stats[key]["iqr"][offset]
            x = left + col * cell_w
            color = color_from_iqr(value, vmax)
            lines.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}" stroke="#ffffff" stroke-width="1"/>')
            text_color = "#ffffff" if math.isfinite(value) and value / vmax > 0.62 else "#222222"
            lines.append(f'<text x="{x + cell_w / 2:.2f}" y="{y + cell_h * 0.66:.2f}" text-anchor="middle" font-family="Arial" font-size="9" fill="{text_color}">{value:.1f}</text>')

    legend_x = left + len(offsets) * cell_w + 38
    legend_y = top
    legend_h = 220
    steps = 80
    for i in range(steps):
        t = i / (steps - 1)
        y = legend_y + (1.0 - t) * legend_h
        color = color_from_iqr(t * vmax, vmax)
        lines.append(f'<rect x="{legend_x}" y="{y:.2f}" width="24" height="{legend_h / steps + 1:.2f}" fill="{color}" stroke="none"/>')
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="24" height="{legend_h}" fill="none" stroke="#777"/>')
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = tick * vmax
        y = legend_y + (1.0 - tick) * legend_h
        lines.append(f'<line x1="{legend_x + 24}" y1="{y:.2f}" x2="{legend_x + 30}" y2="{y:.2f}" stroke="#555"/>')
        lines.append(f'<text x="{legend_x + 35}" y="{y + 4:.2f}" font-family="Arial" font-size="10" fill="#333">{value:.1f}</text>')
    lines.append(f'<text x="{legend_x}" y="{legend_y + legend_h + 24}" font-family="Arial" font-size="11" fill="#333">IQR dB</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate q=4 packet overlay and IQR stability plots.")
    parser.add_argument("--input-long-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves, offsets = load_packet_curves(args.input_long_csv)
    stats = compute_point_stats(curves, offsets)
    write_packet_curves_csv(args.output_dir / "q4_packet_curves.csv", curves, offsets)
    write_iqr_csv(args.output_dir / "q4_subbin_iqr_by_point.csv", stats, offsets)
    write_individual_overlays(args.output_dir, curves, stats, offsets)
    write_overlay_grid_svg(args.output_dir / "q4_packet_overlay_grid.svg", curves, stats, offsets)
    write_iqr_heatmap_svg(args.output_dir / "q4_subbin_iqr_heatmap.svg", stats, offsets)
    print(f"Wrote {len(curves)} point packet overlays and IQR heatmap to {args.output_dir}")


if __name__ == "__main__":
    main()
