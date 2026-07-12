#!/usr/bin/env python3
"""Dependency-free SVG figure for wideband multipath vs LoRa shoulder."""

from __future__ import annotations

import csv
import html
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "v2_output/20260711_wideband_lora_shoulder_link_figure"
PDP_CSV = (
    ROOT
    / "v2_output/20260623_from_raw/step5_chirp_intrinsic_spatial_patterns"
    / "04_chirp_pdp_profiles_long.csv"
)
PATH_CSV = (
    ROOT
    / "v2_output/20260623_from_raw/step6c_chirp_structure_original_minus25"
    / "02_stable_equivalent_paths_with_reference_overlap.csv"
)
SYMBOL_CSV = (
    ROOT
    / "v2_output/20260624_zero_padding_fft_q4_from_trusted_starts"
    / "symbol_peak_summary.csv"
)
SUBBIN_CSV = (
    ROOT
    / "v2_output/20260624_zero_padding_fft_q4_from_trusted_starts"
    / "subbin_spectrum_long.csv"
)
REP_CORRIDOR = 0
REP_LOCATION = 44
W = 1040
H = 520


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: object, default: float = math.nan) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def line(x1, y1, x2, y2, stroke="#111827", width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def text(x, y, content, size=13, fill="#111827", anchor="start", weight=400):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(content)}</text>'
    )


def circle(x, y, r, fill="#304C89", stroke="white", sw=0.8, opacity=1.0):
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
    )


def polyline(points, stroke="#304C89", width=2, opacity=1.0):
    pairs = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{pairs}" fill="none" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"/>'
    )


def polygon(points, fill="#9CA3AF", stroke="none", opacity=1.0):
    pairs = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{pairs}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'


def rect(x, y, w, h, fill="none", stroke="none", opacity=1.0):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'


def scale(domain_min, domain_max, range_min, range_max):
    def mapper(value):
        if domain_max == domain_min:
            return (range_min + range_max) / 2
        return range_min + (value - domain_min) * (range_max - range_min) / (domain_max - domain_min)

    return mapper


def axes(x, y, w, h, x_ticks, y_ticks, xlim, ylim, xlabel, ylabel):
    sx = scale(xlim[0], xlim[1], x, x + w)
    sy = scale(ylim[0], ylim[1], y + h, y)
    out = []
    out.append(rect(x, y, w, h, fill="#FFFFFF", stroke="#D1D5DB"))
    for tick in x_ticks:
        tx = sx(tick)
        out.append(line(tx, y, tx, y + h, stroke="#E5E7EB", width=1))
        out.append(text(tx, y + h + 18, f"{tick:g}", size=11, fill="#374151", anchor="middle"))
    for tick in y_ticks:
        ty = sy(tick)
        out.append(line(x, ty, x + w, ty, stroke="#E5E7EB", width=1))
        out.append(text(x - 8, ty + 4, f"{tick:g}", size=11, fill="#374151", anchor="end"))
    out.append(line(x, y + h, x + w, y + h, stroke="#111827", width=1))
    out.append(line(x, y, x, y + h, stroke="#111827", width=1))
    out.append(text(x + w / 2, y + h + 40, xlabel, size=13, fill="#111827", anchor="middle"))
    out.append(
        f'<text x="{x - 48:.2f}" y="{y + h / 2:.2f}" font-size="13" fill="#111827" '
        f'text-anchor="middle" transform="rotate(-90 {x - 48:.2f},{y + h / 2:.2f})">{esc(ylabel)}</text>'
    )
    return out, sx, sy


def select_lora_symbol() -> dict[str, str]:
    rows = [
        row
        for row in read_csv(SYMBOL_CSV)
        if int(float(row["corridor_id"])) == REP_CORRIDOR
        and int(float(row["position_id"])) == REP_LOCATION
        and int(float(row["q"])) == 4
    ]
    target = sum(fnum(row["secondary_peak_rel_db"]) for row in rows) / len(rows)
    return min(rows, key=lambda row: abs(fnum(row["secondary_peak_rel_db"]) - target))


def select_lora_symbols() -> list[dict[str, str]]:
    return [
        row
        for row in read_csv(SYMBOL_CSV)
        if int(float(row["corridor_id"])) == REP_CORRIDOR
        and int(float(row["position_id"])) == REP_LOCATION
        and int(float(row["q"])) == 4
    ]


def symbol_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["file_name"], row["packet_index"], row["local_symbol_index"]


def lora_symbol_curve(symbol: dict[str, str]) -> list[dict[str, float]]:
    rows = []
    for row in read_csv(SUBBIN_CSV):
        if row["file_name"] != symbol["file_name"]:
            continue
        if row["packet_index"] != symbol["packet_index"]:
            continue
        if row["local_symbol_index"] != symbol["local_symbol_index"]:
            continue
        if int(float(row["q"])) != 4:
            continue
        rows.append({"offset": fnum(row["subbin_offset"]), "db": fnum(row["mag_db_rel_peak"])})
    return sorted(rows, key=lambda row: row["offset"])


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def interpolate_curve(curve: list[dict[str, float]], x_value: float) -> float | None:
    if not curve or x_value < curve[0]["offset"] or x_value > curve[-1]["offset"]:
        return None
    for idx, point in enumerate(curve):
        x1 = point["offset"]
        if abs(x1 - x_value) < 1e-9:
            return point["db"]
        if x1 > x_value and idx > 0:
            prev = curve[idx - 1]
            x0 = prev["offset"]
            y0 = prev["db"]
            y1 = point["db"]
            return y0 + (y1 - y0) * (x_value - x0) / (x1 - x0)
    return None


def lora_aligned_symbol_curves() -> tuple[list[list[dict[str, float]]], list[dict[str, str]]]:
    symbols = select_lora_symbols()
    by_key = {symbol_key(row): row for row in symbols}
    curves = {key: [] for key in by_key}
    for row in read_csv(SUBBIN_CSV):
        key = symbol_key(row)
        if key not in curves or int(float(row["q"])) != 4:
            continue
        peak_offset = fnum(by_key[key]["interpolated_peak_offset_bins"], 0.0)
        curves[key].append(
            {
                "offset": fnum(row["subbin_offset"]) - peak_offset,
                "db": fnum(row["mag_db_rel_peak"]),
            }
        )
    aligned = []
    for key in sorted(curves):
        curve = sorted(curves[key], key=lambda row: row["offset"])
        if curve:
            aligned.append(curve)
    return aligned, symbols


def lora_statistical_spectrum() -> tuple[list[list[dict[str, float]]], list[dict[str, float]], dict[str, float]]:
    curves, symbols = lora_aligned_symbol_curves()
    grid = [-2.0 + 0.25 * idx for idx in range(17)]
    resampled = []
    grouped = {offset: [] for offset in grid}
    for curve in curves:
        sampled_curve = []
        for offset in grid:
            value = interpolate_curve(curve, offset)
            if value is None:
                continue
            sampled_curve.append({"offset": offset, "db": value})
            grouped[offset].append(value)
        if sampled_curve:
            resampled.append(sampled_curve)

    stats = []
    for offset in grid:
        values = grouped[offset]
        if not values:
            continue
        stats.append(
            {
                "offset": offset,
                "median": percentile(values, 0.50),
                "q25": percentile(values, 0.25),
                "q75": percentile(values, 0.75),
                "n": float(len(values)),
            }
        )

    rosl_values = [fnum(row["secondary_peak_rel_db"]) for row in symbols]
    summary = {
        "n_symbols": float(len(symbols)),
        "n_packets": float(len({row["packet_index"] for row in symbols})),
        "mean_rosl": sum(rosl_values) / len(rosl_values),
        "median_rosl": percentile(rosl_values, 0.50),
    }
    return resampled, stats, summary


def panel_a(x, y, w, h):
    out = []
    out.append(text(x, y - 22, "(a) Wideband chirp multipath", size=16, weight=700))
    plot_y = y
    plot_h = h - 52
    axis, sx, sy = axes(
        x + 58,
        plot_y,
        w - 70,
        plot_h,
        [-0.5, 0, 0.5],
        [-40, -30, -20, -10, 0],
        (-0.75, 0.75),
        (-40, 2),
        "Relative delay (us)",
        "Relative power (dB)",
    )
    out.extend(axis)
    pdp = [
        row
        for row in read_csv(PDP_CSV)
        if int(float(row["corridor_id"])) == REP_CORRIDOR
        and int(float(row["location_id"])) == REP_LOCATION
        and -0.75 <= fnum(row["delay_us"]) <= 0.75
    ]
    out.append(
        polyline(
            [(sx(fnum(row["delay_us"])), sy(max(fnum(row["relative_db"]), -40.0))) for row in pdp],
            stroke="#304C89",
            width=2.4,
        )
    )
    paths = [
        row
        for row in read_csv(PATH_CSV)
        if int(float(row["corridor_id"])) == REP_CORRIDOR
        and int(float(row["location_id"])) == REP_LOCATION
        and abs(fnum(row["threshold_db"]) + 25.0) < 1e-6
        and str(row["stable_20pct"]).lower() == "true"
    ]
    paths.sort(key=lambda row: fnum(row["amplitude_db_median"]), reverse=True)
    secondary = paths[0]
    out.append(circle(sx(0), sy(0), 5.5, fill="#111827", stroke="#111827"))
    out.append(text(sx(0) + 14, sy(0) + 15, "strongest path", size=12))
    for row in paths:
        radius = 3.6 + 5.0 * fnum(row["recurrence_fraction"], 0.0)
        out.append(circle(sx(fnum(row["delay_center_us"])), sy(fnum(row["amplitude_db_median"])), radius, fill="#D95F02"))
    sec_x = fnum(secondary["delay_center_us"])
    sec_y = fnum(secondary["amplitude_db_median"])
    out.append(line(sx(sec_x), sy(0), sx(sec_x), sy(sec_y), stroke="#D95F02", width=2))
    out.append(text(sx(sec_x) + 12, sy(sec_y) - 10, "strongest secondary path", size=12, fill="#D95F02"))
    out.append(text(sx(sec_x) + 14, (sy(0) + sy(sec_y)) / 2 + 4, f"rel. {sec_y:.1f} dB", size=12, fill="#D95F02"))
    out.append(text(x + 78, y + 42, f"stable paths K={len(paths)}", size=12, fill="#374151"))
    out.append(text(x + 78, y + 60, "marker size = recurrence", size=11, fill="#6B7280"))
    return out


def panel_b_single_symbol(x, y, w, h):
    out = []
    out.append(text(x, y - 22, "(b) LoRa q=4 spectral shoulder", size=16, weight=700))
    plot_h = h - 52
    axis, sx, sy = axes(
        x + 58,
        y,
        w - 70,
        plot_h,
        [-2, -1, 0, 1, 2],
        [-30, -20, -10, 0],
        (-2.05, 2.05),
        (-30, 1.2),
        "Relative frequency (bin)",
        "Level relative to peak (dB)",
    )
    out.extend(axis)
    symbol = select_lora_symbol()
    curve = lora_symbol_curve(symbol)
    peak_offset = fnum(symbol["interpolated_peak_offset_bins"])
    secondary_offset = fnum(symbol["secondary_peak_offset_bins"])
    secondary_db = fnum(symbol["secondary_peak_rel_db"])
    out.append(rect(sx(peak_offset - 0.25), y, sx(peak_offset + 0.25) - sx(peak_offset - 0.25), plot_h, fill="#9CA3AF", opacity=0.22))
    out.append(polyline([(sx(row["offset"]), sy(row["db"])) for row in curve], stroke="#1B998B", width=2.4))
    for row in curve:
        out.append(circle(sx(row["offset"]), sy(row["db"]), 3.0, fill="#1B998B", stroke="#1B998B"))
    out.append(circle(sx(peak_offset), sy(0), 5.5, fill="#111827", stroke="#111827"))
    out.append(text(sx(peak_offset) - 12, sy(0) - 8, "main peak", size=12, anchor="end"))
    out.append(circle(sx(secondary_offset), sy(secondary_db), 6.2, fill="#D95F02"))
    out.append(line(sx(secondary_offset), sy(0), sx(secondary_offset), sy(secondary_db), stroke="#D95F02", width=2))
    out.append(text(sx(secondary_offset) + 14, sy(secondary_db) + 24, f"ROSL = {secondary_db:.2f} dB", size=12, fill="#D95F02"))
    out.append(text(x + 76, y + plot_h + 66, "ROSL represents an unresolved spectral shoulder,", size=11, fill="#4B5563"))
    out.append(text(x + 76, y + plot_h + 82, "not a resolved multipath component.", size=11, fill="#4B5563"))
    return out


def panel_b_statistical_spectrum(x, y, w, h):
    out = []
    out.append(text(x, y - 22, "(b) LoRa q=4 spectral shoulder statistics", size=16, weight=700))
    plot_h = h - 52
    axis, sx, sy = axes(
        x + 58,
        y,
        w - 70,
        plot_h,
        [-2, -1, 0, 1, 2],
        [-30, -20, -10, 0],
        (-2.05, 2.05),
        (-30, 1.2),
        "Frequency relative to aligned main peak (bin)",
        "Level relative to peak (dB)",
    )
    out.extend(axis)
    curves, stats, summary = lora_statistical_spectrum()
    out.append(rect(sx(-0.25), y, sx(0.25) - sx(-0.25), plot_h, fill="#9CA3AF", opacity=0.22))

    for curve in curves:
        out.append(polyline([(sx(row["offset"]), sy(max(row["db"], -30.0))) for row in curve], stroke="#9CA3AF", width=0.65, opacity=0.16))

    upper = [(sx(row["offset"]), sy(max(row["q75"], -30.0))) for row in stats]
    lower = [(sx(row["offset"]), sy(max(row["q25"], -30.0))) for row in reversed(stats)]
    out.append(polygon(upper + lower, fill="#1B998B", opacity=0.18))
    out.append(polyline([(sx(row["offset"]), sy(max(row["median"], -30.0))) for row in stats], stroke="#1B998B", width=3.0))

    out.append(circle(sx(0), sy(0), 5.5, fill="#111827", stroke="#111827"))
    out.append(text(sx(0) - 12, sy(0) - 8, "aligned main peak", size=12, anchor="end"))
    out.append(rect(x + 70, y + 20, 247, 48, fill="#FFFFFF", opacity=0.84))
    out.append(text(x + 76, y + 42, f"n={summary['n_symbols']:.0f} symbols from {summary['n_packets']:.0f} packets", size=12, fill="#374151"))
    out.append(text(x + 76, y + 60, "gray: individual symbols; band: IQR; line: median", size=11, fill="#6B7280"))
    out.append(text(x + 76, y + plot_h + 66, f"Per-symbol ROSL: mean {summary['mean_rosl']:.2f} dB,", size=11, fill="#4B5563"))
    out.append(text(x + 76, y + plot_h + 82, f"median {summary['median_rosl']:.2f} dB.", size=11, fill="#4B5563"))
    return out


def render(panel_b_func, out_name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    body = []
    body.append(rect(0, 0, W, H, fill="#FFFFFF"))
    body.append(text(24, 32, "Wideband multipath structure and the LoRa spectral shoulder", size=20, weight=700))
    top = 78
    panel_h = 375
    body.extend(panel_a(24, top, 455, panel_h))
    body.extend(panel_b_func(535, top, 455, panel_h))
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
            '<style>text{font-family:Arial, Helvetica, sans-serif;} .small{font-size:11px;}</style>',
            *body,
            "</svg>",
        ]
    )
    out_path = OUT / out_name
    out_path.write_text(svg + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    paths = [
        render(panel_b_single_symbol, "wideband_lora_shoulder_link.svg"),
        render(panel_b_single_symbol, "wideband_lora_shoulder_link_single_symbol.svg"),
        render(panel_b_statistical_spectrum, "wideband_lora_shoulder_link_statistical_spectrum.svg"),
    ]
    for out_path in paths:
        print(out_path)


if __name__ == "__main__":
    main()
