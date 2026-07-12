#!/usr/bin/env python3
"""Figure: wideband multipath structure and LoRa spectral shoulder link."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
POINT_DIM_CSV = (
    ROOT
    / "v2_output/20260711_lora_main_secondary_dimensions"
    / "01_main_secondary_point_dimensions.csv"
)

REP_CORRIDOR = 0
REP_LOCATION = 44


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: object, default: float = math.nan) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def key(row: dict[str, str]) -> tuple[int, int]:
    return int(float(row["corridor_id"])), int(float(row.get("location_id", row.get("position_id"))))


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
        rows.append(
            {
                "offset": fnum(row["subbin_offset"]),
                "db": fnum(row["mag_db_rel_peak"]),
            }
        )
    return sorted(rows, key=lambda row: row["offset"])


def panel_a(ax) -> None:
    pdp = [
        row
        for row in read_csv(PDP_CSV)
        if int(float(row["corridor_id"])) == REP_CORRIDOR
        and int(float(row["location_id"])) == REP_LOCATION
        and -0.75 <= fnum(row["delay_us"]) <= 0.75
    ]
    paths = [
        row
        for row in read_csv(PATH_CSV)
        if int(float(row["corridor_id"])) == REP_CORRIDOR
        and int(float(row["location_id"])) == REP_LOCATION
        and abs(fnum(row["threshold_db"]) + 25.0) < 1e-6
        and str(row["stable_20pct"]).lower() == "true"
    ]
    paths.sort(key=lambda row: fnum(row["amplitude_db_median"]), reverse=True)
    strongest = paths[0]

    ax.plot(
        [fnum(row["delay_us"]) for row in pdp],
        [max(fnum(row["relative_db"]), -55.0) for row in pdp],
        color="#304C89",
        lw=1.9,
    )
    ax.scatter([0], [0], s=58, color="#111827", zorder=4)
    ax.text(0.02, -2.5, "strongest path", fontsize=8.2, color="#111827")

    path_x = [fnum(row["delay_center_us"]) for row in paths]
    path_y = [fnum(row["amplitude_db_median"]) for row in paths]
    path_size = [42 + 120 * fnum(row["recurrence_fraction"], 0.0) for row in paths]
    ax.scatter(path_x, path_y, s=path_size, color="#D95F02", edgecolor="white", lw=0.7, zorder=5)

    sx = fnum(strongest["delay_center_us"])
    sy = fnum(strongest["amplitude_db_median"])
    ax.annotate(
        "",
        xy=(sx, sy),
        xytext=(sx, 0),
        arrowprops={"arrowstyle": "<->", "color": "#D95F02", "lw": 1.1},
    )
    ax.text(sx + 0.035, sy / 2, f"{sy:.1f} dB", fontsize=8.3, color="#D95F02", va="center")
    ax.text(
        -0.72,
        -6.0,
        f"stable paths K={len(paths)}\n(marker size = recurrence)",
        fontsize=8.1,
        color="#374151",
        ha="left",
        va="top",
    )
    ax.set_title("(a) Wideband chirp multipath", loc="left", fontsize=10.5, weight="bold")
    ax.set_xlabel("Relative delay (us)")
    ax.set_ylabel("Relative power (dB)")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-32, 2)
    ax.grid(True, color="#E5E7EB", lw=0.7)


def panel_b(ax) -> None:
    symbol = select_lora_symbol()
    curve = lora_symbol_curve(symbol)
    peak_offset = fnum(symbol["interpolated_peak_offset_bins"])
    secondary_offset = fnum(symbol["secondary_peak_offset_bins"])
    secondary_db = fnum(symbol["secondary_peak_rel_db"])

    ax.plot(
        [row["offset"] for row in curve],
        [row["db"] for row in curve],
        color="#1B998B",
        lw=1.9,
        marker="o",
        ms=3.2,
    )
    ax.axvspan(peak_offset - 0.25, peak_offset + 0.25, color="#9CA3AF", alpha=0.25, lw=0)
    ax.scatter([peak_offset], [0], s=58, color="#111827", zorder=4)
    ax.text(peak_offset + 0.05, -0.65, "main peak", fontsize=8.2, color="#111827")
    ax.scatter([secondary_offset], [secondary_db], s=66, color="#D95F02", edgecolor="white", lw=0.7, zorder=5)
    ax.annotate(
        "",
        xy=(secondary_offset, secondary_db),
        xytext=(secondary_offset, 0),
        arrowprops={"arrowstyle": "<->", "color": "#D95F02", "lw": 1.1},
    )
    ax.text(
        secondary_offset + 0.12,
        secondary_db / 2,
        f"ROSL = {secondary_db:.2f} dB",
        fontsize=8.3,
        color="#D95F02",
        va="center",
    )
    ax.text(
        -1.95,
        -22.5,
        "ROSL represents an unresolved spectral shoulder\nrather than a resolved multipath component.",
        fontsize=7.4,
        color="#4B5563",
        va="bottom",
    )
    ax.set_title("(b) LoRa q=4 spectral shoulder", loc="left", fontsize=10.5, weight="bold")
    ax.set_xlabel("Relative frequency (bin)")
    ax.set_ylabel("Level relative to peak (dB)")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-24, 1.2)
    ax.grid(True, color="#E5E7EB", lw=0.7)


def scatter_panel(ax, rows, x_name, y_name, title, xlabel, stat_text) -> None:
    for row in rows:
        is_ref = str(row["is_reference_point32"]).lower() == "true"
        x = fnum(row[x_name])
        y = fnum(row[y_name])
        if is_ref:
            ax.scatter([x], [y], s=52, facecolor="none", edgecolor="#6B7280", lw=1.4, zorder=3)
            ax.text(x, y + 0.018, "P32", fontsize=7.3, color="#6B7280", ha="center")
        else:
            ax.scatter([x], [y], s=45, color="#304C89", alpha=0.88, edgecolor="white", lw=0.45)
    valid = [(fnum(row[x_name]), fnum(row[y_name])) for row in rows if str(row["is_reference_point32"]).lower() != "true"]
    xs = [item[0] for item in valid]
    ys = [item[1] for item in valid]
    if len(xs) > 1:
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        if denom > 1e-12:
            slope = sum((x - mx) * (y - my) for x, y in valid) / denom
            intercept = my - slope * mx
            x0, x1 = min(xs), max(xs)
            ax.plot([x0, x1], [intercept + slope * x0, intercept + slope * x1], color="#D95F02", lw=1.2)
    ax.text(0.04, 0.08, stat_text, transform=ax.transAxes, fontsize=8.2, color="#111827")
    ax.set_title(title, loc="left", fontsize=9.5, weight="bold")
    ax.set_xlabel(xlabel, fontsize=8.8)
    ax.set_ylabel("ROSL (dB)", fontsize=8.8)
    ax.grid(True, color="#E5E7EB", lw=0.7)


def panel_c(fig, spec) -> None:
    rows = read_csv(POINT_DIM_CSV)
    ax_top = fig.add_subplot(spec[0, 0])
    ax_bottom = fig.add_subplot(spec[1, 0])
    scatter_panel(
        ax_top,
        rows,
        "chirp_secondary_amp_sum",
        "lora_secondary_peak_rel_db_symbol_mean",
        "(c1) Secondary strength vs. ROSL",
        r"Wideband secondary strength  $\sum \sqrt{r_i}|a_i|$",
        r"$\rho=-0.726,\ p=0.00456$" + "\n" + "LOO: [-0.801, -0.680]",
    )
    scatter_panel(
        ax_bottom,
        rows,
        "chirp_stable_path_count",
        "lora_secondary_peak_rel_db_symbol_mean",
        "(c2) Stable path count vs. ROSL",
        "Stable wideband path count",
        r"$\rho=-0.686,\ p=0.00764$",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
        }
    )
    fig = plt.figure(figsize=(14.2, 4.6), dpi=220)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.08, 1.08, 1.18], wspace=0.34)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig, gs[0, 2].subgridspec(2, 1, hspace=0.55))
    fig.suptitle(
        "Wideband multipath structure is linked to the LoRa spectral shoulder",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=12.5,
        weight="bold",
    )
    fig.savefig(OUT / "wideband_lora_shoulder_link.png", bbox_inches="tight")
    fig.savefig(OUT / "wideband_lora_shoulder_link.svg", bbox_inches="tight")
    plt.close(fig)
    print(OUT / "wideband_lora_shoulder_link.png")
    print(OUT / "wideband_lora_shoulder_link.svg")


if __name__ == "__main__":
    main()
