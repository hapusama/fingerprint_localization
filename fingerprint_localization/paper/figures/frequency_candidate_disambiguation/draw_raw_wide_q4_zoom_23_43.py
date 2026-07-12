from __future__ import annotations

import csv
import math
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW_CSV = ROOT / "v2_output/20260623_from_raw/data_processing/lora_frequency_s17_54points.csv"
Q4_CSV = ROOT / "v2_output/20260624_zero_padding_fft_q1_q4_point_compare/subbin_spectrum_long.csv"

QUERY_FILE = "2_1_23_11_2_16.bin"
QUERY_PACKET = 16
TRUE_LOC = 23
AMBIG_LOC = 43
OUTPUT_PREFIX = "raw_wide_q4_zoom_23_43"

WIDTH_IN = 7.2
HEIGHT_IN = 3.45
DPI = 400

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
GRAY = "#666666"
LIGHT_GRAY = "#d9d9d9"
BLACK = "#111111"
WHITE = "#ffffff"
ZOOM_FILL = "#fff0d6"
ZOOM_EDGE = "#b26a00"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/ArialHB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def raw_col(offset: int) -> str:
    return f"preamble_fft_mag_bin_{offset:+d}"


def row_to_db_rel_peak(row: dict[str, str], offsets: list[int]) -> dict[int, float]:
    mags = [max(float(row[raw_col(offset)]), 1e-12) for offset in offsets]
    peak = max(mags)
    return {offset: 20.0 * math.log10(mag / peak) for offset, mag in zip(offsets, mags)}


def summarize(values_by_offset: dict[float, list[float]], offsets: list[float]) -> dict[float, tuple[float, float, int]]:
    out: dict[float, tuple[float, float, int]] = {}
    for offset in offsets:
        values = values_by_offset[offset]
        out[offset] = (mean(values), std(values), len(values))
    return out


def load_raw_rows() -> list[dict]:
    offsets = list(range(-8, 9))
    buckets: dict[str, dict[float, list[float]]] = {
        "query": defaultdict(list),
        "true_template": defaultdict(list),
        "ambiguous_template": defaultdict(list),
    }

    with RAW_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            loc = int(row["position_id"])
            if loc not in (TRUE_LOC, AMBIG_LOC):
                continue
            file_name = row["file_name"]
            packet = int(row["packet_index"])
            is_query = file_name == QUERY_FILE and packet == QUERY_PACKET
            values = row_to_db_rel_peak(row, offsets)

            if is_query:
                for offset, value in values.items():
                    buckets["query"][float(offset)].append(value)
            if loc == TRUE_LOC and not is_query:
                for offset, value in values.items():
                    buckets["true_template"][float(offset)].append(value)
            if loc == AMBIG_LOC:
                for offset, value in values.items():
                    buckets["ambiguous_template"][float(offset)].append(value)

    summaries = {
        kind: summarize(bucket, [float(offset) for offset in offsets])
        for kind, bucket in buckets.items()
    }
    rows: list[dict] = []
    for offset in offsets:
        row = {
            "panel": "raw_q1_wide",
            "q": 1,
            "offset_bin": float(offset),
        }
        for kind in ("query", "true_template", "ambiguous_template"):
            value_mean, value_std, count = summaries[kind][float(offset)]
            row[f"{kind}_mean_db"] = value_mean
            row[f"{kind}_std_db"] = value_std
            row[f"{kind}_count"] = count
        rows.append(row)
    return rows


def load_q4_rows() -> list[dict]:
    buckets: dict[str, dict[float, list[float]]] = {
        "query": defaultdict(list),
        "true_template": defaultdict(list),
        "ambiguous_template": defaultdict(list),
    }

    with Q4_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["q"]) != 4:
                continue
            loc = int(row["position_id"])
            if loc not in (TRUE_LOC, AMBIG_LOC):
                continue
            offset = float(row["subbin_offset"])
            if offset < -2.0 or offset > 2.0:
                continue
            value = float(row["mag_db_rel_peak"])
            file_name = row["file_name"]
            packet = int(row["packet_index"])
            is_query = file_name == QUERY_FILE and packet == QUERY_PACKET

            if is_query:
                buckets["query"][offset].append(value)
            if loc == TRUE_LOC and not is_query:
                buckets["true_template"][offset].append(value)
            if loc == AMBIG_LOC:
                buckets["ambiguous_template"][offset].append(value)

    offsets = sorted(buckets["query"])
    summaries = {kind: summarize(bucket, offsets) for kind, bucket in buckets.items()}
    rows: list[dict] = []
    for offset in offsets:
        row = {
            "panel": "q4_zoom_center",
            "q": 4,
            "offset_bin": offset,
        }
        for kind in ("query", "true_template", "ambiguous_template"):
            value_mean, value_std, count = summaries[kind][offset]
            row[f"{kind}_mean_db"] = value_mean
            row[f"{kind}_std_db"] = value_std
            row[f"{kind}_count"] = count
        rows.append(row)
    return rows


def rmse(rows: list[dict], panel: str, target_key: str, offsets: set[float] | None = None) -> float:
    selected = [
        row
        for row in rows
        if row["panel"] == panel and (offsets is None or row["offset_bin"] in offsets)
    ]
    return math.sqrt(
        sum((row["query_mean_db"] - row[target_key]) ** 2 for row in selected) / len(selected)
    )


def load_series() -> tuple[list[dict], dict[str, float]]:
    rows = load_raw_rows() + load_q4_rows()
    center_offsets = {-2.0, -1.0, 0.0, 1.0, 2.0}
    metrics = {
        "raw_center_rmse_query_to_true_db": rmse(rows, "raw_q1_wide", "true_template_mean_db", center_offsets),
        "raw_center_rmse_query_to_ambiguous_db": rmse(rows, "raw_q1_wide", "ambiguous_template_mean_db", center_offsets),
        "q4_rmse_query_to_true_db": rmse(rows, "q4_zoom_center", "true_template_mean_db"),
        "q4_rmse_query_to_ambiguous_db": rmse(rows, "q4_zoom_center", "ambiguous_template_mean_db"),
    }
    return rows, metrics


def write_data(rows: list[dict], metrics: dict[str, float]) -> Path:
    out = HERE / f"{OUTPUT_PREFIX}_data.csv"
    fields = [
        "query_file",
        "query_packet",
        "true_location",
        "ambiguous_location",
        "raw_center_rmse_query_to_true_db",
        "raw_center_rmse_query_to_ambiguous_db",
        "q4_rmse_query_to_true_db",
        "q4_rmse_query_to_ambiguous_db",
        "panel",
        "q",
        "offset_bin",
        "query_mean_db",
        "query_std_db",
        "query_count",
        "true_template_mean_db",
        "true_template_std_db",
        "true_template_count",
        "ambiguous_template_mean_db",
        "ambiguous_template_std_db",
        "ambiguous_template_count",
        "note",
    ]
    note = (
        "Raw q=1 uses integer FFT bins -8..+8 normalized by each packet local peak; "
        "q=4 zero padding gives finer sampling of the highlighted [-2,+2] center-bin neighborhood "
        "and does not add physical resolution."
    )
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query_file": QUERY_FILE,
                    "query_packet": QUERY_PACKET,
                    "true_location": TRUE_LOC,
                    "ambiguous_location": AMBIG_LOC,
                    **metrics,
                    **row,
                    "note": note,
                }
            )
    return out


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((x - width / 2, y - height / 2), text, font=font, fill=fill)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: str,
    width: int,
    dash: int,
    gap: int,
) -> None:
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        distance = 0.0
        while distance < length:
            end = min(distance + dash, length)
            draw.line(
                (x0 + ux * distance, y0 + uy * distance, x0 + ux * end, y0 + uy * end),
                fill=fill,
                width=width,
            )
            distance += dash + gap


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill: str, width: int) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 12.0 * width / 2.0
    for delta in (math.pi * 0.82, -math.pi * 0.82):
        x = end[0] + head * math.cos(angle + delta)
        y = end[1] + head * math.sin(angle + delta)
        draw.line((end[0], end[1], x, y), fill=fill, width=width)


def render_png(rows: list[dict], metrics: dict[str, float]) -> Path:
    width = int(WIDTH_IN * DPI)
    height = int(HEIGHT_IN * DPI)
    scale = DPI / 100
    image = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(10.6 * scale), bold=True)
    panel_font = load_font(int(8.4 * scale), bold=True)
    label_font = load_font(int(7.3 * scale))
    small_font = load_font(int(6.25 * scale))

    draw.text((42 * scale, 23 * scale), "Raw FFT bins and q=4 center-neighborhood detail", font=title_font, fill=BLACK)
    draw.text(
        (42 * scale, 43 * scale),
        f"RSSI+ ambiguous pair: loc {TRUE_LOC} vs loc {AMBIG_LOC}; query packet is from loc {TRUE_LOC}",
        font=small_font,
        fill=GRAY,
    )

    left = (220, 322, 1310, 1080)
    right = (1590, 322, 2710, 1080)
    y_min, y_max = -40.0, 2.0

    def x_at(box: tuple[float, float, float, float], value: float, x_min: float, x_max: float) -> float:
        x0, _y0, x1, _y1 = box
        return x0 + (value - x_min) / (x_max - x_min) * (x1 - x0)

    def y_at(box: tuple[float, float, float, float], value: float) -> float:
        _x0, y0, _x1, y1 = box
        value = max(y_min, min(y_max, value))
        return y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)

    def axes(
        box: tuple[float, float, float, float],
        title: str,
        x_min: float,
        x_max: float,
        ticks: list[float],
        shade_zoom: bool,
    ) -> None:
        x0, y0, x1, y1 = box
        draw.text((x0, y0 - 31 * scale), title, font=panel_font, fill=BLACK)
        if shade_zoom:
            zx0 = x_at(box, -2.0, x_min, x_max)
            zx1 = x_at(box, 2.0, x_min, x_max)
            draw.rectangle((zx0, y0, zx1, y1), fill=ZOOM_FILL)
            dashed_line(
                draw,
                [(zx0, y0), (zx1, y0), (zx1, y1), (zx0, y1), (zx0, y0)],
                ZOOM_EDGE,
                max(1, int(1.0 * scale)),
                int(5 * scale),
                int(4 * scale),
            )
            draw_centered(draw, (zx0 + zx1) / 2, y0 + 18 * scale, "[-2,+2]", small_font, ZOOM_EDGE)

        for tick in [-40, -30, -20, -10, 0]:
            y = y_at(box, tick)
            draw.line((x0, y, x1, y), fill=LIGHT_GRAY, width=max(1, int(0.75 * scale)))
            tw, th = text_size(draw, str(tick), small_font)
            draw.text((x0 - tw - 8 * scale, y - th / 2), str(tick), font=small_font, fill=GRAY)
        for tick in ticks:
            x = x_at(box, tick, x_min, x_max)
            draw.line((x, y0, x, y1), fill="#eeeeee", width=max(1, int(0.65 * scale)))
            label = f"{tick:g}"
            draw_centered(draw, x, y1 + 18 * scale, label, small_font, BLACK)
        draw.line((x0, y1, x1, y1), fill=BLACK, width=max(1, int(1.15 * scale)))
        draw.line((x0, y0, x0, y1), fill=BLACK, width=max(1, int(1.15 * scale)))

    axes(left, "(a) Raw integer-bin spectrum, wider context", -8.0, 8.0, [-8, -6, -4, -2, 0, 2, 4, 6, 8], True)
    axes(right, "(b) q=4 zoom of the highlighted center bins", -2.0, 2.0, [-2, -1, 0, 1, 2], False)

    raw_rows = [row for row in rows if row["panel"] == "raw_q1_wide"]
    q4_rows = [row for row in rows if row["panel"] == "q4_zoom_center"]
    series = [
        ("query packet", "query_mean_db", BLUE, "solid"),
        (f"loc {TRUE_LOC} template", "true_template_mean_db", GREEN, "solid"),
        (f"loc {AMBIG_LOC} template", "ambiguous_template_mean_db", ORANGE, "dash"),
    ]

    for label, key, color, style in series:
        points = [(x_at(left, row["offset_bin"], -8.0, 8.0), y_at(left, row[key])) for row in raw_rows]
        if style == "dash":
            dashed_line(draw, points, color, max(2, int(1.3 * scale)), int(9 * scale), int(6 * scale))
        else:
            draw.line(points, fill=color, width=max(2, int(1.3 * scale)))
        for x, y in points:
            radius = 3.3 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=max(1, int(0.75 * scale)))

    for label, key, color, style in series:
        points = [(x_at(right, row["offset_bin"], -2.0, 2.0), y_at(right, row[key])) for row in q4_rows]
        if style == "dash":
            dashed_line(draw, points, color, max(2, int(1.35 * scale)), int(10 * scale), int(7 * scale))
        else:
            draw.line(points, fill=color, width=max(2, int(1.35 * scale)))
        for x, y in points:
            radius = 2.25 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    # Overlay raw integer-bin samples inside the zoomed interval as open markers.
    center_raw = [row for row in raw_rows if -2.0 <= row["offset_bin"] <= 2.0]
    for _label, key, color, _style in series:
        for row in center_raw:
            x = x_at(right, row["offset_bin"], -2.0, 2.0)
            y = y_at(right, row[key])
            radius = 5.2 * scale
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=max(1, int(1.0 * scale)))

    left_zoom_center = (x_at(left, 2.0, -8.0, 8.0), left[1] + 70 * scale)
    right_zoom_target = (right[0] - 45 * scale, right[1] + 70 * scale)
    draw_arrow(draw, left_zoom_center, right_zoom_target, ZOOM_EDGE, max(2, int(0.9 * scale)))
    draw.text((left[2] - 350, left[1] + 50 * scale), "zoomed q=4 detail", font=small_font, fill=ZOOM_EDGE)

    lx, ly = right[0] + 10, right[1] - 18 * scale
    for idx, (label, _key, color, style) in enumerate(series):
        x = lx + idx * 330
        y = ly
        if style == "dash":
            dashed_line(draw, [(x, y), (x + 38 * scale, y)], color, max(2, int(1.3 * scale)), int(9 * scale), int(6 * scale))
        else:
            draw.line((x, y, x + 38 * scale, y), fill=color, width=max(2, int(1.3 * scale)))
        draw.text((x + 47 * scale, y - 6 * scale), label, font=small_font, fill=BLACK)

    q4_text = f"q=4 RMSE: loc {TRUE_LOC} {metrics['q4_rmse_query_to_true_db']:.2f} dB, loc {AMBIG_LOC} {metrics['q4_rmse_query_to_ambiguous_db']:.2f} dB"
    draw.text((right[0] + 125, right[3] + 22 * scale), q4_text, font=small_font, fill=GRAY)
    draw.text((right[0] + 340, right[3] + 47 * scale), "q=4 provides finer samples only; no extra physical resolution.", font=small_font, fill=GRAY)

    for box in [left, right]:
        draw_centered(draw, (box[0] + box[2]) / 2, height - 34 * scale, "Offset from aligned center bin", label_font, BLACK)

    ylabel = "Magnitude relative to local peak (dB)"
    scratch = Image.new("RGBA", (420, 80), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), ylabel, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), ylabel, font=label_font, fill=BLACK)
    rotated = text_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (int(24 * scale), int((left[1] + left[3] - rotated.height) / 2)))

    out = HERE / f"{OUTPUT_PREFIX}.png"
    image.convert("RGB").save(out, dpi=(DPI, DPI))
    return out


def render_pdf(png_path: Path) -> Path:
    out = HERE / f"{OUTPUT_PREFIX}.pdf"
    pdf = canvas.Canvas(str(out), pagesize=(WIDTH_IN * 72, HEIGHT_IN * 72))
    pdf.drawImage(ImageReader(str(png_path)), 0, 0, width=WIDTH_IN * 72, height=HEIGHT_IN * 72)
    pdf.showPage()
    pdf.save()
    return out


def main() -> None:
    global QUERY_FILE, QUERY_PACKET, TRUE_LOC, AMBIG_LOC, OUTPUT_PREFIX

    parser = argparse.ArgumentParser(
        description="Draw wide raw-bin and q=4 zoomed spectrum for an RSSI+ ambiguous packet."
    )
    parser.add_argument("--query-bin-file", default=QUERY_FILE)
    parser.add_argument("--packet", type=int, default=QUERY_PACKET)
    parser.add_argument("--true-loc", type=int, default=TRUE_LOC)
    parser.add_argument("--ambig-loc", type=int, default=AMBIG_LOC)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    args = parser.parse_args()

    QUERY_FILE = args.query_bin_file
    QUERY_PACKET = args.packet
    TRUE_LOC = args.true_loc
    AMBIG_LOC = args.ambig_loc
    OUTPUT_PREFIX = args.output_prefix

    rows, metrics = load_series()
    write_data(rows, metrics)
    png = render_png(rows, metrics)
    render_pdf(png)


if __name__ == "__main__":
    main()
