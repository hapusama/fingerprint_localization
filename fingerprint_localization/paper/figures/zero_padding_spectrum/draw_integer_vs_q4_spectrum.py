from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "v2_output/20260624_zero_padding_fft_q1_q4_point_compare/subbin_spectrum_long.csv"

QUERY_FILE = "2_1_23_11_2_16.bin"
QUERY_PACKET_INDEX = 16
POSITION_ID = 23

WIDTH_IN = 7.2
HEIGHT_IN = 3.4
DPI = 400

BLUE = "#1f77b4"
BLUE_LIGHT = "#d8ecf8"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
GREEN_LIGHT = "#dff1df"
GRAY = "#666666"
LIGHT_GRAY = "#d9d9d9"
BLACK = "#111111"
WHITE = "#ffffff"


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


def register_pdf_font() -> tuple[str, str]:
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if Path(regular).exists():
        pdfmetrics.registerFont(TTFont("PlotArial", regular))
    if Path(bold).exists():
        pdfmetrics.registerFont(TTFont("PlotArial-Bold", bold))
    return (
        "PlotArial" if Path(regular).exists() else "Helvetica",
        "PlotArial-Bold" if Path(bold).exists() else "Helvetica-Bold",
    )


def hex_to_color(value: str) -> colors.Color:
    value = value.lstrip("#")
    return colors.Color(
        int(value[0:2], 16) / 255,
        int(value[2:4], 16) / 255,
        int(value[4:6], 16) / 255,
    )


def load_summary() -> list[dict]:
    values: dict[tuple[int, float], list[float]] = defaultdict(list)
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["file_name"] != QUERY_FILE:
                continue
            if int(row["packet_index"]) != QUERY_PACKET_INDEX:
                continue
            q = int(row["q"])
            offset = float(row["subbin_offset"])
            values[(q, offset)].append(float(row["mag_db_rel_peak"]))

    rows: list[dict] = []
    for (q, offset), series in sorted(values.items()):
        rows.append(
            {
                "file_name": QUERY_FILE,
                "position_id": POSITION_ID,
                "packet_index": QUERY_PACKET_INDEX,
                "q": q,
                "subbin_offset": offset,
                "sample_count": len(series),
                "mag_db_mean": sum(series) / len(series),
                "mag_db_std": statistics.pstdev(series) if len(series) > 1 else 0.0,
                "zero_padding_note": "q=4 is finer sampling of the same underlying dechirped spectrum; it does not add physical resolution.",
            }
        )
    if not rows:
        raise RuntimeError(f"No rows found for {QUERY_FILE}, packet {QUERY_PACKET_INDEX}")
    return rows


def write_data(rows: list[dict]) -> Path:
    out = HERE / "integer_vs_q4_spectrum_data.csv"
    fields = [
        "file_name",
        "position_id",
        "packet_index",
        "q",
        "subbin_offset",
        "sample_count",
        "mag_db_mean",
        "mag_db_std",
        "zero_padding_note",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.ImageFont, fill: str) -> None:
    w, h = text_size(draw, text, font)
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def arrow_head(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill: str, scale: float) -> None:
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    length = 7 * scale
    spread = 0.42
    points = [
        (ex, ey),
        (ex - length * math.cos(angle - spread), ey - length * math.sin(angle - spread)),
        (ex - length * math.cos(angle + spread), ey - length * math.sin(angle + spread)),
    ]
    draw.polygon(points, fill=fill)


def render_png(rows: list[dict]) -> Path:
    width = int(WIDTH_IN * DPI)
    height = int(HEIGHT_IN * DPI)
    scale = DPI / 100
    image = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(10.6 * scale), bold=True)
    panel_font = load_font(int(8.8 * scale), bold=True)
    label_font = load_font(int(7.8 * scale))
    small_font = load_font(int(6.7 * scale))

    draw.text((54 * scale, 17 * scale), "Integer-bin FFT samples vs. q=4 zero-padded spectrum", font=title_font, fill=BLACK)
    draw.text((54 * scale, 34 * scale), f"same packet: loc {POSITION_ID}, packet {QUERY_PACKET_INDEX}", font=small_font, fill=GRAY)

    left_box = (58 * scale, 66 * scale, 1360 * scale / 4, height - 132 * scale)
    # Use explicit pixel boxes for stable two-panel layout.
    left = (210, 250, 1280, 1080)
    right = (1540, 250, 2640, 1080)
    y_min, y_max = -32.0, 2.0

    q1 = [row for row in rows if row["q"] == 1]
    q4 = [row for row in rows if row["q"] == 4]

    def draw_axes(box: tuple[int, int, int, int], title: str) -> tuple:
        x0, y0, x1, y1 = box
        draw.text((x0, y0 - 58 * scale), title, font=panel_font, fill=BLACK)
        for tick in [-30, -20, -10, 0]:
            y = y1 - (tick - y_min) / (y_max - y_min) * (y1 - y0)
            draw.line((x0, y, x1, y), fill=LIGHT_GRAY, width=max(1, int(scale)))
            label = str(tick)
            tw, th = text_size(draw, label, small_font)
            draw.text((x0 - tw - 9 * scale, y - th / 2), label, font=small_font, fill=GRAY)
        draw.line((x0, y1, x1, y1), fill=BLACK, width=max(1, int(1.2 * scale)))
        draw.line((x0, y0, x0, y1), fill=BLACK, width=max(1, int(1.2 * scale)))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=0, outline="#eeeeee", width=0)
        return x0, y0, x1, y1

    def x_at(box: tuple[int, int, int, int], value: float) -> float:
        x0, _y0, x1, _y1 = box
        return x0 + (value + 2.0) / 4.0 * (x1 - x0)

    def y_at(box: tuple[int, int, int, int], value: float) -> float:
        _x0, y0, _x1, y1 = box
        return y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)

    for box in [left, right]:
        x0, y0, x1, y1 = box
        # Center-bin neighborhood bracket.
        by = y1 + 36 * scale
        draw.line((x0, by, x1, by), fill=GRAY, width=max(1, int(scale)))
        draw.line((x0, by - 6 * scale, x0, by + 6 * scale), fill=GRAY, width=max(1, int(scale)))
        draw.line((x1, by - 6 * scale, x1, by + 6 * scale), fill=GRAY, width=max(1, int(scale)))
        draw_centered(draw, (x0 + x1) / 2, by + 18 * scale, "center-bin neighborhood (M=2)", small_font, GRAY)

    draw_axes(left, "(a) Original integer-bin samples")
    draw_axes(right, "(b) Zero-padded q=4 sub-bin samples")

    # Left: stem plot for q=1.
    for row in q1:
        x = x_at(left, row["subbin_offset"])
        y = y_at(left, row["mag_db_mean"])
        y_zero = y_at(left, y_min)
        draw.line((x, y_zero, x, y), fill=BLUE, width=max(2, int(1.5 * scale)))
        draw.ellipse((x - 5 * scale, y - 5 * scale, x + 5 * scale, y + 5 * scale), fill=BLUE, outline=WHITE, width=max(1, int(scale)))
        draw_centered(draw, x, left[3] + 18 * scale, f"{row['subbin_offset']:.0f}", small_font, BLACK)

    # Right: dense q=4 curve.
    points = [(x_at(right, row["subbin_offset"]), y_at(right, row["mag_db_mean"])) for row in q4]
    for index in range(len(points) - 1):
        draw.line((points[index][0], points[index][1], points[index + 1][0], points[index + 1][1]), fill=ORANGE, width=max(2, int(1.5 * scale)))
    for row, (x, y) in zip(q4, points):
        draw.ellipse((x - 3.2 * scale, y - 3.2 * scale, x + 3.2 * scale, y + 3.2 * scale), fill=ORANGE)
    for value in [-2, -1, 0, 1, 2]:
        x = x_at(right, value)
        draw.line((x, right[1], x, right[3]), fill="#eeeeee", width=max(1, int(0.75 * scale)))
        draw_centered(draw, x, right[3] + 18 * scale, str(value), small_font, BLACK)

    # Highlight the q=4 peak and integer center.
    peak = max(q4, key=lambda row: row["mag_db_mean"])
    px = x_at(right, peak["subbin_offset"])
    py = y_at(right, peak["mag_db_mean"])
    draw.line((px, py - 38 * scale, px, py - 8 * scale), fill=GREEN, width=max(1, int(scale)))
    arrow_head(draw, (px, py - 38 * scale), (px, py - 8 * scale), GREEN, scale)
    draw.text((px + 8 * scale, py - 42 * scale), f"peak near {peak['subbin_offset']:+.2f}", font=small_font, fill=GREEN)

    cx = x_at(right, 0.0)
    draw.line((cx, right[1], cx, right[3]), fill=BLUE, width=max(1, int(scale)))
    draw.text((cx + 8 * scale, right[1] + 9 * scale), "integer center", font=small_font, fill=BLUE)

    # Axis labels.
    for box in [left, right]:
        x0, y0, x1, y1 = box
        draw_centered(draw, (x0 + x1) / 2, height - 34 * scale, "Offset from aligned center bin", label_font, BLACK)
    ylabel = "Magnitude relative to local peak (dB)"
    scratch = Image.new("RGBA", (360, 60), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), ylabel, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), ylabel, font=label_font, fill=BLACK)
    rotated = text_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (int(22 * scale), int((left[1] + left[3] - rotated.height) / 2)))

    note = "Zero padding provides finer sampling of the same spectrum; it does not add information or physical resolution."
    draw.text((1540, 1200), note, font=small_font, fill=GRAY)

    out = HERE / "integer_vs_q4_spectrum.png"
    image.convert("RGB").save(out, dpi=(DPI, DPI))
    return out


def pdf_center(pdf: canvas.Canvas, x: float, y: float, text: str, font: str, size: float, fill: str) -> None:
    pdf.setFont(font, size)
    pdf.setFillColor(hex_to_color(fill))
    pdf.drawCentredString(x, y, text)


def render_pdf(rows: list[dict]) -> Path:
    font, bold = register_pdf_font()
    width = WIDTH_IN * 72
    height = HEIGHT_IN * 72
    out = HERE / "integer_vs_q4_spectrum.pdf"
    pdf = canvas.Canvas(str(out), pagesize=(width, height))

    q1 = [row for row in rows if row["q"] == 1]
    q4 = [row for row in rows if row["q"] == 4]
    y_min, y_max = -32.0, 2.0
    left = (38, 55, 240, 190)
    right = (292, 55, 500, 190)

    def x_at(box: tuple[float, float, float, float], value: float) -> float:
        x0, _y0, x1, _y1 = box
        return x0 + (value + 2.0) / 4.0 * (x1 - x0)

    def y_at(box: tuple[float, float, float, float], value: float) -> float:
        _x0, y0, _x1, y1 = box
        return y0 + (value - y_min) / (y_max - y_min) * (y1 - y0)

    pdf.setFont(bold, 10.6)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(38, height - 20, "Integer-bin FFT samples vs. q=4 zero-padded spectrum")
    pdf.setFont(font, 6.7)
    pdf.setFillColor(hex_to_color(GRAY))
    pdf.drawString(38, height - 34, f"same packet: loc {POSITION_ID}, packet {QUERY_PACKET_INDEX}")

    def axes(box: tuple[float, float, float, float], title: str) -> None:
        x0, y0, x1, y1 = box
        pdf.setFont(bold, 8.6)
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawString(x0, y1 + 17, title)
        for tick in [-30, -20, -10, 0]:
            y = y_at(box, tick)
            pdf.setStrokeColor(hex_to_color(LIGHT_GRAY))
            pdf.setLineWidth(0.45)
            pdf.line(x0, y, x1, y)
            pdf.setFont(font, 6.5)
            pdf.setFillColor(hex_to_color(GRAY))
            pdf.drawRightString(x0 - 4, y - 2, str(tick))
        pdf.setStrokeColor(hex_to_color(BLACK))
        pdf.setLineWidth(0.8)
        pdf.line(x0, y0, x1, y0)
        pdf.line(x0, y0, x0, y1)
        by = y0 - 16
        pdf.setStrokeColor(hex_to_color(GRAY))
        pdf.setLineWidth(0.5)
        pdf.line(x0, by, x1, by)
        pdf.line(x0, by - 3, x0, by + 3)
        pdf.line(x1, by - 3, x1, by + 3)
        pdf_center(pdf, (x0 + x1) / 2, by - 11, "center-bin neighborhood (M=2)", font, 6.2, GRAY)

    axes(left, "(a) Original integer-bin samples")
    axes(right, "(b) Zero-padded q=4 sub-bin samples")

    pdf.setStrokeColor(hex_to_color(BLUE))
    pdf.setFillColor(hex_to_color(BLUE))
    for row in q1:
        x = x_at(left, row["subbin_offset"])
        y = y_at(left, row["mag_db_mean"])
        pdf.setLineWidth(1.0)
        pdf.line(x, left[1], x, y)
        pdf.circle(x, y, 2.4, stroke=0, fill=1)
        pdf_center(pdf, x, left[1] - 10, f"{row['subbin_offset']:.0f}", font, 6.5, BLACK)

    pdf.setStrokeColor(hex_to_color(ORANGE))
    pdf.setFillColor(hex_to_color(ORANGE))
    points = [(x_at(right, row["subbin_offset"]), y_at(right, row["mag_db_mean"])) for row in q4]
    path = pdf.beginPath()
    path.moveTo(points[0][0], points[0][1])
    for x, y in points[1:]:
        path.lineTo(x, y)
    pdf.setLineWidth(1.0)
    pdf.drawPath(path, stroke=1, fill=0)
    for row, (x, y) in zip(q4, points):
        pdf.circle(x, y, 1.45, stroke=0, fill=1)
    for value in [-2, -1, 0, 1, 2]:
        x = x_at(right, value)
        pdf.setStrokeColor(hex_to_color("#eeeeee"))
        pdf.setLineWidth(0.35)
        pdf.line(x, right[1], x, right[3])
        pdf_center(pdf, x, right[1] - 10, str(value), font, 6.5, BLACK)

    peak = max(q4, key=lambda row: row["mag_db_mean"])
    px = x_at(right, peak["subbin_offset"])
    py = y_at(right, peak["mag_db_mean"])
    pdf.setStrokeColor(hex_to_color(GREEN))
    pdf.setLineWidth(0.8)
    pdf.line(px, py + 19, px, py + 5)
    pdf.setFillColor(hex_to_color(GREEN))
    pdf.setFont(font, 6.4)
    pdf.drawString(px + 5, py + 18, f"peak near {peak['subbin_offset']:+.2f}")
    cx = x_at(right, 0)
    pdf.setStrokeColor(hex_to_color(BLUE))
    pdf.line(cx, right[1], cx, right[3])
    pdf.setFillColor(hex_to_color(BLUE))
    pdf.drawString(cx + 4, right[3] - 8, "integer center")

    pdf_center(pdf, (left[0] + left[2]) / 2, 12, "Offset from aligned center bin", font, 7.6, BLACK)
    pdf_center(pdf, (right[0] + right[2]) / 2, 12, "Offset from aligned center bin", font, 7.6, BLACK)
    pdf.saveState()
    pdf.translate(12, (left[1] + left[3]) / 2)
    pdf.rotate(90)
    pdf_center(pdf, 0, 0, "Magnitude relative to local peak (dB)", font, 7.6, BLACK)
    pdf.restoreState()
    pdf.setFont(font, 6.5)
    pdf.setFillColor(hex_to_color(GRAY))
    pdf.drawString(292, 32, "Zero padding provides finer sampling of the same spectrum; it does not add information or physical resolution.")

    pdf.showPage()
    pdf.save()
    return out


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    rows = load_summary()
    write_data(rows)
    render_png(rows)
    render_pdf(rows)


if __name__ == "__main__":
    main()
