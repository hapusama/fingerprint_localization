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
SPECTRUM_CSV = ROOT / "v2_output/20260624_zero_padding_fft_q1_q4_point_compare/subbin_spectrum_long.csv"

QUERY_FILE = "2_1_23_11_2_16.bin"
QUERY_PACKET = 16
TRUE_LOC = 23
AMBIG_LOC = 43

WIDTH_IN = 7.2
HEIGHT_IN = 3.45
DPI = 400

BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_series() -> tuple[list[dict], dict[str, float]]:
    buckets: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    with SPECTRUM_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            q = int(row["q"])
            loc = int(row["position_id"])
            if q not in (1, 4) or loc not in (TRUE_LOC, AMBIG_LOC):
                continue
            offset = float(row["subbin_offset"])
            value = float(row["mag_db_rel_peak"])
            file_name = row["file_name"]
            packet = int(row["packet_index"])
            if file_name == QUERY_FILE and packet == QUERY_PACKET:
                buckets[("query", q, offset)].append(value)
            if loc == TRUE_LOC and not (file_name == QUERY_FILE and packet == QUERY_PACKET):
                buckets[("loc23_template", q, offset)].append(value)
            if loc == AMBIG_LOC:
                buckets[("loc43_template", q, offset)].append(value)

    rows: list[dict] = []
    for q in (1, 4):
        offsets = sorted(offset for kind, qq, offset in buckets if qq == q and kind == "query")
        for offset in offsets:
            out = {"q": q, "subbin_offset": offset}
            for kind in ("query", "loc23_template", "loc43_template"):
                values = buckets[(kind, q, offset)]
                out[f"{kind}_mean_db"] = mean(values)
                out[f"{kind}_std_db"] = statistics.pstdev(values) if len(values) > 1 else 0.0
                out[f"{kind}_count"] = len(values)
            rows.append(out)

    q4 = [row for row in rows if row["q"] == 4]
    rmse23 = math.sqrt(sum((row["query_mean_db"] - row["loc23_template_mean_db"]) ** 2 for row in q4) / len(q4))
    rmse43 = math.sqrt(sum((row["query_mean_db"] - row["loc43_template_mean_db"]) ** 2 for row in q4) / len(q4))
    metrics = {
        "q4_rmse_query_to_loc23_db": rmse23,
        "q4_rmse_query_to_loc43_db": rmse43,
    }
    return rows, metrics


def write_data(rows: list[dict], metrics: dict[str, float]) -> Path:
    out = HERE / "raw_bin_q4_detail_23_43_data.csv"
    fields = [
        "query_file",
        "query_packet",
        "true_location",
        "ambiguous_location",
        "q4_rmse_query_to_loc23_db",
        "q4_rmse_query_to_loc43_db",
        "q",
        "subbin_offset",
        "query_mean_db",
        "query_std_db",
        "query_count",
        "loc23_template_mean_db",
        "loc23_template_std_db",
        "loc23_template_count",
        "loc43_template_mean_db",
        "loc43_template_std_db",
        "loc43_template_count",
        "note",
    ]
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
                    "q4_rmse_query_to_loc23_db": metrics["q4_rmse_query_to_loc23_db"],
                    "q4_rmse_query_to_loc43_db": metrics["q4_rmse_query_to_loc43_db"],
                    **row,
                    "note": "q=4 zero padding gives finer sampling of the same center-bin neighborhood; it does not add physical resolution.",
                }
            )
    return out


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.ImageFont, fill: str) -> None:
    w, h = text_size(draw, text, font)
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill: str, width: int, dash: int, gap: int) -> None:
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        distance = 0.0
        while distance < length:
            end = min(distance + dash, length)
            draw.line((x0 + ux * distance, y0 + uy * distance, x0 + ux * end, y0 + uy * end), fill=fill, width=width)
            distance += dash + gap


def render_png(rows: list[dict], metrics: dict[str, float]) -> Path:
    width = int(WIDTH_IN * DPI)
    height = int(HEIGHT_IN * DPI)
    scale = DPI / 100
    image = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(10.5 * scale), bold=True)
    panel_font = load_font(int(8.5 * scale), bold=True)
    label_font = load_font(int(7.6 * scale))
    small_font = load_font(int(6.4 * scale))

    draw.text((52 * scale, 24 * scale), "Raw integer bins and q=4 sub-bin detail for ambiguous candidates", font=title_font, fill=BLACK)
    draw.text((52 * scale, 43 * scale), f"query: loc {TRUE_LOC}, packet {QUERY_PACKET}; loc {TRUE_LOC} template excludes the query packet", font=small_font, fill=GRAY)

    left = (220, 330, 1070, 1080)
    right = (1320, 330, 2680, 1080)
    y_min, y_max = -32.0, 2.0

    def x_at(box: tuple[int, int, int, int], value: float) -> float:
        x0, _y0, x1, _y1 = box
        return x0 + (value + 2.0) / 4.0 * (x1 - x0)

    def y_at(box: tuple[int, int, int, int], value: float) -> float:
        _x0, y0, _x1, y1 = box
        return y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)

    def axes(box: tuple[int, int, int, int], title: str, dense: bool) -> None:
        x0, y0, x1, y1 = box
        draw.text((x0, y0 - 30 * scale), title, font=panel_font, fill=BLACK)
        for tick in [-30, -20, -10, 0]:
            y = y_at(box, tick)
            draw.line((x0, y, x1, y), fill=LIGHT_GRAY, width=max(1, int(scale)))
            tw, th = text_size(draw, str(tick), small_font)
            draw.text((x0 - tw - 8 * scale, y - th / 2), str(tick), font=small_font, fill=GRAY)
        for tick in [-2, -1, 0, 1, 2]:
            x = x_at(box, tick)
            draw.line((x, y0, x, y1), fill="#eeeeee", width=max(1, int(0.7 * scale)))
            draw_centered(draw, x, y1 + 18 * scale, str(tick), small_font, BLACK)
        draw.line((x0, y1, x1, y1), fill=BLACK, width=max(1, int(1.2 * scale)))
        draw.line((x0, y0, x0, y1), fill=BLACK, width=max(1, int(1.2 * scale)))
        if dense:
            draw.text((x0 + 390 * scale, y0 + 16 * scale), "integer-bin locations", font=small_font, fill=GRAY)

    axes(left, "(a) Raw integer-bin samples (q=1)", dense=False)
    axes(right, "(b) q=4 zero-padded sub-bin detail", dense=True)

    series = [
        ("query packet", "query_mean_db", BLUE, "solid"),
        (f"loc {TRUE_LOC} template", "loc23_template_mean_db", GREEN, "solid"),
        (f"loc {AMBIG_LOC} template", "loc43_template_mean_db", ORANGE, "dash"),
    ]
    q1 = [row for row in rows if row["q"] == 1]
    q4 = [row for row in rows if row["q"] == 4]

    # q=1 sparse samples: slight x offsets to prevent markers from hiding each other.
    marker_offsets = [-0.035, 0.0, 0.035]
    for idx, (_label, key, color, style) in enumerate(series):
        points = [(x_at(left, row["subbin_offset"] + marker_offsets[idx]), y_at(left, row[key])) for row in q1]
        if style == "dash":
            dashed_line(draw, points, color, max(2, int(1.3 * scale)), int(9 * scale), int(6 * scale))
        else:
            draw.line(points, fill=color, width=max(2, int(1.3 * scale)))
        for x, y in points:
            draw.ellipse((x - 4.5 * scale, y - 4.5 * scale, x + 4.5 * scale, y + 4.5 * scale), fill=color, outline=WHITE, width=max(1, int(scale)))

    # q=4 dense curves.
    for _label, key, color, style in series:
        points = [(x_at(right, row["subbin_offset"]), y_at(right, row[key])) for row in q4]
        if style == "dash":
            dashed_line(draw, points, color, max(2, int(1.4 * scale)), int(10 * scale), int(7 * scale))
        else:
            draw.line(points, fill=color, width=max(2, int(1.4 * scale)))
        for x, y in points:
            draw.ellipse((x - 2.6 * scale, y - 2.6 * scale, x + 2.6 * scale, y + 2.6 * scale), fill=color)

    # Overlay raw integer samples on q=4 panel as open markers to make the zoom relation explicit.
    for _label, key, color, _style in series:
        for row in q1:
            x = x_at(right, row["subbin_offset"])
            y = y_at(right, row[key])
            draw.ellipse((x - 6.0 * scale, y - 6.0 * scale, x + 6.0 * scale, y + 6.0 * scale), outline=color, width=max(1, int(1.2 * scale)))

    # Legend.
    lx, ly = right[0] + 40 * scale, right[1] + 16 * scale
    for idx, (label, _key, color, style) in enumerate(series):
        y = ly + idx * 18 * scale
        if style == "dash":
            dashed_line(draw, [(lx, y), (lx + 38 * scale, y)], color, max(2, int(1.4 * scale)), int(9 * scale), int(6 * scale))
        else:
            draw.line((lx, y, lx + 38 * scale, y), fill=color, width=max(2, int(1.4 * scale)))
        draw.text((lx + 46 * scale, y - 6 * scale), label, font=small_font, fill=BLACK)

    draw.text((right[2] - 470, right[1] + 16 * scale), f"q=4 RMSE to loc {TRUE_LOC}: {metrics['q4_rmse_query_to_loc23_db']:.2f} dB", font=small_font, fill=GRAY)
    draw.text((right[2] - 470, right[1] + 33 * scale), f"q=4 RMSE to loc {AMBIG_LOC}: {metrics['q4_rmse_query_to_loc43_db']:.2f} dB", font=small_font, fill=GRAY)
    draw.text((right[0] + 100, right[3] + 50 * scale), "q=4 adds intermediate samples only; no extra physical resolution.", font=small_font, fill=GRAY)

    for box in [left, right]:
        draw_centered(draw, (box[0] + box[2]) / 2, height - 35 * scale, "Offset from aligned center bin", label_font, BLACK)

    ylabel = "Magnitude relative to local peak (dB)"
    scratch = Image.new("RGBA", (360, 60), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), ylabel, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), ylabel, font=label_font, fill=BLACK)
    rotated = text_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (int(24 * scale), int((left[1] + left[3] - rotated.height) / 2)))

    out = HERE / "raw_bin_q4_detail_23_43.png"
    image.convert("RGB").save(out, dpi=(DPI, DPI))
    return out


def pdf_center(pdf: canvas.Canvas, x: float, y: float, text: str, font: str, size: float, fill: str) -> None:
    pdf.setFont(font, size)
    pdf.setFillColor(hex_to_color(fill))
    pdf.drawCentredString(x, y, text)


def render_pdf(rows: list[dict], metrics: dict[str, float]) -> Path:
    font, bold = register_pdf_font()
    width = WIDTH_IN * 72
    height = HEIGHT_IN * 72
    out = HERE / "raw_bin_q4_detail_23_43.pdf"
    pdf = canvas.Canvas(str(out), pagesize=(width, height))

    pdf.setFont(bold, 10.5)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(38, height - 20, "Raw integer bins and q=4 sub-bin detail for ambiguous candidates")
    pdf.setFont(font, 6.4)
    pdf.setFillColor(hex_to_color(GRAY))
    pdf.drawString(38, height - 34, f"query: loc {TRUE_LOC}, packet {QUERY_PACKET}; loc {TRUE_LOC} template excludes the query packet")

    left = (40, 50, 198, 185)
    right = (250, 50, 500, 185)
    y_min, y_max = -32.0, 2.0
    series = [
        ("query packet", "query_mean_db", BLUE, False),
        (f"loc {TRUE_LOC} template", "loc23_template_mean_db", GREEN, False),
        (f"loc {AMBIG_LOC} template", "loc43_template_mean_db", ORANGE, True),
    ]
    q1 = [row for row in rows if row["q"] == 1]
    q4 = [row for row in rows if row["q"] == 4]

    def x_at(box: tuple[float, float, float, float], value: float) -> float:
        return box[0] + (value + 2.0) / 4.0 * (box[2] - box[0])

    def y_at(box: tuple[float, float, float, float], value: float) -> float:
        return box[1] + (value - y_min) / (y_max - y_min) * (box[3] - box[1])

    def axes(box: tuple[float, float, float, float], title: str) -> None:
        x0, y0, x1, y1 = box
        pdf.setFont(bold, 8.4)
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawString(x0, y1 + 17, title)
        for tick in [-30, -20, -10, 0]:
            y = y_at(box, tick)
            pdf.setStrokeColor(hex_to_color(LIGHT_GRAY))
            pdf.setLineWidth(0.45)
            pdf.line(x0, y, x1, y)
            pdf.setFont(font, 6.3)
            pdf.setFillColor(hex_to_color(GRAY))
            pdf.drawRightString(x0 - 4, y - 2.1, str(tick))
        for tick in [-2, -1, 0, 1, 2]:
            x = x_at(box, tick)
            pdf.setStrokeColor(hex_to_color("#eeeeee"))
            pdf.line(x, y0, x, y1)
            pdf.setFillColor(hex_to_color(BLACK))
            pdf.drawCentredString(x, y0 - 10, str(tick))
        pdf.setStrokeColor(hex_to_color(BLACK))
        pdf.setLineWidth(0.8)
        pdf.line(x0, y0, x1, y0)
        pdf.line(x0, y0, x0, y1)

    axes(left, "(a) Raw integer-bin samples (q=1)")
    axes(right, "(b) q=4 zero-padded sub-bin detail")

    def draw_path(points: list[tuple[float, float]], color: str, dashed: bool) -> None:
        pdf.setStrokeColor(hex_to_color(color))
        pdf.setLineWidth(1.0)
        pdf.setDash(4, 3) if dashed else pdf.setDash()
        path = pdf.beginPath()
        path.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, y)
        pdf.drawPath(path, stroke=1, fill=0)
        pdf.setDash()

    marker_offsets = [-0.035, 0.0, 0.035]
    for idx, (_label, key, color, dashed) in enumerate(series):
        points = [(x_at(left, row["subbin_offset"] + marker_offsets[idx]), y_at(left, row[key])) for row in q1]
        draw_path(points, color, dashed)
        pdf.setFillColor(hex_to_color(color))
        for x, y in points:
            pdf.circle(x, y, 2.0, stroke=0, fill=1)

    for _label, key, color, dashed in series:
        points = [(x_at(right, row["subbin_offset"]), y_at(right, row[key])) for row in q4]
        draw_path(points, color, dashed)
        pdf.setFillColor(hex_to_color(color))
        for x, y in points:
            pdf.circle(x, y, 1.0, stroke=0, fill=1)
        # Raw q=1 samples as open circles in q=4 panel.
        pdf.setStrokeColor(hex_to_color(color))
        for row in q1:
            pdf.circle(x_at(right, row["subbin_offset"]), y_at(right, row[key]), 2.5, stroke=1, fill=0)

    lx, ly = right[0] + 8, right[3] - 10
    for idx, (label, _key, color, dashed) in enumerate(series):
        y = ly - idx * 11
        pdf.setStrokeColor(hex_to_color(color))
        pdf.setLineWidth(1.0)
        pdf.setDash(4, 3) if dashed else pdf.setDash()
        pdf.line(lx, y, lx + 17, y)
        pdf.setDash()
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.setFont(font, 6.3)
        pdf.drawString(lx + 22, y - 2.3, label)

    pdf.setFillColor(hex_to_color(GRAY))
    pdf.setFont(font, 6.3)
    pdf.drawString(right[2] - 108, right[3] - 10, f"q=4 RMSE to loc {TRUE_LOC}: {metrics['q4_rmse_query_to_loc23_db']:.2f} dB")
    pdf.drawString(right[2] - 108, right[3] - 21, f"q=4 RMSE to loc {AMBIG_LOC}: {metrics['q4_rmse_query_to_loc43_db']:.2f} dB")
    pdf.drawString(right[0] + 90, 30, "q=4 adds intermediate samples only; no extra physical resolution.")

    pdf_center(pdf, (left[0] + left[2]) / 2, 15, "Offset from aligned center bin", font, 7.3, BLACK)
    pdf_center(pdf, (right[0] + right[2]) / 2, 15, "Offset from aligned center bin", font, 7.3, BLACK)
    pdf.saveState()
    pdf.translate(13, (left[1] + left[3]) / 2)
    pdf.rotate(90)
    pdf_center(pdf, 0, 0, "Magnitude relative to local peak (dB)", font, 7.3, BLACK)
    pdf.restoreState()

    pdf.showPage()
    pdf.save()
    return out


def main() -> None:
    rows, metrics = load_series()
    write_data(rows, metrics)
    render_png(rows, metrics)
    render_pdf(rows, metrics)


if __name__ == "__main__":
    main()
