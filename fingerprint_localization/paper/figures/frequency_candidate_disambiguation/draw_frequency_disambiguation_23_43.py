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
POSTERIOR_CSV = ROOT / "paper_figures/rssiplus_posterior/rssiplus_posterior_28points_data.csv"

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


def mean_std(values: list[float]) -> tuple[float, float]:
    return (
        sum(values) / len(values),
        statistics.pstdev(values) if len(values) > 1 else 0.0,
    )


def load_posterior_pair() -> dict[int, float]:
    result: dict[int, float] = {}
    with POSTERIOR_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            loc = int(row["reference_location"])
            if loc in (TRUE_LOC, AMBIG_LOC):
                result[loc] = float(row["posterior_probability"])
    return result


def load_curves() -> tuple[list[dict], dict[str, float]]:
    query: dict[float, list[float]] = defaultdict(list)
    template23: dict[float, list[float]] = defaultdict(list)
    template43: dict[float, list[float]] = defaultdict(list)

    with SPECTRUM_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["q"]) != 4:
                continue
            loc = int(row["position_id"])
            if loc not in (TRUE_LOC, AMBIG_LOC):
                continue
            offset = float(row["subbin_offset"])
            value = float(row["mag_db_rel_peak"])
            file_name = row["file_name"]
            packet = int(row["packet_index"])

            if file_name == QUERY_FILE and packet == QUERY_PACKET:
                query[offset].append(value)
            if loc == TRUE_LOC and not (file_name == QUERY_FILE and packet == QUERY_PACKET):
                template23[offset].append(value)
            if loc == AMBIG_LOC:
                template43[offset].append(value)

    offsets = sorted(query)
    rows: list[dict] = []
    for offset in offsets:
        query_mean, query_std = mean_std(query[offset])
        loc23_mean, loc23_std = mean_std(template23[offset])
        loc43_mean, loc43_std = mean_std(template43[offset])
        rows.append(
            {
                "subbin_offset": offset,
                "query_packet_mean_db": query_mean,
                "query_packet_std_db": query_std,
                "loc23_template_mean_db_excluding_query": loc23_mean,
                "loc23_template_std_db_excluding_query": loc23_std,
                "loc43_template_mean_db": loc43_mean,
                "loc43_template_std_db": loc43_std,
            }
        )

    rmse23 = math.sqrt(
        sum((row["query_packet_mean_db"] - row["loc23_template_mean_db_excluding_query"]) ** 2 for row in rows)
        / len(rows)
    )
    rmse43 = math.sqrt(
        sum((row["query_packet_mean_db"] - row["loc43_template_mean_db"]) ** 2 for row in rows)
        / len(rows)
    )
    max23 = max(abs(row["query_packet_mean_db"] - row["loc23_template_mean_db_excluding_query"]) for row in rows)
    max43 = max(abs(row["query_packet_mean_db"] - row["loc43_template_mean_db"]) for row in rows)
    metrics = {
        "rmse_query_to_loc23_db": rmse23,
        "rmse_query_to_loc43_db": rmse43,
        "max_abs_query_to_loc23_db": max23,
        "max_abs_query_to_loc43_db": max43,
    }
    return rows, metrics


def write_data(rows: list[dict], metrics: dict[str, float], posterior: dict[int, float]) -> Path:
    out = HERE / "frequency_disambiguation_23_43_data.csv"
    fields = [
        "query_file",
        "query_packet",
        "true_location",
        "ambiguous_location",
        "posterior_loc23",
        "posterior_loc43",
        "rmse_query_to_loc23_db",
        "rmse_query_to_loc43_db",
        "subbin_offset",
        "query_packet_mean_db",
        "query_packet_std_db",
        "loc23_template_mean_db_excluding_query",
        "loc23_template_std_db_excluding_query",
        "loc43_template_mean_db",
        "loc43_template_std_db",
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
                    "posterior_loc23": posterior[TRUE_LOC],
                    "posterior_loc43": posterior[AMBIG_LOC],
                    "rmse_query_to_loc23_db": metrics["rmse_query_to_loc23_db"],
                    "rmse_query_to_loc43_db": metrics["rmse_query_to_loc43_db"],
                    **row,
                }
            )
    return out


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, font: ImageFont.ImageFont, fill: str) -> None:
    w, h = text_size(draw, text, font)
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def dashed_line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill: str, width: int, dash: int = 10, gap: int = 7) -> None:
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


def render_png(rows: list[dict], metrics: dict[str, float], posterior: dict[int, float]) -> Path:
    width = int(WIDTH_IN * DPI)
    height = int(HEIGHT_IN * DPI)
    scale = DPI / 100
    image = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(10.5 * scale), bold=True)
    panel_font = load_font(int(8.5 * scale), bold=True)
    label_font = load_font(int(7.6 * scale))
    small_font = load_font(int(6.5 * scale))

    draw.text((50 * scale, 24 * scale), "Frequency evidence for RSSI+ ambiguous candidates", font=title_font, fill=BLACK)
    draw.text(
        (50 * scale, 43 * scale),
        f"query: loc {TRUE_LOC}, packet {QUERY_PACKET}; loc {TRUE_LOC} template excludes the query packet",
        font=small_font,
        fill=GRAY,
    )

    # Panel A: posterior pair.
    bar_box = (260, 330, 820, 1080)
    spec_box = (1080, 330, 2660, 1080)

    x0, y0, x1, y1 = bar_box
    draw.text((x0, y0 - 30 * scale), "(a) RSSI+ posterior is nearly tied", font=panel_font, fill=BLACK)
    bar_y_max = 0.16
    for tick in [0.0, 0.05, 0.10, 0.15]:
        y = y1 - tick / bar_y_max * (y1 - y0)
        draw.line((x0, y, x1, y), fill=LIGHT_GRAY, width=max(1, int(scale)))
        label = f"{tick:.2f}".rstrip("0").rstrip(".")
        tw, th = text_size(draw, label, small_font)
        draw.text((x0 - tw - 8 * scale, y - th / 2), label, font=small_font, fill=GRAY)
    draw.line((x0, y1, x1, y1), fill=BLACK, width=max(1, int(1.2 * scale)))
    draw.line((x0, y0, x0, y1), fill=BLACK, width=max(1, int(1.2 * scale)))
    bar_data = [(TRUE_LOC, posterior[TRUE_LOC], GREEN), (AMBIG_LOC, posterior[AMBIG_LOC], ORANGE)]
    slot = (x1 - x0) / 3.2
    for index, (loc, prob, color) in enumerate(bar_data):
        bx = x0 + slot * (index + 0.9)
        by = y1 - prob / bar_y_max * (y1 - y0)
        draw.rectangle((bx - 48 * scale, by, bx + 48 * scale, y1), fill=color)
        draw_centered(draw, bx, by - 14 * scale, f"{prob:.3f}", small_font, GRAY)
        draw_centered(draw, bx, y1 + 18 * scale, str(loc), label_font, BLACK)
    draw_centered(draw, (x0 + x1) / 2, height - 35 * scale, "Candidate location", label_font, BLACK)
    posterior_label = "Posterior"
    tw, th = text_size(draw, posterior_label, label_font)
    draw.text((x0 - tw - 10 * scale, (y0 + y1) / 2 - th / 2), posterior_label, font=label_font, fill=BLACK)

    # Panel B: q=4 frequency curves.
    sx0, sy0, sx1, sy1 = spec_box
    draw.text((sx0, sy0 - 30 * scale), "(b) q=4 preamble spectrum gives structure cue", font=panel_font, fill=BLACK)
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -32.0, 2.0

    def x_at(value: float) -> float:
        return sx0 + (value - x_min) / (x_max - x_min) * (sx1 - sx0)

    def y_at(value: float) -> float:
        return sy1 - (value - y_min) / (y_max - y_min) * (sy1 - sy0)

    for tick in [-30, -20, -10, 0]:
        y = y_at(tick)
        draw.line((sx0, y, sx1, y), fill=LIGHT_GRAY, width=max(1, int(scale)))
        label = str(tick)
        tw, th = text_size(draw, label, small_font)
        draw.text((sx0 - tw - 8 * scale, y - th / 2), label, font=small_font, fill=GRAY)
    for tick in [-2, -1, 0, 1, 2]:
        x = x_at(tick)
        draw.line((x, sy0, x, sy1), fill="#eeeeee", width=max(1, int(0.7 * scale)))
        draw_centered(draw, x, sy1 + 18 * scale, str(tick), small_font, BLACK)
    draw.line((sx0, sy1, sx1, sy1), fill=BLACK, width=max(1, int(1.2 * scale)))
    draw.line((sx0, sy0, sx0, sy1), fill=BLACK, width=max(1, int(1.2 * scale)))

    query_points = [(x_at(row["subbin_offset"]), y_at(row["query_packet_mean_db"])) for row in rows]
    loc23_points = [(x_at(row["subbin_offset"]), y_at(row["loc23_template_mean_db_excluding_query"])) for row in rows]
    loc43_points = [(x_at(row["subbin_offset"]), y_at(row["loc43_template_mean_db"])) for row in rows]

    draw.line(query_points, fill=BLUE, width=max(2, int(1.7 * scale)))
    draw.line(loc23_points, fill=GREEN, width=max(2, int(1.5 * scale)))
    dashed_line(draw, loc43_points, fill=ORANGE, width=max(2, int(1.5 * scale)), dash=int(10 * scale), gap=int(7 * scale))
    for x, y in query_points:
        draw.ellipse((x - 3.1 * scale, y - 3.1 * scale, x + 3.1 * scale, y + 3.1 * scale), fill=BLUE)
    for x, y in loc23_points:
        draw.ellipse((x - 2.5 * scale, y - 2.5 * scale, x + 2.5 * scale, y + 2.5 * scale), fill=GREEN)

    # Emphasize the side where the query differs from loc43 around +1 sub-bin.
    for highlight_offset in [0.75, 1.0]:
        row = min(rows, key=lambda item: abs(item["subbin_offset"] - highlight_offset))
        hx = x_at(row["subbin_offset"])
        yq = y_at(row["query_packet_mean_db"])
        y43 = y_at(row["loc43_template_mean_db"])
        draw.line((hx, yq, hx, y43), fill=RED, width=max(1, int(scale)))
    draw.text(
        (x_at(0.45), y_at(-13.0)),
        "shape gap\nnear right side",
        font=small_font,
        fill=RED,
    )

    # Legend and metrics.
    legend_x = sx0 + 20 * scale
    legend_y = sy0 + 14 * scale
    legend = [
        ("query packet", BLUE, "solid"),
        (f"loc {TRUE_LOC} template", GREEN, "solid"),
        (f"loc {AMBIG_LOC} template", ORANGE, "dash"),
    ]
    for idx, (label, color, style) in enumerate(legend):
        y = legend_y + idx * 18 * scale
        if style == "dash":
            dashed_line(draw, [(legend_x, y), (legend_x + 36 * scale, y)], color, max(2, int(1.4 * scale)), int(8 * scale), int(5 * scale))
        else:
            draw.line((legend_x, y, legend_x + 36 * scale, y), fill=color, width=max(2, int(1.4 * scale)))
        draw.text((legend_x + 44 * scale, y - 6 * scale), label, font=small_font, fill=BLACK)
    draw.text((sx1 - 390, sy0 + 14 * scale), f"RMSE to loc {TRUE_LOC}: {metrics['rmse_query_to_loc23_db']:.2f} dB", font=small_font, fill=GRAY)
    draw.text((sx1 - 390, sy0 + 31 * scale), f"RMSE to loc {AMBIG_LOC}: {metrics['rmse_query_to_loc43_db']:.2f} dB", font=small_font, fill=GRAY)

    draw_centered(draw, (sx0 + sx1) / 2, height - 35 * scale, "Sub-bin offset from aligned center bin", label_font, BLACK)
    ylabel = "Magnitude relative to local peak (dB)"
    # Draw y label for spectrum.
    scratch = Image.new("RGBA", (360, 60), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), ylabel, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), ylabel, font=label_font, fill=BLACK)
    rotated = text_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (int(sx0 - 74 * scale), int((sy0 + sy1 - rotated.height) / 2)))

    out = HERE / "frequency_disambiguation_23_43.png"
    image.convert("RGB").save(out, dpi=(DPI, DPI))
    return out


def render_pdf(rows: list[dict], metrics: dict[str, float], posterior: dict[int, float]) -> Path:
    font, bold = register_pdf_font()
    width = WIDTH_IN * 72
    height = HEIGHT_IN * 72
    out = HERE / "frequency_disambiguation_23_43.pdf"
    pdf = canvas.Canvas(str(out), pagesize=(width, height))

    pdf.setFont(bold, 10.5)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(36, height - 19, "Frequency evidence for RSSI+ ambiguous candidates")
    pdf.setFont(font, 6.5)
    pdf.setFillColor(hex_to_color(GRAY))
    pdf.drawString(36, height - 33, f"query: loc {TRUE_LOC}, packet {QUERY_PACKET}; loc {TRUE_LOC} template excludes the query packet")

    bar = (38, 50, 148, 185)
    spec = (198, 50, 500, 185)

    # Bar panel.
    x0, y0, x1, y1 = bar
    pdf.setFont(bold, 8.4)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(x0, y1 + 17, "(a) RSSI+ posterior is nearly tied")
    ymax = 0.16
    for tick in [0.0, 0.05, 0.10, 0.15]:
        y = y0 + tick / ymax * (y1 - y0)
        pdf.setStrokeColor(hex_to_color(LIGHT_GRAY))
        pdf.setLineWidth(0.45)
        pdf.line(x0, y, x1, y)
        pdf.setFont(font, 6.3)
        pdf.setFillColor(hex_to_color(GRAY))
        pdf.drawRightString(x0 - 4, y - 2.2, f"{tick:.2f}".rstrip("0").rstrip("."))
    pdf.setStrokeColor(hex_to_color(BLACK))
    pdf.setLineWidth(0.8)
    pdf.line(x0, y0, x1, y0)
    pdf.line(x0, y0, x0, y1)
    slot = (x1 - x0) / 3.2
    for idx, (loc, prob, color) in enumerate([(TRUE_LOC, posterior[TRUE_LOC], GREEN), (AMBIG_LOC, posterior[AMBIG_LOC], ORANGE)]):
        bx = x0 + slot * (idx + 0.9)
        by = y0 + prob / ymax * (y1 - y0)
        pdf.setFillColor(hex_to_color(color))
        pdf.rect(bx - 10, y0, 20, by - y0, stroke=0, fill=1)
        pdf.setFont(font, 6.3)
        pdf.setFillColor(hex_to_color(GRAY))
        pdf.drawCentredString(bx, by + 5, f"{prob:.3f}")
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawCentredString(bx, y0 - 11, str(loc))
    pdf.setFont(font, 7.3)
    pdf.drawCentredString((x0 + x1) / 2, 15, "Candidate location")
    pdf.saveState()
    pdf.translate(13, (y0 + y1) / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Posterior")
    pdf.restoreState()

    # Spectrum panel.
    sx0, sy0, sx1, sy1 = spec
    pdf.setFont(bold, 8.4)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(sx0, sy1 + 17, "(b) q=4 preamble spectrum gives structure cue")
    x_min, x_max = -2.0, 2.0
    y_min, y_max = -32.0, 2.0

    def x_at(value: float) -> float:
        return sx0 + (value - x_min) / (x_max - x_min) * (sx1 - sx0)

    def y_at(value: float) -> float:
        return sy0 + (value - y_min) / (y_max - y_min) * (sy1 - sy0)

    for tick in [-30, -20, -10, 0]:
        y = y_at(tick)
        pdf.setStrokeColor(hex_to_color(LIGHT_GRAY))
        pdf.setLineWidth(0.45)
        pdf.line(sx0, y, sx1, y)
        pdf.setFont(font, 6.3)
        pdf.setFillColor(hex_to_color(GRAY))
        pdf.drawRightString(sx0 - 4, y - 2.2, str(tick))
    for tick in [-2, -1, 0, 1, 2]:
        x = x_at(tick)
        pdf.setStrokeColor(hex_to_color("#eeeeee"))
        pdf.setLineWidth(0.35)
        pdf.line(x, sy0, x, sy1)
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawCentredString(x, sy0 - 10, str(tick))
    pdf.setStrokeColor(hex_to_color(BLACK))
    pdf.setLineWidth(0.8)
    pdf.line(sx0, sy0, sx1, sy0)
    pdf.line(sx0, sy0, sx0, sy1)

    def draw_path(points: list[tuple[float, float]], color: str, dashed: bool = False) -> None:
        pdf.setStrokeColor(hex_to_color(color))
        pdf.setLineWidth(1.0)
        if dashed:
            pdf.setDash(4, 3)
        else:
            pdf.setDash()
        path = pdf.beginPath()
        path.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            path.lineTo(x, y)
        pdf.drawPath(path, stroke=1, fill=0)
        pdf.setDash()

    query_points = [(x_at(row["subbin_offset"]), y_at(row["query_packet_mean_db"])) for row in rows]
    loc23_points = [(x_at(row["subbin_offset"]), y_at(row["loc23_template_mean_db_excluding_query"])) for row in rows]
    loc43_points = [(x_at(row["subbin_offset"]), y_at(row["loc43_template_mean_db"])) for row in rows]
    draw_path(query_points, BLUE)
    draw_path(loc23_points, GREEN)
    draw_path(loc43_points, ORANGE, dashed=True)
    pdf.setFillColor(hex_to_color(BLUE))
    for x, y in query_points:
        pdf.circle(x, y, 1.4, stroke=0, fill=1)

    pdf.setFont(font, 6.3)
    pdf.setFillColor(hex_to_color(BLACK))
    lx, ly = sx0 + 8, sy1 - 10
    for idx, (label, color, dashed) in enumerate([
        ("query packet", BLUE, False),
        (f"loc {TRUE_LOC} template", GREEN, False),
        (f"loc {AMBIG_LOC} template", ORANGE, True),
    ]):
        y = ly - idx * 11
        pdf.setStrokeColor(hex_to_color(color))
        pdf.setLineWidth(1.0)
        pdf.setDash(4, 3) if dashed else pdf.setDash()
        pdf.line(lx, y, lx + 17, y)
        pdf.setDash()
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawString(lx + 22, y - 2.4, label)
    pdf.setFillColor(hex_to_color(GRAY))
    pdf.drawString(sx0 + 120, sy1 - 10, f"RMSE to loc {TRUE_LOC}: {metrics['rmse_query_to_loc23_db']:.2f} dB")
    pdf.drawString(sx0 + 120, sy1 - 21, f"RMSE to loc {AMBIG_LOC}: {metrics['rmse_query_to_loc43_db']:.2f} dB")

    pdf.setFont(font, 7.3)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawCentredString((sx0 + sx1) / 2, 15, "Sub-bin offset from aligned center bin")
    pdf.saveState()
    pdf.translate(sx0 - 22, (sy0 + sy1) / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Magnitude relative to local peak (dB)")
    pdf.restoreState()

    pdf.showPage()
    pdf.save()
    return out


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    posterior = load_posterior_pair()
    rows, metrics = load_curves()
    write_data(rows, metrics, posterior)
    render_png(rows, metrics, posterior)
    render_pdf(rows, metrics, posterior)


if __name__ == "__main__":
    main()
