from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
TOP_N = 8
TOP_K = 5
WIDTH_IN = 3.5
HEIGHT_IN = 2.7
DPI = 400

ORANGE = "#ff7f0e"
LIGHT_BLUE = "#b7d7ee"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY_BAR = "#cfcfcf"
GRID = "#d9d9d9"
GRAY = "#666666"
BLACK = "#111111"


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


def read_rows() -> list[dict]:
    path = HERE / "rssiplus_posterior_28points_data.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["reference_location"] = int(row["reference_location"])
        row["rank"] = int(row["rank"])
        row["posterior_probability"] = float(row["posterior_probability"])
        row["true_location"] = int(row["true_location"])
        row["predicted_location"] = int(row["predicted_location"])
        row["is_true_location"] = row["is_true_location"] == "True"
        row["is_rssi_plus_prediction"] = row["is_rssi_plus_prediction"] == "True"
    return sorted(rows, key=lambda item: item["rank"])


def build_plot_rows(rows: list[dict]) -> list[dict]:
    top = rows[:TOP_N]
    other_prob = sum(row["posterior_probability"] for row in rows[TOP_N:])
    top_k_mass = sum(row["posterior_probability"] for row in rows[:TOP_K])
    output: list[dict] = []
    for index, row in enumerate(top, start=1):
        output.append(
            {
                "bar_order": index,
                "bar_label": str(row["reference_location"]),
                "reference_location": row["reference_location"],
                "posterior_probability": row["posterior_probability"],
                "rank": row["rank"],
                "is_other": False,
                "is_top_k": row["rank"] <= TOP_K,
                "is_true_location": row["is_true_location"],
                "is_rssi_plus_prediction": row["is_rssi_plus_prediction"],
                "true_location": row["true_location"],
                "predicted_location": row["predicted_location"],
                "top_k_mass": top_k_mass,
                "others_mass": other_prob,
                "candidate_count": len(rows),
            }
        )
    output.append(
        {
            "bar_order": TOP_N + 1,
            "bar_label": "Others",
            "reference_location": "",
            "posterior_probability": other_prob,
            "rank": "",
            "is_other": True,
            "is_top_k": False,
            "is_true_location": False,
            "is_rssi_plus_prediction": False,
            "true_location": rows[0]["true_location"],
            "predicted_location": rows[0]["predicted_location"],
            "top_k_mass": top_k_mass,
            "others_mass": other_prob,
            "candidate_count": len(rows),
        }
    )
    return output


def write_data(rows: list[dict]) -> Path:
    path = HERE / "rssiplus_topk_28points_data.csv"
    fields = [
        "bar_order",
        "bar_label",
        "reference_location",
        "posterior_probability",
        "rank",
        "is_other",
        "is_top_k",
        "is_true_location",
        "is_rssi_plus_prediction",
        "true_location",
        "predicted_location",
        "top_k_mass",
        "others_mass",
        "candidate_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def marker_star(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: str) -> None:
    points = []
    for index in range(10):
        radius = r if index % 2 == 0 else r * 0.42
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    draw.polygon(points, fill=fill, outline="white")


def marker_triangle(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: str) -> None:
    draw.polygon(
        [(x, y - r), (x - r * 0.9, y + r * 0.8), (x + r * 0.9, y + r * 0.8)],
        fill=fill,
        outline="white",
    )


def render_png(rows: list[dict]) -> Path:
    width = int(WIDTH_IN * DPI)
    height = int(HEIGHT_IN * DPI)
    scale = DPI / 100
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(9.8 * scale), bold=True)
    label_font = load_font(int(7.7 * scale))
    small_font = load_font(int(6.6 * scale))

    left = 48 * scale
    right = 18 * scale
    top = 52 * scale
    bottom = 70 * scale
    x0, x1 = left, width - right
    y0, y1 = top, height - bottom
    y_max = 0.40
    plot_h = y1 - y0
    slot = (x1 - x0) / len(rows)
    bar_w = slot * 0.62

    def y_at(value: float) -> float:
        return y1 - value / y_max * plot_h

    draw.text((x0, 14 * scale), "RSSI+ posterior support (28 reference points)", font=title_font, fill=BLACK)
    draw.text(
        (x0, 31 * scale),
        f"Top-{TOP_K} mass={rows[0]['top_k_mass']:.2f}, Others={rows[-1]['others_mass']:.2f}",
        font=small_font,
        fill=GRAY,
    )

    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        y = y_at(tick)
        draw.line((x0, y, x1, y), fill=GRID, width=max(1, int(scale)))
        label = f"{tick:.1f}"
        tw, th = text_size(draw, label, small_font)
        draw.text((x0 - tw - 7 * scale, y - th / 2), label, font=small_font, fill=GRAY)
    draw.line((x0, y1, x1, y1), fill=BLACK, width=max(1, int(1.1 * scale)))
    draw.line((x0, y0, x0, y1), fill=BLACK, width=max(1, int(1.1 * scale)))

    for index, row in enumerate(rows):
        x = x0 + slot * (index + 0.5)
        y = y_at(row["posterior_probability"])
        fill = GRAY_BAR if row["is_other"] else (ORANGE if row["is_top_k"] else LIGHT_BLUE)
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, y1), fill=fill)
        prob_label = f"{row['posterior_probability']:.2f}"
        tw, th = text_size(draw, prob_label, small_font)
        draw.text((x - tw / 2, y - th - 3 * scale), prob_label, font=small_font, fill=GRAY)
        label = row["bar_label"]
        tw, th = text_size(draw, label, small_font)
        draw.text((x - tw / 2, y1 + 8 * scale), label, font=small_font, fill=BLACK)
        if row["is_rssi_plus_prediction"]:
            marker_triangle(draw, x - 4.5 * scale, y - 15 * scale, 4.8 * scale, RED)
        if row["is_true_location"]:
            marker_star(draw, x + 4.5 * scale, y - 15 * scale, 5.6 * scale, GREEN)

    xlabel = "Ranked candidate location"
    ylabel = "Posterior probability / mass"
    tw, _ = text_size(draw, xlabel, label_font)
    draw.text((x0 + (x1 - x0) / 2 - tw / 2, height - 33 * scale), xlabel, font=label_font, fill=BLACK)

    scratch = Image.new("RGBA", (220, 50), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), ylabel, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), ylabel, font=label_font, fill=BLACK)
    rotated = text_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (int(13 * scale), int((y0 + y1 - rotated.height) / 2)))

    out = HERE / "rssiplus_topk_28points.png"
    image.convert("RGB").save(out, dpi=(DPI, DPI))
    return out


def pdf_star(pdf: canvas.Canvas, x: float, y: float, r: float, fill: str) -> None:
    path = pdf.beginPath()
    points = []
    for index in range(10):
        radius = r if index % 2 == 0 else r * 0.42
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    path.moveTo(points[0][0], points[0][1])
    for px, py in points[1:]:
        path.lineTo(px, py)
    path.close()
    pdf.setFillColor(hex_to_color(fill))
    pdf.setStrokeColor(colors.white)
    pdf.drawPath(path, stroke=1, fill=1)


def pdf_triangle(pdf: canvas.Canvas, x: float, y: float, r: float, fill: str) -> None:
    path = pdf.beginPath()
    path.moveTo(x, y + r)
    path.lineTo(x - r * 0.9, y - r * 0.8)
    path.lineTo(x + r * 0.9, y - r * 0.8)
    path.close()
    pdf.setFillColor(hex_to_color(fill))
    pdf.setStrokeColor(colors.white)
    pdf.drawPath(path, stroke=1, fill=1)


def render_pdf(rows: list[dict]) -> Path:
    font, bold = register_pdf_font()
    width = WIDTH_IN * 72
    height = HEIGHT_IN * 72
    out = HERE / "rssiplus_topk_28points.pdf"
    pdf = canvas.Canvas(str(out), pagesize=(width, height))
    left = 48 / 100 * 72
    right = 18 / 100 * 72
    top = height - 52 / 100 * 72
    bottom = 70 / 100 * 72
    x0, x1 = left, width - right
    y0, y1 = bottom, top
    y_max = 0.40
    plot_h = y1 - y0
    slot = (x1 - x0) / len(rows)
    bar_w = slot * 0.62

    def y_at(value: float) -> float:
        return y0 + value / y_max * plot_h

    pdf.setFont(bold, 9.8)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(x0, height - 16, "RSSI+ posterior support (28 reference points)")
    pdf.setFont(font, 6.6)
    pdf.setFillColor(hex_to_color(GRAY))
    pdf.drawString(x0, height - 29, f"Top-{TOP_K} mass={rows[0]['top_k_mass']:.2f}, Others={rows[-1]['others_mass']:.2f}")

    for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
        y = y_at(tick)
        pdf.setStrokeColor(hex_to_color(GRID))
        pdf.setLineWidth(0.45)
        pdf.line(x0, y, x1, y)
        pdf.setFillColor(hex_to_color(GRAY))
        pdf.setFont(font, 6.6)
        pdf.drawRightString(x0 - 4, y - 2.4, f"{tick:.1f}")
    pdf.setStrokeColor(hex_to_color(BLACK))
    pdf.setLineWidth(0.8)
    pdf.line(x0, y0, x1, y0)
    pdf.line(x0, y0, x0, y1)

    for index, row in enumerate(rows):
        x = x0 + slot * (index + 0.5)
        y = y_at(row["posterior_probability"])
        fill = GRAY_BAR if row["is_other"] else (ORANGE if row["is_top_k"] else LIGHT_BLUE)
        pdf.setFillColor(hex_to_color(fill))
        pdf.rect(x - bar_w / 2, y0, bar_w, y - y0, stroke=0, fill=1)
        pdf.setFillColor(hex_to_color(GRAY))
        pdf.setFont(font, 6.3)
        pdf.drawCentredString(x, y + 4.2, f"{row['posterior_probability']:.2f}")
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawCentredString(x, y0 - 10, str(row["bar_label"]))
        if row["is_rssi_plus_prediction"]:
            pdf_triangle(pdf, x - 3.2, y + 12, 3.4, RED)
        if row["is_true_location"]:
            pdf_star(pdf, x + 3.2, y + 12, 3.8, GREEN)

    pdf.setFont(font, 7.7)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawCentredString(x0 + (x1 - x0) / 2, 22, "Ranked candidate location")
    pdf.saveState()
    pdf.translate(14, y0 + (y1 - y0) / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Posterior probability / mass")
    pdf.restoreState()
    pdf.showPage()
    pdf.save()
    return out


def main() -> None:
    plot_rows = build_plot_rows(read_rows())
    write_data(plot_rows)
    render_png(plot_rows)
    render_pdf(plot_rows)


if __name__ == "__main__":
    main()
