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
WIDTH_IN = 7.2
HEIGHT_IN = 4.2
DPI = 400

BLUE = "#1f77b4"
LIGHT_BLUE = "#b7d7ee"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#666666"
LIGHT_GRAY = "#d9d9d9"
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


def read_scope(scope: str) -> list[dict]:
    path = HERE / f"rssiplus_posterior_{scope}_data.csv"
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
        row["is_top_k"] = row["is_top_k"] == "True"
    return sorted(rows, key=lambda item: item["rank"])


def comparison_rows(scopes: list[str]) -> list[dict]:
    output: list[dict] = []
    for scope in scopes:
        rows = read_scope(scope)
        top = rows[:TOP_N]
        other_prob = sum(row["posterior_probability"] for row in rows[TOP_N:])
        top_k_mass = sum(row["posterior_probability"] for row in rows[:TOP_K])
        top_n_mass = sum(row["posterior_probability"] for row in rows[:TOP_N])
        for index, row in enumerate(top, start=1):
            output.append(
                {
                    "scope": scope,
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
                    "top_n_mass": top_n_mass,
                    "others_mass": other_prob,
                    "candidate_count": len(rows),
                }
            )
        output.append(
            {
                "scope": scope,
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
                "top_n_mass": top_n_mass,
                "others_mass": other_prob,
                "candidate_count": len(rows),
            }
        )
    return output


def write_data(rows: list[dict]) -> Path:
    path = HERE / "rssiplus_topk_comparison_data.csv"
    fields = [
        "scope",
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
        "top_n_mass",
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
    title_font = load_font(int(10.5 * scale), bold=True)
    label_font = load_font(int(8.4 * scale))
    small_font = load_font(int(7.1 * scale))
    panel_font = load_font(int(9.2 * scale), bold=True)

    title = f"RSSI+ posterior support: ranked candidates vs. remaining mass"
    draw.text((48 * scale, 14 * scale), title, font=title_font, fill=BLACK)

    panel_top = 76 * scale
    panel_bottom = height - 120 * scale
    panel_gap = 48 * scale
    left_margin = 62 * scale
    right_margin = 28 * scale
    panel_width = (width - left_margin - right_margin - panel_gap) / 2
    y_max = 0.40

    scopes = ["54points", "28points"]
    scope_titles = {
        "54points": "All 54 reference points",
        "28points": "Filtered 28 reference points",
    }

    for panel_index, scope in enumerate(scopes):
        panel_rows = [row for row in rows if row["scope"] == scope]
        x0 = left_margin + panel_index * (panel_width + panel_gap)
        x1 = x0 + panel_width
        y0 = panel_top
        y1 = panel_bottom
        plot_h = y1 - y0
        slot = panel_width / len(panel_rows)
        bar_w = slot * 0.62

        def y_at(value: float) -> float:
            return y1 - value / y_max * plot_h

        for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
            y = y_at(tick)
            draw.line((x0, y, x1, y), fill=LIGHT_GRAY, width=max(1, int(scale)))
            if panel_index == 0:
                label = f"{tick:.1f}"
                tw, th = text_size(draw, label, small_font)
                draw.text((x0 - tw - 8 * scale, y - th / 2), label, font=small_font, fill=GRAY)

        draw.line((x0, y1, x1, y1), fill=BLACK, width=max(1, int(1.2 * scale)))
        draw.line((x0, y0, x0, y1), fill=BLACK, width=max(1, int(1.2 * scale)))

        top_k_mass = panel_rows[0]["top_k_mass"]
        others_mass = panel_rows[-1]["others_mass"]
        panel_title = scope_titles[scope]
        summary = f"Top-{TOP_K} mass={top_k_mass:.2f}, Others={others_mass:.2f}"
        draw.text((x0, y0 - 34 * scale), panel_title, font=panel_font, fill=BLACK)
        draw.text((x0, y0 - 18 * scale), summary, font=small_font, fill=GRAY)

        for index, row in enumerate(panel_rows):
            x = x0 + slot * (index + 0.5)
            y = y_at(row["posterior_probability"])
            fill = "#cfcfcf" if row["is_other"] else (ORANGE if row["is_top_k"] else LIGHT_BLUE)
            draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, y1), fill=fill)
            draw.line((x, y1, x, y1 + 3 * scale), fill=BLACK, width=max(1, int(scale)))
            label = row["bar_label"]
            tw, th = text_size(draw, label, small_font)
            draw.text((x - tw / 2, y1 + 9 * scale), label, font=small_font, fill=BLACK)

            prob_label = f"{row['posterior_probability']:.2f}"
            tw, th = text_size(draw, prob_label, small_font)
            draw.text((x - tw / 2, y - th - 4 * scale), prob_label, font=small_font, fill=GRAY)

            if row["is_rssi_plus_prediction"]:
                marker_triangle(draw, x - 5 * scale, y - 16 * scale, 5.0 * scale, RED)
            if row["is_true_location"]:
                marker_star(draw, x + 5 * scale, y - 16 * scale, 6.0 * scale, GREEN)

        xlabel = "Ranked candidate location"
        tw, th = text_size(draw, xlabel, label_font)
        draw.text((x0 + panel_width / 2 - tw / 2, height - 76 * scale), xlabel, font=label_font, fill=BLACK)

    ylabel = "Posterior probability / mass"
    scratch = Image.new("RGBA", (260, 60), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), ylabel, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), ylabel, font=label_font, fill=BLACK)
    rotated = text_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (int(18 * scale), int((panel_top + panel_bottom - rotated.height) / 2)))

    legend_x = 220 * scale
    legend_y = height - 45 * scale
    legend_items = [
        ("Top-5 candidates", ORANGE, "circle"),
        ("rank 6-8", LIGHT_BLUE, "square"),
        ("Others", "#cfcfcf", "square"),
        ("true location", GREEN, "star"),
        ("RSSI+ Top-1", RED, "triangle"),
    ]
    item_offsets = [0, 105, 185, 255, 335]
    for idx, (label, color, kind) in enumerate(legend_items):
        x = legend_x + item_offsets[idx] * scale
        y = legend_y
        if kind == "circle":
            draw.ellipse((x, y, x + 9 * scale, y + 9 * scale), fill=color)
        elif kind == "square":
            draw.rectangle((x, y, x + 9 * scale, y + 9 * scale), fill=color)
        elif kind == "star":
            marker_star(draw, x + 5 * scale, y + 5 * scale, 5 * scale, color)
        else:
            marker_triangle(draw, x + 5 * scale, y + 5 * scale, 4.6 * scale, color)
        draw.text((x + 13 * scale, y - 1 * scale), label, font=small_font, fill=BLACK)

    out = HERE / "rssiplus_topk_comparison.png"
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
    out = HERE / "rssiplus_topk_comparison.pdf"
    pdf = canvas.Canvas(str(out), pagesize=(width, height))
    pdf.setFont(bold, 10.5)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(35, height - 20, "RSSI+ posterior support: ranked candidates vs. remaining mass")

    panel_top = height - 76 / 100 * 72
    panel_bottom = 120 / 100 * 72
    panel_gap = 48 / 100 * 72
    left_margin = 62 / 100 * 72
    right_margin = 28 / 100 * 72
    panel_width = (width - left_margin - right_margin - panel_gap) / 2
    y_max = 0.40
    scopes = ["54points", "28points"]
    scope_titles = {
        "54points": "All 54 reference points",
        "28points": "Filtered 28 reference points",
    }

    for panel_index, scope in enumerate(scopes):
        panel_rows = [row for row in rows if row["scope"] == scope]
        x0 = left_margin + panel_index * (panel_width + panel_gap)
        x1 = x0 + panel_width
        y0 = panel_bottom
        y1 = panel_top
        plot_h = y1 - y0
        slot = panel_width / len(panel_rows)
        bar_w = slot * 0.62

        def y_at(value: float) -> float:
            return y0 + value / y_max * plot_h

        for tick in [0.0, 0.1, 0.2, 0.3, 0.4]:
            y = y_at(tick)
            pdf.setStrokeColor(hex_to_color(LIGHT_GRAY))
            pdf.setLineWidth(0.5)
            pdf.line(x0, y, x1, y)
            if panel_index == 0:
                pdf.setFont(font, 7.0)
                pdf.setFillColor(hex_to_color(GRAY))
                pdf.drawRightString(x0 - 5, y - 2.5, f"{tick:.1f}")

        pdf.setStrokeColor(hex_to_color(BLACK))
        pdf.setLineWidth(0.8)
        pdf.line(x0, y0, x1, y0)
        pdf.line(x0, y0, x0, y1)
        pdf.setFont(bold, 9.2)
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawString(x0, y1 + 20, scope_titles[scope])
        pdf.setFont(font, 7.1)
        pdf.setFillColor(hex_to_color(GRAY))
        pdf.drawString(
            x0,
            y1 + 8,
            f"Top-{TOP_K} mass={panel_rows[0]['top_k_mass']:.2f}, Others={panel_rows[-1]['others_mass']:.2f}",
        )

        for index, row in enumerate(panel_rows):
            x = x0 + slot * (index + 0.5)
            y = y_at(row["posterior_probability"])
            fill = "#cfcfcf" if row["is_other"] else (ORANGE if row["is_top_k"] else LIGHT_BLUE)
            pdf.setFillColor(hex_to_color(fill))
            pdf.rect(x - bar_w / 2, y0, bar_w, y - y0, stroke=0, fill=1)
            pdf.setFillColor(hex_to_color(GRAY))
            pdf.setFont(font, 6.8)
            pdf.drawCentredString(x, y + 5, f"{row['posterior_probability']:.2f}")
            pdf.setFillColor(hex_to_color(BLACK))
            pdf.drawCentredString(x, y0 - 12, str(row["bar_label"]))
            if row["is_rssi_plus_prediction"]:
                pdf_triangle(pdf, x - 3.5, y + 16, 3.8, RED)
            if row["is_true_location"]:
                pdf_star(pdf, x + 3.5, y + 16, 4.3, GREEN)

        pdf.setFont(font, 8.4)
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawCentredString(x0 + panel_width / 2, 55, "Ranked candidate location")

    pdf.saveState()
    pdf.translate(18, panel_bottom + (panel_top - panel_bottom) / 2)
    pdf.rotate(90)
    pdf.setFont(font, 8.4)
    pdf.drawCentredString(0, 0, "Posterior probability / mass")
    pdf.restoreState()

    pdf.setFont(font, 7.1)
    legend_x = 145
    legend_y = 30
    legend_items = [
        ("Top-5 candidates", ORANGE),
        ("rank 6-8", LIGHT_BLUE),
        ("Others", "#cfcfcf"),
        ("true", GREEN),
        ("Top-1", RED),
    ]
    item_offsets = [0, 80, 145, 198, 252]
    for idx, (label, color) in enumerate(legend_items):
        x = legend_x + item_offsets[idx]
        pdf.setFillColor(hex_to_color(color))
        if idx == 3:
            pdf_star(pdf, x + 4, legend_y + 4, 4, color)
        elif idx == 4:
            pdf_triangle(pdf, x + 4, legend_y + 4, 3.7, color)
        else:
            pdf.rect(x, legend_y, 8, 8, stroke=0, fill=1)
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawString(x + 11, legend_y + 1, label)

    pdf.showPage()
    pdf.save()
    return out


def main() -> None:
    rows = comparison_rows(["54points", "28points"])
    write_data(rows)
    render_png(rows)
    render_pdf(rows)


if __name__ == "__main__":
    main()
