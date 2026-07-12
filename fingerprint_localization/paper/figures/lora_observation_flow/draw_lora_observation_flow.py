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
WIDTH_IN = 7.2
HEIGHT_IN = 3.4
DPI = 400

BLUE = "#1f77b4"
BLUE_LIGHT = "#d8ecf8"
ORANGE = "#ff7f0e"
ORANGE_LIGHT = "#ffe1bd"
GREEN = "#2ca02c"
GREEN_LIGHT = "#dff1df"
GRAY = "#666666"
LIGHT_GRAY = "#e6e6e6"
MID_GRAY = "#bdbdbd"
BLACK = "#111111"
WHITE = "#ffffff"


NODES = [
    {
        "id": "tag",
        "type": "device",
        "label": "Commercial\nLoRa tag",
        "x": 0.25,
        "y": 1.24,
        "w": 0.78,
        "h": 0.58,
        "fill": WHITE,
        "stroke": MID_GRAY,
        "branch": "input",
    },
    {
        "id": "gateway",
        "type": "device",
        "label": "Single\ngateway",
        "x": 1.28,
        "y": 1.24,
        "w": 0.78,
        "h": 0.58,
        "fill": WHITE,
        "stroke": MID_GRAY,
        "branch": "input",
    },
    {
        "id": "packet",
        "type": "process",
        "label": "Standard LoRa\npacket",
        "x": 2.28,
        "y": 1.24,
        "w": 0.88,
        "h": 0.58,
        "fill": "#f7f7f7",
        "stroke": MID_GRAY,
        "branch": "input",
    },
    {
        "id": "demod",
        "type": "process",
        "label": "Packet\ndemod.",
        "x": 3.55,
        "y": 0.55,
        "w": 0.78,
        "h": 0.50,
        "fill": ORANGE_LIGHT,
        "stroke": ORANGE,
        "branch": "rssi_plus",
    },
    {
        "id": "rssi_snr",
        "type": "feature",
        "label": "RSSI / SNR",
        "x": 4.55,
        "y": 0.55,
        "w": 0.78,
        "h": 0.50,
        "fill": ORANGE_LIGHT,
        "stroke": ORANGE,
        "branch": "rssi_plus",
    },
    {
        "id": "rssi_plus",
        "type": "feature",
        "label": "RSSI+",
        "x": 5.55,
        "y": 0.55,
        "w": 0.68,
        "h": 0.50,
        "fill": ORANGE_LIGHT,
        "stroke": ORANGE,
        "branch": "rssi_plus",
    },
    {
        "id": "posterior",
        "type": "output",
        "label": "Location\nposterior",
        "x": 6.42,
        "y": 0.45,
        "w": 0.64,
        "h": 0.70,
        "fill": ORANGE_LIGHT,
        "stroke": ORANGE,
        "branch": "rssi_plus",
    },
    {
        "id": "preamble",
        "type": "process",
        "label": "Preamble\nI/Q",
        "x": 3.55,
        "y": 2.03,
        "w": 0.78,
        "h": 0.50,
        "fill": GREEN_LIGHT,
        "stroke": GREEN,
        "branch": "spectrum",
    },
    {
        "id": "dechirp",
        "type": "process",
        "label": "Dechirp",
        "x": 4.55,
        "y": 2.03,
        "w": 0.78,
        "h": 0.50,
        "fill": GREEN_LIGHT,
        "stroke": GREEN,
        "branch": "spectrum",
    },
    {
        "id": "fft",
        "type": "process",
        "label": "FFT",
        "x": 5.55,
        "y": 2.03,
        "w": 0.68,
        "h": 0.50,
        "fill": GREEN_LIGHT,
        "stroke": GREEN,
        "branch": "spectrum",
    },
    {
        "id": "shape",
        "type": "output",
        "label": "Center-bin\nspectrum",
        "x": 6.42,
        "y": 1.93,
        "w": 0.64,
        "h": 0.70,
        "fill": GREEN_LIGHT,
        "stroke": GREEN,
        "branch": "spectrum",
    },
    {
        "id": "power_obs",
        "type": "annotation",
        "label": "Power-statistical\nobservation",
        "x": 3.64,
        "y": 0.08,
        "w": 2.45,
        "h": 0.27,
        "fill": WHITE,
        "stroke": WHITE,
        "branch": "rssi_plus",
    },
    {
        "id": "local_obs",
        "type": "annotation",
        "label": "Local propagation-\nstructure observation",
        "x": 3.56,
        "y": 2.78,
        "w": 2.80,
        "h": 0.28,
        "fill": WHITE,
        "stroke": WHITE,
        "branch": "spectrum",
    },
]

EDGES = [
    ("tag", "gateway", "", "input"),
    ("gateway", "packet", "", "input"),
    ("packet", "demod", "", "rssi_plus"),
    ("demod", "rssi_snr", "", "rssi_plus"),
    ("rssi_snr", "rssi_plus", "", "rssi_plus"),
    ("rssi_plus", "posterior", "", "rssi_plus"),
    ("packet", "preamble", "", "spectrum"),
    ("preamble", "dechirp", "", "spectrum"),
    ("dechirp", "fft", "", "spectrum"),
    ("fft", "shape", "", "spectrum"),
]


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


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def hex_to_color(value: str) -> colors.Color:
    r, g, b = hex_to_rgb(value)
    return colors.Color(r / 255, g / 255, b / 255)


def pt(value: float) -> float:
    return value * DPI


def pdf_pt(value: float) -> float:
    return value * 72


def center(node: dict) -> tuple[float, float]:
    return node["x"] + node["w"] / 2, node["y"] + node["h"] / 2


def find_node(node_id: str) -> dict:
    return next(node for node in NODES if node["id"] == node_id)


def text_lines(draw: ImageDraw.ImageDraw, label: str, font: ImageFont.ImageFont) -> list[tuple[str, int, int]]:
    lines = []
    for line in label.split("\n"):
        bbox = draw.textbbox((0, 0), line, font=font)
        lines.append((line, bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return lines


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    label: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    x0, y0, x1, y1 = box
    lines = text_lines(draw, label, font)
    line_gap = font.size * 0.22
    total_h = sum(line[2] for line in lines) + line_gap * (len(lines) - 1)
    y = y0 + (y1 - y0 - total_h) / 2
    for line, w, h in lines:
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=font, fill=fill)
        y += h + line_gap


def arrow_head(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], fill: str, scale: float) -> None:
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    length = 8.0 * scale
    spread = 0.42
    points = [
        (ex, ey),
        (ex - length * math.cos(angle - spread), ey - length * math.sin(angle - spread)),
        (ex - length * math.cos(angle + spread), ey - length * math.sin(angle + spread)),
    ]
    draw.polygon(points, fill=fill)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: str,
    width: int,
    scale: float,
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=width)
    arrow_head(draw, start, end, fill, scale)


def edge_points(source: dict, target: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    sx, sy = center(source)
    tx, ty = center(target)
    if abs(tx - sx) >= abs(ty - sy):
        start = (source["x"] + source["w"], sy) if tx > sx else (source["x"], sy)
        end = (target["x"], ty) if tx > sx else (target["x"] + target["w"], ty)
    else:
        start = (sx, source["y"] + source["h"]) if ty > sy else (sx, source["y"])
        end = (tx, target["y"]) if ty > sy else (tx, target["y"] + target["h"])
    return start, end


def draw_radio_waves(draw: ImageDraw.ImageDraw, cx: float, cy: float, color: str, scale: float) -> None:
    for radius in [0.14, 0.22, 0.30]:
        bbox = (
            cx - radius * DPI,
            cy - radius * DPI,
            cx + radius * DPI,
            cy + radius * DPI,
        )
        draw.arc(bbox, start=-35, end=35, fill=color, width=max(2, int(1.2 * scale)))


def render_png() -> Path:
    width = int(WIDTH_IN * DPI)
    height = int(HEIGHT_IN * DPI)
    scale = DPI / 100
    image = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(10.6 * scale), bold=True)
    node_font = load_font(int(7.4 * scale), bold=False)
    node_bold = load_font(int(7.6 * scale), bold=True)
    small_font = load_font(int(6.5 * scale))
    ann_font = load_font(int(8.1 * scale), bold=True)

    draw.text(
        (pt(0.25), pt(0.14)),
        "Two observations from one standard LoRa packet",
        font=title_font,
        fill=BLACK,
    )

    # Branch background bands.
    draw.rounded_rectangle((pt(3.32), pt(0.38), pt(7.14), pt(1.28)), radius=pt(0.06), fill="#fff8f0", outline="#f0c28d", width=max(1, int(0.7 * scale)))
    draw.rounded_rectangle((pt(3.32), pt(1.86), pt(7.14), pt(2.76)), radius=pt(0.06), fill="#f4fbf4", outline="#a8d6a8", width=max(1, int(0.7 * scale)))

    node_by_id = {node["id"]: node for node in NODES}
    for source_id, target_id, label, branch in EDGES:
        source = node_by_id[source_id]
        target = node_by_id[target_id]
        start, end = edge_points(source, target)
        color = ORANGE if branch == "rssi_plus" else GREEN if branch == "spectrum" else GRAY
        draw_arrow(
            draw,
            (pt(start[0]), pt(start[1])),
            (pt(end[0]) - (5 * scale if end[0] > start[0] else 0), pt(end[1])),
            color,
            max(2, int(1.25 * scale)),
            scale,
        )
        if label:
            mx = (pt(start[0]) + pt(end[0])) / 2
            my = (pt(start[1]) + pt(end[1])) / 2 - 18 * scale
            bbox = draw.textbbox((0, 0), label, font=small_font)
            draw.text((mx - (bbox[2] - bbox[0]) / 2, my), label, font=small_font, fill=GRAY)

    # Split guide from packet to branches.
    packet = find_node("packet")
    px, py = center(packet)
    draw.line((pt(px), pt(0.99), pt(px), pt(2.08)), fill=LIGHT_GRAY, width=max(1, int(scale)))

    for node in NODES:
        if node["type"] == "annotation":
            x0, y0, x1, y1 = pt(node["x"]), pt(node["y"]), pt(node["x"] + node["w"]), pt(node["y"] + node["h"])
            color = ORANGE if node["branch"] == "rssi_plus" else GREEN
            draw_centered_text(draw, (x0, y0, x1, y1), node["label"], ann_font, color)
            continue

        x0, y0 = pt(node["x"]), pt(node["y"])
        x1, y1 = pt(node["x"] + node["w"]), pt(node["y"] + node["h"])
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=pt(0.06),
            fill=node["fill"],
            outline=node["stroke"],
            width=max(2, int(1.2 * scale)),
        )
        font = node_bold if node["type"] in {"output", "feature"} else node_font
        draw_centered_text(draw, (x0, y0, x1, y1), node["label"], font, BLACK)

    # Minimal device glyphs.
    for node_id, color in [("tag", BLUE), ("gateway", BLUE)]:
        node = find_node(node_id)
        x0, y0 = pt(node["x"]), pt(node["y"])
        draw.rectangle((x0 + 10 * scale, y0 + 11 * scale, x0 + 24 * scale, y0 + 38 * scale), fill=color, outline=None)
        draw.rounded_rectangle((x0 + 14 * scale, y0 + 4 * scale, x0 + 20 * scale, y0 + 11 * scale), radius=2 * scale, fill=color)
    draw_radio_waves(draw, pt(1.03), pt(1.52), BLUE, scale)

    # Compact equations under outputs.
    draw.text((pt(6.48), pt(1.17)), r"p_R(l|x_R)", font=small_font, fill=ORANGE)
    draw.text((pt(6.50), pt(2.65)), r"x_S = |Y[k0-M:k0+M]|", font=small_font, fill=GREEN)

    out = HERE / "lora_observation_flow.png"
    image.convert("RGB").save(out, dpi=(DPI, DPI))
    return out


def pdf_centered_text(pdf: canvas.Canvas, box: tuple[float, float, float, float], label: str, font: str, size: float, fill: str) -> None:
    x0, y0, x1, y1 = box
    pdf.setFont(font, size)
    pdf.setFillColor(hex_to_color(fill))
    lines = label.split("\n")
    line_h = size * 1.15
    total_h = line_h * len(lines)
    y = y0 + (y1 - y0 + total_h) / 2 - line_h
    for line in lines:
        pdf.drawCentredString((x0 + x1) / 2, y, line)
        y -= line_h


def pdf_arrow_head(pdf: canvas.Canvas, start: tuple[float, float], end: tuple[float, float], fill: str) -> None:
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    length = 5.5
    spread = 0.42
    path = pdf.beginPath()
    path.moveTo(ex, ey)
    path.lineTo(ex - length * math.cos(angle - spread), ey - length * math.sin(angle - spread))
    path.lineTo(ex - length * math.cos(angle + spread), ey - length * math.sin(angle + spread))
    path.close()
    pdf.setFillColor(hex_to_color(fill))
    pdf.drawPath(path, stroke=0, fill=1)


def render_pdf() -> Path:
    font, bold = register_pdf_font()
    width = WIDTH_IN * 72
    height = HEIGHT_IN * 72
    out = HERE / "lora_observation_flow.pdf"
    pdf = canvas.Canvas(str(out), pagesize=(width, height))

    def x(value: float) -> float:
        return pdf_pt(value)

    def y(value: float) -> float:
        return height - pdf_pt(value)

    pdf.setFont(bold, 10.6)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(x(0.25), y(0.22), "Two observations from one standard LoRa packet")

    pdf.setFillColor(hex_to_color("#fff8f0"))
    pdf.setStrokeColor(hex_to_color("#f0c28d"))
    pdf.roundRect(x(3.32), y(1.28), pdf_pt(3.82), pdf_pt(0.90), 4.2, stroke=1, fill=1)
    pdf.setFillColor(hex_to_color("#f4fbf4"))
    pdf.setStrokeColor(hex_to_color("#a8d6a8"))
    pdf.roundRect(x(3.32), y(2.76), pdf_pt(3.82), pdf_pt(0.90), 4.2, stroke=1, fill=1)

    node_by_id = {node["id"]: node for node in NODES}
    for source_id, target_id, label, branch in EDGES:
        source = node_by_id[source_id]
        target = node_by_id[target_id]
        start, end = edge_points(source, target)
        color = ORANGE if branch == "rssi_plus" else GREEN if branch == "spectrum" else GRAY
        start_pdf = (x(start[0]), y(start[1]))
        end_pdf = (x(end[0]) - (3.0 if end[0] > start[0] else 0), y(end[1]))
        pdf.setStrokeColor(hex_to_color(color))
        pdf.setLineWidth(0.9)
        pdf.line(start_pdf[0], start_pdf[1], end_pdf[0], end_pdf[1])
        pdf_arrow_head(pdf, start_pdf, end_pdf, color)
        if label:
            pdf.setFont(font, 6.5)
            pdf.setFillColor(hex_to_color(GRAY))
            pdf.drawCentredString((start_pdf[0] + end_pdf[0]) / 2, (start_pdf[1] + end_pdf[1]) / 2 + 10, label)

    packet = find_node("packet")
    px, _ = center(packet)
    pdf.setStrokeColor(hex_to_color(LIGHT_GRAY))
    pdf.setLineWidth(0.7)
    pdf.line(x(px), y(0.99), x(px), y(2.08))

    for node in NODES:
        if node["type"] == "annotation":
            x0, y0 = x(node["x"]), y(node["y"] + node["h"])
            x1, y1 = x(node["x"] + node["w"]), y(node["y"])
            color = ORANGE if node["branch"] == "rssi_plus" else GREEN
            pdf_centered_text(pdf, (x0, y0, x1, y1), node["label"], bold, 8.1, color)
            continue
        x0, y0 = x(node["x"]), y(node["y"] + node["h"])
        w, h = pdf_pt(node["w"]), pdf_pt(node["h"])
        pdf.setFillColor(hex_to_color(node["fill"]))
        pdf.setStrokeColor(hex_to_color(node["stroke"]))
        pdf.setLineWidth(0.9)
        pdf.roundRect(x0, y0, w, h, 4.2, stroke=1, fill=1)
        use_font = bold if node["type"] in {"output", "feature"} else font
        pdf_centered_text(pdf, (x0, y0, x0 + w, y0 + h), node["label"], use_font, 7.4, BLACK)

    for node_id in ["tag", "gateway"]:
        node = find_node(node_id)
        x0, y0 = x(node["x"]), y(node["y"] + node["h"])
        pdf.setFillColor(hex_to_color(BLUE))
        pdf.rect(x0 + 7, y0 + 8, 10, 19, stroke=0, fill=1)
        pdf.roundRect(x0 + 10, y0 + 27, 5, 5, 1.5, stroke=0, fill=1)

    pdf.setFont(font, 6.5)
    pdf.setFillColor(hex_to_color(ORANGE))
    pdf.drawString(x(6.48), y(1.24), "p_R(l|x_R)")
    pdf.setFillColor(hex_to_color(GREEN))
    pdf.drawString(x(6.50), y(2.72), "x_S = |Y[k0-M:k0+M]|")

    pdf.showPage()
    pdf.save()
    return out


def write_data() -> Path:
    out = HERE / "lora_observation_flow_data.csv"
    fields = [
        "kind",
        "id",
        "source",
        "target",
        "label",
        "x_in",
        "y_in",
        "width_in",
        "height_in",
        "branch",
        "type",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for node in NODES:
            writer.writerow(
                {
                    "kind": "node",
                    "id": node["id"],
                    "source": "",
                    "target": "",
                    "label": node["label"].replace("\n", " "),
                    "x_in": node["x"],
                    "y_in": node["y"],
                    "width_in": node["w"],
                    "height_in": node["h"],
                    "branch": node["branch"],
                    "type": node["type"],
                }
            )
        for index, (source, target, label, branch) in enumerate(EDGES, start=1):
            writer.writerow(
                {
                    "kind": "edge",
                    "id": f"edge_{index}",
                    "source": source,
                    "target": target,
                    "label": label,
                    "x_in": "",
                    "y_in": "",
                    "width_in": "",
                    "height_in": "",
                    "branch": branch,
                    "type": "arrow",
                }
            )
    return out


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    write_data()
    render_png()
    render_pdf()


if __name__ == "__main__":
    main()
