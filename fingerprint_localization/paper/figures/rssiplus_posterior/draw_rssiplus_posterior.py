from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import portrait
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_figures" / "rssiplus_posterior"

RSSI_PLUS_FEATURES = [
    "snr",
    "realtime_average_rssi",
    "median_rssi",
    "mode_rssi",
    "rssi_variance",
    "residual",
]

SCOPES = {
    "54points": ROOT
    / "v2_output/20260623_from_raw/data_processing/rssi_plus_packet_level_54points.csv",
    "28points": ROOT
    / "v2_output/20260623_from_raw/step1_rssi_confusion_28packet_shared_pl_01/01_paired_packet_features_28points.csv",
}

QUERY_FILE = "2_1_23_11_2_16.txt"
QUERY_PACKET_INDEX = 16
TOP_K = 5

WIDTH_IN = 7.2
HEIGHT_IN = 4.2
DPI = 400

BLUE = "#1f77b4"
LIGHT_BLUE = "#b7d7ee"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY = "#555555"
GRID = "#d9d9d9"
BLACK = "#111111"


@dataclass
class PosteriorResult:
    scope: str
    source_csv: Path
    query_file: str
    query_packet_index: int
    true_location: int
    predicted_location: int
    temperature: float
    rows: list[dict]


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


def register_pdf_font() -> str:
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if Path(regular).exists():
        pdfmetrics.registerFont(TTFont("PlotArial", regular))
    if Path(bold).exists():
        pdfmetrics.registerFont(TTFont("PlotArial-Bold", bold))
    return "PlotArial" if Path(regular).exists() else "Helvetica"


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            file_name = row.get("file_name") or row.get("file_name_rssi")
            if not file_name:
                continue
            try:
                parsed = {
                    "file_name": file_name,
                    "packet_index": int(row["packet_index"]),
                    "location_id": int(row["location_id"]),
                }
                for column in RSSI_PLUS_FEATURES:
                    parsed[column] = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(parsed)
    return rows


def standardization_stats(rows: list[dict]) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for column in RSSI_PLUS_FEATURES:
        values = [float(row[column]) for row in rows]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(var) or 1.0
        means.append(mean)
        stds.append(std)
    return means, stds


def scaled_vector(row: dict, means: list[float], stds: list[float]) -> list[float]:
    return [
        (float(row[column]) - means[index]) / stds[index]
        for index, column in enumerate(RSSI_PLUS_FEATURES)
    ]


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def candidate_distances(rows: list[dict], query_index: int) -> list[tuple[int, float]]:
    query = rows[query_index]
    train = [row for index, row in enumerate(rows) if index != query_index]
    means, stds = standardization_stats(train)
    query_vector = scaled_vector(query, means, stds)

    by_location: dict[int, list[list[float]]] = {}
    for row in train:
        by_location.setdefault(int(row["location_id"]), []).append(
            scaled_vector(row, means, stds)
        )

    distances: list[tuple[int, float]] = []
    for location, vectors in by_location.items():
        centroid = [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(RSSI_PLUS_FEATURES))
        ]
        distances.append((location, euclidean(query_vector, centroid)))
    return sorted(distances)


def estimate_temperature(rows: list[dict]) -> float:
    nearest: list[float] = []
    for query_index in range(len(rows)):
        distances = candidate_distances(rows, query_index)
        if distances:
            nearest.append(min(distance for _, distance in distances))
    nearest.sort()
    if not nearest:
        return 1.0
    mid = len(nearest) // 2
    return nearest[mid] if len(nearest) % 2 else 0.5 * (nearest[mid - 1] + nearest[mid])


def compute_posterior(scope: str, path: Path) -> PosteriorResult:
    rows = read_rows(path)
    query_index = next(
        index
        for index, row in enumerate(rows)
        if row["file_name"] == QUERY_FILE and row["packet_index"] == QUERY_PACKET_INDEX
    )
    query = rows[query_index]
    temperature = estimate_temperature(rows)
    distances = candidate_distances(rows, query_index)
    min_distance = min(distance for _, distance in distances)
    weights = [
        math.exp(-(distance - min_distance) / max(temperature, 1e-12))
        for _, distance in distances
    ]
    weight_sum = sum(weights)
    ranked = sorted(
        [
            {
                "scope": scope,
                "source_csv": str(path),
                "query_file": query["file_name"],
                "query_packet_index": query["packet_index"],
                "true_location": int(query["location_id"]),
                "reference_location": int(location),
                "plot_index": index + 1,
                "rssi_plus_distance": float(distance),
                "posterior_probability": float(weight / weight_sum),
                "temperature_T": float(temperature),
            }
            for index, ((location, distance), weight) in enumerate(zip(distances, weights))
        ],
        key=lambda item: item["posterior_probability"],
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
        item["is_top_k"] = rank <= TOP_K

    predicted_location = int(ranked[0]["reference_location"])
    by_location = {int(item["reference_location"]): item for item in ranked}
    plot_rows = [by_location[location] for location, _ in sorted(distances)]
    for item in plot_rows:
        item["predicted_location"] = predicted_location
        item["is_true_location"] = int(item["reference_location"]) == int(query["location_id"])
        item["is_rssi_plus_prediction"] = int(item["reference_location"]) == predicted_location
        item["feature_columns"] = ";".join(RSSI_PLUS_FEATURES)

    return PosteriorResult(
        scope=scope,
        source_csv=path,
        query_file=query["file_name"],
        query_packet_index=int(query["packet_index"]),
        true_location=int(query["location_id"]),
        predicted_location=predicted_location,
        temperature=float(temperature),
        rows=plot_rows,
    )


def nice_ymax(max_prob: float) -> float:
    return max(0.20, math.ceil(max_prob * 1.18 / 0.05) * 0.05)


def write_csv_table(result: PosteriorResult) -> Path:
    path = OUT / f"rssiplus_posterior_{result.scope}_data.csv"
    fields = [
        "scope",
        "source_csv",
        "query_file",
        "query_packet_index",
        "true_location",
        "predicted_location",
        "reference_location",
        "plot_index",
        "rssi_plus_distance",
        "posterior_probability",
        "rank",
        "is_top_k",
        "is_true_location",
        "is_rssi_plus_prediction",
        "temperature_T",
        "feature_columns",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result.rows:
            writer.writerow({field: row[field] for field in fields})
    return path


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_rotated_text(
    image: Image.Image,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    angle: int = 90,
    anchor: str = "center",
) -> None:
    scratch = Image.new("RGBA", (120, 60), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), text, font=font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), text, font=font, fill=fill)
    rotated = text_image.rotate(angle, expand=True)
    x, y = xy
    if anchor == "center":
        x -= rotated.width / 2
        y -= rotated.height / 2
    image.alpha_composite(rotated, (int(x), int(y)))


def marker_star(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: str) -> None:
    points = []
    for index in range(10):
        radius = r if index % 2 == 0 else r * 0.42
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    draw.polygon(points, fill=fill, outline="white")


def marker_triangle(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: str) -> None:
    draw.polygon([(x, y - r), (x - r * 0.9, y + r * 0.8), (x + r * 0.9, y + r * 0.8)], fill=fill, outline="white")


def render_png(result: PosteriorResult, y_max: float) -> Path:
    width = int(WIDTH_IN * DPI)
    height = int(HEIGHT_IN * DPI)
    scale = DPI / 100
    image = Image.new("RGBA", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(int(8.5 * scale))
    small_font = load_font(int(7.2 * scale))
    title_font = load_font(int(10.5 * scale), bold=True)
    label_font = load_font(int(9 * scale))

    left = int(72 * scale)
    right = int(24 * scale)
    top = int(42 * scale)
    bottom = int(94 * scale)
    plot_left, plot_top = left, top
    plot_right, plot_bottom = width - right, height - bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    values = [row["posterior_probability"] for row in result.rows]
    locations = [row["reference_location"] for row in result.rows]
    count = len(result.rows)
    slot = plot_width / count
    bar_width = max(2, slot * 0.58)

    def x_at(index: int) -> float:
        return plot_left + slot * (index + 0.5)

    def y_at(value: float) -> float:
        return plot_bottom - value / y_max * plot_height

    for tick in [i * 0.05 for i in range(int(y_max / 0.05) + 1)]:
        y = y_at(tick)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=max(1, int(scale)))
        label = f"{tick:.2f}".rstrip("0").rstrip(".")
        tw, th = text_size(draw, label, small_font)
        draw.text((plot_left - tw - 8 * scale, y - th / 2), label, font=small_font, fill=GRAY)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=BLACK, width=max(1, int(1.2 * scale)))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=BLACK, width=max(1, int(1.2 * scale)))

    for index, value in enumerate(values):
        x = x_at(index)
        y = y_at(value)
        row = result.rows[index]
        fill = LIGHT_BLUE
        if row["is_top_k"]:
            fill = "#ffd8a8"
        draw.rectangle((x - bar_width / 2, y, x + bar_width / 2, plot_bottom), fill=fill, outline=None)

    points = [(x_at(index), y_at(value)) for index, value in enumerate(values)]
    draw.line(points, fill=BLUE, width=max(2, int(1.7 * scale)), joint="curve")
    for x, y in points:
        draw.ellipse((x - 2.1 * scale, y - 2.1 * scale, x + 2.1 * scale, y + 2.1 * scale), fill=BLUE)

    top_rows = [row for row in result.rows if row["is_top_k"]]
    for row in top_rows:
        index = int(row["plot_index"]) - 1
        x = x_at(index)
        y = y_at(row["posterior_probability"])
        draw.ellipse((x - 4.7 * scale, y - 4.7 * scale, x + 4.7 * scale, y + 4.7 * scale), fill=ORANGE, outline="white", width=max(1, int(scale)))

    for row in result.rows:
        index = int(row["plot_index"]) - 1
        x = x_at(index)
        y = y_at(row["posterior_probability"])
        if row["is_rssi_plus_prediction"]:
            for dash_y in range(int(plot_top), int(plot_bottom), int(12 * scale)):
                draw.line((x, dash_y, x, min(plot_bottom, dash_y + 6 * scale)), fill=RED, width=max(1, int(1.2 * scale)))
            marker_triangle(draw, x, y - 11 * scale, 5.8 * scale, RED)
        if row["is_true_location"]:
            for dash_y in range(int(plot_top), int(plot_bottom), int(12 * scale)):
                draw.line((x + 2 * scale, dash_y, x + 2 * scale, min(plot_bottom, dash_y + 6 * scale)), fill=GREEN, width=max(1, int(1.2 * scale)))
            marker_star(draw, x + 2 * scale, y - 4 * scale, 7.0 * scale, GREEN)

    for index, location in enumerate(locations):
        x = x_at(index)
        draw.line((x, plot_bottom, x, plot_bottom + 3 * scale), fill=BLACK, width=max(1, int(scale)))
        draw_rotated_text(image, (x, plot_bottom + 26 * scale), str(location), small_font, BLACK)

    title = f"RSSI+ posterior distribution ({result.scope.replace('points', ' reference points')})"
    draw.text((plot_left, int(13 * scale)), title, font=title_font, fill=BLACK)
    subtitle = (
        f"query: loc {result.true_location}, packet {result.query_packet_index}; "
        f"T={result.temperature:.3f}; Top-{TOP_K} highlighted"
    )
    draw.text((plot_left, int(28 * scale)), subtitle, font=small_font, fill=GRAY)

    xlabel = "Reference location ID"
    ylabel = "Posterior probability  p_R(l | x_R)"
    tw, th = text_size(draw, xlabel, label_font)
    draw.text((plot_left + plot_width / 2 - tw / 2, height - 28 * scale), xlabel, font=label_font, fill=BLACK)
    draw_rotated_text(image, (24 * scale, plot_top + plot_height / 2), ylabel, label_font, BLACK, angle=90)

    legend_x = plot_right - 220 * scale
    legend_y = plot_top + 8 * scale
    legend_items = [
        ("distribution", BLUE),
        (f"Top-{TOP_K} candidates", ORANGE),
        ("true location", GREEN),
        ("RSSI+ prediction", RED),
    ]
    for idx, (text, color) in enumerate(legend_items):
        y = legend_y + idx * 17 * scale
        if idx == 0:
            draw.line((legend_x, y + 5 * scale, legend_x + 22 * scale, y + 5 * scale), fill=color, width=max(2, int(1.8 * scale)))
        elif idx == 1:
            draw.ellipse((legend_x + 7 * scale, y, legend_x + 17 * scale, y + 10 * scale), fill=color, outline="white")
        elif idx == 2:
            marker_star(draw, legend_x + 12 * scale, y + 5 * scale, 5 * scale, color)
        else:
            marker_triangle(draw, legend_x + 12 * scale, y + 5 * scale, 4.8 * scale, color)
        draw.text((legend_x + 30 * scale, y - 2 * scale), text, font=small_font, fill=BLACK)

    out = OUT / f"rssiplus_posterior_{result.scope}.png"
    image.convert("RGB").save(out, dpi=(DPI, DPI))
    return out


def hex_to_color(value: str) -> colors.Color:
    value = value.lstrip("#")
    return colors.Color(int(value[0:2], 16) / 255, int(value[2:4], 16) / 255, int(value[4:6], 16) / 255)


def render_pdf(result: PosteriorResult, y_max: float) -> Path:
    font_name = register_pdf_font()
    font_bold = "PlotArial-Bold" if "PlotArial-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
    width = WIDTH_IN * 72
    height = HEIGHT_IN * 72
    out = OUT / f"rssiplus_posterior_{result.scope}.pdf"
    pdf = canvas.Canvas(str(out), pagesize=portrait((width, height)))
    pdf.setLineJoin(1)

    left, right, top, bottom = 72 / 100 * 72, 24 / 100 * 72, 42 / 100 * 72, 94 / 100 * 72
    plot_left, plot_top = left, height - top
    plot_right, plot_bottom = width - right, bottom
    plot_width = plot_right - plot_left
    plot_height = plot_top - plot_bottom
    count = len(result.rows)
    slot = plot_width / count
    bar_width = max(1.0, slot * 0.58)

    def x_at(index: int) -> float:
        return plot_left + slot * (index + 0.5)

    def y_at(value: float) -> float:
        return plot_bottom + value / y_max * plot_height

    pdf.setFont(font_bold, 10.5)
    pdf.setFillColor(hex_to_color(BLACK))
    pdf.drawString(plot_left, height - 17, f"RSSI+ posterior distribution ({result.scope.replace('points', ' reference points')})")
    pdf.setFont(font_name, 7.2)
    pdf.setFillColor(hex_to_color(GRAY))
    pdf.drawString(
        plot_left,
        height - 31,
        f"query: loc {result.true_location}, packet {result.query_packet_index}; T={result.temperature:.3f}; Top-{TOP_K} highlighted",
    )

    pdf.setFont(font_name, 7.2)
    for tick_index in range(int(y_max / 0.05) + 1):
        tick = tick_index * 0.05
        y = y_at(tick)
        pdf.setStrokeColor(hex_to_color(GRID))
        pdf.setLineWidth(0.5)
        pdf.line(plot_left, y, plot_right, y)
        pdf.setFillColor(hex_to_color(GRAY))
        label = f"{tick:.2f}".rstrip("0").rstrip(".")
        pdf.drawRightString(plot_left - 5, y - 2.4, label)

    pdf.setStrokeColor(hex_to_color(BLACK))
    pdf.setLineWidth(0.8)
    pdf.line(plot_left, plot_bottom, plot_right, plot_bottom)
    pdf.line(plot_left, plot_bottom, plot_left, plot_top)

    for index, row in enumerate(result.rows):
        x = x_at(index)
        y = y_at(row["posterior_probability"])
        pdf.setFillColor(hex_to_color("#ffd8a8" if row["is_top_k"] else LIGHT_BLUE))
        pdf.rect(x - bar_width / 2, plot_bottom, bar_width, y - plot_bottom, stroke=0, fill=1)

    pdf.setStrokeColor(hex_to_color(BLUE))
    pdf.setLineWidth(1.2)
    points = [(x_at(index), y_at(row["posterior_probability"])) for index, row in enumerate(result.rows)]
    path = pdf.beginPath()
    path.moveTo(points[0][0], points[0][1])
    for x, y in points[1:]:
        path.lineTo(x, y)
    pdf.drawPath(path, stroke=1, fill=0)
    pdf.setFillColor(hex_to_color(BLUE))
    for x, y in points:
        pdf.circle(x, y, 1.5, stroke=0, fill=1)

    for row in result.rows:
        index = int(row["plot_index"]) - 1
        x = x_at(index)
        y = y_at(row["posterior_probability"])
        if row["is_top_k"]:
            pdf.setFillColor(hex_to_color(ORANGE))
            pdf.setStrokeColor(colors.white)
            pdf.circle(x, y, 3.4, stroke=1, fill=1)
        if row["is_rssi_plus_prediction"]:
            pdf.setStrokeColor(hex_to_color(RED))
            pdf.setDash(4, 4)
            pdf.line(x, plot_bottom, x, plot_top)
            pdf.setDash()
            pdf.setFillColor(hex_to_color(RED))
            pdf.setStrokeColor(colors.white)
            pdf.line(x, y + 8, x - 4, y + 14)
            pdf.line(x - 4, y + 14, x + 4, y + 14)
            pdf.line(x + 4, y + 14, x, y + 8)
        if row["is_true_location"]:
            pdf.setStrokeColor(hex_to_color(GREEN))
            pdf.setDash(4, 4)
            pdf.line(x + 1.5, plot_bottom, x + 1.5, plot_top)
            pdf.setDash()
            pdf.setFillColor(hex_to_color(GREEN))
            pdf.setStrokeColor(colors.white)
            star = pdf.beginPath()
            pts = []
            for idx in range(10):
                radius = 5.0 if idx % 2 == 0 else 2.1
                angle = -math.pi / 2 + idx * math.pi / 5
                pts.append((x + 1.5 + radius * math.cos(angle), y + 3 + radius * math.sin(angle)))
            star.moveTo(pts[0][0], pts[0][1])
            for px, py in pts[1:]:
                star.lineTo(px, py)
            star.close()
            pdf.drawPath(star, stroke=1, fill=1)

    pdf.setFont(font_name, 7.0)
    pdf.setFillColor(hex_to_color(BLACK))
    for index, row in enumerate(result.rows):
        x = x_at(index)
        pdf.setStrokeColor(hex_to_color(BLACK))
        pdf.line(x, plot_bottom, x, plot_bottom - 2.5)
        pdf.saveState()
        pdf.translate(x + 2.2, plot_bottom - 7)
        pdf.rotate(90)
        pdf.drawString(0, 0, str(row["reference_location"]))
        pdf.restoreState()

    pdf.setFont(font_name, 9.0)
    pdf.drawCentredString(plot_left + plot_width / 2, 19, "Reference location ID")
    pdf.saveState()
    pdf.translate(17, plot_bottom + plot_height / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Posterior probability  p_R(l | x_R)")
    pdf.restoreState()

    legend_x = plot_right - 158
    legend_y = plot_top - 15
    legend_items = [
        ("distribution", BLUE),
        (f"Top-{TOP_K} candidates", ORANGE),
        ("true location", GREEN),
        ("RSSI+ prediction", RED),
    ]
    pdf.setFont(font_name, 7.4)
    for idx, (text, color) in enumerate(legend_items):
        y = legend_y - idx * 12
        pdf.setFillColor(hex_to_color(color))
        pdf.setStrokeColor(hex_to_color(color))
        if idx == 0:
            pdf.setLineWidth(1.4)
            pdf.line(legend_x, y, legend_x + 16, y)
        elif idx == 1:
            pdf.circle(legend_x + 8, y, 3.2, stroke=0, fill=1)
        elif idx == 2:
            pdf.circle(legend_x + 8, y, 3.2, stroke=0, fill=1)
        else:
            pdf.line(legend_x + 8, y + 3.2, legend_x + 4.5, y - 3.2)
            pdf.line(legend_x + 4.5, y - 3.2, legend_x + 11.5, y - 3.2)
            pdf.line(legend_x + 11.5, y - 3.2, legend_x + 8, y + 3.2)
        pdf.setFillColor(hex_to_color(BLACK))
        pdf.drawString(legend_x + 22, y - 2.5, text)

    pdf.showPage()
    pdf.save()
    return out


def write_summary(results: list[PosteriorResult]) -> Path:
    path = OUT / "rssiplus_posterior_summary.csv"
    fields = [
        "scope",
        "source_csv",
        "query_file",
        "query_packet_index",
        "true_location",
        "predicted_location",
        "temperature_T",
        "reference_location_count",
        "top_k",
        "top_k_locations",
        "top_k_probabilities",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            ranked = sorted(result.rows, key=lambda row: row["rank"])
            top = ranked[:TOP_K]
            writer.writerow(
                {
                    "scope": result.scope,
                    "source_csv": str(result.source_csv),
                    "query_file": result.query_file,
                    "query_packet_index": result.query_packet_index,
                    "true_location": result.true_location,
                    "predicted_location": result.predicted_location,
                    "temperature_T": result.temperature,
                    "reference_location_count": len(result.rows),
                    "top_k": TOP_K,
                    "top_k_locations": ";".join(str(row["reference_location"]) for row in top),
                    "top_k_probabilities": ";".join(f"{row['posterior_probability']:.8f}" for row in top),
                }
            )
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = [compute_posterior(scope, path) for scope, path in SCOPES.items()]
    y_max = nice_ymax(max(row["posterior_probability"] for result in results for row in result.rows))
    for result in results:
        write_csv_table(result)
        render_png(result, y_max)
        render_pdf(result, y_max)
    write_summary(results)


if __name__ == "__main__":
    main()
