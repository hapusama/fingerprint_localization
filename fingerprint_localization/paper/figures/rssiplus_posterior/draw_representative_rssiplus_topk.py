from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RSSI_CSV = (
    ROOT
    / "v2_output/20260623_from_raw/step1_rssi_confusion_28packet_shared_pl_01"
    / "01_paired_packet_features_28points.csv"
)

RSSI_PLUS_FEATURES = [
    "snr",
    "realtime_average_rssi",
    "median_rssi",
    "mode_rssi",
    "rssi_variance",
    "residual",
]

WIDTH_IN = 7.2
HEIGHT_IN = 3.45
DPI = 400
TOP_N = 8
TOP_K = 5

BLUE = "#1f77b4"
LIGHT_BLUE = "#b7d7ee"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
RED = "#d62728"
GRAY_BAR = "#cfcfcf"
GRID = "#d9d9d9"
GRAY = "#666666"
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


def read_rows() -> list[dict]:
    rows: list[dict] = []
    with RSSI_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed = {
                "file_name": row["file_name_rssi"],
                "packet_index": int(row["packet_index"]),
                "location_id": int(row["location_id"]),
            }
            for column in RSSI_PLUS_FEATURES:
                parsed[column] = float(row[column])
            rows.append(parsed)
    return rows


def standardization_stats(rows: list[dict]) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for column in RSSI_PLUS_FEATURES:
        values = [row[column] for row in rows]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def scaled_vector(row: dict, means: list[float], stds: list[float]) -> list[float]:
    return [
        (row[column] - means[index]) / stds[index]
        for index, column in enumerate(RSSI_PLUS_FEATURES)
    ]


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def candidate_distances(rows: list[dict], query_index: int) -> list[tuple[int, float]]:
    train = [row for index, row in enumerate(rows) if index != query_index]
    means, stds = standardization_stats(train)
    query_vector = scaled_vector(rows[query_index], means, stds)

    by_location: dict[int, list[list[float]]] = defaultdict(list)
    for row in train:
        by_location[int(row["location_id"])].append(scaled_vector(row, means, stds))

    distances: list[tuple[int, float]] = []
    for location, vectors in by_location.items():
        centroid = [
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(RSSI_PLUS_FEATURES))
        ]
        distances.append((location, euclidean(query_vector, centroid)))
    return sorted(distances, key=lambda item: item[1])


def estimate_temperature(rows: list[dict]) -> float:
    nearest = [candidate_distances(rows, index)[0][1] for index in range(len(rows))]
    nearest.sort()
    mid = len(nearest) // 2
    return nearest[mid] if len(nearest) % 2 else 0.5 * (nearest[mid - 1] + nearest[mid])


def compute_posterior(query_file: str, packet_index: int) -> tuple[list[dict], dict]:
    rows = read_rows()
    query_index = next(
        index
        for index, row in enumerate(rows)
        if row["file_name"] == query_file and row["packet_index"] == packet_index
    )
    query = rows[query_index]
    temperature = estimate_temperature(rows)
    distances = candidate_distances(rows, query_index)
    min_distance = distances[0][1]
    weights = [math.exp(-(distance - min_distance) / max(temperature, 1e-12)) for _, distance in distances]
    weight_sum = sum(weights)
    ranked = sorted(
        [
            {
                "reference_location": location,
                "rssi_plus_distance": distance,
                "posterior_probability": weight / weight_sum,
            }
            for (location, distance), weight in zip(distances, weights)
        ],
        key=lambda item: item["posterior_probability"],
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    predicted = int(ranked[0]["reference_location"])
    true_location = int(query["location_id"])
    top1 = ranked[0]
    top2 = ranked[1]
    meta = {
        "query_file": query_file,
        "packet_index": packet_index,
        "true_location": true_location,
        "predicted_location": predicted,
        "temperature_T": temperature,
        "top1_location": int(top1["reference_location"]),
        "top2_location": int(top2["reference_location"]),
        "top1_probability": float(top1["posterior_probability"]),
        "top2_probability": float(top2["posterior_probability"]),
        "top1_top2_gap": abs(float(top1["posterior_probability"]) - float(top2["posterior_probability"])),
        "feature_columns": ";".join(RSSI_PLUS_FEATURES),
    }
    return ranked, meta


def write_data(rows: list[dict], meta: dict, output_prefix: str) -> Path:
    out = HERE / f"{output_prefix}_data.csv"
    fields = [
        "query_file",
        "packet_index",
        "true_location",
        "predicted_location",
        "top1_location",
        "top2_location",
        "top1_probability",
        "top2_probability",
        "top1_top2_gap",
        "temperature_T",
        "reference_location",
        "rank",
        "rssi_plus_distance",
        "posterior_probability",
        "is_true_location",
        "is_rssi_plus_prediction",
        "is_top_k",
        "feature_columns",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **meta,
                    **row,
                    "is_true_location": int(row["reference_location"]) == int(meta["true_location"]),
                    "is_rssi_plus_prediction": int(row["reference_location"]) == int(meta["predicted_location"]),
                    "is_top_k": int(row["rank"]) <= TOP_K,
                }
            )
    return out


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def marker_star(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: str) -> None:
    points = []
    for index in range(10):
        radius = r if index % 2 == 0 else r * 0.42
        angle = -math.pi / 2 + index * math.pi / 5
        points.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
    draw.polygon(points, fill=fill, outline=WHITE)


def marker_triangle(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: str) -> None:
    draw.polygon(
        [(x, y - r), (x - r * 0.9, y + r * 0.8), (x + r * 0.9, y + r * 0.8)],
        fill=fill,
        outline=WHITE,
    )


def render_png(rows: list[dict], meta: dict, output_prefix: str) -> Path:
    width = int(WIDTH_IN * DPI)
    height = int(HEIGHT_IN * DPI)
    scale = DPI / 100
    image = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    title_font = load_font(int(10.6 * scale), bold=True)
    label_font = load_font(int(7.3 * scale))
    small_font = load_font(int(6.25 * scale))

    top = rows[:TOP_N]
    others_mass = sum(row["posterior_probability"] for row in rows[TOP_N:])
    top_k_mass = sum(row["posterior_probability"] for row in rows[:TOP_K])
    plot_rows = [
        {
            **row,
            "bar_label": str(row["reference_location"]),
            "is_other": False,
        }
        for row in top
    ]
    plot_rows.append(
        {
            "reference_location": -1,
            "rank": "",
            "posterior_probability": others_mass,
            "bar_label": "Others",
            "is_other": True,
        }
    )

    x0, x1 = 240, width - 120
    y0, y1 = 270, height - 265
    y_max = max(0.30, math.ceil(max(row["posterior_probability"] for row in plot_rows) * 1.25 / 0.05) * 0.05)
    slot = (x1 - x0) / len(plot_rows)
    bar_w = slot * 0.60

    def y_at(value: float) -> float:
        return y1 - value / y_max * (y1 - y0)

    draw.text((72 * scale, 23 * scale), "RSSI+ posterior support for the representative packet", font=title_font, fill=BLACK)
    subtitle = (
        f"query: loc {meta['true_location']}, packet {meta['packet_index']}; "
        f"Top-1/Top-2 gap={meta['top1_top2_gap']:.4f}, Top-{TOP_K} mass={top_k_mass:.2f}"
    )
    draw.text((72 * scale, 43 * scale), subtitle, font=small_font, fill=GRAY)

    tick = 0.0
    while tick <= y_max + 1e-9:
        y = y_at(tick)
        draw.line((x0, y, x1, y), fill=GRID, width=max(1, int(0.75 * scale)))
        label = f"{tick:.2f}"
        tw, th = text_size(draw, label, small_font)
        draw.text((x0 - tw - 8 * scale, y - th / 2), label, font=small_font, fill=GRAY)
        tick += 0.05

    draw.line((x0, y1, x1, y1), fill=BLACK, width=max(1, int(1.1 * scale)))
    draw.line((x0, y0, x0, y1), fill=BLACK, width=max(1, int(1.1 * scale)))

    for index, row in enumerate(plot_rows):
        x = x0 + slot * (index + 0.5)
        value = row["posterior_probability"]
        y = y_at(value)
        loc = int(row["reference_location"]) if row["reference_location"] != -1 else -1
        if row["is_other"]:
            fill = GRAY_BAR
        elif loc == int(meta["true_location"]):
            fill = GREEN
        elif loc == int(meta["top2_location"]) or loc == int(meta["top1_location"]):
            fill = ORANGE
        else:
            fill = LIGHT_BLUE
        draw.rectangle((x - bar_w / 2, y, x + bar_w / 2, y1), fill=fill)
        label = f"{value:.3f}" if value < 0.1 else f"{value:.2f}"
        tw, th = text_size(draw, label, small_font)
        draw.text((x - tw / 2, y - th - 4 * scale), label, font=small_font, fill=GRAY)
        tw, th = text_size(draw, row["bar_label"], small_font)
        draw.text((x - tw / 2, y1 + 10 * scale), row["bar_label"], font=small_font, fill=BLACK)
        if not row["is_other"] and loc == int(meta["predicted_location"]):
            marker_triangle(draw, x - 7 * scale, y - 17 * scale, 4.8 * scale, RED)
        if not row["is_other"] and loc == int(meta["true_location"]):
            marker_star(draw, x + 7 * scale, y - 17 * scale, 5.6 * scale, GREEN)

    draw.text((x1 - 480, y0 + 16 * scale), "green star: true location", font=small_font, fill=GRAY)
    draw.text((x1 - 480, y0 + 34 * scale), "red triangle: RSSI+ Top-1", font=small_font, fill=GRAY)

    xlabel = "Ranked candidate location"
    ylabel = "Posterior probability / mass"
    tw, _ = text_size(draw, xlabel, label_font)
    draw.text((x0 + (x1 - x0) / 2 - tw / 2, height - 100), xlabel, font=label_font, fill=BLACK)

    scratch = Image.new("RGBA", (420, 80), (255, 255, 255, 0))
    scratch_draw = ImageDraw.Draw(scratch)
    bbox = scratch_draw.textbbox((0, 0), ylabel, font=label_font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((4 - bbox[0], 4 - bbox[1]), ylabel, font=label_font, fill=BLACK)
    rotated = text_image.rotate(90, expand=True)
    image.alpha_composite(rotated, (70, int((y0 + y1 - rotated.height) / 2)))

    out = HERE / f"{output_prefix}.png"
    image.convert("RGB").save(out, dpi=(DPI, DPI))
    return out


def render_pdf(png_path: Path, output_prefix: str) -> Path:
    out = HERE / f"{output_prefix}.pdf"
    pdf = canvas.Canvas(str(out), pagesize=(WIDTH_IN * 72, HEIGHT_IN * 72))
    pdf.drawImage(ImageReader(str(png_path)), 0, 0, width=WIDTH_IN * 72, height=HEIGHT_IN * 72)
    pdf.showPage()
    pdf.save()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw representative RSSI+ Top-K posterior.")
    parser.add_argument("--query-file", default="2_0_37_11_2_16.txt")
    parser.add_argument("--packet", type=int, default=6)
    parser.add_argument("--output-prefix", default="representative_rssiplus_topk_loc37_pkt6")
    args = parser.parse_args()

    rows, meta = compute_posterior(args.query_file, args.packet)
    write_data(rows, meta, args.output_prefix)
    png = render_png(rows, meta, args.output_prefix)
    render_pdf(png, args.output_prefix)


if __name__ == "__main__":
    main()
