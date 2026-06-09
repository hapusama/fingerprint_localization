#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a spectrogram for the received LFM upchirp test capture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DONG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = DONG_ROOT / "outputs" / "captures" / "chime_test_rx_fc32.bin"
DEFAULT_OUTPUT = DONG_ROOT / "outputs" / "analysis" / "chime_test_spectrogram.png"


def viridis_rgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    stops = np.array(
        [
            [0.267, 0.005, 0.329],
            [0.283, 0.141, 0.458],
            [0.254, 0.265, 0.530],
            [0.207, 0.372, 0.553],
            [0.164, 0.471, 0.558],
            [0.128, 0.567, 0.551],
            [0.135, 0.659, 0.518],
            [0.267, 0.749, 0.441],
            [0.478, 0.821, 0.318],
            [0.741, 0.873, 0.150],
            [0.993, 0.906, 0.144],
        ],
        dtype=np.float32,
    )
    pos = values * (len(stops) - 1)
    idx = np.floor(pos).astype(np.int32)
    idx = np.clip(idx, 0, len(stops) - 2)
    frac = (pos - idx)[..., None]
    rgb = stops[idx] * (1.0 - frac) + stops[idx + 1] * frac
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def render(args: argparse.Namespace) -> Path:
    infile = Path(args.infile)
    if not infile.exists():
        raise FileNotFoundError(infile)

    fs = float(args.fs)
    start = max(0, int(round(args.skip * fs)))
    count = max(1, int(round(args.duration * fs)))
    src = np.memmap(infile, dtype=np.complex64, mode="r")
    stop = min(src.size, start + count)
    if stop - start < 32:
        raise ValueError("Not enough samples to render")
    x = np.asarray(src[start:stop], dtype=np.complex64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    nperseg = min(int(args.nperseg), x.size)
    hop = max(1, int(args.hop))
    nfft = 1 << int(np.ceil(np.log2(max(nperseg, int(args.nfft)))))
    starts = np.arange(0, x.size - nperseg + 1, hop)
    if starts.size == 0:
        starts = np.array([0])
    window = np.hanning(nperseg).astype(np.float32)
    frames = np.empty((starts.size, nperseg), dtype=np.complex64)
    for row, frame_start in enumerate(starts):
        frames[row] = x[frame_start : frame_start + nperseg] * window

    spec = np.abs(np.fft.fft(frames, n=nfft, axis=1)).T
    spec = np.fft.fftshift(spec, axes=0)
    spec = np.maximum(spec, 1e-15)
    spec_db = 20.0 * np.log10(spec / np.max(spec))
    spec_db = np.clip(spec_db, args.db_floor, args.db_ceil)
    norm = (spec_db - args.db_floor) / max(args.db_ceil - args.db_floor, 1e-6)
    rgb = np.flipud(viridis_rgb(norm))

    plot_w = int(args.width)
    plot_h = int(args.height)
    left = 92
    right = 126
    top = 58
    bottom = 64
    img_w = left + plot_w + right
    img_h = top + plot_h + bottom

    image = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    spec_img = Image.fromarray(rgb, mode="RGB").resize((plot_w, plot_h), resampling)
    image.paste(spec_img, (left, top))

    total_s = x.size / fs
    freq_min = -fs / 2.0
    freq_max = fs / 2.0

    def x_at(t_s: float) -> int:
        return left + int(round(t_s / total_s * plot_w))

    def y_at(freq_hz: float) -> int:
        frac = (freq_max - freq_hz) / (freq_max - freq_min)
        return top + int(round(frac * plot_h))

    axis_color = (18, 18, 18)
    grid_color = (225, 225, 225)
    for tick in np.linspace(0.0, total_s, 7):
        x_pos = x_at(float(tick))
        draw.line([(x_pos, top), (x_pos, top + plot_h)], fill=grid_color)
        label = f"{(args.skip + tick) * 1e3:.1f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x_pos - (bbox[2] - bbox[0]) / 2, top + plot_h + 9), label, fill=axis_color, font=font)
    for freq in np.linspace(freq_min, freq_max, 5):
        y_pos = y_at(float(freq))
        draw.line([(left, y_pos), (left + plot_w, y_pos)], fill=grid_color)
        label = f"{freq / 1e3:.0f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((left - 10 - (bbox[2] - bbox[0]), y_pos - 5), label, fill=axis_color, font=font)

    # Overlay the expected transmitted upchirp timing. It should align with the bright ridge.
    overlay_color = (255, 60, 60)
    period = float(args.period)
    chirp_duration = float(args.chirp_duration)
    chirp_bw = float(args.chirp_bw)
    first_period = int(np.floor(args.skip / period)) if period > 0 else 0
    periods = int(np.ceil((args.skip + total_s) / period)) + 1
    for period_idx in range(first_period, periods):
        t0_abs = period_idx * period
        t0 = t0_abs - args.skip
        t1 = t0 + chirp_duration
        if t1 < 0 or t0 > total_s:
            continue
        clipped_t0 = max(0.0, t0)
        clipped_t1 = min(total_s, t1)
        if clipped_t1 <= clipped_t0:
            continue
        frac0 = (clipped_t0 - t0) / chirp_duration
        frac1 = (clipped_t1 - t0) / chirp_duration
        f0 = -chirp_bw / 2.0 + frac0 * chirp_bw
        f1 = -chirp_bw / 2.0 + frac1 * chirp_bw
        draw.line([(x_at(clipped_t0), y_at(f0)), (x_at(clipped_t1), y_at(f1))], fill=overlay_color, width=2)
        draw.line([(x_at(clipped_t0), top), (x_at(clipped_t0), top + plot_h)], fill=(255, 255, 255), width=1)

    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=axis_color, width=1)
    title = "Received LFM Upchirp Spectrogram"
    subtitle = f"fs={fs / 1e6:.3f} Msps, chirp_bw={chirp_bw / 1e6:.3f} MHz, period={period * 1e3:.1f} ms"
    draw.text((left, 18), title, fill=axis_color, font=font)
    draw.text((left, 34), subtitle, fill=(70, 70, 70), font=font)
    draw.text((left + plot_w / 2 - 36, img_h - 30), "Time (ms)", fill=axis_color, font=font)
    draw.text((8, top + plot_h / 2 - 8), "Frequency (kHz)", fill=axis_color, font=font)

    cbar_x = left + plot_w + 36
    cbar_w = 20
    cbar_values = np.linspace(1.0, 0.0, plot_h, dtype=np.float32).reshape(plot_h, 1)
    cbar_img = Image.fromarray(viridis_rgb(cbar_values), mode="RGB").resize((cbar_w, plot_h), resampling)
    image.paste(cbar_img, (cbar_x, top))
    draw.rectangle([cbar_x, top, cbar_x + cbar_w, top + plot_h], outline=axis_color, width=1)
    for tick in np.linspace(args.db_floor, args.db_ceil, 6):
        y_pos = top + int(round((args.db_ceil - tick) / (args.db_ceil - args.db_floor) * plot_h))
        draw.line([(cbar_x + cbar_w, y_pos), (cbar_x + cbar_w + 5, y_pos)], fill=axis_color)
        draw.text((cbar_x + cbar_w + 9, y_pos - 5), f"{tick:.0f} dB", fill=axis_color, font=font)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a received chirp spectrogram PNG")
    parser.add_argument("--infile", default=str(DEFAULT_CAPTURE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fs", type=float, default=2e6)
    parser.add_argument("--chirp-bw", type=float, default=1e6)
    parser.add_argument("--chirp-duration", type=float, default=1e-3)
    parser.add_argument("--period", type=float, default=20e-3)
    parser.add_argument("--skip", type=float, default=0.0, help="Seconds to skip from the capture start")
    parser.add_argument("--duration", type=float, default=0.12, help="Seconds to render")
    parser.add_argument("--nperseg", type=int, default=512)
    parser.add_argument("--hop", type=int, default=64)
    parser.add_argument("--nfft", type=int, default=2048)
    parser.add_argument("--db-floor", type=float, default=-80.0)
    parser.add_argument("--db-ceil", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=520)
    return parser.parse_args()


def main() -> None:
    out = render(parse_args())
    print(f"[spectrogram] saved {out}")


if __name__ == "__main__":
    main()
