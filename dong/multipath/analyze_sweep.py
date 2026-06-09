#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze stepped-frequency captures into frequency response and CIR/PDP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from waveforms import correlation_channel_estimate


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(x), 1e-15))


def db10(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x), 1e-30))


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        out = {key: data[key] for key in data.files}
    metadata = out.get("metadata", "{}")
    if isinstance(metadata, np.ndarray):
        metadata = metadata.item() if metadata.shape == () else "{}"
    out["metadata_dict"] = json.loads(str(metadata))
    return out


def estimate_response(captures: np.ndarray, probe: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.zeros(captures.shape[0], dtype=np.complex128)
    peak_index = np.zeros(captures.shape[0], dtype=np.int64)
    metric = np.zeros(captures.shape[0], dtype=np.float64)
    for index, samples in enumerate(captures):
        h[index], peak_index[index], metric[index] = correlation_channel_estimate(samples, probe)
    return h, peak_index, metric


def synthetic_cir(freqs: np.ndarray, h: np.ndarray, n_fft: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(freqs)
    freqs = freqs[order]
    h = h[order]
    if freqs.size < 2:
        raise ValueError("at least two frequency steps are required")
    step = float(np.median(np.diff(freqs)))
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(256, freqs.size * 8))))
    window = np.hanning(freqs.size)
    cir = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(h * window, axes=0), n=n_fft), axes=0)
    delays = (np.arange(n_fft) - n_fft // 2) / (n_fft * step)
    return delays, cir


def write_frequency_csv(path: Path, freqs: np.ndarray, h: np.ndarray, peak_index: np.ndarray, metric: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frequency_hz", "magnitude_db", "phase_rad", "peak_index", "match_metric"])
        for freq, value, peak, score in zip(freqs, h, peak_index, metric):
            writer.writerow([f"{freq:.6f}", f"{db20(abs(value)):.6f}", f"{np.angle(value):.9f}", int(peak), f"{score:.9f}"])


def write_cir_csv(path: Path, delays: np.ndarray, cir: np.ndarray) -> None:
    power = np.abs(cir) ** 2
    power_db = db10(power / max(float(np.max(power)), 1e-30))
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["delay_s", "delay_ns", "relative_power_db", "magnitude", "phase_rad"])
        for delay, db, value in zip(delays, power_db, cir):
            writer.writerow([f"{delay:.12e}", f"{delay * 1e9:.6f}", f"{db:.6f}", f"{abs(value):.12e}", f"{np.angle(value):.9f}"])


def simple_plot(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    color: tuple[int, int, int] = (16, 92, 160),
) -> None:
    width, height = 1100, 520
    left, right, top, bottom = 92, 38, 58, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        raise ValueError("nothing to plot")
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    pad = 0.08 * (ymax - ymin)
    ymin -= pad
    ymax += pad

    axis = (30, 30, 30)
    grid = (225, 225, 225)
    draw.text((left, 20), title, fill=axis, font=font)
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=axis)

    for i in range(6):
        frac = i / 5
        xpix = left + int(round(frac * plot_w))
        ypix = top + plot_h - int(round(frac * plot_h))
        draw.line([(xpix, top), (xpix, top + plot_h)], fill=grid)
        draw.line([(left, ypix), (left + plot_w, ypix)], fill=grid)
        xv = xmin + frac * (xmax - xmin)
        yv = ymin + frac * (ymax - ymin)
        draw.text((xpix - 24, top + plot_h + 9), f"{xv:g}", fill=axis, font=font)
        draw.text((8, ypix - 6), f"{yv:g}", fill=axis, font=font)

    points = []
    for xv, yv in zip(x, y):
        px = left + int(round((xv - xmin) / (xmax - xmin) * plot_w))
        py = top + plot_h - int(round((yv - ymin) / (ymax - ymin) * plot_h))
        points.append((px, py))
    if len(points) >= 2:
        draw.line(points, fill=color, width=2)
    for px, py in points:
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)
    draw.text((left + plot_w // 2 - 40, height - 30), x_label, fill=axis, font=font)
    draw.text((10, top - 25), y_label, fill=axis, font=font)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate channel response and synthetic CIR from a sweep npz")
    parser.add_argument("--input", required=True, help="Measurement .npz from capture_sweep.py")
    parser.add_argument("--calibration", help="Optional loopback calibration .npz")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/PNG/summary outputs")
    parser.add_argument("--nfft", type=int, default=0, help="CIR IFFT length. 0 chooses automatically")
    parser.add_argument("--max-delay-us", type=float, default=8.0, help="PDP plot delay window around 0 in us")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(in_path)
    freqs = np.asarray(data["frequencies_hz"], dtype=np.float64)
    captures = np.asarray(data["captures"], dtype=np.complex64)
    probe = np.asarray(data["probe"], dtype=np.complex64)
    h, peak_index, metric = estimate_response(captures, probe)

    calibration_used = False
    if args.calibration:
        cal = load_npz(Path(args.calibration))
        cal_freqs = np.asarray(cal["frequencies_hz"], dtype=np.float64)
        if cal_freqs.size != freqs.size or np.max(np.abs(cal_freqs - freqs)) > 1e-3:
            raise ValueError("calibration frequencies do not match the measurement sweep")
        h_cal, _, metric_cal = estimate_response(np.asarray(cal["captures"], dtype=np.complex64), np.asarray(cal["probe"], dtype=np.complex64))
        h_cal_safe = np.where(np.abs(h_cal) > 1e-15, h_cal, 1e-15 + 0j)
        h = h / h_cal_safe
        metric = np.minimum(metric, metric_cal)
        calibration_used = True

    n_fft = args.nfft if args.nfft > 0 else None
    delays, cir = synthetic_cir(freqs, h, n_fft)
    power = np.abs(cir) ** 2
    rel_db = db10(power / max(float(np.max(power)), 1e-30))
    dominant = np.argsort(power)[-8:][::-1]

    write_frequency_csv(out_dir / "frequency_response.csv", freqs, h, peak_index, metric)
    write_cir_csv(out_dir / "cir.csv", delays, cir)

    simple_plot(
        out_dir / "frequency_response.png",
        freqs / 1e6,
        db20(np.abs(h)),
        "Synthetic Sweep Frequency Response",
        "Frequency (MHz)",
        "Magnitude (dB)",
    )
    mask = np.abs(delays * 1e6) <= args.max_delay_us
    simple_plot(
        out_dir / "pdp.png",
        delays[mask] * 1e6,
        rel_db[mask],
        "Synthetic-Bandwidth Power Delay Profile",
        "Delay (us)",
        "Relative power (dB)",
        color=(170, 64, 32),
    )

    step_hz = float(np.median(np.diff(np.sort(freqs))))
    span_hz = float(np.max(freqs) - np.min(freqs) + step_hz)
    summary = {
        "input": str(in_path),
        "calibration": args.calibration or "",
        "calibration_used": calibration_used,
        "frequency_steps": int(freqs.size),
        "step_hz": step_hz,
        "synthetic_span_hz": span_hz,
        "nominal_delay_resolution_s": 1.0 / span_hz,
        "nominal_delay_resolution_ns": 1.0e9 / span_hz,
        "unambiguous_delay_s": 1.0 / step_hz,
        "unambiguous_delay_us": 1.0e6 / step_hz,
        "dominant_taps": [
            {
                "delay_ns": float(delays[i] * 1e9),
                "relative_power_db": float(rel_db[i]),
                "magnitude": float(abs(cir[i])),
                "phase_rad": float(np.angle(cir[i])),
            }
            for i in dominant
        ],
        "mean_match_metric": float(np.mean(metric)),
        "min_match_metric": float(np.min(metric)),
        "source_metadata": data.get("metadata_dict", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[analyze] wrote {out_dir}")
    print(f"[analyze] synthetic span: {span_hz / 1e6:.3f} MHz")
    print(f"[analyze] nominal delay resolution: {summary['nominal_delay_resolution_ns']:.1f} ns")
    print(f"[analyze] unambiguous delay: {summary['unambiguous_delay_us']:.2f} us")
    print(f"[analyze] calibration used: {calibration_used}")
    for tap in summary["dominant_taps"][:4]:
        print(f"[tap] {tap['delay_ns']:9.1f} ns  {tap['relative_power_db']:7.2f} dB")


if __name__ == "__main__":
    main()
