#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one repeated fc32 period for the Chime-style chirp TX GRC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DONG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DONG_ROOT / "inputs" / "chime_test_tx_period_fc32.bin"


def generate_waveform(
    output: str | Path = DEFAULT_OUTPUT,
    fs: float = 2e6,
    chirp_bw: float = 1e6,
    chirp_duration: float = 1e-3,
    period: float = 20e-3,
    amplitude: float = 0.2,
) -> dict[str, float | int | Path]:
    if chirp_bw <= 0 or chirp_bw >= fs:
        raise ValueError("chirp_bw must be positive and smaller than fs")
    if period < chirp_duration:
        raise ValueError("period must be >= chirp_duration")

    # 一个周期内只在前 chirp_duration 放 LFM，上升扫频；剩余时间补零作为保护间隔。
    n_chirp = max(1, int(round(fs * chirp_duration)))
    n_period = max(n_chirp, int(round(fs * period)))
    t = np.arange(n_chirp, dtype=np.float64) / fs
    sweep_rate = chirp_bw / chirp_duration
    phase = 2.0 * np.pi * ((-chirp_bw / 2.0) * t + 0.5 * sweep_rate * t * t)
    waveform = np.zeros(n_period, dtype=np.complex64)
    waveform[:n_chirp] = (amplitude * np.exp(1j * phase)).astype(np.complex64)

    # GNU Radio 的 file_source 会按 complex64/fc32 原样读取这个二进制文件。
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    waveform.tofile(out)
    return {
        "path": out,
        "samples": waveform.size,
        "bytes": out.stat().st_size,
        "chirp_samples": n_chirp,
        "peak_amplitude": float(np.max(np.abs(waveform))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 20 ms fc32 LFM upchirp period")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fs", type=float, default=2e6)
    parser.add_argument("--chirp-bw", type=float, default=1e6)
    parser.add_argument("--chirp-duration", type=float, default=1e-3)
    parser.add_argument("--period", type=float, default=20e-3)
    parser.add_argument("--amplitude", type=float, default=0.2)
    args = parser.parse_args()

    stats = generate_waveform(
        output=args.output,
        fs=args.fs,
        chirp_bw=args.chirp_bw,
        chirp_duration=args.chirp_duration,
        period=args.period,
        amplitude=args.amplitude,
    )
    print(f"[waveform] wrote {stats['path']}")
    print(f"[waveform] samples: {stats['samples']}")
    print(f"[waveform] bytes: {stats['bytes']}")
    print(f"[waveform] chirp samples: {stats['chirp_samples']}")
    print(f"[waveform] peak amplitude: {stats['peak_amplitude']:.6f}")


if __name__ == "__main__":
    main()
