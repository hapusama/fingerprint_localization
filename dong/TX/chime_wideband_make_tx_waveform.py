#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate one repeated fc32 period for the Chime-style chirp TX GRC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DONG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DONG_ROOT / "inputs" / "chime_test_tx_period_fc32.bin"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 20 ms fc32 LFM upchirp period")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fs", type=float, default=2e6)
    parser.add_argument("--chirp-bw", type=float, default=1e6)
    parser.add_argument("--chirp-duration", type=float, default=1e-3)
    parser.add_argument("--period", type=float, default=20e-3)
    parser.add_argument("--amplitude", type=float, default=0.2)
    args = parser.parse_args()

    if args.chirp_bw <= 0 or args.chirp_bw >= args.fs:
        raise ValueError("chirp_bw must be positive and smaller than fs")
    if args.period < args.chirp_duration:
        raise ValueError("period must be >= chirp_duration")

    n_chirp = max(1, int(round(args.fs * args.chirp_duration)))
    n_period = max(n_chirp, int(round(args.fs * args.period)))
    t = np.arange(n_chirp, dtype=np.float64) / args.fs
    sweep_rate = args.chirp_bw / args.chirp_duration
    phase = 2.0 * np.pi * ((-args.chirp_bw / 2.0) * t + 0.5 * sweep_rate * t * t)
    waveform = np.zeros(n_period, dtype=np.complex64)
    waveform[:n_chirp] = (args.amplitude * np.exp(1j * phase)).astype(np.complex64)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    waveform.tofile(out)
    print(f"[waveform] wrote {out}")
    print(f"[waveform] samples: {waveform.size}")
    print(f"[waveform] bytes: {out.stat().st_size}")
    print(f"[waveform] chirp samples: {n_chirp}")
    print(f"[waveform] peak amplitude: {float(np.max(np.abs(waveform))):.6f}")


if __name__ == "__main__":
    main()
