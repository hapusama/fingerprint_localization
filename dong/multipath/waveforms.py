#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe waveform helpers for stepped-frequency channel sounding."""

from __future__ import annotations

import numpy as np


def make_chirp_probe(
    sample_rate: float,
    probe_duration: float,
    bandwidth: float,
    amplitude: float = 0.1,
    guard_duration: float = 0.002,
    repeats: int = 3,
) -> np.ndarray:
    """Create a repeated baseband LFM chirp probe with short zero guards.

    The probe sweeps from -bandwidth/2 to +bandwidth/2. A Hann taper keeps
    edges gentle enough for lab use and for cleaner correlation peaks.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if probe_duration <= 0:
        raise ValueError("probe_duration must be positive")
    if bandwidth <= 0 or bandwidth >= sample_rate:
        raise ValueError("bandwidth must be positive and smaller than sample_rate")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    n_probe = max(32, int(round(sample_rate * probe_duration)))
    n_guard = max(0, int(round(sample_rate * guard_duration)))
    t = np.arange(n_probe, dtype=np.float64) / sample_rate
    sweep_rate = bandwidth / probe_duration
    phase = 2.0 * np.pi * ((-bandwidth / 2.0) * t + 0.5 * sweep_rate * t * t)
    chirp = np.exp(1j * phase).astype(np.complex64)
    chirp *= np.hanning(n_probe).astype(np.float32)
    chirp *= np.float32(amplitude)

    guard = np.zeros(n_guard, dtype=np.complex64)
    burst = np.concatenate((guard, chirp, guard))
    return np.tile(burst, int(repeats)).astype(np.complex64, copy=False)


def correlation_channel_estimate(rx: np.ndarray, probe: np.ndarray) -> tuple[complex, int, float]:
    """Estimate one complex channel coefficient by matched filtering.

    Returns:
        h_hat: complex correlation peak normalized by probe energy
        peak_index: sample index where the probe starts in rx
        peak_metric: normalized magnitude of the correlation peak
    """
    rx = np.asarray(rx, dtype=np.complex64)
    probe = np.asarray(probe, dtype=np.complex64)
    if rx.size < probe.size:
        raise ValueError("rx capture is shorter than the probe")

    n_corr = rx.size + probe.size - 1
    n_fft = 1 << int(np.ceil(np.log2(max(1, n_corr))))
    corr = np.fft.ifft(
        np.fft.fft(rx, n_fft) * np.conj(np.fft.fft(probe, n_fft))
    )[: rx.size]
    valid = corr[: max(1, rx.size - probe.size + 1)]
    peak_index = int(np.argmax(np.abs(valid)))
    energy = float(np.vdot(probe, probe).real)
    if energy <= 0:
        raise ValueError("probe has zero energy")
    h_hat = complex(valid[peak_index] / energy)
    rx_energy = float(np.vdot(rx[peak_index : peak_index + probe.size], rx[peak_index : peak_index + probe.size]).real)
    denom = np.sqrt(max(energy * rx_energy, 1e-30))
    peak_metric = float(np.abs(valid[peak_index]) / denom)
    return h_hat, peak_index, peak_metric

