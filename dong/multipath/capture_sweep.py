#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stepped-frequency USRP capture for synthetic-bandwidth multipath tests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from gnuradio import blocks, gr, uhd

from waveforms import make_chirp_probe


DONG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = DONG_ROOT / "outputs" / "multipath" / "sweep_capture.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep a B200/B210 over many narrow channels and save IQ captures."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output .npz file")
    parser.add_argument("--device-args", default="", help='UHD device args. Leave empty for auto, e.g. "serial=2512552"')
    parser.add_argument("--rx-antenna", default="TX/RX", choices=["TX/RX", "RX2"])
    parser.add_argument("--tx-antenna", default="TX/RX")
    parser.add_argument("--center", type=float, default=487.7e6, help="Sweep center frequency in Hz")
    parser.add_argument("--span", type=float, default=2.0e6, help="Total synthetic span in Hz")
    parser.add_argument("--step", type=float, default=125e3, help="Frequency step in Hz")
    parser.add_argument("--samp-rate", type=float, default=500e3, help="Per-step sample rate in samples/s")
    parser.add_argument("--rf-bandwidth", type=float, default=500e3, help="USRP analog bandwidth in Hz")
    parser.add_argument("--duration", type=float, default=0.08, help="RX capture duration per frequency in seconds")
    parser.add_argument("--settle", type=float, default=0.08, help="Delay after each retune in seconds")
    parser.add_argument("--rx-gain", type=float, default=0.5, help="Normalized RX gain, 0..1")
    parser.add_argument("--otw-format", default="sc16", choices=["sc16", "sc8"], help="UHD wire format")
    parser.add_argument("--tx-enable", action="store_true", help="Transmit the chirp probe while receiving")
    parser.add_argument("--tx-gain", type=float, default=0.15, help="Normalized TX gain, 0..1")
    parser.add_argument("--tx-amplitude", type=float, default=0.05, help="Complex baseband amplitude for TX probe")
    parser.add_argument("--probe-duration", type=float, default=0.008, help="One chirp duration in seconds")
    parser.add_argument("--probe-bandwidth", type=float, default=200e3, help="Baseband chirp bandwidth in Hz")
    parser.add_argument("--probe-guard", type=float, default=0.002, help="Zero guard before and after each chirp in seconds")
    parser.add_argument("--probe-repeats", type=int, default=4, help="Number of chirp repetitions per step")
    parser.add_argument("--clock-source", default="internal", choices=["internal", "external", "gpsdo"])
    parser.add_argument("--save-raw", action="store_true", help="Save all raw per-step IQ samples in the npz")
    return parser.parse_args()


def frequency_grid(center: float, span: float, step: float) -> np.ndarray:
    if span <= 0 or step <= 0:
        raise ValueError("span and step must be positive")
    n_steps = int(np.floor(span / step)) + 1
    start = center - step * (n_steps - 1) / 2.0
    return start + step * np.arange(n_steps, dtype=np.float64)


def make_usrp_source(args: argparse.Namespace) -> uhd.usrp_source:
    src = uhd.usrp_source(
        args.device_args,
        uhd.stream_args(cpu_format="fc32", otw_format=args.otw_format, channels=[0]),
    )
    src.set_clock_source(args.clock_source, 0)
    src.set_samp_rate(args.samp_rate)
    src.set_antenna(args.rx_antenna, 0)
    src.set_bandwidth(args.rf_bandwidth, 0)
    src.set_normalized_gain(args.rx_gain, 0)
    src.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
    return src


def make_usrp_sink(args: argparse.Namespace) -> uhd.usrp_sink:
    sink = uhd.usrp_sink(
        args.device_args,
        uhd.stream_args(cpu_format="fc32", otw_format=args.otw_format, channels=[0]),
        "",
    )
    sink.set_clock_source(args.clock_source, 0)
    sink.set_samp_rate(args.samp_rate)
    sink.set_antenna(args.tx_antenna, 0)
    sink.set_bandwidth(args.rf_bandwidth, 0)
    sink.set_normalized_gain(args.tx_gain, 0)
    return sink


def capture_one_frequency(
    args: argparse.Namespace,
    freq: float,
    probe: np.ndarray,
) -> np.ndarray:
    nsamps = int(round(args.samp_rate * args.duration))
    if nsamps <= 0:
        raise ValueError("duration is too short")

    src = make_usrp_source(args)
    src.set_center_freq(freq, 0)
    time.sleep(args.settle)

    head = blocks.head(gr.sizeof_gr_complex, nsamps)
    rx_sink = blocks.vector_sink_c()
    tb = gr.top_block()
    tb.connect(src, head, rx_sink)

    if args.tx_enable:
        tx_sink = make_usrp_sink(args)
        tx_sink.set_center_freq(freq, 0)
        tx_waveform = np.asarray(probe, dtype=np.complex64)
        if tx_waveform.size < nsamps:
            repeats = int(np.ceil(nsamps / tx_waveform.size))
            tx_waveform = np.tile(tx_waveform, repeats)
        tx_waveform = tx_waveform[:nsamps].astype(np.complex64, copy=False)
        tx_src = blocks.vector_source_c(tx_waveform.tolist(), False)
        tb.connect(tx_src, tx_sink)

    tb.run()
    return np.asarray(rx_sink.data(), dtype=np.complex64)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    freqs = frequency_grid(args.center, args.span, args.step)
    probe = make_chirp_probe(
        sample_rate=args.samp_rate,
        probe_duration=args.probe_duration,
        bandwidth=args.probe_bandwidth,
        amplitude=args.tx_amplitude,
        guard_duration=args.probe_guard,
        repeats=args.probe_repeats,
    )

    print(f"[sweep] {freqs.size} steps, {freqs[0] / 1e6:.6f} to {freqs[-1] / 1e6:.6f} MHz")
    print(f"[sweep] per-step capture {args.duration:.3f} s at {args.samp_rate / 1e3:.1f} kS/s")
    print(f"[sweep] tx {'enabled' if args.tx_enable else 'disabled'}, device args: {args.device_args or '<auto>'}")

    captures = []
    rms = []
    peaks = []
    for index, freq in enumerate(freqs, start=1):
        print(f"[sweep] step {index:03d}/{freqs.size:03d}: {freq / 1e6:.6f} MHz", flush=True)
        samples = capture_one_frequency(args, float(freq), probe)
        captures.append(samples if args.save_raw else samples.astype(np.complex64, copy=True))
        mag = np.abs(samples)
        rms.append(float(np.sqrt(np.mean(mag * mag))) if samples.size else 0.0)
        peaks.append(float(np.max(mag)) if samples.size else 0.0)

    metadata = {
        "created_unix": time.time(),
        "device_args": args.device_args,
        "rx_antenna": args.rx_antenna,
        "tx_antenna": args.tx_antenna,
        "center_hz": args.center,
        "span_hz": args.span,
        "step_hz": args.step,
        "sample_rate_hz": args.samp_rate,
        "rf_bandwidth_hz": args.rf_bandwidth,
        "duration_s": args.duration,
        "settle_s": args.settle,
        "rx_gain_normalized": args.rx_gain,
        "otw_format": args.otw_format,
        "tx_enable": args.tx_enable,
        "tx_gain_normalized": args.tx_gain,
        "tx_amplitude": args.tx_amplitude,
        "probe_duration_s": args.probe_duration,
        "probe_bandwidth_hz": args.probe_bandwidth,
        "probe_guard_s": args.probe_guard,
        "probe_repeats": args.probe_repeats,
        "clock_source": args.clock_source,
    }
    np.savez_compressed(
        output,
        frequencies_hz=freqs.astype(np.float64),
        captures=np.asarray(captures, dtype=np.complex64),
        probe=probe.astype(np.complex64),
        rms=np.asarray(rms, dtype=np.float64),
        peaks=np.asarray(peaks, dtype=np.float64),
        metadata=json.dumps(metadata, indent=2),
    )
    print(f"[sweep] saved {output}")


if __name__ == "__main__":
    main()
