#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a two-USRP CHIME-style TX/RX self-test and analyze the 10 s capture."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path


DONG_ROOT = Path(__file__).resolve().parent
TX_DIR = DONG_ROOT / "TX"
RX_DIR = DONG_ROOT / "RX"
for path in (TX_DIR, RX_DIR):
    sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start TX first, capture RX, then run matched-filter analysis.")
    parser.add_argument("--waveform-file", default=str(DONG_ROOT / "inputs" / "chime_test_tx_period_fc32.bin"))
    parser.add_argument("--capture-file", default="")
    parser.add_argument("--csv-out", default=str(DONG_ROOT / "outputs" / "analysis" / "chime_self_test_paths.csv"))
    parser.add_argument("--json-out", default=str(DONG_ROOT / "outputs" / "analysis" / "chime_self_test_summary.json"))
    parser.add_argument("--fs", type=float, default=20e6)
    parser.add_argument("--chirp-bw", type=float, default=18e6)
    parser.add_argument("--chirp-duration", type=float, default=1e-3)
    parser.add_argument("--period", type=float, default=20e-3)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--center", type=float, default=487.7e6)
    parser.add_argument("--rf-bandwidth", type=float, default=20e6)
    parser.add_argument("--tx-seconds", type=float, default=50.0)
    parser.add_argument("--rx-seconds", type=float, default=10.05)
    parser.add_argument("--tx-lead-time", type=float, default=1.0)
    parser.add_argument("--tx-gain-db", type=float, default=100.0)
    parser.add_argument("--rx-gain-db", type=float, default=40.0)
    parser.add_argument("--tx-device-addr", default="serial=2512552,num_send_frames=1024")
    parser.add_argument("--rx-device-addr", default="serial=2603160,num_recv_frames=512")
    parser.add_argument("--tx-antenna", default="TX/RX")
    parser.add_argument("--rx-antenna", default="TX/RX")
    parser.add_argument("--skip-waveform", action="store_true", help="Use the existing waveform file.")
    parser.add_argument("--no-tx", action="store_true", help="Capture RX only. Use this to measure the noise floor.")
    parser.add_argument("--no-analyze", action="store_true", help="Only capture, do not run matched-filter analysis.")
    parser.add_argument("--max-segments", type=int, default=500, help="Analyze this many 20 ms periods; 500 is exactly 10.00 s, 0 means all full periods.")
    parser.add_argument("--corr-gate", type=float, default=0.25)
    parser.add_argument("--require-trusted-ratio", type=float, default=0.95)
    return parser.parse_args()


def make_capture_path(raw_path: str) -> Path:
    if raw_path:
        return Path(raw_path).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("/Users/siri/Desktop/data") / f"self_rx_test_{timestamp}.bin"


def build_tx_top_block(args: argparse.Namespace, waveform_file: Path):
    from gnuradio import blocks, gr, uhd
    import pmt

    tb = gr.top_block("CHIME Wideband Self-Test TX", catch_exceptions=True)
    usrp = uhd.usrp_sink(
        ",".join((args.tx_device_addr, "")),
        uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]),
        "",
    )
    usrp.set_clock_source("internal", 0)
    usrp.set_samp_rate(args.fs)
    usrp.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
    usrp.set_center_freq(args.center, 0)
    usrp.set_antenna(args.tx_antenna, 0)
    usrp.set_bandwidth(args.rf_bandwidth, 0)
    usrp.set_gain(args.tx_gain_db, 0)

    source = blocks.file_source(gr.sizeof_gr_complex, str(waveform_file), True, 0, 0)
    source.set_begin_tag(pmt.PMT_NIL)
    head = blocks.head(gr.sizeof_gr_complex, int(round(args.fs * args.tx_seconds)))

    tb.connect(source, head)
    tb.connect(head, usrp)
    return tb


def build_rx_top_block(args: argparse.Namespace, capture_file: Path):
    from gnuradio import blocks, gr, uhd

    capture_file.parent.mkdir(parents=True, exist_ok=True)

    tb = gr.top_block("CHIME Wideband Self-Test RX", catch_exceptions=True)
    usrp = uhd.usrp_source(
        ",".join((args.rx_device_addr, "")),
        uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]),
    )
    usrp.set_clock_source("internal", 0)
    usrp.set_samp_rate(args.fs)
    usrp.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)
    usrp.set_center_freq(args.center, 0)
    usrp.set_antenna(args.rx_antenna, 0)
    usrp.set_bandwidth(args.rf_bandwidth, 0)
    usrp.set_rx_agc(False, 0)
    usrp.set_gain(args.rx_gain_db, 0)

    head = blocks.head(gr.sizeof_gr_complex, int(round(args.fs * args.rx_seconds)))
    sink = blocks.file_sink(gr.sizeof_gr_complex, str(capture_file), False)
    sink.set_unbuffered(False)

    tb.connect(usrp, head)
    tb.connect(head, sink)
    return tb


def main() -> None:
    args = parse_args()
    waveform_file = Path(args.waveform_file).expanduser()
    capture_file = make_capture_path(args.capture_file)

    if args.tx_seconds < args.rx_seconds + args.tx_lead_time:
        raise SystemExit("tx_seconds must cover tx_lead_time + rx_seconds")

    if not args.skip_waveform and not args.no_tx:
        from chime_wideband_make_tx_waveform import generate_waveform

        stats = generate_waveform(
            output=waveform_file,
            fs=args.fs,
            chirp_bw=args.chirp_bw,
            chirp_duration=args.chirp_duration,
            period=args.period,
            amplitude=args.amplitude,
        )
        print(f"[waveform] ready: {stats['path']} ({stats['samples']} samples, {stats['bytes']} bytes)")
    elif not args.no_tx and not waveform_file.exists():
        raise FileNotFoundError(waveform_file)

    tx_tb = None
    try:
        if args.no_tx:
            print("[tx] disabled")
        else:
            print(f"[tx] starting {args.tx_seconds:.1f} s on {args.tx_device_addr}")
            tx_tb = build_tx_top_block(args, waveform_file)
            tx_tb.start()
            print(f"[rx] waiting {args.tx_lead_time:.3f} s for TX to settle")
            time.sleep(max(0.0, args.tx_lead_time))

        print(f"[rx] capturing {args.rx_seconds:.1f} s on {args.rx_device_addr}")
        rx_tb = build_rx_top_block(args, capture_file)
        rx_tb.start()
        rx_tb.wait()
        print(f"[capture] saved: {capture_file}")
    finally:
        if tx_tb is not None:
            print("[tx] stopping")
            tx_tb.stop()
            tx_tb.wait()

    if args.no_analyze:
        return

    from chime_wideband_analyze import analyze

    analysis_args = argparse.Namespace(
        infile=str(capture_file),
        fs=args.fs,
        chirp_bw=args.chirp_bw,
        chirp_duration=args.chirp_duration,
        period=args.period,
        corr_gate=args.corr_gate,
        threshold_db=-20.0,
        min_gap_us=0.10,
        pre_delay_us=1.0,
        max_delay_us=8.0,
        max_peaks=8,
        max_segments=args.max_segments,
        csv_out=args.csv_out,
        json_out=args.json_out,
    )
    summary = analyze(analysis_args)
    segments = max(int(summary["segments"]), 1)
    trusted = int(summary["trusted_segments"])
    trusted_ratio = trusted / segments
    print(f"[analyze] trusted: {trusted}/{segments} ({trusted_ratio:.3f})")
    print(f"[analyze] mean corr: {summary['mean_corr_score']:.3f}")
    print(f"[analyze] max corr: {summary['max_corr_score']:.3f}")
    print(f"[analyze] csv: {summary['csv_out']}")

    if trusted_ratio < args.require_trusted_ratio:
        raise SystemExit(
            f"trusted ratio {trusted_ratio:.3f} is below required {args.require_trusted_ratio:.3f}"
        )


if __name__ == "__main__":
    main()
