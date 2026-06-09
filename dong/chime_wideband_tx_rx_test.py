#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the Chime wideband TX and RX flowgraphs together for a quick link test."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path


DONG_ROOT = Path(__file__).resolve().parent
TX_DIR = DONG_ROOT / "TX"
RX_DIR = DONG_ROOT / "RX"
for path in (TX_DIR, RX_DIR):
    sys.path.insert(0, str(path))


def run_top_block(name: str, make_tb, started: threading.Event, errors: queue.Queue) -> None:
    tb = None
    try:
        tb = make_tb()
        # start() 只负责启动 GNU Radio 调度线程；实际数据流在底层线程里运行。
        tb.start()
        started.set()
        tb.wait()
        print(f"[{name}] finished")
    except BaseException as exc:
        errors.put((name, exc))
        started.set()
    finally:
        if tb is not None:
            try:
                tb.stop()
                tb.wait()
            except BaseException:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start RX and TX in two threads, then analyze the capture.")
    parser.add_argument("--waveform-file", default=str(DONG_ROOT / "inputs" / "chime_test_tx_period_fc32.bin"))
    parser.add_argument("--capture-file", default=str(DONG_ROOT / "outputs" / "captures" / "chime_test_rx_fc32.bin"))
    parser.add_argument("--csv-out", default=str(DONG_ROOT / "outputs" / "analysis" / "chime_test_paths.csv"))
    parser.add_argument("--json-out", default=str(DONG_ROOT / "outputs" / "analysis" / "chime_test_summary.json"))
    parser.add_argument("--fs", type=float, default=2e6)
    parser.add_argument("--chirp-bw", type=float, default=1e6)
    parser.add_argument("--chirp-duration", type=float, default=1e-3)
    parser.add_argument("--period", type=float, default=20e-3)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--center", type=float, default=487.7e6)
    parser.add_argument("--rf-bandwidth", type=float, default=2e6)
    parser.add_argument("--tx-seconds", type=float, default=10.0)
    parser.add_argument("--rx-seconds", type=float, default=12.0)
    parser.add_argument("--tx-start-delay", type=float, default=0.5)
    parser.add_argument("--start-timeout", type=float, default=60.0)
    parser.add_argument("--tx-gain-db", type=float, default=10.0)
    parser.add_argument("--rx-gain-db", type=float, default=20.0)
    parser.add_argument("--tx-device-addr", default="serial=2512552")
    parser.add_argument("--rx-device-addr", default="serial=2603160")
    parser.add_argument("--tx-antenna", default="TX/RX")
    parser.add_argument("--rx-antenna", default="TX/RX")
    parser.add_argument("--skip-waveform", action="store_true", help="Use the existing waveform file.")
    parser.add_argument("--no-tx", action="store_true", help="Start RX only. Use this to measure the noise floor.")
    parser.add_argument("--no-analyze", action="store_true", help="Only capture, do not run matched-filter analysis.")
    parser.add_argument("--max-segments", type=int, default=0, help="0 means analyze all full periods.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    waveform_file = Path(args.waveform_file)
    capture_file = Path(args.capture_file)

    if not args.skip_waveform and not args.no_tx:
        from chime_wideband_make_tx_waveform import generate_waveform

        # 每次联调前重新生成模板文件，避免 TX 读到过期参数的 bin 文件。
        stats = generate_waveform(
            output=waveform_file,
            fs=args.fs,
            chirp_bw=args.chirp_bw,
            chirp_duration=args.chirp_duration,
            period=args.period,
            amplitude=args.amplitude,
        )
        print(f"[waveform] ready: {stats['path']} ({stats['bytes']} bytes)")
    elif not args.no_tx and not waveform_file.exists():
        raise FileNotFoundError(waveform_file)

    errors: queue.Queue = queue.Queue()
    rx_started = threading.Event()
    tx_started = threading.Event()

    try:
        from chime_wideband_tx import chime_wideband_tx
        from chime_wideband_rx import chime_wideband_rx
    except ModuleNotFoundError as exc:
        if exc.name == "gnuradio":
            raise SystemExit("GNU Radio is not available in this Python. Run this with the radioconda/GNU Radio python.") from exc
        raise

    def make_rx() -> chime_wideband_rx:
        return chime_wideband_rx(
            samp_rate=args.fs,
            capture_seconds=args.rx_seconds,
            rx_gain_db=args.rx_gain_db,
            rx_device_addr=args.rx_device_addr,
            rx_antenna=args.rx_antenna,
            rf_bandwidth=args.rf_bandwidth,
            output_file=capture_file,
            center_freq=args.center,
        )

    def make_tx() -> chime_wideband_tx:
        return chime_wideband_tx(
            tx_seconds=args.tx_seconds,
            samp_rate=args.fs,
            waveform_file=waveform_file,
            tx_gain_db=args.tx_gain_db,
            tx_device_addr=args.tx_device_addr,
            tx_antenna=args.tx_antenna,
            rf_bandwidth=args.rf_bandwidth,
            center_freq=args.center,
        )

    rx_thread = threading.Thread(target=run_top_block, args=("rx", make_rx, rx_started, errors), daemon=False)
    tx_thread = threading.Thread(target=run_top_block, args=("tx", make_tx, tx_started, errors), daemon=False)

    print("[rx] starting")
    rx_thread.start()
    if not rx_started.wait(timeout=args.start_timeout):
        raise TimeoutError(f"RX did not start within {args.start_timeout:.1f} seconds")
    if not errors.empty():
        name, exc = errors.get()
        raise RuntimeError(f"{name} failed before TX start") from exc

    if args.no_tx:
        print("[tx] disabled")
    else:
        # RX 先启动，给硬件调谐和文件 sink 留出时间，再启动 TX。
        print(f"[tx] starting after {args.tx_start_delay:.3f} s")
        time.sleep(max(0.0, args.tx_start_delay))
        tx_thread.start()
        tx_thread.join()
    rx_thread.join()

    if not errors.empty():
        name, exc = errors.get()
        raise RuntimeError(f"{name} failed") from exc

    print(f"[capture] saved: {capture_file}")
    if args.no_analyze:
        return

    from chime_wideband_analyze import analyze

    analysis_args = argparse.Namespace(
        infile=str(capture_file),
        fs=args.fs,
        chirp_bw=args.chirp_bw,
        chirp_duration=args.chirp_duration,
        period=args.period,
        corr_gate=0.25,
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
    print(f"[analyze] trusted: {summary['trusted_segments']}/{summary['segments']}")
    print(f"[analyze] max corr: {summary['max_corr_score']:.3f}")
    print(f"[analyze] csv: {summary['csv_out']}")


if __name__ == "__main__":
    main()
