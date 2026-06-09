#!/usr/bin/env python3
import argparse
import time

import numpy as np
from gnuradio import blocks, gr, uhd


def dbfs(x):
    return 20.0 * np.log10(np.maximum(x, 1e-12))


def capture(args):
    nsamps = int(args.samp_rate * args.duration)

    src = uhd.usrp_source(
        ",".join((f"serial={args.serial}", "")),
        uhd.stream_args(cpu_format="fc32", otw_format="sc8", channels=[0]),
    )
    src.set_clock_source("internal", 0)
    src.set_samp_rate(args.samp_rate)
    src.set_center_freq(args.freq, 0)
    src.set_antenna(args.antenna, 0)
    src.set_bandwidth(args.bandwidth, 0)
    src.set_normalized_gain(args.gain, 0)
    src.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

    head = blocks.head(gr.sizeof_gr_complex, nsamps)
    sink = blocks.vector_sink_c()

    tb = gr.top_block()
    tb.connect(src, head, sink)
    tb.run()

    samples = np.asarray(sink.data(), dtype=np.complex64)
    if samples.size == 0:
        raise RuntimeError("No samples captured from USRP")

    mag = np.abs(samples)
    mean_db = float(np.mean(dbfs(mag)))
    rms_db = float(20.0 * np.log10(np.sqrt(np.mean(np.abs(samples) ** 2)) + 1e-12))
    peak_db = float(np.max(dbfs(mag)))

    nfft = min(65536, samples.size)
    win = np.hanning(nfft).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft(samples[:nfft] * win))
    power = 20.0 * np.log10(np.abs(spec) / max(np.sum(win), 1e-12) + 1e-12)
    peak_bin = int(np.argmax(power))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / args.samp_rate))
    peak_offset = float(freqs[peak_bin])
    peak_freq = args.freq + peak_offset

    print(f"Captured samples : {samples.size}")
    print(f"Center frequency : {args.freq / 1e6:.6f} MHz")
    print(f"Antenna          : {args.antenna}")
    print(f"Gain             : {args.gain:.2f} normalized")
    print(f"Mean power       : {mean_db:.1f} dBFS")
    print(f"RMS power        : {rms_db:.1f} dBFS")
    print(f"Peak sample      : {peak_db:.1f} dBFS")
    print(f"FFT peak offset  : {peak_offset / 1e3:.1f} kHz")
    print(f"FFT peak freq    : {peak_freq / 1e6:.6f} MHz")


def main():
    parser = argparse.ArgumentParser(description="Quick B210 receive power probe")
    parser.add_argument("--serial", default="30AA048")
    parser.add_argument("--freq", type=float, default=487700000.0)
    parser.add_argument("--samp-rate", type=float, default=500000.0)
    parser.add_argument("--bandwidth", type=float, default=200000.0)
    parser.add_argument("--gain", type=float, default=0.6)
    parser.add_argument("--antenna", default="TX/RX", choices=["TX/RX", "RX2"])
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()
    capture(args)


if __name__ == "__main__":
    main()
