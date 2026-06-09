#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Chime Wideband RX
# Author: guang
# Description: Low-rate raw IQ receiver for Chime-style chirp sounding
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import threading
from pathlib import Path




class chime_wideband_rx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Chime Wideband RX", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.dong_root = dong_root = Path(__file__).resolve().parents[1]
        self.samp_rate = samp_rate = 2000000
        self.capture_seconds = capture_seconds = 10
        self.rx_gain_db = rx_gain_db = 20
        self.rx_device_addr = rx_device_addr = "serial=2603160"
        self.rf_bandwidth = rf_bandwidth = 2000000
        self.output_file = output_file = str(dong_root / "outputs" / "captures" / "chime_test_rx_fc32.bin")
        self.center_freq = center_freq = 487700000
        self.capture_samples = capture_samples = int(samp_rate * capture_seconds)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join((rx_device_addr, "")),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args="",
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0.set_clock_source('internal', 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_source_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0.set_bandwidth(rf_bandwidth, 0)
        self.uhd_usrp_source_0.set_rx_agc(False, 0)
        self.uhd_usrp_source_0.set_gain(rx_gain_db, 0)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, capture_samples)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, output_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_head_0, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_capture_samples(int(self.samp_rate * self.capture_seconds))
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_capture_seconds(self):
        return self.capture_seconds

    def set_capture_seconds(self, capture_seconds):
        self.capture_seconds = capture_seconds
        self.set_capture_samples(int(self.samp_rate * self.capture_seconds))

    def get_rx_gain_db(self):
        return self.rx_gain_db

    def set_rx_gain_db(self, rx_gain_db):
        self.rx_gain_db = rx_gain_db
        self.uhd_usrp_source_0.set_gain(self.rx_gain_db, 0)

    def get_rx_device_addr(self):
        return self.rx_device_addr

    def set_rx_device_addr(self, rx_device_addr):
        self.rx_device_addr = rx_device_addr

    def get_rf_bandwidth(self):
        return self.rf_bandwidth

    def set_rf_bandwidth(self, rf_bandwidth):
        self.rf_bandwidth = rf_bandwidth
        self.uhd_usrp_source_0.set_bandwidth(self.rf_bandwidth, 0)

    def get_output_file(self):
        return self.output_file

    def set_output_file(self, output_file):
        self.output_file = output_file
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        self.blocks_file_sink_0.open(self.output_file)

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.uhd_usrp_source_0.set_center_freq(self.center_freq, 0)

    def get_capture_samples(self):
        return self.capture_samples

    def set_capture_samples(self, capture_samples):
        self.capture_samples = capture_samples
        self.blocks_head_0.set_length(self.capture_samples)




def main(top_block_cls=chime_wideband_rx, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    tb.wait()


if __name__ == '__main__':
    main()
