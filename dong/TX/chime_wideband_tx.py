#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Chime Wideband TX
# Author: guang
# Description: Low-rate Chime-style LFM upchirp transmitter using a repeated fc32 waveform file
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
import pmt
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




class chime_wideband_tx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Chime Wideband TX", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.dong_root = dong_root = Path(__file__).resolve().parents[1]
        self.tx_seconds = tx_seconds = 10
        self.samp_rate = samp_rate = 2000000
        self.waveform_file = waveform_file = str(dong_root / "inputs" / "chime_test_tx_period_fc32.bin")
        self.tx_samples = tx_samples = int(samp_rate * tx_seconds)
        self.tx_gain_db = tx_gain_db = 10
        self.tx_device_addr = tx_device_addr = "serial=2512552"
        self.rf_bandwidth = rf_bandwidth = 2000000
        self.center_freq = center_freq = 487700000

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join((tx_device_addr, "")),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args="",
                channels=list(range(0,1)),
            ),
            "",
        )
        self.uhd_usrp_sink_0.set_clock_source('internal', 0)
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_sink_0.set_center_freq(center_freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_bandwidth(rf_bandwidth, 0)
        self.uhd_usrp_sink_0.set_gain(tx_gain_db, 0)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, tx_samples)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, waveform_file, True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_head_0, 0), (self.uhd_usrp_sink_0, 0))


    def get_tx_seconds(self):
        return self.tx_seconds

    def set_tx_seconds(self, tx_seconds):
        self.tx_seconds = tx_seconds
        self.set_tx_samples(int(self.samp_rate * self.tx_seconds))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_tx_samples(int(self.samp_rate * self.tx_seconds))
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_waveform_file(self):
        return self.waveform_file

    def set_waveform_file(self, waveform_file):
        self.waveform_file = waveform_file
        self.blocks_file_source_0.open(self.waveform_file, True)

    def get_tx_samples(self):
        return self.tx_samples

    def set_tx_samples(self, tx_samples):
        self.tx_samples = tx_samples
        self.blocks_head_0.set_length(self.tx_samples)

    def get_tx_gain_db(self):
        return self.tx_gain_db

    def set_tx_gain_db(self, tx_gain_db):
        self.tx_gain_db = tx_gain_db
        self.uhd_usrp_sink_0.set_gain(self.tx_gain_db, 0)

    def get_tx_device_addr(self):
        return self.tx_device_addr

    def set_tx_device_addr(self, tx_device_addr):
        self.tx_device_addr = tx_device_addr

    def get_rf_bandwidth(self):
        return self.rf_bandwidth

    def set_rf_bandwidth(self, rf_bandwidth):
        self.rf_bandwidth = rf_bandwidth
        self.uhd_usrp_sink_0.set_bandwidth(self.rf_bandwidth, 0)

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 0)




def main(top_block_cls=chime_wideband_tx, options=None):
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
