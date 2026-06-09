#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Multipath Chirp TX
# Author: guang
# Description: TX-only chirp probe flowgraph for synthetic-bandwidth multipath sounding
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
import multipath_chirp_tx_epy_chirp_source as epy_chirp_source  # embedded python block
import threading




class multipath_chirp_tx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Multipath Chirp TX", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.tx_seconds = tx_seconds = 10
        self.samp_rate = samp_rate = 500000
        self.tx_samps = tx_samps = int(samp_rate * tx_seconds)
        self.tx_gain = tx_gain = 0.10
        self.rf_bandwidth = rf_bandwidth = 500000
        self.repeat_pause = repeat_pause = 0.010
        self.guard_duration = guard_duration = 0.002
        self.device_addr = device_addr = ""
        self.chirp_duration = chirp_duration = 0.008
        self.chirp_bw = chirp_bw = 200000
        self.center_freq = center_freq = 915000000
        self.amplitude = amplitude = 0.05

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join((device_addr, "")),
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
        self.uhd_usrp_sink_0.set_normalized_gain(tx_gain, 0)
        self.epy_chirp_source = epy_chirp_source.blk(sample_rate=samp_rate, chirp_bw=chirp_bw, chirp_duration=chirp_duration, guard_duration=guard_duration, repeat_pause=repeat_pause, amplitude=amplitude)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, tx_samps)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.epy_chirp_source, 0), (self.blocks_head_0, 0))


    def get_tx_seconds(self):
        return self.tx_seconds

    def set_tx_seconds(self, tx_seconds):
        self.tx_seconds = tx_seconds
        self.set_tx_samps(int(self.samp_rate * self.tx_seconds))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_tx_samps(int(self.samp_rate * self.tx_seconds))
        self.epy_chirp_source.sample_rate = self.samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_tx_samps(self):
        return self.tx_samps

    def set_tx_samps(self, tx_samps):
        self.tx_samps = tx_samps
        self.blocks_head_0.set_length(self.tx_samps)

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_normalized_gain(self.tx_gain, 0)

    def get_rf_bandwidth(self):
        return self.rf_bandwidth

    def set_rf_bandwidth(self, rf_bandwidth):
        self.rf_bandwidth = rf_bandwidth
        self.uhd_usrp_sink_0.set_bandwidth(self.rf_bandwidth, 0)

    def get_repeat_pause(self):
        return self.repeat_pause

    def set_repeat_pause(self, repeat_pause):
        self.repeat_pause = repeat_pause
        self.epy_chirp_source.repeat_pause = self.repeat_pause

    def get_guard_duration(self):
        return self.guard_duration

    def set_guard_duration(self, guard_duration):
        self.guard_duration = guard_duration
        self.epy_chirp_source.guard_duration = self.guard_duration

    def get_device_addr(self):
        return self.device_addr

    def set_device_addr(self, device_addr):
        self.device_addr = device_addr

    def get_chirp_duration(self):
        return self.chirp_duration

    def set_chirp_duration(self, chirp_duration):
        self.chirp_duration = chirp_duration
        self.epy_chirp_source.chirp_duration = self.chirp_duration

    def get_chirp_bw(self):
        return self.chirp_bw

    def set_chirp_bw(self, chirp_bw):
        self.chirp_bw = chirp_bw
        self.epy_chirp_source.chirp_bw = self.chirp_bw

    def get_center_freq(self):
        return self.center_freq

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 0)

    def get_amplitude(self):
        return self.amplitude

    def set_amplitude(self, amplitude):
        self.amplitude = amplitude
        self.epy_chirp_source.amplitude = self.amplitude




def main(top_block_cls=multipath_chirp_tx, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
