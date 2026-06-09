#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: guang
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import dong_epy_block_0 as epy_block_0  # embedded python block
import sip
import threading
from pathlib import Path



class dong(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "dong")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.output_dir = output_dir = Path(__file__).resolve().parent / "outputs" / "captures"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.samp_rate = samp_rate = 500000
        self.EXPID = EXPID = r"\2_0_53_11_2_16"
        self.samp_time = samp_time = int(samp_rate*1)
        self.NOP = NOP = int(samp_rate*1)
        self.IQ_Root = IQ_Root = str(output_dir / EXPID.lstrip("\\/"))
        self.CF = CF = 487700000
        self.BW = BW = 125000

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_source_0_0_0 = uhd.usrp_source(
            ",".join(('serial=2603160', '')),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc8",
                args='',
                channels=list(range(0,1)),
            ),
        )
        self.uhd_usrp_source_0_0_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0_0_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_source_0_0_0.set_center_freq(CF, 0)
        self.uhd_usrp_source_0_0_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_source_0_0_0.set_bandwidth(BW, 0)
        self.uhd_usrp_source_0_0_0.set_normalized_gain(00.6, 0)
        self.qtgui_time_sink_x_1_4_0 = qtgui.time_sink_f(
            samp_time, #size
            samp_rate, #samp_rate
            "LoRa", #name
            1, #number of inputs
            None # parent
        )
        self.qtgui_time_sink_x_1_4_0.set_update_time(0.1)
        self.qtgui_time_sink_x_1_4_0.set_y_axis(-100, 10)

        self.qtgui_time_sink_x_1_4_0.set_y_label('Power', "dBFS")

        self.qtgui_time_sink_x_1_4_0.enable_tags(False)
        self.qtgui_time_sink_x_1_4_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1_4_0.enable_autoscale(False)
        self.qtgui_time_sink_x_1_4_0.enable_grid(True)
        self.qtgui_time_sink_x_1_4_0.enable_axis_labels(True)
        self.qtgui_time_sink_x_1_4_0.enable_control_panel(True)
        self.qtgui_time_sink_x_1_4_0.enable_stem_plot(False)


        labels = ['Signal 1', 'Signal 2', 'Signal 3', 'Signal 4', 'Signal 5',
            'Signal 6', 'Signal 7', 'Signal 8', 'Signal 9', 'Signal 10']
        widths = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        colors = ['blue', 'red', 'green', 'black', 'cyan',
            'magenta', 'yellow', 'dark red', 'dark green', 'dark blue']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0]
        styles = [1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]
        markers = [-1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1]


        for i in range(1):
            if len(labels[i]) == 0:
                self.qtgui_time_sink_x_1_4_0.set_line_label(i, "Data {0}".format(i))
            else:
                self.qtgui_time_sink_x_1_4_0.set_line_label(i, labels[i])
            self.qtgui_time_sink_x_1_4_0.set_line_width(i, widths[i])
            self.qtgui_time_sink_x_1_4_0.set_line_color(i, colors[i])
            self.qtgui_time_sink_x_1_4_0.set_line_style(i, styles[i])
            self.qtgui_time_sink_x_1_4_0.set_line_marker(i, markers[i])
            self.qtgui_time_sink_x_1_4_0.set_line_alpha(i, alphas[i])

        self._qtgui_time_sink_x_1_4_0_win = sip.wrapinstance(self.qtgui_time_sink_x_1_4_0.qwidget(), Qt.QWidget)
        self.top_layout.addWidget(self._qtgui_time_sink_x_1_4_0_win)
        self.epy_block_0 = epy_block_0.blk(example_param=1.0)
        self.blocks_file_sink_0_0_1 = blocks.file_sink(gr.sizeof_gr_complex*1, IQ_Root + ".bin", False)
        self.blocks_file_sink_0_0_1.set_unbuffered(True)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.epy_block_0, 0), (self.qtgui_time_sink_x_1_4_0, 0))
        self.connect((self.uhd_usrp_source_0_0_0, 0), (self.blocks_file_sink_0_0_1, 0))
        self.connect((self.uhd_usrp_source_0_0_0, 0), (self.epy_block_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "dong")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_NOP(int(self.samp_rate*1))
        self.set_samp_time(int(self.samp_rate*1))
        self.qtgui_time_sink_x_1_4_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0_0_0.set_samp_rate(self.samp_rate)

    def get_EXPID(self):
        return self.EXPID

    def set_EXPID(self, EXPID):
        self.EXPID = EXPID
        self.set_IQ_Root(str(self.output_dir / self.EXPID.lstrip("\\/")))

    def get_samp_time(self):
        return self.samp_time

    def set_samp_time(self, samp_time):
        self.samp_time = samp_time

    def get_NOP(self):
        return self.NOP

    def set_NOP(self, NOP):
        self.NOP = NOP

    def get_IQ_Root(self):
        return self.IQ_Root

    def set_IQ_Root(self, IQ_Root):
        self.IQ_Root = IQ_Root
        Path(self.IQ_Root).parent.mkdir(parents=True, exist_ok=True)
        self.blocks_file_sink_0_0_1.open(self.IQ_Root + ".bin")

    def get_CF(self):
        return self.CF

    def set_CF(self, CF):
        self.CF = CF
        self.uhd_usrp_source_0_0_0.set_center_freq(self.CF, 0)
        self.uhd_usrp_source_0_0_0.set_center_freq(self.CF, 1)
        self.uhd_usrp_source_0_0_0.set_center_freq(self.CF, 2)
        self.uhd_usrp_source_0_0_0.set_center_freq(self.CF, 3)

    def get_BW(self):
        return self.BW

    def set_BW(self, BW):
        self.BW = BW
        self.uhd_usrp_source_0_0_0.set_bandwidth(self.BW, 0)
        self.uhd_usrp_source_0_0_0.set_bandwidth(self.BW, 1)
        self.uhd_usrp_source_0_0_0.set_bandwidth(self.BW, 2)




def main(top_block_cls=dong, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
