#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# GNU Radio version: 3.10.12.0

from gnuradio import analog
import math
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from gnuradio import soapy
from gnuradio.filter import pfb
import std_t98_multi_sync as sync_word_corr  # embedded python block
import threading

from firdes import make_rx_taps

class test3(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Test 3", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # 1. SDR / RF Parameters
        ##################################################
        self.rf_samp_rate = rf_samp_rate = 1.2e6
        self.rf_freq = rf_freq = 351.29375e6
        self.freq_offset = freq_offset = 0
        self.freq_err_offset = freq_err_offset = -340
        
        self.sdr_tuner_gain = sdr_tuner_gain = 30
        self.sdr_agc_enabled = sdr_agc_enabled = False
        self.sdr_biastee_enabled = sdr_biastee_enabled = False
        self.sdr_freq_corr = sdr_freq_corr = 0

        ##################################################
        # 2. Resampling & Channelization Rates
        ##################################################
        # Stage 1: Initial Decimation
        self.resamp1_interp = resamp1_interp = 1
        self.resamp1_decim = resamp1_decim = 4  
        self.samp_rate_post_resamp1 = samp_rate_post_resamp1 = rf_samp_rate * resamp1_interp / resamp1_decim
        
        # Stage 2: PFB Channelizer
        self.pfb_num_channels = pfb_num_channels = 48
        self.num_channels = num_channels = 30
        self.samp_rate_post_pfb = samp_rate_post_pfb = samp_rate_post_resamp1 / pfb_num_channels
        
        # Stage 3: Second Resampling (per channel)
        self.resamp2_interp = resamp2_interp = 10
        self.resamp2_decim = resamp2_decim = 1
        self.demod_samp_rate = demod_samp_rate = samp_rate_post_pfb * resamp2_interp / resamp2_decim

        ##################################################
        # 3. Demodulation Parameters
        ##################################################
        self.fsk_dev = fsk_dev = 315
        self.fm_demod_gain = fm_demod_gain = demod_samp_rate / (2 * math.pi * fsk_dev)
        
        self.squelch_threshold = squelch_threshold = -25
        self.squelch_alpha = squelch_alpha = 1

        ##################################################
        # 4. FSK & Timing Sync Parameters
        ##################################################
        self.baud_rate = baud_rate = 2400
        self.sps = sps = demod_samp_rate / baud_rate 
        
        self.excess_bw = excess_bw = 0.2
        self.filter_ntaps_per_sym = filter_ntaps_per_sym = 20
        self.filter_ntaps = filter_ntaps = int(sps) * filter_ntaps_per_sym 

        self.sym_sync_loop_bw = sym_sync_loop_bw = 0.06
        self.sym_sync_damping = sym_sync_damping = 1.1
        self.sym_sync_ted_gain = sym_sync_ted_gain = 0.1
        self.sym_sync_max_dev = sym_sync_max_dev = 0.02
        self.sym_sync_osps = sym_sync_osps = 1
        self.sym_sync_interp = sym_sync_interp = 128

        ##################################################
        # 5. Gains and Miscellaneous
        ##################################################
        self.post_filt_gain = post_filt_gain = 0.23
        self.post_sync_gain = post_sync_gain = 5
        self.rotator_phase_inc = rotator_phase_inc = 0.0
        self.throttle_max_items_per_block = throttle_max_items_per_block = 0

        ##################################################
        # 6. PFB Taps Generation
        ##################################################
        pfb_cutoff = samp_rate_post_pfb / 2.0
        pfb_trans_bw = pfb_cutoff * 0.5 
        
        self.pfb_taps = filter.firdes.low_pass_2(
            1.0,
            samp_rate_post_resamp1,
            pfb_cutoff,
            pfb_trans_bw,
            80.0,
            window.WIN_BLACKMAN_HARRIS
        )

        ##################################################
        # 7. Sync Word Correlator Setup
        ##################################################
        self.zmq_ip = zmq_ip = "127.0.0.1"
        self.zmq_port = zmq_port = 5555
        self.sync_error_threshold = sync_error_threshold = 0.2
        self.sync_packet_len = sync_packet_len = 192

        ##################################################
        # Blocks
        ##################################################
        self.soapy_rtlsdr_source_0 = None
        dev = 'driver=rtlsdr'
        stream_args = 'bufflen=16384'
        tune_args = ['']
        settings = ['']

        def _set_soapy_rtlsdr_source_0_gain_mode(channel, agc):
            self.soapy_rtlsdr_source_0.set_gain_mode(channel, agc)
            if not agc:
                  self.soapy_rtlsdr_source_0.set_gain(channel, self._soapy_rtlsdr_source_0_gain_value)
        self.set_soapy_rtlsdr_source_0_gain_mode = _set_soapy_rtlsdr_source_0_gain_mode

        def _set_soapy_rtlsdr_source_0_gain(channel, name, gain):
            self._soapy_rtlsdr_source_0_gain_value = gain
            if not self.soapy_rtlsdr_source_0.get_gain_mode(channel):
                self.soapy_rtlsdr_source_0.set_gain(channel, gain)
        self.set_soapy_rtlsdr_source_0_gain = _set_soapy_rtlsdr_source_0_gain

        def _set_soapy_rtlsdr_source_0_bias(bias):
            if 'biastee' in self._soapy_rtlsdr_source_0_setting_keys:
                self.soapy_rtlsdr_source_0.write_setting('biastee', bias)
        self.set_soapy_rtlsdr_source_0_bias = _set_soapy_rtlsdr_source_0_bias

        self.soapy_rtlsdr_source_0 = soapy.source(dev, "fc32", 1, '', stream_args, tune_args, settings)
        self._soapy_rtlsdr_source_0_setting_keys =[a.key for a in self.soapy_rtlsdr_source_0.get_setting_info()]

        self.soapy_rtlsdr_source_0.set_sample_rate(0, rf_samp_rate)
        self.soapy_rtlsdr_source_0.set_frequency(0, (rf_freq + freq_offset + freq_err_offset))
        self.soapy_rtlsdr_source_0.set_frequency_correction(0, sdr_freq_corr)
        self.set_soapy_rtlsdr_source_0_bias(bool(sdr_biastee_enabled))
        self._soapy_rtlsdr_source_0_gain_value = sdr_tuner_gain
        self.set_soapy_rtlsdr_source_0_gain_mode(0, bool(sdr_agc_enabled))
        self.set_soapy_rtlsdr_source_0_gain(0, 'TUNER', sdr_tuner_gain)

        self.blocks_throttle_1 = blocks.throttle(gr.sizeof_gr_complex*1, rf_samp_rate, True, throttle_max_items_per_block)
        self.blocks_freqshift_cc_0 = blocks.rotator_cc(rotator_phase_inc)

        self.rational_resampler_1 = filter.rational_resampler_ccc(
                interpolation=resamp1_interp,
                decimation=resamp1_decim,
                taps=[],
                fractional_bw=0)
        
        ##################################################
        # PFB Channelizer & Mapping
        ##################################################
        self.pfb_channelizer_ccf_0 = pfb.channelizer_ccf(
            pfb_num_channels,
            self.pfb_taps,
            1.0
        )
        
        # CH1(-93.75k) = Bin 33, CH15(-6.25k) = Bin 47
        # CH16(0) = Bin 0
        # CH17(+6.25k) = Bin 1, CH30(+87.5k) = Bin 14
        self.channel_map = [
            # Port 0-14  (CH1-15) -> Bin 33-47
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            # Port 15    (CH16) -> Bin 0
            0,
            # Port 16-29 (CH17-30) -> Bin 1-14
            1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
            # Port 30-47 (Unused) -> Bin 15-32 (Null Sink)
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ]
        self.pfb_channelizer_ccf_0.set_channel_map(self.channel_map)
        
        ##################################################
        # Per-Channel Blocks
        ##################################################
        self.simple_squelch = [None] * num_channels
        self.rational_resampler_2 = [None] * num_channels
        self.analog_quadrature_demod =[None] * num_channels
        self.fft_filter = [None] * num_channels
        self.multiply_const_1 = [None] * num_channels
        self.symbol_sync = [None] * num_channels
        self.multiply_const_2 = [None] * num_channels
        
        for ch in range(num_channels):
            self.simple_squelch[ch] = analog.simple_squelch_cc((squelch_threshold), squelch_alpha)
            
            self.rational_resampler_2[ch] = filter.rational_resampler_ccc(
                interpolation=resamp2_interp,
                decimation=resamp2_decim,
                taps=[],
                fractional_bw=0)
                
            self.analog_quadrature_demod[ch] = analog.quadrature_demod_cf(fm_demod_gain)
            
            rx_taps = make_rx_taps(demod_samp_rate, baud_rate, excess_bw, filter_ntaps)
            self.fft_filter[ch] = filter.fft_filter_fff(1, rx_taps, 1)
            self.fft_filter[ch].declare_sample_delay(0)
            
            self.multiply_const_1[ch] = blocks.multiply_const_ff(post_filt_gain)
            
            self.symbol_sync[ch] = digital.symbol_sync_ff(
                digital.TED_GARDNER,
                sps,
                sym_sync_loop_bw,
                sym_sync_damping,
                sym_sync_ted_gain,
                sym_sync_max_dev,
                sym_sync_osps,
                digital.constellation_bpsk().base(),
                digital.IR_MMSE_8TAP,
                sym_sync_interp,[])
                
            self.multiply_const_2[ch] = blocks.multiply_const_ff(post_sync_gain)

        self.sync_word_corr = sync_word_corr.sync_word_correlator(
            num_channels=num_channels,
            zmq_ip=zmq_ip,
            zmq_port=zmq_port,
            sync_word=None,
            error_threshold_ratio=sync_error_threshold,
            packet_len=sync_packet_len)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.soapy_rtlsdr_source_0, 0), (self.blocks_throttle_1, 0))
        self.connect((self.blocks_throttle_1, 0), (self.blocks_freqshift_cc_0, 0))
        self.connect((self.blocks_freqshift_cc_0, 0), (self.rational_resampler_1, 0))
        self.connect((self.rational_resampler_1, 0), (self.pfb_channelizer_ccf_0, 0))

        for ch in range(num_channels):
            self.connect((self.pfb_channelizer_ccf_0, ch), (self.simple_squelch[ch], 0))
            self.connect((self.simple_squelch[ch], 0), (self.rational_resampler_2[ch], 0))
            self.connect((self.rational_resampler_2[ch], 0), (self.analog_quadrature_demod[ch], 0))
            self.connect((self.analog_quadrature_demod[ch], 0), (self.fft_filter[ch], 0))
            self.connect((self.fft_filter[ch], 0), (self.multiply_const_1[ch], 0))
            self.connect((self.multiply_const_1[ch], 0), (self.symbol_sync[ch], 0))
            self.connect((self.symbol_sync[ch], 0), (self.multiply_const_2[ch], 0))
            self.connect((self.multiply_const_2[ch], 0), (self.sync_word_corr, ch))
        
        self.null_sinks = []
        for ch in range(num_channels, pfb_num_channels):
            ns = blocks.null_sink(gr.sizeof_gr_complex)
            self.null_sinks.append(ns)
            self.connect((self.pfb_channelizer_ccf_0, ch), (ns, 0))

def main(top_block_cls=test3, options=None):
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