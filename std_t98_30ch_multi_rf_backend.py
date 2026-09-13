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
from gnuradio.filter import pfb
import std_t98_multi_sync as sync_word_corr  # embedded python block
import threading

from firdes import make_rx_taps
from core.rf.backend_config import BackendConfig, derive_rates
from core.rf.soapy_source import open_source

class test3(gr.top_block):

    def __init__(self, config: BackendConfig | None = None, replay=None):
        gr.top_block.__init__(self, "Test 3", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        self.config = config = config or BackendConfig()
        sdr_cfg = config.sdr
        rates = derive_rates(sdr_cfg, config.channelizer)

        ##################################################
        # 1. SDR / RF Parameters
        ##################################################
        self.rf_samp_rate = rf_samp_rate = sdr_cfg.sample_rate
        self.rf_freq = rf_freq = sdr_cfg.center_freq
        self.freq_offset = freq_offset = sdr_cfg.freq_offset
        self.freq_err_offset = freq_err_offset = sdr_cfg.resolved_freq_err_offset()

        self.sdr_tuner_gain = sdr_tuner_gain = sdr_cfg.tuner_gain
        self.sdr_agc_enabled = sdr_agc_enabled = sdr_cfg.agc
        self.sdr_biastee_enabled = sdr_biastee_enabled = sdr_cfg.bias_tee
        self.sdr_freq_corr = sdr_freq_corr = sdr_cfg.freq_correction

        ##################################################
        # 2. Resampling & Channelization Rates (derived from sample rate)
        ##################################################
        # Stage 1: Initial resampling to the channelizer input rate
        self.resamp1_interp = resamp1_interp = rates.resamp1_interp
        self.resamp1_decim = resamp1_decim = rates.resamp1_decim
        self.samp_rate_post_resamp1 = samp_rate_post_resamp1 = rates.samp_rate_post_resamp1

        # Stage 2: PFB Channelizer
        self.pfb_num_channels = pfb_num_channels = rates.pfb_num_channels
        self.num_channels = num_channels = config.channelizer.num_channels
        self.samp_rate_post_pfb = samp_rate_post_pfb = rates.samp_rate_post_pfb

        # Stage 3: Second Resampling (per channel)
        self.resamp2_interp = resamp2_interp = rates.resamp2_interp
        self.resamp2_decim = resamp2_decim = rates.resamp2_decim
        self.demod_samp_rate = demod_samp_rate = rates.demod_samp_rate

        ##################################################
        # 3. Demodulation Parameters
        ##################################################
        self.fsk_dev = fsk_dev = 315
        self.fm_demod_gain = fm_demod_gain = demod_samp_rate / (2 * math.pi * fsk_dev)
        
        self.squelch_threshold = squelch_threshold = config.demod.squelch_threshold
        self.squelch_alpha = squelch_alpha = config.demod.squelch_alpha

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
        self.sync_error_threshold_ratio = sync_error_threshold_ratio = 0.2
        self.sync_packet_len = sync_packet_len = 192

        ##################################################
        # Blocks
        ##################################################
        self.replay = replay
        if replay:
            # A recording from std_t98_record_iq.py is already past stage 1, so
            # it joins the flowgraph at the channelizer. This lets the whole
            # four-process stack be exercised on a machine with no SDR at all,
            # deterministically and as often as needed.
            self.file_source = blocks.file_source(
                gr.sizeof_gr_complex, str(replay), False)
            self.replay_throttle = blocks.throttle(
                gr.sizeof_gr_complex, samp_rate_post_resamp1, True, 0)
        else:
            self._source = open_source(sdr_cfg)
            self.soapy_source_0 = self._source.source

            # Names kept for anything that reached into the flowgraph before
            # the device handling moved into core.rf.soapy_source.
            self.set_soapy_source_0_gain_mode = self._source.set_gain_mode
            self.set_soapy_source_0_gain = self._source.set_gain
            self.set_soapy_source_0_bias = self._source.set_bias
            self.soapy_rtlsdr_source_0 = self.soapy_source_0

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
        
        # Ports 0..neg-1 -> top FFT bins (negative freqs, CH below centre),
        # port neg -> bin 0 (centre channel), then low positive bins, then the
        # unused bins (attached to null sinks below). Derived from the channel
        # counts so it tracks pfb_num_channels / num_channels automatically.
        self.channel_map = rates.channel_map
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
            sync_word=None,
            error_threshold_ratio=sync_error_threshold_ratio,
            packet_len=sync_packet_len)

        ##################################################
        # Connections
        ##################################################
        if self.replay:
            self.connect((self.file_source, 0), (self.replay_throttle, 0))
            self.connect((self.replay_throttle, 0), (self.pfb_channelizer_ccf_0, 0))
        else:
            # No throttle here: the SDR's sample clock already paces the
            # flowgraph, and GNU Radio's throttle is explicitly not meant to
            # share a graph with a hardware source.
            self.connect((self.soapy_source_0, 0), (self.blocks_freqshift_cc_0, 0))
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


def _parse_args(argv=None):
    import argparse

    from core.rf.backend_config import add_config_arguments

    parser = argparse.ArgumentParser(
        description="STD-T98 30ch multi-channel RF backend (SoapySDR)."
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--replay",
        help="Replay a recording from tools/std_t98_record_iq.py instead of "
        "opening the SDR, so the stack can be exercised without hardware.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and derived rates, then exit without "
        "opening the SDR or starting the flowgraph.",
    )
    return parser.parse_args(argv)


def _print_dry_run(config):
    rates = derive_rates(config.sdr, config.channelizer)
    print("[sdr]")
    for field_name in config.sdr.__dataclass_fields__:
        print(f"  {field_name} = {getattr(config.sdr, field_name)!r}")
    # Resolved view of the two settings that decide whether the device opens
    # at all, so a failing driver can be diagnosed without starting the SDR.
    print("[sdr.resolved]")
    print(f"  device_string = {config.sdr.device_string()!r}")
    print(f"  freq_err_offset = {config.sdr.resolved_freq_err_offset():+.1f} Hz")
    print(f"  tuned_freq = {config.sdr.tuned_freq():.0f} Hz")
    print(f"  stream_args = {config.sdr.resolved_stream_args()!r}")
    print(f"  antenna = {config.sdr.antenna or '(driver default)'}")
    bandwidth = config.sdr.resolved_bandwidth()
    print(f"  bandwidth = {bandwidth if bandwidth else '(device default)'}")
    print("[demod]")
    for field_name in config.demod.__dataclass_fields__:
        print(f"  {field_name} = {getattr(config.demod, field_name)!r}")
    print("[channelizer]")
    for field_name in config.channelizer.__dataclass_fields__:
        print(f"  {field_name} = {getattr(config.channelizer, field_name)!r}")
    print("[derived]")
    print(f"  resamp1 = {rates.resamp1_interp}/{rates.resamp1_decim}")
    print(f"  samp_rate_post_resamp1 = {rates.samp_rate_post_resamp1}")
    print(f"  samp_rate_post_pfb (bin width) = {rates.samp_rate_post_pfb}")
    print(f"  demod_samp_rate = {rates.demod_samp_rate}")
    print(f"  bin_width_error = {rates.bin_width_error_hz:+.4f} Hz")
    print(f"  channel_map = {rates.channel_map}")


def main(top_block_cls=test3, options=None):
    from core.rf.backend_config import load_config

    args = _parse_args()
    config = load_config(args)
    if args.dry_run:
        _print_dry_run(config)
        return

    tb = top_block_cls(config=config, replay=args.replay)
    if args.replay:
        print(f"replaying {args.replay} instead of opening the SDR")

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