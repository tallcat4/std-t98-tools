#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-channel diagnostic scope for the STD-T98 receiver.

Shows what one channel looks like at each stage of the real receive chain, so
a channel that decodes nothing can be told apart from a channel with no signal
on it, a mistuned one, and one whose symbol timing never locks.

Four views, left to right through the chain:

  1. RF spectrum + waterfall over the whole captured band -- is anything
     transmitting, and where does it sit relative to the tuned centre?
  2. Channel spectrum after channelisation and the second resampler -- is the
     carrier centred in its 6.25 kHz slot, or offset?
  3. Eye diagram of the matched-filter output -- is there a usable eye to
     sample, and how far open is it?
  4. Recovered symbols after the timing loop -- did symbol sync lock?

The DSP is lifted from std_t98_30ch_multi_rf_backend so what is on screen is
what the backend sees; only the sinks differ.

    python3 tools/std_t98_channel_scope.py --driver uhd --antenna TX/RX \\
        --gain 10 --channel 1
"""

import math
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5 import Qt
import sip

from gnuradio import analog, blocks, digital, filter, gr, qtgui
from gnuradio.fft import window
from gnuradio.filter import pfb

from core.rf.backend_config import (
    add_config_arguments,
    derive_rates,
    load_config,
)
from core.rf.soapy_source import open_source
from firdes import make_rx_taps


class ChannelScope(gr.top_block, Qt.QWidget):
    def __init__(self, config, channel, use_squelch, eye_sps=None, replay=None):
        gr.top_block.__init__(self, "STD-T98 Channel Scope", catch_exceptions=True)
        Qt.QWidget.__init__(self)

        sdr_cfg = config.sdr
        rates = derive_rates(sdr_cfg, config.channelizer)
        num_channels = config.channelizer.num_channels
        spacing = config.channelizer.channel_spacing

        if not 0 <= channel < num_channels:
            raise ValueError(
                f"channel must be 0..{num_channels - 1} (got {channel})"
            )

        tuned_freq = sdr_cfg.tuned_freq()
        channel_freq = tuned_freq + (channel - num_channels // 2) * spacing

        self.setWindowTitle(
            f"STD-T98 ch{channel} (登録局 ch{channel + 1}) "
            f"{channel_freq / 1e6:.5f} MHz"
        )
        layout = Qt.QVBoxLayout(self)
        top = Qt.QHBoxLayout()
        bottom = Qt.QHBoxLayout()
        layout.addLayout(top)
        layout.addLayout(bottom)

        # ---------------- demodulation parameters (mirrors the backend) -----
        demod_samp_rate = rates.demod_samp_rate
        fsk_dev = 315
        fm_demod_gain = demod_samp_rate / (2 * math.pi * fsk_dev)
        baud_rate = 2400
        sps = demod_samp_rate / baud_rate
        excess_bw = 0.2
        filter_ntaps = int(sps) * 20
        post_filt_gain = 0.23
        post_sync_gain = 5

        # ---------------- source ------------------------------------------
        self.replay = replay
        if replay:
            # A recording from std_t98_record_iq.py is already past stage 1, so
            # it feeds the channelizer directly and needs a throttle to play at
            # the rate it was captured at.
            self.file_source = blocks.file_source(
                gr.sizeof_gr_complex, str(replay), True)
            self.throttle = blocks.throttle(
                gr.sizeof_gr_complex, rates.samp_rate_post_resamp1, True, 0)
        else:
            # Same helper the backend uses, so the device is configured
            # identically -- rate validation and snapping, driver-aware stream
            # args and bandwidth, antenna checking, and the gain fallbacks.
            self._source = open_source(sdr_cfg)
            self.source = self._source.source
            self.rotator = blocks.rotator_cc(0.0)
            self.resamp1 = filter.rational_resampler_ccc(
                interpolation=rates.resamp1_interp,
                decimation=rates.resamp1_decim,
                taps=[], fractional_bw=0)

        # ---------------- channelisation ----------------------------------
        pfb_cutoff = rates.samp_rate_post_pfb / 2.0
        pfb_taps = filter.firdes.low_pass_2(
            1.0, rates.samp_rate_post_resamp1, pfb_cutoff, pfb_cutoff * 0.5,
            80.0, window.WIN_BLACKMAN_HARRIS)
        self.channelizer = pfb.channelizer_ccf(
            rates.pfb_num_channels, pfb_taps, 1.0)
        self.channelizer.set_channel_map(rates.channel_map)

        # Only one channel is examined; the rest must still be consumed.
        self.nulls = []
        for port in range(rates.pfb_num_channels):
            if port == channel:
                continue
            sink = blocks.null_sink(gr.sizeof_gr_complex)
            self.nulls.append(sink)
            self.connect((self.channelizer, port), (sink, 0))

        # ---------------- one channel's demodulator ------------------------
        self.squelch = analog.simple_squelch_cc(-25, 1)
        self.resamp2 = filter.rational_resampler_ccc(
            interpolation=rates.resamp2_interp,
            decimation=rates.resamp2_decim,
            taps=[], fractional_bw=0)
        self.quad_demod = analog.quadrature_demod_cf(fm_demod_gain)
        self.rx_filter = filter.fft_filter_fff(
            1, make_rx_taps(demod_samp_rate, baud_rate, excess_bw, filter_ntaps), 1)
        self.rx_filter.declare_sample_delay(0)
        self.filt_gain = blocks.multiply_const_ff(post_filt_gain)
        self.symbol_sync = digital.symbol_sync_ff(
            digital.TED_GARDNER, sps, 0.06, 1.1, 0.1, 0.02, 1,
            digital.constellation_bpsk().base(), digital.IR_MMSE_8TAP, 128, [])
        self.sync_gain = blocks.multiply_const_ff(post_sync_gain)

        # ---------------- sinks -------------------------------------------
        rf_view_rate = (rates.samp_rate_post_resamp1 if replay
                        else sdr_cfg.sample_rate)
        self.rf_spectrum = qtgui.freq_sink_c(
            4096, window.WIN_BLACKMAN_hARRIS, tuned_freq, rf_view_rate,
            "RF spectrum (all 30 channels)", 1, None)
        self.rf_spectrum.set_y_axis(-130, 0)
        self.rf_spectrum.enable_grid(True)
        self.rf_spectrum.set_fft_average(0.2)
        top.addWidget(sip.wrapinstance(self.rf_spectrum.qwidget(), Qt.QWidget))

        self.rf_waterfall = qtgui.waterfall_sink_c(
            2048, window.WIN_BLACKMAN_hARRIS, tuned_freq, rf_view_rate,
            "RF waterfall", 1, None)
        self.rf_waterfall.set_intensity_range(-130, -30)
        top.addWidget(sip.wrapinstance(self.rf_waterfall.qwidget(), Qt.QWidget))

        self.ch_spectrum = qtgui.freq_sink_c(
            1024, window.WIN_BLACKMAN_hARRIS, channel_freq, demod_samp_rate,
            f"ch{channel} after channelisation ({demod_samp_rate/1e3:.1f} kHz)",
            1, None)
        self.ch_spectrum.set_y_axis(-130, 10)
        self.ch_spectrum.enable_grid(True)
        top.addWidget(sip.wrapinstance(self.ch_spectrum.qwidget(), Qt.QWidget))

        # The first argument is the working buffer, not the trace length --
        # passing two symbols' worth leaves the sink with too little to work
        # with and it draws the axes but never a trace at all. The display is
        # always two symbols wide, set by samp_per_symbol.
        # The chain runs at 62500/2400 = 26.0417 samples per symbol and the
        # sink only takes an integer, so each 2-symbol trace ends 0.0032 of a
        # symbol short of the last one and the traces walk sideways instead of
        # overlaying into an eye. Resample the display tap -- and only the
        # display tap -- onto an exact integer rate so successive traces land
        # on top of each other.
        self.eye_sps = eye_sps or int(round(sps))
        eye_rate = baud_rate * self.eye_sps
        self.eye_resamp = filter.mmse_resampler_ff(0.0, demod_samp_rate / eye_rate)
        # Scale to the units the sync correlator works in, so the eye's levels
        # can be read directly against the sync word's +/-1 and +/-3 and match
        # the symbol plot beside it.
        self.eye_gain = blocks.multiply_const_ff(post_sync_gain)
        # The buffer sets how many traces get overlaid, and an eye only takes
        # shape once enough of them accumulate to fill in every transition
        # path: at 1024 samples there are 20 traces and the picture reads as
        # random squiggles, while 8192 gives about 157 and the openings appear.
        self.eye = qtgui.eye_sink_f(8192, eye_rate, 1, None)
        self.eye.set_samp_per_symbol(self.eye_sps)
        # Room for the +/-3 outer levels and their overshoot, and for some
        # residual frequency-error DC on top.
        self.eye.set_y_axis(-5.0, 5.0)
        self.eye.enable_grid(True)
        self.eye.set_update_time(0.10)
        bottom.addWidget(sip.wrapinstance(self.eye.qwidget(), Qt.QWidget))

        self.symbols = qtgui.time_sink_f(
            256, baud_rate, "recovered symbols (after timing recovery)", 1, None)
        self.symbols.set_y_axis(-6, 6)
        self.symbols.enable_grid(True)
        self.symbols.set_update_time(0.05)
        bottom.addWidget(sip.wrapinstance(self.symbols.qwidget(), Qt.QWidget))

        # ---------------- wiring ------------------------------------------
        if self.replay:
            self.connect((self.file_source, 0), (self.throttle, 0))
            self.connect((self.throttle, 0), (self.channelizer, 0))
            self.connect((self.throttle, 0), (self.rf_spectrum, 0))
            self.connect((self.throttle, 0), (self.rf_waterfall, 0))
        else:
            self.connect((self.source, 0), (self.rotator, 0))
            self.connect((self.source, 0), (self.rf_spectrum, 0))
            self.connect((self.source, 0), (self.rf_waterfall, 0))
            self.connect((self.rotator, 0), (self.resamp1, 0))
            self.connect((self.resamp1, 0), (self.channelizer, 0))

        if use_squelch:
            self.connect((self.channelizer, channel), (self.squelch, 0))
            self.connect((self.squelch, 0), (self.resamp2, 0))
        else:
            self.connect((self.channelizer, channel), (self.resamp2, 0))

        self.connect((self.resamp2, 0), (self.ch_spectrum, 0))
        self.connect((self.resamp2, 0), (self.quad_demod, 0))
        self.connect((self.quad_demod, 0), (self.rx_filter, 0))
        self.connect((self.rx_filter, 0), (self.filt_gain, 0))
        self.connect((self.filt_gain, 0), (self.eye_resamp, 0))
        self.connect((self.eye_resamp, 0), (self.eye_gain, 0))
        self.connect((self.eye_gain, 0), (self.eye, 0))
        self.connect((self.filt_gain, 0), (self.symbol_sync, 0))
        self.connect((self.symbol_sync, 0), (self.sync_gain, 0))
        self.connect((self.sync_gain, 0), (self.symbols, 0))

        if replay:
            print(f"replaying     {replay}")
        print(f"tuned to      {tuned_freq/1e6:.5f} MHz @ {sdr_cfg.sample_rate/1e6:.3f} Msps")
        print(f"watching      ch{channel} (登録局 ch{channel+1}) = {channel_freq/1e6:.5f} MHz")
        print(f"demod rate    {demod_samp_rate:.0f} Hz, {sps:.4f} samples/symbol")
        print(f"squelch       {'on (-25 dB)' if use_squelch else 'bypassed'}")
        print(f"eye           {self.eye_sps} samples/symbol, display resampled "
              f"{demod_samp_rate:.0f} -> {eye_rate} Hz for a stable trace")
        print(f"freq err      {sdr_cfg.resolved_freq_err_offset():+.0f} Hz "
              f"-> tuned {sdr_cfg.tuned_freq()/1e6:.5f} MHz")

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Single-channel spectrum and eye-diagram scope for STD-T98.",
    )
    add_config_arguments(parser)
    parser.add_argument(
        "--channel", type=int, default=0,
        help="Channel to examine, 0-29 (0 = 登録局 ch1 = 351.20000 MHz).")
    parser.add_argument(
        "--no-squelch", dest="squelch", action="store_false", default=True,
        help="Bypass the squelch, so a weak signal is still visible.")
    parser.add_argument(
        "--replay", default=None,
        help="Replay a recording from std_t98_record_iq.py instead of opening "
        "the SDR. Lets a capture be examined without the radio.")
    parser.add_argument(
        "--eye-sps", type=int, default=None,
        help="Samples per symbol for the eye display. Defaults to the chain's "
        "rate rounded to an integer (26); the true rate is fractional, so the "
        "eye drifts at that setting. A smaller value gives a stable picture.")
    args = parser.parse_args()

    config = load_config(args)

    qapp = Qt.QApplication(sys.argv)
    tb = ChannelScope(config, args.channel, args.squelch, args.eye_sps,
                      args.replay)
    tb.start()
    tb.show()

    def quit_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, quit_handler)
    signal.signal(signal.SIGTERM, quit_handler)
    # Let Python process signals while Qt owns the loop.
    timer = Qt.QTimer()
    timer.start(200)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == "__main__":
    main()
