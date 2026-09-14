#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record the channelizer's input to a file for offline analysis.

Capturing once and analysing offline beats re-transmitting for every question:
the recording holds all 30 channels, so channel choice, frequency offset,
symbol levels and timing can all be re-examined afterwards without touching
the radio again.

The tap is after the stage-1 resampler, at pfb_num_channels * channel_spacing
(300 kHz by default), stored as interleaved complex float32 -- the format
GNU Radio's file source and numpy both read directly.

    python3 tools/std_t98_record_iq.py --sample-rate 2000000 \\
        --antenna TX/RX --gain-element PGA --gain 30 \\
        --freq-err-offset 1030 --seconds 30 -o capture.cf32
"""

import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gnuradio import blocks, filter, gr
from gnuradio.fft import window
from gnuradio.filter import pfb

from core.rf.backend_config import add_config_arguments, derive_rates, load_config
from core.rf.uhd_source import open_source


class Recorder(gr.top_block):
    def __init__(self, config, path):
        gr.top_block.__init__(self, "STD-T98 IQ recorder", catch_exceptions=True)

        sdr_cfg = config.sdr
        self.rates = rates = derive_rates(sdr_cfg, config.channelizer)

        self._source = open_source(sdr_cfg)
        self.source = self._source.source

        self.rotator = blocks.rotator_cc(0.0)
        self.resamp1 = filter.rational_resampler_ccc(
            interpolation=rates.resamp1_interp,
            decimation=rates.resamp1_decim,
            taps=[], fractional_bw=0)
        self.sink = blocks.file_sink(gr.sizeof_gr_complex, str(path), False)
        self.sink.set_unbuffered(False)

        self.connect(self.source, self.rotator, self.resamp1, self.sink)

        self.tuned_freq = sdr_cfg.tuned_freq()
        self.record_rate = rates.samp_rate_post_resamp1


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Record STD-T98 channelizer input for offline analysis.")
    add_config_arguments(parser)
    parser.add_argument("--seconds", type=float, default=30.0,
                        help="How long to record.")
    parser.add_argument("-o", "--output", default="capture.cf32",
                        help="Output file (interleaved complex float32).")
    args = parser.parse_args()

    config = load_config(args)
    path = Path(args.output).resolve()
    tb = Recorder(config, path)

    rate = tb.record_rate
    size_mb = rate * 8 * args.seconds / 1e6
    print(f"tuned to    {tb.tuned_freq/1e6:.5f} MHz")
    print(f"recording   {rate:.0f} Hz complex float32 -> {path}")
    print(f"expected    {args.seconds:.0f} s, about {size_mb:.0f} MB")
    print(f"channels    {config.channelizer.num_channels} of them, "
          f"{config.channelizer.channel_spacing} Hz apart, all captured")

    stopping = False

    def stop(sig=None, frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    tb.start()
    print("\n>>> RECORDING -- key up now <<<\n", flush=True)
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline and not stopping:
        remaining = deadline - time.monotonic()
        print(f"\r  {remaining:5.1f}s left", end="", flush=True)
        time.sleep(0.5)
    tb.stop()
    tb.wait()

    written = path.stat().st_size if path.exists() else 0
    print(f"\n\nwrote {written/1e6:.1f} MB "
          f"({written/8/rate:.1f} s at {rate:.0f} Hz)")


if __name__ == "__main__":
    main()
