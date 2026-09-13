#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure a recording from std_t98_record_iq.py.

Answers, from one capture and without the radio, the questions that come up
when a channel will not decode:

  * is anything transmitting, on which channel, and for how long?
  * how far off frequency is it? (this is what to put in --freq-err-offset)
  * what is the transmitter's real symbol rate?
  * does the sync correlator actually fire, and how much margin is there?

Each of these was needed to bring the receiver up on a USRP B210, and each
failed in a way that looks identical from the outside -- a receiver that runs
perfectly and decodes nothing.

    python3 tools/std_t98_analyse_capture.py capture.cf32
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rf.backend_config import BackendConfig
from firdes import make_rx_taps

# A frequency error reaches the sync correlator as a DC level on the symbol
# stream. These are the constants that turn one into the other.
FSK_DEV = 315.0
POST_FILT_GAIN = 0.23
POST_SYNC_GAIN = 5.0
HZ_PER_SYMBOL_UNIT = FSK_DEV / (POST_FILT_GAIN * POST_SYNC_GAIN)

SYNC_WORD = np.array([-3., 1., -3., 3., -3., -3., 3., 3., -1., 3.])
SYNC_THRESHOLD = float(SYNC_WORD @ SYNC_WORD) * 0.2


def find_activity(iq, rate, spacing, num_channels):
    """Per-channel peak level and the span of time each was active."""
    nfft, hop = 2048, int(rate * 0.05)
    win = np.hanning(nfft)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / rate))
    offsets = [(k - num_channels // 2) * spacing for k in range(num_channels)]
    masks = [np.abs(freqs - o) <= spacing / 2 for o in offsets]

    frames = []
    for start in range(0, len(iq) - nfft, hop):
        psd = np.abs(np.fft.fftshift(np.fft.fft(iq[start:start + nfft] * win))) ** 2
        frames.append([10 * np.log10(psd[m].mean() + 1e-30) for m in masks])
    return np.array(frames), np.arange(len(frames)) * hop / rate


def channelize(iq, rate, offset, spacing, interp):
    """Bring one channel to DC and up to the demodulation rate."""
    from scipy import signal as sig

    t = np.arange(len(iq)) / rate
    shifted = iq * np.exp(-2j * np.pi * offset * t).astype(np.complex64)
    lp = sig.firwin(301, spacing / 2, fs=rate)
    narrow = sig.lfilter(lp, 1.0, shifted)[::int(rate // spacing)]
    return sig.resample_poly(narrow, interp, 1).astype(np.complex64)


def measure_symbol_rate(filt, demod_rate, baud):
    """Track the best sampling phase over time; its slope is the clock error."""
    sps = demod_rate / baud
    block = int(demod_rate * 0.5)
    phases, times = [], []
    steps = 208
    for k in range(len(filt) // block):
        seg = filt[k * block:(k + 1) * block]
        scores = np.empty(steps)
        for i in range(steps):
            idx = i * sps / steps + np.arange(int((len(seg) - sps) / sps)) * sps
            scores[i] = np.mean(np.abs(np.interp(idx, np.arange(len(seg)), seg)))
        phases.append(int(np.argmax(scores)) * sps / steps)
        times.append(k * 0.5)
    if len(phases) < 3:
        return None, None
    unwrapped = np.unwrap(np.array(phases) / sps * 2 * np.pi) / (2 * np.pi) * sps
    slope = np.polyfit(times, unwrapped, 1)[0]
    true_sps = sps - slope / (demod_rate / sps)
    return slope, demod_rate / true_sps


def count_sync(filt, demod_rate, baud):
    """Run the real timing recovery, then look for the sync word."""
    from gnuradio import blocks, digital, gr

    tb = gr.top_block()
    src = blocks.vector_source_f(filt.astype(np.float32).tolist(), False, 1, [])
    ss = digital.symbol_sync_ff(
        digital.TED_GARDNER, demod_rate / baud, 0.06, 1.1, 0.1, 0.02, 1,
        digital.constellation_bpsk().base(), digital.IR_MMSE_8TAP, 128, [])
    gain = blocks.multiply_const_ff(POST_SYNC_GAIN)
    snk = blocks.vector_sink_f()
    tb.connect(src, ss, gain, snk)
    tb.run()

    symbols = np.array(snk.data(), dtype=np.float64)
    if len(symbols) < len(SYNC_WORD):
        return symbols, 0, np.inf
    win = np.lib.stride_tricks.sliding_window_view(symbols, len(SYNC_WORD))
    sse = ((win - SYNC_WORD) ** 2).sum(axis=1)
    return symbols, int((sse <= SYNC_THRESHOLD).sum()), float(sse.min())


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure a std_t98_record_iq.py capture.")
    parser.add_argument("capture", help="Recording (complex float32).")
    parser.add_argument("--channel", type=int, default=None,
                        help="Channel to demodulate. Default: the strongest.")
    parser.add_argument("--baud", type=float, default=2400.0)
    args = parser.parse_args()

    cfg = BackendConfig()
    spacing = cfg.channelizer.channel_spacing
    num = cfg.channelizer.num_channels
    rate = float(cfg.channelizer.pfb_num_channels * spacing)
    demod_rate = float(cfg.channelizer.resamp2_interp * spacing)

    iq = np.fromfile(args.capture, dtype=np.complex64)
    print(f"{args.capture}: {len(iq)} samples = {len(iq)/rate:.1f}s at {rate:.0f} Hz")
    peak = float(np.abs(iq).max())
    print(f"peak |sample| {peak:.4f}"
          + ("  -- CLIPPING, reduce --gain" if peak > 0.95 else ""))

    frames, times = find_activity(iq, rate, spacing, num)
    floor = float(np.median(frames))
    peaks = frames.max(axis=0)
    print(f"\nnoise floor {floor:.1f} dB")
    active = [k for k in range(num) if peaks[k] - floor > 6]
    if not active:
        print("no channel rose more than 6 dB above the floor -- nothing to decode")
        return
    for k in active:
        on = frames[:, k] > floor + 6
        print(f"  ch{k:<2d} (登録局 ch{k+1:<2d}) {peaks[k] - floor:5.1f} dB above floor, "
              f"active {times[on][0]:5.2f}-{times[on][-1]:5.2f}s")

    ch = args.channel if args.channel is not None else max(active, key=lambda k: peaks[k])
    on = frames[:, ch] > floor + 6
    start, stop = times[on][0], times[on][-1]
    print(f"\n--- ch{ch} (登録局 ch{ch+1}), {start:.1f}-{stop:.1f}s ---")

    seg = iq[int(start * rate):int(stop * rate)]
    chan = channelize(seg, rate, (ch - num // 2) * spacing, spacing,
                      cfg.channelizer.resamp2_interp)

    demod = np.angle(chan[1:] * np.conj(chan[:-1])) * demod_rate / (2 * np.pi * FSK_DEV)
    taps = make_rx_taps(demod_rate, args.baud, 0.2,
                        int(demod_rate / args.baud) * 20)
    filt = np.convolve(demod, taps, mode="same") * POST_FILT_GAIN

    dc = float(filt.mean()) * POST_SYNC_GAIN
    print(f"frequency offset  {dc * HZ_PER_SYMBOL_UNIT:+8.0f} Hz "
          f"(DC {dc:+.2f} on the symbol stream)")
    print(f"                  adjust --freq-err-offset by this much to null it")
    print(f"                  sync tolerates about "
          f"{np.sqrt(SYNC_THRESHOLD/len(SYNC_WORD))*HZ_PER_SYMBOL_UNIT:.0f} Hz")

    slope, baud = measure_symbol_rate(filt, demod_rate, args.baud)
    if baud:
        print(f"symbol rate       {baud:9.4f} baud "
              f"({(baud/args.baud - 1)*1e6:+.1f} ppm), "
              f"phase drift {slope:+.3f} samples/s")

    symbols, hits, best = count_sync(filt, demod_rate, args.baud)
    print(f"symbol levels     RMS {np.sqrt(np.mean(symbols**2)):.2f} "
          f"(expect ~2.24 for equiprobable +/-1 and +/-3)")
    print(f"sync detections   {hits} (best SSE {best:.2f}, threshold "
          f"{SYNC_THRESHOLD})")
    if hits == 0:
        print("\n  no sync. In order of likelihood: the frequency offset above,")
        print("  then gain (check the level), then the wrong channel.")


if __name__ == "__main__":
    main()
