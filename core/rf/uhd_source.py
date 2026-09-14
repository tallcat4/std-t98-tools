"""Opening, configuring, and self-checking the UHD (USRP) source.

Kept out of :mod:`core.rf.backend_config`, which stays importable without GNU
Radio, and shared by the backend flowgraph and the diagnostic scope so both
get identical device handling: the same gain/antenna/sample-rate resolution.
A tool that configures the device slightly differently from the backend is a
tool that cannot be trusted to diagnose the backend.

This replaced a SoapySDR-based version of the same module: the backend now
targets USRP hardware through UHD directly (``gnuradio.uhd``), which is the
only way to get two things SoapySDR's generic wrapper cannot: a documented,
structured stream-discontinuity signal (:class:`OverflowTap`, below) and
UHD's real sensor tree (:func:`collect_health_snapshot`), instead of parsing
the single-character stream-health codes UHD prints to raw stderr.
"""

from __future__ import annotations

import time

import numpy as np
import pmt
from gnuradio import gr, uhd

from core.rf.backend_config import SdrConfig


class ConfiguredSource:
    """A UHD source plus the gain setter that knows about named gain stages."""

    def __init__(self, source, gain_element):
        self.source = source
        self.gain_element = gain_element
        try:
            self.gain_names = list(source.get_gain_names(0))
        except Exception:
            self.gain_names = []

    def set_gain(self, gain):
        """Prefer a named gain stage when the device (and config) name one;
        fall back to the overall gain otherwise."""
        if self.gain_element and self.gain_element in self.gain_names:
            self.source.set_gain(gain, self.gain_element, 0)
        else:
            self.source.set_gain(gain, 0)

    def set_gain_mode(self, agc):
        # Not every daughterboard exposes AGC; leave unsupported boards alone
        # rather than erroring, same as the other capability-dependent setters.
        try:
            self.source.set_rx_agc(bool(agc), 0)
        except Exception:
            pass


def _resolve_sample_rate(source, requested):
    """Return the rate to actually ask the device for.

    UHD boards derive their achievable rates from a master clock divider
    rather than advertising a short fixed list, so ``meta_range_t.clip()`` --
    which always returns the nearest achievable value, clamping instead of
    raising -- is the right primitive here (there is no finite list to
    enumerate the way SoapySDR's RTL-SDR backend used to advertise one).
    """
    try:
        return source.get_samp_rates().clip(requested)
    except Exception:
        return requested  # Device does not advertise a rate range.


def _set_antenna(source, antenna):
    """Select an RX port, refusing an unavailable name up front.

    Getting this wrong produces no error and no warning at runtime -- just a
    receiver that never hears anything -- so it is worth failing loudly.
    """
    try:
        available = list(source.get_antennas(0))
    except Exception:
        available = []
    if available and antenna not in available:
        raise ValueError(
            f"Antenna {antenna!r} is not available on this device. "
            f"Choose one of: {available}"
        )
    source.set_antenna(antenna, 0)


def open_source(sdr_cfg: SdrConfig) -> ConfiguredSource:
    """Open and fully configure the source described by ``sdr_cfg``."""
    stream_args = uhd.stream_args(cpu_format="fc32", otw_format="sc16")
    source = uhd.usrp_source(sdr_cfg.device_string(), stream_args)
    configured = ConfiguredSource(source, sdr_cfg.gain_element)

    # Only touch the antenna when asked: a single-input board has nothing to
    # select, and the device's own default is right for most others.
    if sdr_cfg.antenna:
        _set_antenna(source, sdr_cfg.antenna)

    requested = sdr_cfg.sample_rate
    source.set_samp_rate(_resolve_sample_rate(source, requested))
    # Belt and braces: clip() should already land on what get_samp_rate()
    # reports back, but the channelizer geometry is derived from the
    # requested rate, so a silent mismatch would mistune every channel.
    actual = source.get_samp_rate()
    if abs(actual - requested) > max(1.0, requested * 1e-6):
        raise ValueError(
            f"Device serves {actual:.0f} Hz, not the requested "
            f"{requested:.0f} Hz. Re-run with --sample-rate {actual:.0f} so "
            "the channelizer matches, or pick a rate the device supports "
            "exactly."
        )

    bandwidth = sdr_cfg.resolved_bandwidth()
    if bandwidth:
        try:
            source.set_bandwidth(bandwidth, 0)
        except Exception:
            # Not every board exposes a tunable analog filter.
            pass

    source.set_center_freq(sdr_cfg.tuned_freq(), 0)
    configured.set_gain_mode(sdr_cfg.agc)
    configured.set_gain(sdr_cfg.tuner_gain)  # ignored by the device while AGC is on

    return configured


# ---------------------------------------------------------------------------
# Live health self-check: real UHD queries, not "a packet arrived once".
# ---------------------------------------------------------------------------
_FREQUENCY_TOLERANCE_HZ = 10.0


def collect_health_snapshot(source, sdr_cfg: SdrConfig) -> dict:
    """Re-read the live device state and compare it against what was asked
    for, plus whatever sensors this specific board exposes.

    Defensive throughout: this runs periodically against a live, streaming
    device from a background thread, and a missing capability (a getter an
    unfamiliar board does not implement, a sensor it does not have) must be
    reported as absent, never raised. The result is a plain JSON-safe dict,
    ready to hand to :class:`~core.pipeline.runtime_status.StatusPublisher`.
    """
    snapshot: dict = {}

    try:
        actual_rate = source.get_samp_rate()
    except Exception:
        actual_rate = None
    expected_rate = sdr_cfg.sample_rate
    snapshot["sample_rate_hz"] = actual_rate
    snapshot["sample_rate_ok"] = actual_rate is not None and abs(
        actual_rate - expected_rate
    ) <= max(1.0, expected_rate * 1e-6)

    try:
        actual_freq = source.get_center_freq(0)
    except Exception:
        actual_freq = None
    expected_freq = sdr_cfg.tuned_freq()
    snapshot["frequency_hz"] = actual_freq
    snapshot["frequency_ok"] = (
        actual_freq is not None and abs(actual_freq - expected_freq) <= _FREQUENCY_TOLERANCE_HZ
    )

    try:
        actual_antenna = source.get_antenna(0)
    except Exception:
        actual_antenna = None
    snapshot["antenna"] = actual_antenna
    snapshot["antenna_ok"] = sdr_cfg.antenna is None or actual_antenna == sdr_cfg.antenna

    sensors: dict = {}
    try:
        for name in source.get_mboard_sensor_names(0):
            try:
                sensors[name] = str(source.get_mboard_sensor(name, 0))
            except Exception:
                continue
    except Exception:
        pass
    try:
        for name in source.get_sensor_names(0):
            try:
                sensors[name] = str(source.get_sensor(name, 0))
            except Exception:
                continue
    except Exception:
        pass
    snapshot["sensors"] = sensors

    # lo_locked is the one sensor that is a meaningful pass/fail gate on any
    # board that has it: if the LO never locked, the receiver is not tuned to
    # what get_center_freq() claims. ref_locked is deliberately NOT gated on
    # here -- it reads unlocked whenever the clock source is "internal" (the
    # default with nothing external plugged in), which is normal, not a fault.
    lo_locked_ok = True
    try:
        lo_locked_ok = bool(source.get_sensor("lo_locked", 0).to_bool())
    except Exception:
        pass  # Board has no lo_locked sensor -- nothing to gate on.
    snapshot["lo_locked_ok"] = lo_locked_ok

    checks = (
        ("sample_rate", snapshot["sample_rate_ok"]),
        ("frequency", snapshot["frequency_ok"]),
        ("antenna", snapshot["antenna_ok"]),
        ("lo_locked", lo_locked_ok),
    )
    problems = [label for label, ok in checks if not ok]
    snapshot["ok"] = not problems
    snapshot["summary"] = "OK" if not problems else "drift: " + ", ".join(problems)
    return snapshot


class OverflowTap(gr.sync_block):
    """Passthrough tap that counts UHD's stream re-stamps after the first.

    UHD emits an rx_time/rx_rate/rx_freq tag set at stream start and again
    after any discontinuity in the sample stream -- gr-uhd's own docs group
    these together: "A timestamp tag is produced at start() and after
    overflows." Tapped right at the source, before any resampling or
    channelization that could shift tag offsets or drop them, so this
    reflects the device's own view of stream continuity, not a guess parsed
    out of console text.
    """

    def __init__(self):
        gr.sync_block.__init__(
            self, name="overflow_tap",
            in_sig=[np.complex64], out_sig=[np.complex64],
        )
        self.discontinuity_count = 0
        self.last_discontinuity_at = None
        self._seen_first_rx_time = False

    def work(self, input_items, output_items):
        sample_count = len(input_items[0])
        for tag in self.get_tags_in_window(0, 0, sample_count):
            if pmt.symbol_to_string(tag.key) != "rx_time":
                continue
            if not self._seen_first_rx_time:
                self._seen_first_rx_time = True
                continue
            self.discontinuity_count += 1
            self.last_discontinuity_at = time.monotonic()

        output_items[0][:sample_count] = input_items[0][:sample_count]
        return sample_count
