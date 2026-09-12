"""Opening and configuring the SoapySDR source.

Kept out of :mod:`core.rf.backend_config`, which stays importable without GNU
Radio, and shared by the backend flowgraph and the diagnostic scope so both get
identical device handling: the driver-aware defaults, the capability fallbacks
for devices without named gains / AGC / bias tee, and above all the sample-rate
checks. A tool that configures the device slightly differently from the backend
is a tool that cannot be trusted to diagnose the backend.
"""

from __future__ import annotations

from gnuradio import soapy

from core.rf.backend_config import nearest_sample_rates


class ConfiguredSource:
    """A SoapySDR source plus the runtime setters that depend on its
    capabilities."""

    def __init__(self, source, gain_element):
        self.source = source
        self.gain_element = gain_element

        self.setting_keys = [a.key for a in source.get_setting_info()]
        try:
            self.gain_names = list(source.list_gains(0))
        except Exception:
            self.gain_names = []
        try:
            self.has_agc = bool(source.has_gain_mode(0))
        except Exception:
            self.has_agc = True

        self._gain_value = 0.0

    # -- runtime controls ---------------------------------------------------
    def apply_manual_gain(self, channel, gain):
        """Prefer a named gain element when the device (and config) name one;
        fall back to the overall gain otherwise."""
        if self.gain_element and self.gain_element in self.gain_names:
            self.source.set_gain(channel, self.gain_element, gain)
        else:
            self.source.set_gain(channel, gain)

    def set_gain_mode(self, channel, agc):
        # Not every SoapySDR device exposes an automatic gain mode.
        if not self.has_agc:
            return
        self.source.set_gain_mode(channel, agc)
        if not agc:
            self.apply_manual_gain(channel, self._gain_value)

    def set_gain(self, channel, gain):
        self._gain_value = gain
        if self.has_agc and self.source.get_gain_mode(channel):
            return
        self.apply_manual_gain(channel, gain)

    def set_bias(self, enabled):
        if "biastee" in self.setting_keys:
            self.source.write_setting("biastee", enabled)


def _resolve_sample_rate(source, requested):
    """Return the rate to actually ask the device for.

    Devices advertise exact values such as 8e6/7 = 1230769.230769 and refuse
    anything else, so a user typing the rounded figure is snapped onto the
    advertised one rather than rejected. If nothing matches, the error names
    the nearest supported rates -- the raw device list runs to hundreds of
    entries, which is no answer to "what should I use?".
    """
    try:
        ranges = source.get_sample_rate_range(0)
    except Exception:
        return requested  # Driver does not advertise its rates.

    if not ranges:
        return requested

    tolerance = max(1.0, requested * 1e-6)
    discrete = []
    for entry in ranges:
        low, high = entry.minimum(), entry.maximum()
        if low == high:
            discrete.append(low)
            if abs(requested - low) <= tolerance:
                return low
        elif low <= requested <= high:
            return requested

    if not discrete:
        return requested  # Continuous ranges only; let Soapy complain.

    suggestions = nearest_sample_rates(discrete, requested)
    raise ValueError(
        f"This device cannot sample at {requested:.0f} Hz. "
        "Nearest rates this device supports: "
        + ", ".join(f"{rate:.0f}" for rate in suggestions)
    )


def _set_antenna(source, antenna):
    """Select an RX port, refusing an unavailable name up front.

    Getting this wrong produces no error and no warning at runtime -- just a
    receiver that never hears anything -- so it is worth failing loudly.
    """
    try:
        available = list(source.list_antennas(0))
    except Exception:
        available = []
    if available and antenna not in available:
        raise ValueError(
            f"Antenna {antenna!r} is not available on this device. "
            f"Choose one of: {available}"
        )
    source.set_antenna(0, antenna)


def open_source(sdr_cfg) -> ConfiguredSource:
    """Open and fully configure the source described by ``sdr_cfg``."""
    configured = ConfiguredSource(
        soapy.source(
            sdr_cfg.device_string(), "fc32", 1, "",
            sdr_cfg.resolved_stream_args(), [""], [""],
        ),
        sdr_cfg.gain_element,
    )
    source = configured.source

    # Only touch the antenna when asked: single-input devices have nothing to
    # select, and the driver's own default is right for most others.
    if sdr_cfg.antenna:
        _set_antenna(source, sdr_cfg.antenna)

    requested = sdr_cfg.sample_rate
    source.set_sample_rate(0, _resolve_sample_rate(source, requested))
    # Some drivers round instead of refusing. The channelizer geometry is
    # derived from the requested rate, so a silent substitution would mistune
    # every channel.
    actual = source.get_sample_rate(0)
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
            source.set_bandwidth(0, bandwidth)
        except Exception:
            # Not every device exposes a tunable analog filter.
            pass

    source.set_frequency(
        0, sdr_cfg.center_freq + sdr_cfg.freq_offset + sdr_cfg.freq_err_offset
    )
    if sdr_cfg.freq_correction:
        try:
            source.set_frequency_correction(0, sdr_cfg.freq_correction)
        except Exception:
            pass

    configured.set_bias(bool(sdr_cfg.bias_tee))
    configured._gain_value = sdr_cfg.tuner_gain
    configured.set_gain_mode(0, bool(sdr_cfg.agc))
    configured.set_gain(0, sdr_cfg.tuner_gain)

    return configured
