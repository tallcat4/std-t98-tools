import json

from core.rf.backend_config import SdrConfig
from core.rf.uhd_source import ConfiguredSource, collect_health_snapshot


class FakeSensor:
    def __init__(self, value, text):
        self._value = value
        self._text = text

    def to_bool(self):
        return self._value

    def __str__(self):
        return self._text


class FakeSource:
    """Mimics enough of uhd.usrp_source for collect_health_snapshot.

    Deliberately partial -- real boards vary in what they expose (most have
    no sensors at all), so the health check must degrade gracefully rather
    than assume any of this is present.
    """

    def __init__(
        self,
        samp_rate=2_000_000.0,
        center_freq=351_293_750.0,
        antenna="TX/RX",
        mboard_sensors=None,
        rf_sensors=None,
    ):
        self._samp_rate = samp_rate
        self._center_freq = center_freq
        self._antenna = antenna
        self._mboard_sensors = mboard_sensors or {}
        self._rf_sensors = rf_sensors or {}

    def get_samp_rate(self):
        return self._samp_rate

    def get_center_freq(self, chan):
        return self._center_freq

    def get_antenna(self, chan):
        return self._antenna

    def get_mboard_sensor_names(self, mboard):
        return list(self._mboard_sensors)

    def get_mboard_sensor(self, name, mboard):
        return self._mboard_sensors[name]

    def get_sensor_names(self, chan):
        return list(self._rf_sensors)

    def get_sensor(self, name, chan):
        return self._rf_sensors[name]


def test_healthy_snapshot_matches_configured_values():
    source = FakeSource(
        rf_sensors={
            "lo_locked": FakeSensor(True, "LO: locked"),
            "temp": FakeSensor(38.4, "temp: 38.4 C"),
        },
        mboard_sensors={"ref_locked": FakeSensor(False, "Ref: unlocked")},
    )
    sdr_cfg = SdrConfig(sample_rate=2_000_000.0, center_freq=351_293_750.0, antenna="TX/RX")

    snapshot = collect_health_snapshot(source, sdr_cfg)

    assert snapshot["ok"] is True
    assert snapshot["summary"] == "OK"
    assert snapshot["sample_rate_ok"] is True
    assert snapshot["frequency_ok"] is True
    assert snapshot["antenna_ok"] is True
    assert snapshot["lo_locked_ok"] is True
    assert snapshot["sensors"]["lo_locked"] == "LO: locked"
    # ref_locked is informational only -- unlocked-on-internal-clock is normal.
    assert snapshot["sensors"]["ref_locked"] == "Ref: unlocked"


def test_snapshot_is_json_serializable():
    source = FakeSource()
    sdr_cfg = SdrConfig(sample_rate=2_000_000.0)
    json.dumps(collect_health_snapshot(source, sdr_cfg))  # must not raise


def test_sample_rate_drift_is_flagged():
    source = FakeSource(samp_rate=1_999_000.0)
    sdr_cfg = SdrConfig(sample_rate=2_000_000.0)

    snapshot = collect_health_snapshot(source, sdr_cfg)

    assert snapshot["sample_rate_ok"] is False
    assert snapshot["ok"] is False
    assert "sample_rate" in snapshot["summary"]


def test_frequency_drift_beyond_tolerance_is_flagged():
    source = FakeSource(center_freq=351_293_750.0 + 50.0)
    sdr_cfg = SdrConfig(center_freq=351_293_750.0)

    snapshot = collect_health_snapshot(source, sdr_cfg)

    assert snapshot["frequency_ok"] is False
    assert "frequency" in snapshot["summary"]


def test_tiny_frequency_residual_is_not_flagged():
    # UHD's DDS leaves sub-Hz residuals even on a successful tune.
    source = FakeSource(center_freq=351_293_750.0 - 0.75)
    sdr_cfg = SdrConfig(center_freq=351_293_750.0)

    snapshot = collect_health_snapshot(source, sdr_cfg)

    assert snapshot["frequency_ok"] is True


def test_antenna_mismatch_is_flagged():
    source = FakeSource(antenna="RX2")
    sdr_cfg = SdrConfig(antenna="TX/RX")

    snapshot = collect_health_snapshot(source, sdr_cfg)

    assert snapshot["antenna_ok"] is False
    assert "antenna" in snapshot["summary"]


def test_antenna_not_configured_is_never_flagged():
    source = FakeSource(antenna="RX2")
    sdr_cfg = SdrConfig(antenna=None)

    snapshot = collect_health_snapshot(source, sdr_cfg)

    assert snapshot["antenna_ok"] is True


def test_lo_unlocked_is_flagged():
    source = FakeSource(rf_sensors={"lo_locked": FakeSensor(False, "LO: unlocked")})
    sdr_cfg = SdrConfig()

    snapshot = collect_health_snapshot(source, sdr_cfg)

    assert snapshot["lo_locked_ok"] is False
    assert snapshot["ok"] is False
    assert "lo_locked" in snapshot["summary"]


def test_board_with_no_sensors_at_all_is_not_penalized():
    # Most SoapySDR/UHD boards report few or no sensors -- lo_locked missing
    # must not itself count as unhealthy.
    source = FakeSource(rf_sensors={}, mboard_sensors={})
    sdr_cfg = SdrConfig()

    snapshot = collect_health_snapshot(source, sdr_cfg)

    assert snapshot["lo_locked_ok"] is True
    assert snapshot["sensors"] == {}


def test_getters_that_raise_are_reported_as_unknown_not_crashed():
    class BrokenSource:
        def get_samp_rate(self):
            raise RuntimeError("not supported")

        def get_center_freq(self, chan):
            raise RuntimeError("not supported")

        def get_antenna(self, chan):
            raise RuntimeError("not supported")

        def get_mboard_sensor_names(self, mboard):
            raise RuntimeError("not supported")

        def get_sensor_names(self, chan):
            raise RuntimeError("not supported")

        def get_sensor(self, name, chan):
            raise RuntimeError("not supported")

    snapshot = collect_health_snapshot(BrokenSource(), SdrConfig())

    assert snapshot["sample_rate_hz"] is None
    assert snapshot["sample_rate_ok"] is False
    assert snapshot["frequency_hz"] is None
    assert snapshot["antenna"] is None
    assert snapshot["sensors"] == {}
    # lo_locked is only gated when the sensor is actually readable.
    assert snapshot["lo_locked_ok"] is True
    assert snapshot["ok"] is False


class RecordingAgcSource(FakeSource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agc_calls = []

    def set_rx_agc(self, enable, chan):
        self.agc_calls.append((enable, chan))


def test_set_gain_mode_calls_set_rx_agc():
    source = RecordingAgcSource()
    configured = ConfiguredSource(source, gain_element="")
    configured.set_gain_mode(True)
    assert source.agc_calls == [(True, 0)]
    configured.set_gain_mode(False)
    assert source.agc_calls == [(True, 0), (False, 0)]


def test_set_gain_mode_is_defensive_on_unsupported_boards():
    class NoAgcSource(FakeSource):
        def set_rx_agc(self, enable, chan):
            raise RuntimeError("AGC not supported on this daughterboard")

    configured = ConfiguredSource(NoAgcSource(), gain_element="")
    configured.set_gain_mode(True)  # must not raise
