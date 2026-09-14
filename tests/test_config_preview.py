from textwrap import dedent

from app.config_preview import (
    parse_uhd_find,
    preview_config,
)


def _write(tmp_path, text):
    path = tmp_path / "cfg.toml"
    path.write_text(dedent(text))
    return path


def test_preview_empty_path_uses_defaults():
    result = preview_config("")
    assert result.ok is True
    assert "built-in defaults" in result.summary


def test_preview_missing_file_is_error():
    result = preview_config("/no/such/file.toml")
    assert result.ok is False
    assert "not found" in result.error.lower()


def test_preview_resolves_uhd_config(tmp_path):
    path = _write(tmp_path, """
        [sdr]
        sample_rate = 2000000
        antenna = "TX/RX"
        tuner_gain = 30
        gain_element = "PGA"
        freq_err_offset = 1030
        [demod]
        squelch_threshold = -40
        sync_error_threshold_ratio = 0.3
    """)
    result = preview_config(str(path))
    assert result.ok is True
    assert "TX/RX" in result.summary
    assert "Sync thr    : 0.3" in result.summary
    assert "PGA" in result.summary
    assert "+1030 Hz" in result.summary
    assert "2,000,000 Hz" in result.summary
    assert result.warnings == []


def test_preview_offset_reflected_in_tuned_freq(tmp_path):
    path = _write(tmp_path, """
        [sdr]
        center_freq = 351293750
    """)
    base = preview_config(str(path))
    assert "+0 Hz" in base.summary

    path.write_text('[sdr]\ncenter_freq = 351293750\nfreq_err_offset = 1030\n')
    over = preview_config(str(path))
    assert "+1030 Hz" in over.summary


def test_shipped_devices_reference_config_loads():
    # devices/usrp-b210.toml is no longer used by the GUI (it always edits a
    # single fixed settings file, see app.settings_store), but it remains a
    # valid --config reference for CLI users, so it must stay loadable.
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "devices" / "usrp-b210.toml"
    assert preview_config(str(path)).ok is True


def test_parse_uhd_find_extracts_devices():
    output = """
    --------------------------------------------------
    -- UHD Device 0
    --------------------------------------------------
    Device Address:
        serial: 5IWG5D5
        name: LibreSDR_B220mini
        product: B210
        type: b200
    """
    devices = parse_uhd_find(output)
    assert len(devices) == 1
    assert devices[0].device_type == "b200"
    assert devices[0].serial == "5IWG5D5"
    assert devices[0].label == "LibreSDR_B220mini"
    assert devices[0].extra["product"] == "B210"


def test_parse_uhd_find_extracts_multiple_devices():
    output = """
    Device Address:
        serial: AAA111
        type: b200

    Device Address:
        serial: BBB222
        type: x300
    """
    devices = parse_uhd_find(output)
    assert [d.serial for d in devices] == ["AAA111", "BBB222"]
    assert [d.device_type for d in devices] == ["b200", "x300"]


def test_parse_uhd_find_empty_output_is_no_devices():
    assert parse_uhd_find("") == []


def test_preview_shows_agc_instead_of_gain_when_enabled(tmp_path):
    path = _write(tmp_path, """
        [sdr]
        agc = true
        tuner_gain = 30
        gain_element = "PGA"
    """)
    result = preview_config(str(path))
    assert result.ok is True
    assert "Gain        : AGC" in result.summary
    assert "PGA" not in result.summary
