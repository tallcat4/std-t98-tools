from textwrap import dedent

from app.config_preview import parse_soapy_find, preview_config


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
        driver = "uhd"
        sample_rate = 2000000
        antenna = "TX/RX"
        agc = false
        tuner_gain = 30
        gain_element = "PGA"
        freq_err_offset = 1030
        [demod]
        squelch_threshold = -40
    """)
    result = preview_config(str(path))
    assert result.ok is True
    assert "driver=uhd" in result.summary
    assert "TX/RX" in result.summary
    assert "+1030 Hz" in result.summary
    assert "2,000,000 Hz" in result.summary
    assert result.warnings == []


def test_parse_soapy_find_extracts_devices():
    output = """
    Found device 0
      driver = audio
      label = Built-in Audio

    Found device 1
      driver = uhd
      label = B210 5IWG5D5
      serial = 5IWG5D5
      type = b200
    """
    devices = parse_soapy_find(output)
    assert [d.driver for d in devices] == ["audio", "uhd"]
    assert devices[1].serial == "5IWG5D5"
    assert devices[1].extra["type"] == "b200"
