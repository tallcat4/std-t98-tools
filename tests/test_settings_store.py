from textwrap import dedent

from app.config_preview import preview_config
from app.settings_store import (
    ensure_settings_file,
    read_demod_value,
    read_sdr_value,
    set_toml_scalar,
    settings_path,
    write_demod_value,
    write_sdr_value,
)


def test_settings_path_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert settings_path() == tmp_path / "std-t98" / "settings.toml"


def test_ensure_settings_file_creates_an_empty_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = ensure_settings_file()
    assert path == settings_path()
    assert path.exists()
    assert path.read_text() == ""
    # Calling again with an existing (non-empty) file must not clobber it.
    path.write_text('[sdr]\nantenna = "TX/RX"\n')
    ensure_settings_file()
    assert 'antenna = "TX/RX"' in path.read_text()


def test_set_toml_scalar_replaces_existing_key():
    text = '[sdr]\nantenna = "TX/RX"\nfreq_err_offset = 0\n'
    out = set_toml_scalar(text, "sdr", "freq_err_offset", 1030)
    assert 'freq_err_offset = 1030' in out
    assert 'antenna = "TX/RX"' in out
    assert out.count("freq_err_offset") == 1


def test_set_toml_scalar_inserts_into_existing_section_keeping_comments():
    text = dedent("""\
        # USRP B210
        [sdr]
        antenna = "TX/RX"

        [demod]
        squelch_threshold = -40
    """)
    out = set_toml_scalar(text, "sdr", "freq_err_offset", 1030)
    assert "# USRP B210" in out                 # comment preserved
    assert "freq_err_offset = 1030" in out
    # inserted under [sdr], before [demod]
    assert out.index("freq_err_offset") < out.index("[demod]")


def test_set_toml_scalar_creates_missing_section():
    out = set_toml_scalar('[demod]\nsquelch_threshold = -40\n', "sdr", "freq_err_offset", -340)
    assert "[sdr]" in out
    assert "freq_err_offset = -340" in out


def test_set_toml_scalar_removes_key_when_none():
    text = '[sdr]\nantenna = "TX/RX"\nfreq_err_offset = 1030\n'
    out = set_toml_scalar(text, "sdr", "freq_err_offset", None)
    assert "freq_err_offset" not in out
    assert 'antenna = "TX/RX"' in out


def test_sdr_value_write_and_read_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert read_sdr_value("freq_err_offset") is None
    write_sdr_value("freq_err_offset", 1030)
    assert read_sdr_value("freq_err_offset") == 1030
    # The result must still be a valid, resolvable backend config.
    result = preview_config(str(settings_path()))
    assert result.ok is True
    assert "+1030 Hz" in result.summary

    write_sdr_value("antenna", "TX/RX")
    assert read_sdr_value("antenna") == "TX/RX"

    # Clearing removes it again.
    write_sdr_value("freq_err_offset", None)
    assert read_sdr_value("freq_err_offset") is None
    # Unrelated keys survive.
    assert read_sdr_value("antenna") == "TX/RX"


def test_demod_value_write_and_read_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert read_demod_value("squelch_threshold") is None
    write_demod_value("squelch_threshold", -60)
    assert read_demod_value("squelch_threshold") == -60

    result = preview_config(str(settings_path()))
    assert result.ok is True

    write_demod_value("squelch_threshold", None)
    assert read_demod_value("squelch_threshold") is None


def test_write_sdr_value_creates_the_settings_file(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert not settings_path().exists()
    write_sdr_value("tuner_gain", 30)
    assert settings_path().exists()
    assert read_sdr_value("tuner_gain") == 30
