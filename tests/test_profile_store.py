from textwrap import dedent

from app.config_preview import preview_config
from app.profile_store import (
    create_profile,
    list_profiles,
    profiles_dir,
    read_freq_err_offset,
    set_toml_scalar,
    write_freq_err_offset,
)


def test_profiles_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert profiles_dir() == tmp_path / "std-t98" / "profiles"


def test_set_toml_scalar_replaces_existing_key():
    text = '[sdr]\ndriver = "uhd"\nfreq_err_offset = 0\n'
    out = set_toml_scalar(text, "sdr", "freq_err_offset", 1030)
    assert 'freq_err_offset = 1030' in out
    assert 'driver = "uhd"' in out
    assert out.count("freq_err_offset") == 1


def test_set_toml_scalar_inserts_into_existing_section_keeping_comments():
    text = dedent("""\
        # USRP B210
        [sdr]
        driver = "uhd"

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
    text = '[sdr]\ndriver = "uhd"\nfreq_err_offset = 1030\n'
    out = set_toml_scalar(text, "sdr", "freq_err_offset", None)
    assert "freq_err_offset" not in out
    assert 'driver = "uhd"' in out


def test_write_and_read_roundtrip_stays_loadable(tmp_path):
    profile = tmp_path / "b210.toml"
    profile.write_text('# USRP B210\n[sdr]\ndriver = "uhd"\nsample_rate = 2000000\n')
    write_freq_err_offset(profile, 1030)
    assert read_freq_err_offset(profile) == 1030
    # The result must still be a valid, resolvable backend config.
    result = preview_config(str(profile))
    assert result.ok is True
    assert "+1030 Hz" in result.summary
    # Clearing removes it again.
    write_freq_err_offset(profile, None)
    assert read_freq_err_offset(profile) is None


def test_create_profile_copies_template_and_lists(tmp_path):
    template = tmp_path / "tmpl.toml"
    template.write_text('# USRP B210\n[sdr]\ndriver = "uhd"\n')
    dest_dir = tmp_path / "profiles"

    created = create_profile(template, "my-b210", directory=dest_dir)
    assert created == dest_dir / "my-b210.toml"
    assert "# USRP B210" in created.read_text()  # comments copied

    names = [p.name for p in list_profiles(dest_dir)]
    assert names == ["my-b210"]

    # A second profile of the same name is refused (caller confirms overwrite).
    import pytest

    with pytest.raises(FileExistsError):
        create_profile(template, "my-b210", directory=dest_dir)
