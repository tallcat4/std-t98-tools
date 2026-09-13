import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from app.main_window import MainWindow  # noqa: E402
from core.pipeline.multi_stack_dashboard import ChannelView, ProcessView  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])
    yield app


def test_channel_grid_has_fixed_thirty_cards(qapp):
    window = MainWindow()
    assert len(window._cards) == 30
    # Backend publishes 1-based channel ids (channel_index + 1), so cards are
    # keyed 1..30 and the label is that id verbatim.
    assert [card.channel_id for card in window._cards] == list(range(1, 31))
    assert window._cards[0]._title.text() == "CH 01"
    assert window._cards[29]._title.text() == "CH 30"


def test_none_profile_disables_calibration(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    window._reload_profiles(select_path=None)
    assert window._current_profile_path() is None
    assert not window._freq_err.isEnabled()


def test_calibration_writes_into_profile_file(qapp, tmp_path, monkeypatch):
    from app.profile_store import create_profile, read_freq_err_offset

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    template = tmp_path / "tmpl.toml"
    template.write_text('# USRP B210\n[sdr]\ndriver = "uhd"\nsample_rate = 2000000\n')
    profile = create_profile(template, "unit1")  # -> $XDG/std-t98/profiles/unit1.toml

    window = MainWindow()
    window._reload_profiles(select_path=str(profile))
    assert window._current_profile_path() == str(profile)
    assert window._freq_err.isEnabled()

    window._freq_err.setText("1030")
    window._save_calibration_to_profile()
    assert read_freq_err_offset(profile) == 1030

    # Re-selecting the profile mirrors the file's value back into the field.
    window._reload_profiles(select_path=None)
    assert window._freq_err.text() == ""
    window._reload_profiles(select_path=str(profile))
    assert window._freq_err.text() == "1030"


def test_external_profile_is_listed_and_selected(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    external = tmp_path / "my-b210.toml"
    external.write_text('[sdr]\ndriver = "uhd"\n')

    window = MainWindow()
    window._reload_profiles(select_path=str(external))
    assert window._current_profile_path() == str(external)
    labels = [window._profile_combo.itemText(i) for i in range(window._profile_combo.count())]
    assert any("external" in label for label in labels)


def test_settings_panel_toggles(qapp):
    window = MainWindow()
    assert window._settings_panel.isVisibleTo(window) is True
    window._settings_toggle.setChecked(False)
    assert window._settings_panel.isVisibleTo(window) is False


def test_squelch_slider_seeds_from_profile_and_defaults_without_one(qapp, tmp_path, monkeypatch):
    from app.main_window import DEFAULT_SQUELCH_THRESHOLD
    from app.profile_store import create_profile

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    template = tmp_path / "tmpl.toml"
    template.write_text('[sdr]\ndriver = "uhd"\n[demod]\nsquelch_threshold = -60\n')
    profile = create_profile(template, "unit1")

    window = MainWindow()
    window._reload_profiles(select_path=None)
    assert window._squelch_slider.value() == int(round(DEFAULT_SQUELCH_THRESHOLD))

    window._reload_profiles(select_path=str(profile))
    assert window._squelch_slider.value() == -60
    assert window._squelch_value_label.text() == "-60 dB"


def test_squelch_save_button_writes_into_profile_and_needs_a_profile(qapp, tmp_path, monkeypatch):
    from app.profile_store import create_profile, read_squelch_threshold

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    template = tmp_path / "tmpl.toml"
    template.write_text('[sdr]\ndriver = "uhd"\n[demod]\nsquelch_threshold = -60\n')
    profile = create_profile(template, "unit1")

    window = MainWindow()
    window._reload_profiles(select_path=None)
    assert not window._squelch_save_button.isEnabled()

    window._reload_profiles(select_path=str(profile))
    assert window._squelch_save_button.isEnabled()
    window._squelch_slider.setValue(-35)
    window._save_squelch_to_profile()
    assert read_squelch_threshold(profile) == -35


def test_squelch_controls_disabled_in_services_only_mode(qapp, tmp_path, monkeypatch):
    from app.profile_store import create_profile

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    template = tmp_path / "tmpl.toml"
    template.write_text('[sdr]\ndriver = "uhd"\n')
    profile = create_profile(template, "unit1")

    window = MainWindow()
    window._reload_profiles(select_path=str(profile))
    assert window._squelch_slider.isEnabled()

    window._services_only.setChecked(True)
    assert not window._squelch_slider.isEnabled()
    assert not window._squelch_save_button.isEnabled()


def test_squelch_slider_stays_enabled_while_running(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    window._set_running(True)
    assert window._squelch_slider.isEnabled()

    calls = []

    class FakeSupervisor:
        def set_squelch(self, value):
            calls.append(value)
            return True

    window.supervisor = FakeSupervisor()
    window._squelch_slider.setValue(-55)
    assert calls == [-55.0]
    window._settings_toggle.setChecked(True)
    assert window._settings_panel.isVisibleTo(window) is True


def test_refresh_marks_open_channel_and_secret_cache(qapp):
    window = MainWindow()

    process_views = [ProcessView(name=n, python_executable="/x", script_name=f"{n}.py")
                     for n in ("protocol", "secret", "audio", "backend")]
    process_views[3].state = "EXITED"

    class _Sup:
        pass

    sup = _Sup()
    sup.process_views = process_views
    window._rebuild_process_badges(sup)

    channel = ChannelView(
        channel_id=3,
        rx_status="OPEN",
        protocol_status="Traffic->IPC",
        audio_status="Playing",
        secret_status="Global Cache Hit",
        secret_key=42,
        secret_cache_keys=(42, 77),
        csm="200446991",
        sacch={"CallStat": 1, "UserCode": 1, "MakerCode": 2},
        last_update=time.time(),
    )
    window._refresh_from_state(process_views, {3: channel}, show_debug=False)

    cards_by_id = {card.channel_id: card for card in window._cards}
    open_card = cards_by_id[3]
    assert open_card._rx.text() == "OPEN"
    assert open_card.property("rx") == "open"
    assert open_card._title.text() == "CH 03"
    closed_card = cards_by_id[1]
    assert closed_card._rx.text() == "CLOSE"
    assert closed_card.property("rx") == "close"
    assert "00042" in window._cache_label.text()
    assert "00077" in window._cache_label.text()
    assert window._process_badges["backend"]._state.text() == "EXITED"
