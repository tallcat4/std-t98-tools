import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5.QtWidgets")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from app.main_window import MainWindow, ProcessBadge, _PROCESS_STATE_COLOURS  # noqa: E402
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


def test_sdr_form_starts_empty_with_no_settings_file(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    assert window._antenna_field.currentText() == ""
    assert window._sample_rate_field.text() == ""
    assert window._freq_err.text() == ""
    # A settings file is always created so the backend has something to load.
    from app.settings_store import settings_path

    assert settings_path().exists()


def test_sdr_form_seeds_from_existing_settings_file(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from app.settings_store import settings_path

    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text(
        '# USRP B210\n[sdr]\nantenna = "TX/RX"\nsample_rate = 2000000\n'
        'tuner_gain = 30\ngain_element = "PGA"\nfreq_err_offset = 1030\n'
    )

    window = MainWindow()
    assert window._antenna_field.currentText() == "TX/RX"
    assert window._sample_rate_field.text() == "2000000"
    assert window._gain_field.text() == "30"
    assert window._gain_element_field.text() == "PGA"
    assert window._freq_err.text() == "1030"


def test_editing_a_field_writes_into_the_settings_file(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from app.settings_store import read_sdr_value

    window = MainWindow()
    window._freq_err.setText("1030")
    assert read_sdr_value("freq_err_offset") == 1030

    window._antenna_field.setCurrentText("RX2")
    assert read_sdr_value("antenna") == "RX2"


def test_invalid_numeric_field_is_not_written_and_warns(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from app.settings_store import read_sdr_value

    window = MainWindow()
    window._sample_rate_field.setText("not-a-number")
    assert read_sdr_value("sample_rate") is None
    assert "invalid number" in window._preview_note.text()


def test_sdr_fields_disabled_while_running(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    assert window._antenna_field.isEnabled()
    assert window._detect_button.isEnabled()

    window._set_running(True)
    assert not window._antenna_field.isEnabled()
    assert not window._sample_rate_field.isEnabled()
    assert not window._detect_button.isEnabled()


def test_agc_checkbox_seeds_from_settings_and_writes_through(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from app.settings_store import read_sdr_value

    window = MainWindow()
    assert window._agc_field.isChecked() is False
    assert window._gain_field.isEnabled()

    window._agc_field.setChecked(True)
    assert read_sdr_value("agc") is True
    # While AGC is on, the device ignores manual gain -- grey it out.
    assert not window._gain_field.isEnabled()

    window._agc_field.setChecked(False)
    assert read_sdr_value("agc") is False
    assert window._gain_field.isEnabled()


def test_agc_checkbox_seeds_true_from_existing_settings_file(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from app.settings_store import settings_path

    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text("[sdr]\nagc = true\n")

    window = MainWindow()
    assert window._agc_field.isChecked() is True
    assert not window._gain_field.isEnabled()


def test_agc_checkbox_disabled_while_running(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    window._set_running(True)
    assert not window._agc_field.isEnabled()
    assert not window._gain_field.isEnabled()


def test_settings_panel_toggles(qapp):
    window = MainWindow()
    assert window._settings_panel.isVisibleTo(window) is True
    window._settings_toggle.setChecked(False)
    assert window._settings_panel.isVisibleTo(window) is False


def test_process_badge_shows_waiting_hint_while_starting_and_silent(qapp):
    badge = ProcessBadge("backend")
    badge.update_from(ProcessView(name="backend", python_executable="/x", script_name="backend.py", state="STARTING"))
    assert badge._state.text() == "STARTING"
    assert "waiting for first report" in badge._detail.text()
    assert badge._detail.isVisibleTo(badge)


def test_process_badge_shows_metrics_once_reported(qapp):
    badge = ProcessBadge("backend")
    view = ProcessView(name="backend", python_executable="/x", script_name="backend.py", state="RUNNING")
    view.detail = "sync=12 ipc=12/0 mode=sse active=1"
    badge.update_from(view)
    assert badge._state.text() == "RUNNING"
    assert badge._detail.text() == "sync=12 ipc=12/0 mode=sse active=1"
    assert badge._detail.isVisibleTo(badge)


def test_process_badge_shows_healthy_status(qapp):
    badge = ProcessBadge("backend")
    view = ProcessView(name="backend", python_executable="/x", script_name="backend.py", state="RUNNING")
    view.health_ok = True
    view.health = "OK"
    badge.update_from(view)
    assert badge._health.text() == "✓ OK"
    assert badge._health.isVisibleTo(badge)
    # A healthy, RUNNING process keeps the plain RUNNING (green) look.
    assert "#2e7d32" in badge._state.styleSheet()


def test_process_badge_flags_degraded_backend_while_running(qapp):
    badge = ProcessBadge("backend")
    view = ProcessView(name="backend", python_executable="/x", script_name="backend.py", state="RUNNING")
    view.health_ok = False
    view.health = "drift: antenna"
    badge.update_from(view)
    assert badge._health.text() == "⚠ drift: antenna"
    assert badge._health.isVisibleTo(badge)
    # Alive is not the same as healthy: reuse the STARTING amber, not green.
    assert badge._state.styleSheet() == f"color: {_PROCESS_STATE_COLOURS['STARTING']}; font-weight: 700;"


def test_process_badge_hides_health_line_when_none_reported(qapp):
    badge = ProcessBadge("backend")
    view = ProcessView(name="backend", python_executable="/x", script_name="backend.py", state="RUNNING")
    badge.update_from(view)
    assert not badge._health.isVisibleTo(badge)


def test_squelch_slider_seeds_from_settings_and_defaults_without_one(qapp, tmp_path, monkeypatch):
    from app.main_window import DEFAULT_SQUELCH_THRESHOLD

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    assert window._squelch_slider.value() == int(round(DEFAULT_SQUELCH_THRESHOLD))

    from app.settings_store import settings_path

    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text('[demod]\nsquelch_threshold = -60\n')
    window2 = MainWindow()
    assert window2._squelch_slider.value() == -60
    assert window2._squelch_value_label.text() == "-60 dB"


def test_squelch_save_button_writes_into_settings_file(qapp, tmp_path, monkeypatch):
    from app.settings_store import read_demod_value

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    assert window._squelch_save_button.isEnabled()
    window._squelch_slider.setValue(-35)
    window._save_squelch_to_settings()
    assert read_demod_value("squelch_threshold") == -35


def test_squelch_controls_disabled_in_services_only_mode(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    window = MainWindow()
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


def test_sync_threshold_slider_seeds_from_settings_and_defaults_without_one(qapp, tmp_path, monkeypatch):
    from app.main_window import DEFAULT_SYNC_THRESHOLD_RATIO

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    assert window._sync_threshold_ratio() == pytest.approx(DEFAULT_SYNC_THRESHOLD_RATIO)
    assert window._sync_threshold_value_label.text() == "0.20"

    from app.settings_store import settings_path

    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text('[demod]\nsync_error_threshold_ratio = 0.35\n')
    window2 = MainWindow()
    assert window2._sync_threshold_slider.value() == 35
    assert window2._sync_threshold_ratio() == pytest.approx(0.35)
    assert window2._sync_threshold_value_label.text() == "0.35"


def test_sync_threshold_slider_ignores_a_nonsensical_saved_value(qapp, tmp_path, monkeypatch):
    from app.main_window import DEFAULT_SYNC_THRESHOLD_RATIO

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from app.settings_store import settings_path

    settings_path().parent.mkdir(parents=True, exist_ok=True)
    settings_path().write_text('[demod]\nsync_error_threshold_ratio = 0\n')
    window = MainWindow()
    assert window._sync_threshold_ratio() == pytest.approx(DEFAULT_SYNC_THRESHOLD_RATIO)


def test_sync_threshold_save_button_writes_into_settings_file(qapp, tmp_path, monkeypatch):
    from app.settings_store import read_demod_value

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    assert window._sync_threshold_save_button.isEnabled()
    window._sync_threshold_slider.setValue(30)
    window._save_sync_threshold_to_settings()
    assert read_demod_value("sync_error_threshold_ratio") == pytest.approx(0.3)


def test_sync_threshold_controls_disabled_in_services_only_mode(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    window = MainWindow()
    assert window._sync_threshold_slider.isEnabled()

    window._services_only.setChecked(True)
    assert not window._sync_threshold_slider.isEnabled()
    assert not window._sync_threshold_save_button.isEnabled()


def test_sync_threshold_slider_pushes_live_ratio_while_running(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    window = MainWindow()
    window._set_running(True)
    assert window._sync_threshold_slider.isEnabled()

    calls = []

    class FakeSupervisor:
        def set_sync_threshold_ratio(self, value):
            calls.append(value)
            return True

    window.supervisor = FakeSupervisor()
    window._sync_threshold_slider.setValue(45)
    assert calls == [pytest.approx(0.45)]
    assert window._sync_threshold_value_label.text() == "0.45"


def test_start_passes_both_sliders_as_backend_overrides(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    import app.main_window as main_window_mod

    captured = {}

    class FakeSupervisor:
        mode_label = "fake"
        process_views = []

        def __init__(self, repo_root, services_only, backend_args):
            captured["backend_args"] = backend_args

        def resolve(self):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(main_window_mod, "StackSupervisor", FakeSupervisor)
    window = MainWindow()
    window._squelch_slider.setValue(-40)
    window._sync_threshold_slider.setValue(25)
    window.start_stack()
    args = captured["backend_args"]
    assert args[args.index("--squelch") + 1] == "-40"
    assert args[args.index("--sync-threshold-ratio") + 1] == "0.25"
    window.stop_stack()


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
