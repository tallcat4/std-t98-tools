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


def test_device_combo_lists_presets_and_sets_config_path(qapp):
    window = MainWindow()
    items = [window._device_combo.itemText(i) for i in range(window._device_combo.count())]
    assert items[0] == "(custom)"
    assert any("USRP B210" in text for text in items)

    b210 = next(
        i for i in range(window._device_combo.count())
        if "USRP B210" in window._device_combo.itemText(i)
    )
    window._device_combo.setCurrentIndex(b210)
    assert window._config_path.text().endswith("devices/usrp-b210.toml")

    # Editing the path away from a preset falls back to "(custom)".
    window._config_path.setText("/tmp/not-a-preset.toml")
    assert window._device_combo.currentIndex() == 0


def test_calibration_saves_and_restores_per_config(qapp, tmp_path):
    window = MainWindow()
    # Isolate persistence from the real user settings.
    window._settings = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)

    cfg_a = tmp_path / "a.toml"
    cfg_a.write_text('[sdr]\ndriver = "uhd"\n')
    cfg_b = tmp_path / "b.toml"
    cfg_b.write_text('[sdr]\ndriver = "rtlsdr"\n')

    window._config_path.setText(str(cfg_a))
    window._freq_err.setText("1030")
    window._save_current_calibration()

    # Switching to another config shows its (absent) calibration...
    window._config_path.setText(str(cfg_b))
    assert window._freq_err.text() == ""
    # ...and switching back restores the saved one.
    window._config_path.setText(str(cfg_a))
    assert window._freq_err.text() == "1030"


def test_invalid_calibration_is_not_saved(qapp, tmp_path):
    window = MainWindow()
    window._settings = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.IniFormat)
    cfg = tmp_path / "a.toml"
    cfg.write_text('[sdr]\ndriver = "uhd"\n')
    window._config_path.setText(str(cfg))
    window._freq_err.setText("not-a-number")
    window._save_current_calibration()  # should be a no-op, not raise
    assert window._load_calibrations() == {}


def test_settings_panel_toggles(qapp):
    window = MainWindow()
    assert window._settings_panel.isVisibleTo(window) is True
    window._settings_toggle.setChecked(False)
    assert window._settings_panel.isVisibleTo(window) is False
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
