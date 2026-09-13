# -*- coding: utf-8 -*-
"""Main window for the STD-T98 desktop front-end.

This is the GUI counterpart of the terminal launcher: it drives a
``StackSupervisor`` and renders the same ChannelView / ProcessView state the
launcher prints, but as a live channel grid instead of a scrolling dashboard.
The supervisor owns the process lifecycle; the window only starts/stops it and
polls it from a QTimer so the Qt event loop never blocks.
"""

import os
import shlex
import time
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from app.config_preview import detect_sdrs, list_device_presets, preview_config
from app.profile_store import (
    create_profile,
    list_profiles,
    profiles_dir,
    read_freq_err_offset,
    read_squelch_threshold,
    write_freq_err_offset,
    write_squelch_threshold,
)
from core.pipeline.multi_stack_dashboard import (
    ChannelView,
    _format_csm,
    _format_sacch,
    _format_secret_cache,
)
from core.pipeline.stack_supervisor import StackSupervisor
from core.rf.backend_config import CONFIG_ENV_VAR, DemodConfig

CHANNEL_COUNT = 30
POLL_INTERVAL_MS = 100
SQUELCH_RANGE_DB = (-100, 0)
DEFAULT_SQUELCH_THRESHOLD = DemodConfig().squelch_threshold

_PROCESS_STATE_COLOURS = {
    "RUNNING": "#2e7d32",
    "STARTING": "#b8860b",
    "EXITED": "#c62828",
}


class ProcessBadge(QtWidgets.QFrame):
    """One line of the process-health strip: name, coloured state, pid, and
    (once available) its own status-socket metrics -- e.g. the RF backend's
    "sync=.../ipc=.../active=..." summary, which only appears once the
    flowgraph is actually pulling samples. While STARTING that line is empty,
    which is itself the readiness signal: a live but silent backend means the
    SDR is still opening (firmware/FPGA load, USB re-enumeration, ...), not
    that it is stuck.
    """

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setObjectName("processBadge")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(2)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(8)
        self._name = QtWidgets.QLabel(name)
        self._name.setStyleSheet("font-weight: 600;")
        self._state = QtWidgets.QLabel("--")
        self._pid = QtWidgets.QLabel("")
        self._pid.setStyleSheet("color: #888;")

        header.addWidget(self._name)
        header.addWidget(self._state)
        header.addStretch(1)
        header.addWidget(self._pid)
        outer.addLayout(header)

        self._detail = QtWidgets.QLabel("")
        self._detail.setStyleSheet("color: #888; font-family: monospace; font-size: 10px;")
        self._detail.hide()
        outer.addWidget(self._detail)

    def update_from(self, process_view):
        self._state.setText(process_view.state)
        colour = _PROCESS_STATE_COLOURS.get(process_view.state, "#b8860b")
        self._state.setStyleSheet(f"color: {colour}; font-weight: 700;")
        self._pid.setText(f"pid {process_view.pid}" if process_view.pid is not None else "")
        if process_view.state == "STARTING" and not process_view.detail:
            self._detail.setText("waiting for first report (e.g. SDR still opening)…")
            self._detail.show()
        elif process_view.detail:
            self._detail.setText(process_view.detail)
            self._detail.show()
        else:
            self._detail.hide()


class ChannelCard(QtWidgets.QFrame):
    """A single channel tile. Green border/badge while RX is OPEN."""

    def __init__(self, channel_id, parent=None):
        super().__init__(parent)
        self.channel_id = channel_id
        self.setObjectName("channelCard")
        self.setProperty("rx", "close")

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setVerticalSpacing(2)
        layout.setColumnStretch(1, 1)

        # channel_id is the 1-based registered channel number the backend
        # publishes (sync_word_correlator emits channel_index + 1), so it is
        # already the user-facing channel number -- show it as-is.
        self._title = QtWidgets.QLabel(f"CH {channel_id:02d}")
        self._title.setStyleSheet("font-weight: 700; font-size: 13px;")
        self._rx = QtWidgets.QLabel("CLOSE")
        self._rx.setAlignment(QtCore.Qt.AlignCenter)
        self._rx.setObjectName("rxBadge")
        self._rx.setProperty("rx", "close")

        self._protocol = QtWidgets.QLabel("Waiting")
        self._audio = QtWidgets.QLabel("Idle")
        self._secret = QtWidgets.QLabel("key=00000 | Idle")
        self._csm = QtWidgets.QLabel("CSM: -")
        self._sacch = QtWidgets.QLabel("SACCH: -")
        self._age = QtWidgets.QLabel("")
        self._age.setStyleSheet("color: #888;")
        self._debug = QtWidgets.QLabel("")
        self._debug.setStyleSheet("color: #777; font-family: monospace; font-size: 10px;")
        self._debug.setWordWrap(True)
        self._debug.hide()

        for label in (self._protocol, self._audio, self._secret, self._csm, self._sacch):
            label.setStyleSheet("font-size: 11px;")

        header = QtWidgets.QHBoxLayout()
        header.addWidget(self._title)
        header.addStretch(1)
        header.addWidget(self._age)

        layout.addLayout(header, 0, 0, 1, 2)
        layout.addWidget(self._rx, 1, 0, 1, 2)
        layout.addWidget(self._protocol, 2, 0, 1, 2)
        layout.addWidget(self._audio, 3, 0, 1, 2)
        layout.addWidget(self._csm, 4, 0, 1, 2)
        layout.addWidget(self._sacch, 5, 0, 1, 2)
        layout.addWidget(self._secret, 6, 0, 1, 2)
        layout.addWidget(self._debug, 7, 0, 1, 2)

    def _set_rx_property(self, value):
        if self.property("rx") != value:
            self.setProperty("rx", value)
            self._rx.setProperty("rx", value)
            for widget in (self, self._rx):
                widget.style().unpolish(widget)
                widget.style().polish(widget)

    def update_from(self, channel, show_debug):
        if channel is None:
            self._set_rx_property("close")
            self._rx.setText("CLOSE")
            self._protocol.setText("Waiting")
            self._audio.setText("Idle")
            self._secret.setText("key=00000 | Idle")
            self._csm.setText("CSM: -")
            self._sacch.setText("SACCH: -")
            self._age.setText("")
            self._debug.hide()
            return

        is_open = channel.rx_status == "OPEN"
        self._set_rx_property("open" if is_open else "close")
        self._rx.setText("OPEN" if is_open else "CLOSE")
        self._protocol.setText(f"Proto: {channel.protocol_status}")
        self._audio.setText(f"Audio: {channel.audio_status}")
        self._secret.setText(f"key={channel.secret_key:05d} | {channel.secret_status}")
        self._csm.setText(f"CSM: {_format_csm(channel)}")
        self._sacch.setText(f"SACCH: {_format_sacch(channel)}")
        age = max(0.0, time.time() - channel.last_update)
        self._age.setText(f"{age:4.1f}s")

        if show_debug:
            parts = [
                f"RF: {channel.rf_debug}" if channel.rf_debug else "",
                f"Proto: {channel.protocol_debug}" if channel.protocol_debug else "",
                f"Audio: {channel.audio_debug}" if channel.audio_debug else "",
            ]
            text = "\n".join(part for part in parts if part)
            self._debug.setText(text)
            self._debug.setVisible(bool(text))
        else:
            self._debug.hide()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, repo_root=None, parent=None):
        super().__init__(parent)
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
        self.supervisor = None
        self._running = False
        self._syncing_profile = False
        self._external_profiles = []  # profiles opened from outside profiles_dir()

        self._settings = QtCore.QSettings("std-t98-tools", "receiver")
        # Restore the last profile; seed from the env var the backend/launcher
        # read, so an existing STD_T98_BACKEND_CONFIG still works on first run.
        initial_profile = (
            self._settings.value("profile_path", "", type=str)
            or os.environ.get(CONFIG_ENV_VAR, "")
        )

        self.setWindowTitle("STD-T98 Multi Receiver")
        self.resize(1100, 820)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_stylesheet()
        self._reload_profiles(select_path=initial_profile or None)
        self._set_running(False)

    # --- UI construction ---------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Control bar (row 1): run controls
        controls = QtWidgets.QHBoxLayout()
        self._start_button = QtWidgets.QPushButton("Start")
        self._stop_button = QtWidgets.QPushButton("Stop")
        self._start_button.clicked.connect(self.start_stack)
        self._stop_button.clicked.connect(self.stop_stack)

        self._services_only = QtWidgets.QCheckBox("Services only")
        self._services_only.setToolTip("Assume the RF backend is started separately.")
        self._services_only.toggled.connect(self._update_preview)
        self._services_only.toggled.connect(self._refresh_squelch_controls_enabled)
        self._show_debug = QtWidgets.QCheckBox("Debug metrics")

        self._settings_toggle = QtWidgets.QToolButton()
        self._settings_toggle.setText("Settings")
        self._settings_toggle.setCheckable(True)
        self._settings_toggle.setChecked(True)
        self._settings_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._settings_toggle.setArrowType(QtCore.Qt.DownArrow)
        self._settings_toggle.toggled.connect(self._on_settings_toggled)

        controls.addWidget(self._start_button)
        controls.addWidget(self._stop_button)
        controls.addWidget(self._services_only)
        controls.addWidget(self._show_debug)
        controls.addStretch(1)
        controls.addWidget(self._settings_toggle)
        root.addLayout(controls)

        # Squelch row: a live control, so it stays outside the collapsible
        # Settings panel below and remains usable while the stack is running.
        squelch_row = QtWidgets.QHBoxLayout()
        squelch_row.addWidget(QtWidgets.QLabel("Squelch:"))
        self._squelch_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._squelch_slider.setRange(*SQUELCH_RANGE_DB)
        self._squelch_slider.setValue(int(round(DEFAULT_SQUELCH_THRESHOLD)))
        self._squelch_slider.valueChanged.connect(self._on_squelch_changed)
        squelch_row.addWidget(self._squelch_slider, 1)
        self._squelch_value_label = QtWidgets.QLabel(f"{int(round(DEFAULT_SQUELCH_THRESHOLD))} dB")
        self._squelch_value_label.setMinimumWidth(48)
        squelch_row.addWidget(self._squelch_value_label)
        self._squelch_save_button = QtWidgets.QPushButton("Save")
        self._squelch_save_button.setToolTip(
            "Write the current value into the selected profile's [demod].squelch_threshold."
        )
        self._squelch_save_button.clicked.connect(self._save_squelch_to_profile)
        squelch_row.addWidget(self._squelch_save_button)
        root.addLayout(squelch_row)

        # Collapsible settings panel: config picker, resolved preview, extra
        # backend args. Not needed while receiving, so it folds away on Start.
        self._settings_panel = QtWidgets.QWidget()
        panel = QtWidgets.QVBoxLayout(self._settings_panel)
        panel.setContentsMargins(0, 0, 0, 0)

        # Profile picker (layer B). A profile is a per-unit config file; the
        # (none) entry runs the backend's built-in defaults. New profiles are
        # copied from a device template; per-unit values live in the file.
        profile_row = QtWidgets.QHBoxLayout()
        self._profile_combo = QtWidgets.QComboBox()
        self._profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        self._new_profile_button = QtWidgets.QPushButton("New from device…")
        self._new_profile_button.clicked.connect(self._new_profile_from_device)
        self._open_profile_button = QtWidgets.QPushButton("Open other…")
        self._open_profile_button.clicked.connect(self._open_other_profile)
        self._detect_button = QtWidgets.QPushButton("Detect SDRs")
        self._detect_button.clicked.connect(self._detect_sdrs)
        profile_row.addWidget(QtWidgets.QLabel("Profile:"))
        profile_row.addWidget(self._profile_combo, 1)
        profile_row.addWidget(self._new_profile_button)
        profile_row.addWidget(self._open_profile_button)
        profile_row.addWidget(self._detect_button)
        panel.addLayout(profile_row)

        # Per-unit calibration, stored in the selected profile file itself.
        calib_row = QtWidgets.QHBoxLayout()
        self._freq_err = QtWidgets.QLineEdit()
        self._freq_err.setPlaceholderText("(none)")
        self._freq_err.setMaximumWidth(120)
        self._freq_err.textChanged.connect(self._update_preview)
        self._freq_err.editingFinished.connect(self._save_calibration_to_profile)
        calib_row.addWidget(QtWidgets.QLabel("Freq err offset:"))
        calib_row.addWidget(self._freq_err)
        calib_row.addWidget(QtWidgets.QLabel("Hz — per-unit calibration, saved into the profile."))
        calib_row.addStretch(1)
        panel.addLayout(calib_row)

        self._preview = QtWidgets.QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setObjectName("preview")
        self._preview.setFixedHeight(150)
        self._preview.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        panel.addWidget(self._preview)
        self._preview_note = QtWidgets.QLabel("")
        self._preview_note.setObjectName("previewNote")
        self._preview_note.setWordWrap(True)
        panel.addWidget(self._preview_note)

        backend_row = QtWidgets.QHBoxLayout()
        self._backend_args = QtWidgets.QLineEdit()
        self._backend_args.setPlaceholderText("Extra backend args, e.g. --replay capture.cf32 (squelch has its own slider above)")
        backend_row.addWidget(QtWidgets.QLabel("Backend:"))
        backend_row.addWidget(self._backend_args, 1)
        panel.addLayout(backend_row)

        root.addWidget(self._settings_panel)

        # Process health strip
        self._process_row = QtWidgets.QHBoxLayout()
        self._process_row.setSpacing(8)
        self._process_badges = {}
        process_container = QtWidgets.QFrame()
        process_container.setObjectName("processStrip")
        process_container.setLayout(self._process_row)
        self._process_row.addStretch(1)
        root.addWidget(process_container)

        # Secret cache line
        self._cache_label = QtWidgets.QLabel("Secret Cache: (empty)")
        self._cache_label.setObjectName("cacheLabel")
        root.addWidget(self._cache_label)

        # Channel grid (fixed 30 cards for a stable, scannable layout)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        grid_host = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_host)
        grid.setSpacing(8)
        self._cards = []
        columns = 5
        # Channel ids published by the backend are 1-based (1..CHANNEL_COUNT).
        for index in range(CHANNEL_COUNT):
            channel_id = index + 1
            card = ChannelCard(channel_id)
            self._cards.append(card)
            grid.addWidget(card, index // columns, index % columns)
        scroll.setWidget(grid_host)
        root.addWidget(scroll, 1)

        self._status_bar = self.statusBar()
        self._status_bar.showMessage("Stopped.")

    def _apply_stylesheet(self):
        self.setStyleSheet(
            """
            QFrame#channelCard {
                border: 1px solid #444;
                border-radius: 6px;
                background: rgba(127,127,127,0.06);
            }
            QFrame#channelCard[rx="open"] {
                border: 2px solid #2e7d32;
                background: rgba(46,125,50,0.10);
            }
            QLabel#rxBadge {
                border-radius: 4px;
                padding: 2px 0;
                font-weight: 700;
                color: #bbb;
                background: rgba(127,127,127,0.18);
            }
            QLabel#rxBadge[rx="open"] {
                color: white;
                background: #2e7d32;
            }
            QFrame#processStrip {
                border: 1px solid #444;
                border-radius: 6px;
            }
            QFrame#processBadge { border: none; }
            QLabel#cacheLabel { color: #a15fb0; font-weight: 600; }
            QPlainTextEdit#preview { font-family: monospace; font-size: 11px; }
            QLabel#previewNote[level="error"] { color: #c62828; font-weight: 600; }
            QLabel#previewNote[level="warn"] { color: #b8860b; font-weight: 600; }
            """
        )

    # --- lifecycle ---------------------------------------------------------
    def _set_running(self, running):
        self._running = running
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._services_only.setEnabled(not running)
        self._backend_args.setEnabled(not running)
        self._profile_combo.setEnabled(not running)
        self._new_profile_button.setEnabled(not running)
        self._open_profile_button.setEnabled(not running)
        self._detect_button.setEnabled(not running)
        self._refresh_freq_field_enabled()
        self._refresh_squelch_controls_enabled()

    # --- settings panel ----------------------------------------------------
    def _on_settings_toggled(self, checked):
        self._settings_panel.setVisible(checked)
        self._settings_toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )

    # --- profiles (layer B: per-unit config files) ------------------------
    def _current_profile_path(self):
        """Selected profile path (str), or None for built-in defaults."""
        return self._profile_combo.currentData()

    def _append_profile_item(self, path):
        self._profile_combo.addItem(f"{Path(path).stem} (external)", str(path))

    def _reload_profiles(self, select_path=None):
        self._syncing_profile = True
        self._profile_combo.clear()
        self._profile_combo.addItem("(none — built-in defaults)", None)
        for profile in list_profiles():
            self._profile_combo.addItem(profile.name, str(profile.path))
        for path in self._external_profiles:
            self._append_profile_item(path)
        self._syncing_profile = False
        self._select_profile_path(select_path)

    def _select_profile_path(self, path):
        target = os.path.abspath(os.path.expanduser(path)) if path else ""
        index = 0  # "(none)"
        for candidate in range(1, self._profile_combo.count()):
            data = self._profile_combo.itemData(candidate)
            if data and target and os.path.abspath(data) == target:
                index = candidate
                break
        else:
            # A path outside the profiles dir: list it as an external entry.
            if target and os.path.exists(target):
                known = {os.path.abspath(p) for p in self._external_profiles}
                if target not in known:
                    self._external_profiles.append(target)
                    self._append_profile_item(target)
                    index = self._profile_combo.count() - 1
        self._syncing_profile = True
        self._profile_combo.setCurrentIndex(index)
        self._syncing_profile = False
        self._on_profile_changed()

    def _on_profile_selected(self, _index):
        if self._syncing_profile:
            return
        self._on_profile_changed()

    def _on_profile_changed(self):
        path = self._current_profile_path()
        offset = read_freq_err_offset(path) if path else None
        # Programmatic setText fires textChanged (-> preview), not editingFinished,
        # so mirroring the file into the field does not write it straight back.
        self._freq_err.setText("" if offset is None else f"{offset:g}")

        squelch = read_squelch_threshold(path) if path else None
        # blockSignals so seeding the slider from the file does not push a
        # live control message (there is nothing running to push to anyway,
        # since the profile combo is disabled while the stack is running).
        self._squelch_slider.blockSignals(True)
        self._squelch_slider.setValue(
            int(round(squelch if squelch is not None else DEFAULT_SQUELCH_THRESHOLD))
        )
        self._squelch_slider.blockSignals(False)
        self._squelch_value_label.setText(f"{self._squelch_slider.value()} dB")

        self._refresh_freq_field_enabled()
        self._refresh_squelch_controls_enabled()
        self._update_preview()

    def _refresh_freq_field_enabled(self):
        # Calibration lives in the profile file, so it needs a selected profile.
        has_profile = self._current_profile_path() is not None
        self._freq_err.setEnabled((not self._running) and has_profile)
        self._freq_err.setToolTip(
            "" if has_profile else "Create or open a profile to save calibration."
        )

    def _refresh_squelch_controls_enabled(self):
        # The slider stays live (and enabled) while running -- that is the
        # whole point -- but it needs a managed backend to have any effect.
        manages_backend = not self._services_only.isChecked()
        self._squelch_slider.setEnabled(manages_backend)
        has_profile = self._current_profile_path() is not None
        self._squelch_save_button.setEnabled(manages_backend and has_profile)
        self._squelch_save_button.setToolTip(
            "Write the current value into the selected profile's [demod].squelch_threshold."
            if has_profile else "Create or open a profile to save the squelch value."
        )

    def _on_squelch_changed(self, value):
        self._squelch_value_label.setText(f"{value} dB")
        if self._running and self.supervisor is not None:
            self.supervisor.set_squelch(float(value))

    def _save_squelch_to_profile(self):
        path = self._current_profile_path()
        if path is None:
            return
        try:
            write_squelch_threshold(path, float(self._squelch_slider.value()))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Save squelch", f"Could not write to profile:\n{exc}")
            return
        self._update_preview()

    def _new_profile_from_device(self):
        presets = list_device_presets(self.repo_root / "devices")
        if not presets:
            QtWidgets.QMessageBox.warning(self, "New profile", "No device templates in devices/.")
            return
        names = [preset.name for preset in presets]
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "New profile", "Device template:", names, 0, False
        )
        if not ok:
            return
        preset = presets[names.index(choice)]
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New profile", "Profile name:", text=Path(preset.path).stem
        )
        name = name.strip()
        if not ok or not name:
            return
        try:
            dest = create_profile(preset.path, name)
        except FileExistsError:
            resp = QtWidgets.QMessageBox.question(
                self, "Overwrite?", f"A profile named '{name}' exists. Overwrite it?"
            )
            if resp != QtWidgets.QMessageBox.Yes:
                return
            (profiles_dir() / f"{name}.toml").unlink()
            dest = create_profile(preset.path, name)
        self._reload_profiles(select_path=str(dest))

    def _open_other_profile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open profile", str(profiles_dir()), "TOML config (*.toml);;All files (*)"
        )
        if path:
            self._reload_profiles(select_path=path)

    def _parse_freq_err(self):
        """Field value as a float, or None if empty. Raises ValueError if invalid."""
        text = self._freq_err.text().strip()
        if not text:
            return None
        return float(text)

    def _save_calibration_to_profile(self):
        path = self._current_profile_path()
        if path is None:
            return
        try:
            value = self._parse_freq_err()
        except ValueError:
            return  # invalid text: leave the file untouched
        try:
            write_freq_err_offset(path, value)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Save calibration", f"Could not write to profile:\n{exc}")
            return
        self._update_preview()

    def _update_preview(self):
        # In services-only mode the RF backend is started elsewhere, so the
        # config here is irrelevant -- say so instead of resolving it.
        if self._services_only.isChecked():
            self._preview.setPlainText("Services only — the RF backend (and its config) is started separately.")
            self._preview_note.setText("")
            return

        try:
            override = self._parse_freq_err()
            freq_err_note = ""
        except ValueError:
            override = None
            freq_err_note = "Freq err offset must be a number."

        preview = preview_config(self._current_profile_path() or "", freq_err_offset_override=override)
        if freq_err_note:
            self._preview.setPlainText(preview.summary)
            self._preview_note.setText(freq_err_note)
            self._preview_note.setProperty("level", "warn")
            self._preview_note.style().unpolish(self._preview_note)
            self._preview_note.style().polish(self._preview_note)
            return
        if not preview.ok:
            self._preview.setPlainText("")
            self._preview_note.setText(preview.error)
            self._preview_note.setProperty("level", "error")
        else:
            self._preview.setPlainText(preview.summary)
            if preview.warnings:
                self._preview_note.setText("⚠ " + "\n⚠ ".join(preview.warnings))
                self._preview_note.setProperty("level", "warn")
            else:
                self._preview_note.setText("")
                self._preview_note.setProperty("level", "ok")
        self._preview_note.style().unpolish(self._preview_note)
        self._preview_note.style().polish(self._preview_note)

    def _detect_sdrs(self):
        devices, error = detect_sdrs()
        if devices is None:
            QtWidgets.QMessageBox.warning(self, "Detect SDRs", error)
            return
        if not devices:
            QtWidgets.QMessageBox.information(
                self, "Detect SDRs", error or "No SoapySDR devices found."
            )
            return

        lines = []
        for device in devices:
            bits = [f"driver={device.driver}"]
            if device.label:
                bits.append(f"label={device.label}")
            if device.serial:
                bits.append(f"serial={device.serial}")
            lines.append("  • " + "  ".join(bits))

        # Tell the user whether the selected profile's driver is present.
        note = ""
        path = self._current_profile_path()
        if path:
            config = preview_config(path)
            if config.ok:
                driver = config.summary.splitlines()[0].split("driver=")[-1].split(",")[0]
                present = {device.driver for device in devices}
                if driver in present:
                    note = f"\nProfile driver '{driver}' is connected."
                else:
                    note = (
                        f"\n⚠ Profile driver '{driver}' is NOT among the connected devices "
                        f"({', '.join(sorted(present))})."
                    )

        QtWidgets.QMessageBox.information(
            self, "Detected SDRs", "\n".join(lines) + note
        )

    def start_stack(self):
        if self.supervisor is not None:
            return

        try:
            extra_args = shlex.split(self._backend_args.text().strip())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Backend args", f"Could not parse backend args: {exc}")
            return

        services_only = self._services_only.isChecked()
        backend_args = list(extra_args)
        if not services_only:
            try:
                self._parse_freq_err()  # validate before writing / starting
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Cannot start", "Freq err offset must be a number (Hz), or empty."
                )
                return

            # The slider is a plain CLI override (not written to the profile
            # unless Save is clicked), so Start always uses exactly what it
            # shows, whether or not that has been persisted yet.
            squelch_args = ["--squelch", str(self._squelch_slider.value())]

            path = self._current_profile_path()
            if path is not None:
                if not os.path.exists(os.path.expanduser(path)):
                    QtWidgets.QMessageBox.critical(
                        self, "Cannot start", f"Profile not found:\n{path}"
                    )
                    return
                # The profile file is the source of truth: fold the field into it,
                # then the backend reads the calibration from --config.
                self._save_calibration_to_profile()
                backend_args = ["--config", str(path), *squelch_args, *extra_args]
                self._settings.setValue("profile_path", str(path))
            else:
                backend_args = [*squelch_args, *extra_args]
                self._settings.setValue("profile_path", "")

        supervisor = StackSupervisor(
            repo_root=self.repo_root,
            services_only=services_only,
            backend_args=backend_args,
        )

        try:
            supervisor.resolve()
        except RuntimeError as exc:
            # The most common first-run failure: a missing interpreter/import.
            # The supervisor's message names exactly what was searched.
            QtWidgets.QMessageBox.critical(self, "Cannot start", str(exc))
            return

        self._rebuild_process_badges(supervisor)
        supervisor.start()
        self.supervisor = supervisor
        self._set_running(True)
        self._settings_toggle.setChecked(False)  # fold settings away while receiving
        self._status_bar.showMessage(f"Running ({supervisor.mode_label}).")
        self._timer.start()

    def stop_stack(self):
        self._timer.stop()
        if self.supervisor is not None:
            self.supervisor.stop()
            self.supervisor = None
        self._set_running(False)
        self._status_bar.showMessage("Stopped.")

    def _rebuild_process_badges(self, supervisor):
        for badge in self._process_badges.values():
            badge.setParent(None)
        self._process_badges = {}
        for index, process_view in enumerate(supervisor.process_views):
            badge = ProcessBadge(process_view.name)
            self._process_badges[process_view.name] = badge
            self._process_row.insertWidget(index, badge)

    # --- polling / rendering ----------------------------------------------
    def _on_tick(self):
        supervisor = self.supervisor
        if supervisor is None:
            return
        supervisor.poll(timeout_ms=0)
        self._refresh_from_state(
            supervisor.process_views, supervisor.channels, self._show_debug.isChecked()
        )
        if supervisor.exit_message is not None:
            self._timer.stop()
            message = supervisor.exit_message
            self.supervisor = None
            supervisor.stop()
            self._set_running(False)
            self._status_bar.showMessage("A process exited.")
            QtWidgets.QMessageBox.warning(self, "Process exited", message)

    def _refresh_from_state(self, process_views, channels, show_debug):
        for process_view in process_views:
            badge = self._process_badges.get(process_view.name)
            if badge is not None:
                badge.update_from(process_view)

        for card in self._cards:
            card.update_from(channels.get(card.channel_id), show_debug)

        self._cache_label.setText(f"Secret Cache: {_format_secret_cache(channels)}")

    # --- Qt events ---------------------------------------------------------
    def closeEvent(self, event):
        self.stop_stack()
        super().closeEvent(event)
