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
from core.pipeline.multi_stack_dashboard import (
    ChannelView,
    _format_csm,
    _format_sacch,
    _format_secret_cache,
)
from core.pipeline.stack_supervisor import StackSupervisor
from core.rf.backend_config import CONFIG_ENV_VAR

CHANNEL_COUNT = 30
POLL_INTERVAL_MS = 100

_PROCESS_STATE_COLOURS = {
    "RUNNING": "#2e7d32",
    "STARTING": "#b8860b",
    "EXITED": "#c62828",
}


class ProcessBadge(QtWidgets.QFrame):
    """One line of the process-health strip: name, coloured state, pid."""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setObjectName("processBadge")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._name = QtWidgets.QLabel(name)
        self._name.setStyleSheet("font-weight: 600;")
        self._state = QtWidgets.QLabel("--")
        self._pid = QtWidgets.QLabel("")
        self._pid.setStyleSheet("color: #888;")

        layout.addWidget(self._name)
        layout.addWidget(self._state)
        layout.addStretch(1)
        layout.addWidget(self._pid)

    def update_from(self, process_view):
        self._state.setText(process_view.state)
        colour = _PROCESS_STATE_COLOURS.get(process_view.state, "#b8860b")
        self._state.setStyleSheet(f"color: {colour}; font-weight: 700;")
        self._pid.setText(f"pid {process_view.pid}" if process_view.pid is not None else "")


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

        # Remember the last config across launches; fall back to the same env
        # var the backend/launcher read, so an existing workflow still works.
        self._settings = QtCore.QSettings("std-t98-tools", "receiver")
        self._initial_config_path = (
            self._settings.value("config_path", "", type=str)
            or os.environ.get(CONFIG_ENV_VAR, "")
        )

        self.setWindowTitle("STD-T98 Multi Receiver")
        self.resize(1100, 820)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_stylesheet()
        self._set_running(False)
        self._on_config_path_changed()

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

        # Collapsible settings panel: config picker, resolved preview, extra
        # backend args. Not needed while receiving, so it folds away on Start.
        self._settings_panel = QtWidgets.QWidget()
        panel = QtWidgets.QVBoxLayout(self._settings_panel)
        panel.setContentsMargins(0, 0, 0, 0)

        # Device preset picker: choosing a device fills in the config path.
        self._syncing_device = False
        device_row = QtWidgets.QHBoxLayout()
        self._device_combo = QtWidgets.QComboBox()
        self._device_combo.addItem("(custom)", None)
        for preset in list_device_presets(self.repo_root / "devices"):
            self._device_combo.addItem(preset.name, str(preset.path))
            if preset.description:
                self._device_combo.setItemData(
                    self._device_combo.count() - 1, preset.description, QtCore.Qt.ToolTipRole
                )
        self._device_combo.currentIndexChanged.connect(self._on_device_selected)
        device_row.addWidget(QtWidgets.QLabel("Device:"))
        device_row.addWidget(self._device_combo)
        device_row.addStretch(1)
        panel.addLayout(device_row)

        config_row = QtWidgets.QHBoxLayout()
        self._config_path = QtWidgets.QLineEdit(self._initial_config_path)
        self._config_path.setPlaceholderText("(no config — built-in RTL-SDR defaults)")
        self._config_path.textChanged.connect(self._on_config_path_changed)
        self._browse_button = QtWidgets.QPushButton("Browse…")
        self._browse_button.clicked.connect(self._browse_config)
        self._detect_button = QtWidgets.QPushButton("Detect SDRs")
        self._detect_button.clicked.connect(self._detect_sdrs)
        config_row.addWidget(QtWidgets.QLabel("Config:"))
        config_row.addWidget(self._config_path, 1)
        config_row.addWidget(self._browse_button)
        config_row.addWidget(self._detect_button)
        panel.addLayout(config_row)

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
        self._backend_args.setPlaceholderText("Extra backend args, e.g. --replay capture.cf32 --squelch -40")
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
        self._start_button.setEnabled(not running)
        self._stop_button.setEnabled(running)
        self._services_only.setEnabled(not running)
        self._backend_args.setEnabled(not running)
        self._config_path.setEnabled(not running)
        self._browse_button.setEnabled(not running)
        self._detect_button.setEnabled(not running)
        self._device_combo.setEnabled(not running)

    # --- settings panel ----------------------------------------------------
    def _on_settings_toggled(self, checked):
        self._settings_panel.setVisible(checked)
        self._settings_toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )

    def _on_config_path_changed(self):
        self._update_preview()
        self._sync_device_combo()

    def _on_device_selected(self, _index):
        if self._syncing_device:
            return
        path = self._device_combo.currentData()
        if path:
            self._config_path.setText(path)

    def _sync_device_combo(self):
        """Point the Device combo at the preset matching the config path, else custom."""
        text = self._config_path.text().strip()
        current = os.path.abspath(os.path.expanduser(text)) if text else ""
        index = 0  # "(custom)"
        if current:
            for candidate in range(1, self._device_combo.count()):
                data = self._device_combo.itemData(candidate)
                if data and os.path.abspath(data) == current:
                    index = candidate
                    break
        self._syncing_device = True
        self._device_combo.setCurrentIndex(index)
        self._syncing_device = False

    def _update_preview(self):
        # In services-only mode the RF backend is started elsewhere, so the
        # config here is irrelevant -- say so instead of resolving it.
        if self._services_only.isChecked():
            self._preview.setPlainText("Services only — the RF backend (and its config) is started separately.")
            self._preview_note.setText("")
            return

        preview = preview_config(self._config_path.text().strip())
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

    def _browse_config(self):
        start_dir = ""
        current = self._config_path.text().strip()
        if current:
            start_dir = str(Path(current).expanduser().parent)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select backend config", start_dir, "TOML config (*.toml);;All files (*)"
        )
        if path:
            self._config_path.setText(path)

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

        # Tell the user whether the config's driver is actually present.
        note = ""
        preview_path = self._config_path.text().strip()
        if preview_path:
            config = preview_config(preview_path)
            if config.ok:
                driver = config.summary.splitlines()[0].split("driver=")[-1].split(",")[0]
                present = {device.driver for device in devices}
                if driver in present:
                    note = f"\nConfig driver '{driver}' is connected."
                else:
                    note = (
                        f"\n⚠ Config driver '{driver}' is NOT among the connected devices "
                        f"({', '.join(sorted(present))})."
                    )

        QtWidgets.QMessageBox.information(
            self, "Detected SDRs", "\n".join(lines) + note
        )

    def _resolved_config_path(self):
        """The config path to use, or None. Raises FileNotFoundError if set but missing."""
        text = self._config_path.text().strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.exists():
            raise FileNotFoundError(str(path))
        return path

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
                config_path = self._resolved_config_path()
            except FileNotFoundError as exc:
                QtWidgets.QMessageBox.critical(
                    self, "Cannot start", f"Config file not found:\n{exc}"
                )
                return
            if config_path is not None:
                # --config first so extra args (e.g. an explicit --squelch) win.
                backend_args = ["--config", str(config_path), *extra_args]
                self._settings.setValue("config_path", str(config_path))

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
