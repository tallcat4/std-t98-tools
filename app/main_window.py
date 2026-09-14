# -*- coding: utf-8 -*-
"""Main window for the STD-T98 desktop front-end.

This is the GUI counterpart of the terminal launcher: it drives a
``StackSupervisor`` and renders the same ChannelView / ProcessView state the
launcher prints, but as a live channel grid instead of a scrolling dashboard.
The supervisor owns the process lifecycle; the window only starts/stops it and
polls it from a QTimer so the Qt event loop never blocks.
"""

import shlex
import time
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from app.config_preview import detect_sdrs, preview_config
from app.settings_store import (
    ensure_settings_file,
    read_demod_value,
    read_sdr_value,
    settings_path,
    write_demod_value,
    write_sdr_value,
)
from core.pipeline.multi_stack_dashboard import (
    ChannelView,
    _format_csm,
    _format_sacch,
    _format_secret_cache,
)
from core.pipeline.stack_supervisor import StackSupervisor
from core.rf.backend_config import DemodConfig, load_config_file

CHANNEL_COUNT = 30
POLL_INTERVAL_MS = 100
SQUELCH_RANGE_DB = (-100, 0)
DEFAULT_SQUELCH_THRESHOLD = DemodConfig().squelch_threshold
_SDR_NUMERIC_FIELDS = {"sample_rate", "tuner_gain", "bandwidth", "freq_err_offset"}


def _format_field_value(value):
    """A settings value as plain text for a form field -- e.g. 2000000, not
    the 2e+06 that Python's ``:g`` format would give a whole-numbered float."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

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

        self._health = QtWidgets.QLabel("")
        self._health.setStyleSheet("font-size: 10px;")
        self._health.hide()
        outer.addWidget(self._health)

    def update_from(self, process_view):
        self._state.setText(process_view.state)
        colour = _PROCESS_STATE_COLOURS.get(process_view.state, "#b8860b")
        # A real self-check catching a problem overrides the green "RUNNING"
        # look with the same amber used for STARTING: alive is not the same
        # as healthy.
        if process_view.state == "RUNNING" and process_view.health_ok is False:
            colour = _PROCESS_STATE_COLOURS["STARTING"]
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

        if process_view.health:
            ok = process_view.health_ok
            prefix = "?" if ok is None else ("✓" if ok else "⚠")
            colour = "#888" if ok is None else ("#2e7d32" if ok else "#c62828")
            self._health.setText(f"{prefix} {process_view.health}")
            self._health.setStyleSheet(f"color: {colour}; font-size: 10px;")
            self._health.show()
        else:
            self._health.hide()


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

        self.setWindowTitle("STD-T98 Multi Receiver")
        self.resize(1100, 820)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(POLL_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

        self._build_ui()
        self._apply_stylesheet()
        self._load_settings_into_form()
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
            "Write the current value into the settings file's [demod].squelch_threshold."
        )
        self._squelch_save_button.clicked.connect(self._save_squelch_to_settings)
        squelch_row.addWidget(self._squelch_save_button)
        root.addLayout(squelch_row)

        # Collapsible settings panel: a direct-edit form over the single fixed
        # settings file (app.settings_store), resolved preview, extra backend
        # args. Not needed while receiving, so it folds away on Start. There
        # is only one kind of device now (a USRP over UHD), so there is
        # nothing to pick a profile *of* -- every field here edits [sdr] in
        # that one file immediately, no separate save step.
        self._settings_panel = QtWidgets.QWidget()
        panel = QtWidgets.QVBoxLayout(self._settings_panel)
        panel.setContentsMargins(0, 0, 0, 0)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self._antenna_field = QtWidgets.QComboBox()
        self._antenna_field.setEditable(True)
        self._antenna_field.addItems(["", "TX/RX", "RX2"])
        self._antenna_field.setToolTip("RX antenna port. Empty = device default.")
        form.addRow("Antenna:", self._antenna_field)

        self._sample_rate_field = QtWidgets.QLineEdit()
        self._sample_rate_field.setPlaceholderText("1200000 (default)")
        form.addRow("Sample rate (Hz):", self._sample_rate_field)

        gain_row = QtWidgets.QHBoxLayout()
        self._gain_field = QtWidgets.QLineEdit()
        self._gain_field.setPlaceholderText("30 (default)")
        self._agc_field = QtWidgets.QCheckBox("AGC")
        self._agc_field.setToolTip(
            "Hardware automatic gain control. While on, the device ignores "
            "Gain/Gain element. Not every board supports it."
        )
        self._agc_field.toggled.connect(self._on_agc_changed)
        gain_row.addWidget(self._gain_field, 1)
        gain_row.addWidget(self._agc_field)
        form.addRow("Gain (dB):", gain_row)

        self._gain_element_field = QtWidgets.QLineEdit()
        self._gain_element_field.setPlaceholderText("PGA on a B210; empty = overall gain")
        form.addRow("Gain element:", self._gain_element_field)

        self._bandwidth_field = QtWidgets.QLineEdit()
        self._bandwidth_field.setPlaceholderText("(follows sample rate)")
        form.addRow("Bandwidth (Hz):", self._bandwidth_field)

        device_args_row = QtWidgets.QHBoxLayout()
        self._device_args_field = QtWidgets.QLineEdit()
        self._device_args_field.setPlaceholderText("(auto-detect if only one USRP)")
        self._detect_button = QtWidgets.QPushButton("Detect SDRs")
        self._detect_button.clicked.connect(self._detect_sdrs)
        device_args_row.addWidget(self._device_args_field, 1)
        device_args_row.addWidget(self._detect_button)
        form.addRow("Device args:", device_args_row)

        self._freq_err = QtWidgets.QLineEdit()
        self._freq_err.setPlaceholderText("(none)")
        form.addRow("Freq err offset (Hz):", self._freq_err)

        panel.addLayout(form)

        self._sdr_field_widgets = {
            "antenna": self._antenna_field,
            "sample_rate": self._sample_rate_field,
            "tuner_gain": self._gain_field,
            "gain_element": self._gain_element_field,
            "bandwidth": self._bandwidth_field,
            "device_args": self._device_args_field,
            "freq_err_offset": self._freq_err,
        }
        self._antenna_field.editTextChanged.connect(lambda _text: self._on_sdr_field_changed("antenna"))
        for key, widget in self._sdr_field_widgets.items():
            if widget is self._antenna_field:
                continue
            widget.textChanged.connect(lambda _text, k=key: self._on_sdr_field_changed(k))

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
        self._refresh_sdr_fields_enabled()
        self._refresh_squelch_controls_enabled()

    # --- settings panel ----------------------------------------------------
    def _on_settings_toggled(self, checked):
        self._settings_panel.setVisible(checked)
        self._settings_toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
        )

    # --- SDR settings form (single fixed settings file) --------------------
    def _load_settings_into_form(self):
        ensure_settings_file()
        for key, widget in self._sdr_field_widgets.items():
            text = _format_field_value(read_sdr_value(key))
            widget.blockSignals(True)
            if isinstance(widget, QtWidgets.QComboBox):
                widget.setCurrentText(text)
            else:
                widget.setText(text)
            widget.blockSignals(False)

        self._agc_field.blockSignals(True)
        self._agc_field.setChecked(bool(read_sdr_value("agc")))
        self._agc_field.blockSignals(False)
        self._gain_field.setEnabled(not self._agc_field.isChecked())

        squelch = read_demod_value("squelch_threshold")
        self._squelch_slider.blockSignals(True)
        self._squelch_slider.setValue(
            int(round(squelch if squelch is not None else DEFAULT_SQUELCH_THRESHOLD))
        )
        self._squelch_slider.blockSignals(False)
        self._squelch_value_label.setText(f"{self._squelch_slider.value()} dB")

        self._update_preview()

    def _field_text(self, key):
        widget = self._sdr_field_widgets[key]
        return widget.currentText() if isinstance(widget, QtWidgets.QComboBox) else widget.text()

    def _parse_sdr_field(self, key):
        """The field's value, coerced for storage. Raises ValueError if a
        numeric field holds unparsable text."""
        text = self._field_text(key).strip()
        if not text:
            return None
        if key in _SDR_NUMERIC_FIELDS:
            return float(text)
        return text

    def _validate_sdr_fields(self):
        """Raises ValueError if any numeric field holds unparsable text."""
        for key in _SDR_NUMERIC_FIELDS:
            text = self._field_text(key).strip()
            if text:
                float(text)

    def _on_sdr_field_changed(self, key):
        try:
            value = self._parse_sdr_field(key)
        except ValueError:
            self._update_preview()  # still show the "invalid number" warning
            return
        write_sdr_value(key, value)
        self._update_preview()

    def _on_agc_changed(self, checked):
        write_sdr_value("agc", checked)
        self._gain_field.setEnabled(not checked and not self._running)
        self._update_preview()

    def _refresh_sdr_fields_enabled(self):
        enabled = not self._running
        for widget in self._sdr_field_widgets.values():
            widget.setEnabled(enabled)
        self._agc_field.setEnabled(enabled)
        self._gain_field.setEnabled(enabled and not self._agc_field.isChecked())
        self._detect_button.setEnabled(enabled)

    def _refresh_squelch_controls_enabled(self):
        # The slider stays live (and enabled) while running -- that is the
        # whole point -- but it needs a managed backend to have any effect.
        manages_backend = not self._services_only.isChecked()
        self._squelch_slider.setEnabled(manages_backend)
        self._squelch_save_button.setEnabled(manages_backend)

    def _on_squelch_changed(self, value):
        self._squelch_value_label.setText(f"{value} dB")
        if self._running and self.supervisor is not None:
            self.supervisor.set_squelch(float(value))

    def _save_squelch_to_settings(self):
        try:
            write_demod_value("squelch_threshold", float(self._squelch_slider.value()))
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Save squelch", f"Could not write settings:\n{exc}")
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
            self._validate_sdr_fields()
            field_note = ""
        except ValueError:
            field_note = "One of the SDR fields has an invalid number."

        preview = preview_config(str(settings_path()))
        if field_note:
            self._preview.setPlainText(preview.summary)
            self._preview_note.setText(field_note)
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
                self, "Detect SDRs", error or "No UHD devices found."
            )
            return

        lines = []
        for device in devices:
            bits = []
            if device.device_type:
                bits.append(f"type={device.device_type}")
            if device.label:
                bits.append(f"label={device.label}")
            if device.serial:
                bits.append(f"serial={device.serial}")
            lines.append("  • " + "  ".join(bits))

        # Tell the user whether the settings' device_args names a serial that
        # is actually among the connected devices (structured, not scraped
        # from the preview text, since device_args is free-form UHD syntax).
        note = ""
        try:
            device_args = load_config_file(settings_path()).sdr.device_args
        except Exception:
            device_args = ""
        present = {device.serial for device in devices if device.serial}
        if device_args:
            matched = next((serial for serial in present if serial in device_args), None)
            if matched:
                note = f"\nSettings device_args mentions connected serial '{matched}'."
            elif present:
                note = (
                    f"\n⚠ Settings device_args ({device_args!r}) does not mention any "
                    f"connected serial ({', '.join(sorted(present))})."
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
                self._validate_sdr_fields()
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Cannot start", "One of the SDR fields has an invalid number."
                )
                return

            # Flush anything not yet written (e.g. a field mid-edit that never
            # lost focus) before the backend reads the settings file.
            for key in self._sdr_field_widgets:
                self._on_sdr_field_changed(key)

            # The slider is a plain CLI override (not written to the settings
            # file unless Save is clicked), so Start always uses exactly what
            # it shows, whether or not that has been persisted yet.
            squelch_args = ["--squelch", str(self._squelch_slider.value())]
            backend_args = ["--config", str(settings_path()), *squelch_args, *extra_args]

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
