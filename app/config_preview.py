# -*- coding: utf-8 -*-
"""Qt-independent helpers for the GUI's config panel.

``preview_config`` resolves a backend TOML the same way the backend's
``--dry-run`` does, but in-process (backend_config imports no GNU Radio), so the
window can show what a config will actually do before starting anything.
``detect_sdrs`` wraps ``uhd_find_devices`` so the window can list connected
USRPs and tell whether the profile's device_args names one of them.
"""

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from core.rf.backend_config import derive_rates, load_config_file


@dataclass
class ConfigPreview:
    ok: bool
    summary: str = ""
    error: str = ""
    warnings: list[str] = field(default_factory=list)


def preview_config(path) -> ConfigPreview:
    """Resolve a backend config path into a human-readable summary.

    An empty path means "no --config": the backend falls back to its built-in
    defaults (any connected USRP, 1.2 MHz). A missing or invalid file is
    reported as an error the window can show before the user hits Start.
    """
    if not path or not str(path).strip():
        return ConfigPreview(ok=True, summary="No config file — built-in defaults (any connected USRP, 1.2 MHz).")

    config_path = Path(str(path)).expanduser()
    if not config_path.exists():
        return ConfigPreview(ok=False, error=f"Config file not found: {config_path}")

    try:
        config = load_config_file(config_path)
    except Exception as exc:  # tomllib / coercion errors
        return ConfigPreview(ok=False, error=f"Could not read config: {exc}")

    sdr = config.sdr
    try:
        rates = derive_rates(sdr, config.channelizer)
    except Exception as exc:
        return ConfigPreview(ok=False, error=f"Invalid channelizer / rate settings: {exc}")

    bandwidth = sdr.resolved_bandwidth()
    if sdr.agc:
        gain_desc = "AGC"
    else:
        gain_desc = f"{sdr.tuner_gain:g} dB ({sdr.gain_element or 'overall'})"

    lines = [
        f"Device      : {sdr.device_string() or '(any connected USRP)'}",
        f"Sample rate : {sdr.sample_rate:,.0f} Hz",
        f"Tuned freq  : {sdr.tuned_freq():,.0f} Hz  (err {sdr.resolved_freq_err_offset():+g} Hz)",
        f"Antenna     : {sdr.antenna or '(device default)'}",
        f"Bandwidth   : {f'{bandwidth:,.0f} Hz' if bandwidth else '(device default)'}",
        f"Gain        : {gain_desc}",
        f"Squelch     : {config.demod.squelch_threshold:g} dB",
        f"Channels    : {config.channelizer.num_channels}"
        f"  (bin width err {rates.bin_width_error_hz:+.3f} Hz)",
    ]

    return ConfigPreview(ok=True, summary="\n".join(lines))


@dataclass
class SdrDevice:
    device_type: str = ""   # UHD's "type" key, e.g. "b200", "x300"
    label: str = ""
    serial: str = ""
    extra: dict = field(default_factory=dict)


def parse_uhd_find(output: str) -> list[SdrDevice]:
    """Parse ``uhd_find_devices`` output into a list of devices.

    Each device is a "Device Address:" header followed by indented
    ``key: value`` lines. Only the fields we surface are kept named; the
    rest go into ``extra``.
    """
    devices = []
    current = None
    for raw in output.splitlines():
        line = raw.strip()
        if line == "Device Address:":
            current = SdrDevice()
            devices.append(current)
            continue
        if current is None or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "type":
            current.device_type = value
        elif key in ("name", "product") and not current.label:
            current.label = value
        elif key == "serial":
            current.serial = value
        else:
            current.extra[key] = value
    return devices


def detect_sdrs(timeout=40):
    """Return (devices, error). ``devices`` is None when uhd_find_devices is absent."""
    executable = shutil.which("uhd_find_devices")
    if executable is None:
        return None, "uhd_find_devices not found (install the UHD host utilities)."
    try:
        result = subprocess.run(
            [executable],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return [], "uhd_find_devices timed out."
    return parse_uhd_find(result.stdout), ""
