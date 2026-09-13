# -*- coding: utf-8 -*-
"""Qt-independent helpers for the GUI's config panel.

``preview_config`` resolves a backend TOML the same way the backend's
``--dry-run`` does, but in-process (backend_config imports no GNU Radio), so the
window can show what a config will actually do before starting anything.
``detect_sdrs`` wraps ``SoapySDRUtil --find`` so the window can list connected
devices and tell whether the config's driver is among them.
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
    defaults (rtlsdr / 1.2 MHz). A missing or invalid file is reported as an
    error the window can show before the user hits Start.
    """
    if not path or not str(path).strip():
        return ConfigPreview(
            ok=True,
            summary="No config file — built-in defaults (rtlsdr / 1.2 MHz).",
        )

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
    stream_args = sdr.resolved_stream_args()
    if sdr.agc:
        gain_desc = "AGC"
    else:
        element = sdr.gain_element or "overall"
        gain_desc = f"{sdr.tuner_gain:g} dB ({element})"

    lines = [
        f"Device      : {sdr.device_string()}",
        f"Sample rate : {sdr.sample_rate:,.0f} Hz",
        f"Tuned freq  : {sdr.tuned_freq():,.0f} Hz  (err {sdr.resolved_freq_err_offset():+g} Hz)",
        f"Antenna     : {sdr.antenna or '(driver default)'}",
        f"Bandwidth   : {f'{bandwidth:,.0f} Hz' if bandwidth else '(device default)'}",
        f"Stream args : {stream_args or '(none)'}",
        f"Gain        : {gain_desc}",
        f"Squelch     : {config.demod.squelch_threshold:g} dB",
        f"Channels    : {config.channelizer.num_channels}"
        f"  (bin width err {rates.bin_width_error_hz:+.3f} Hz)",
    ]

    return ConfigPreview(ok=True, summary="\n".join(lines))


@dataclass
class SdrDevice:
    driver: str = ""
    label: str = ""
    serial: str = ""
    extra: dict = field(default_factory=dict)


def parse_soapy_find(output: str) -> list[SdrDevice]:
    """Parse ``SoapySDRUtil --find`` output into a list of devices.

    Devices are separated by blank lines; each carries ``key = value`` lines.
    Only the fields we surface are kept named; the rest go into ``extra``.
    """
    devices = []
    current = None
    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith("Found device"):
            current = SdrDevice()
            devices.append(current)
            continue
        if current is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if key == "driver":
            current.driver = value
        elif key == "label":
            current.label = value
        elif key == "serial":
            current.serial = value
        else:
            current.extra[key] = value
    return devices


def detect_sdrs(timeout=40):
    """Return (devices, error). ``devices`` is None when SoapySDRUtil is absent."""
    executable = shutil.which("SoapySDRUtil")
    if executable is None:
        return None, "SoapySDRUtil not found (install soapysdr-tools)."
    try:
        result = subprocess.run(
            [executable, "--find"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return [], "SoapySDRUtil --find timed out."
    return parse_soapy_find(result.stdout), ""
