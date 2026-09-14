# -*- coding: utf-8 -*-
"""The GUI's single settings file.

The backend targets one kind of device now (a USRP over UHD), so there is
nothing left to choose between -- no per-model templates, no picking which
profile file to open. The GUI reads and writes exactly one fixed-path TOML
file, and the settings form edits it directly. The backend still takes a
single ``--config`` file (that stays the only way to pass settings across
the process boundary to the subprocess); this is simply the one file the
GUI always uses, never something the user picks.

tomllib is read-only, so writing a value back edits just that one line,
preserving the rest of the file's comments and structure.
"""

import json
import os
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


def settings_path() -> Path:
    """The GUI's fixed settings file: $XDG_CONFIG_HOME/std-t98/settings.toml."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(Path.home(), ".config")
    return Path(base) / "std-t98" / "settings.toml"


def ensure_settings_file() -> Path:
    """Create an empty settings file if none exists yet, and return its path."""
    path = settings_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return path


def _read_value(section, key):
    if tomllib is None:  # pragma: no cover
        return None
    path = settings_path()
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        return None
    value = data.get(section, {}).get(key)
    return value if isinstance(value, (int, float, str)) else None


def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, str):
        return json.dumps(value)  # TOML basic strings are JSON-compatible
    return str(value)


def set_toml_scalar(text: str, section: str, key: str, value) -> str:
    """Set ``[section] key = value`` in a TOML string, preserving comments.

    Replaces the key in place if present, inserts it under an existing section
    header otherwise, and appends the section if it is missing. ``value=None``
    removes the key. Only handles top-level sections and simple scalars, which
    is all the settings file needs.
    """
    lines = text.splitlines()
    header = f"[{section}]"
    section_start = None
    section_end = len(lines)

    for index, line in enumerate(lines):
        if line.strip() == header:
            section_start = index
            for later in range(index + 1, len(lines)):
                if lines[later].strip().startswith("["):
                    section_end = later
                    break
            break

    def is_key_line(line):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return False
        name = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        return name == key

    if section_start is not None:
        key_index = next(
            (i for i in range(section_start + 1, section_end) if is_key_line(lines[i])),
            None,
        )
        if value is None:
            if key_index is not None:
                del lines[key_index]
        elif key_index is not None:
            lines[key_index] = f"{key} = {_format_scalar(value)}"
        else:
            lines.insert(section_start + 1, f"{key} = {_format_scalar(value)}")
    elif value is not None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(header)
        lines.append(f"{key} = {_format_scalar(value)}")

    return "\n".join(lines) + "\n"


def _write_value(section, key, value) -> None:
    path = ensure_settings_file()
    text = path.read_text(encoding="utf-8")
    path.write_text(set_toml_scalar(text, section, key, value), encoding="utf-8")


def read_sdr_value(key):
    """A value from [sdr] in the settings file, or None if unset/unreadable."""
    return _read_value("sdr", key)


def write_sdr_value(key, value) -> None:
    """Write (or clear, if value is None) a [sdr] key in the settings file."""
    _write_value("sdr", key, value)


def read_demod_value(key):
    """A value from [demod] in the settings file, or None if unset/unreadable."""
    return _read_value("demod", key)


def write_demod_value(key, value) -> None:
    """Write (or clear, if value is None) a [demod] key in the settings file."""
    _write_value("demod", key, value)
