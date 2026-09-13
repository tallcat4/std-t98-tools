# -*- coding: utf-8 -*-
"""Per-unit device profiles: the layer-B text configs.

Presets in devices/ are model definitions (layer A, shared, unchanging). A
*profile* is a copy of one that a user keeps per radio, carrying the per-unit
values -- above all the frequency calibration -- as plain TOML. The backend
takes a single --config file, so a profile is a complete config, not an overlay.

tomllib is read-only, so writing a calibration back edits just the one
``[sdr].freq_err_offset`` line, preserving the template's comments. That is all
the GUI needs to write; nothing here rewrites a whole TOML document.
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@dataclass
class Profile:
    path: Path
    name: str


def profiles_dir() -> Path:
    """Default profile directory: $XDG_CONFIG_HOME/std-t98/profiles (or ~/.config)."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(Path.home(), ".config")
    return Path(base) / "std-t98" / "profiles"


def list_profiles(directory=None) -> list[Profile]:
    """Profiles in ``directory`` (default: profiles_dir()), named by filename."""
    directory = Path(directory) if directory is not None else profiles_dir()
    if not directory.is_dir():
        return []
    return [Profile(path=path, name=path.stem) for path in sorted(directory.glob("*.toml"))]


def create_profile(template_path, name, directory=None) -> Path:
    """Copy a device template into a new profile ``<directory>/<name>.toml``.

    The copy keeps the template's comments. Raises FileExistsError if a profile
    of that name already exists, so the caller can confirm an overwrite.
    """
    directory = Path(directory) if directory is not None else profiles_dir()
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{name}.toml"
    if dest.exists():
        raise FileExistsError(str(dest))
    shutil.copy2(template_path, dest)
    return dest


def read_freq_err_offset(profile_path):
    """The profile's [sdr].freq_err_offset, or None if unset/unreadable."""
    if tomllib is None:  # pragma: no cover
        return None
    try:
        with open(profile_path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        return None
    value = data.get("sdr", {}).get("freq_err_offset")
    return value if isinstance(value, (int, float)) else None


def read_squelch_threshold(profile_path):
    """The profile's [demod].squelch_threshold, or None if unset/unreadable."""
    if tomllib is None:  # pragma: no cover
        return None
    try:
        with open(profile_path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, ValueError):
        return None
    value = data.get("demod", {}).get("squelch_threshold")
    return value if isinstance(value, (int, float)) else None


def _format_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return repr(value) if isinstance(value, str) else str(value)


def set_toml_scalar(text: str, section: str, key: str, value) -> str:
    """Set ``[section] key = value`` in a TOML string, preserving comments.

    Replaces the key in place if present, inserts it under an existing section
    header otherwise, and appends the section if it is missing. ``value=None``
    removes the key. Only handles top-level sections and simple scalars, which
    is all a device profile needs.
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


def write_freq_err_offset(profile_path, value) -> None:
    """Write (or clear, if value is None) [sdr].freq_err_offset in the profile."""
    path = Path(profile_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(set_toml_scalar(text, "sdr", "freq_err_offset", value), encoding="utf-8")


def write_squelch_threshold(profile_path, value) -> None:
    """Write (or clear, if value is None) [demod].squelch_threshold in the profile."""
    path = Path(profile_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(set_toml_scalar(text, "demod", "squelch_threshold", value), encoding="utf-8")
