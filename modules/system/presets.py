"""Named copies of the GUI settings, stored beside config.yaml.

``config.yaml`` stays the live file the window saves on close. A preset is an
extra snapshot under ``presets/<name>.yaml`` in that same folder: save the
current settings, list them, load one back onto the window, or delete one.
Loading does not bypass ``save_config`` — the caller applies the dict to the
controls and lets the normal save write config.yaml.
"""
from __future__ import annotations

import os
import re

import yaml

from modules.system.app_paths import config_path

_UNSAFE = re.compile(r"[^A-Za-z0-9 _.-]+")


def presets_dir(config_file: str | None = None) -> str:
    """Folder next to the live config: ``<config dir>/presets``."""
    cfg = config_file or config_path("config.yaml")
    return os.path.join(os.path.dirname(os.path.abspath(cfg)), "presets")


def safe_preset_name(name: str) -> str:
    """A single path segment, or '' when the name cannot be a filename."""
    text = str(name or "").strip().replace("\\", " ").replace("/", " ")
    text = _UNSAFE.sub("", text).strip(" .")
    if text in {"", ".", ".."}:
        return ""
    return text[:80].rstrip(" .")


def preset_path(name: str, config_file: str | None = None) -> str:
    safe = safe_preset_name(name)
    if not safe:
        raise ValueError("preset name is empty")
    return os.path.join(presets_dir(config_file), safe + ".yaml")


def list_presets(config_file: str | None = None) -> list[str]:
    """Preset names, sorted, without the ``.yaml`` suffix."""
    folder = presets_dir(config_file)
    if not os.path.isdir(folder):
        return []
    names = []
    for entry in os.listdir(folder):
        stem, ext = os.path.splitext(entry)
        if ext.lower() in {".yaml", ".yml"} and safe_preset_name(stem) == stem:
            names.append(stem)
    names.sort(key=str.lower)
    return names


def save_preset(name: str, data: dict, config_file: str | None = None) -> str:
    """Write ``data`` (the same mapping ``save_config`` builds). Returns the path."""
    if not isinstance(data, dict):
        raise ValueError("preset data must be a mapping")
    path = preset_path(name, config_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(data, handle, sort_keys=False, allow_unicode=True)
    return path


def load_preset(name: str, config_file: str | None = None) -> dict:
    path = preset_path(name, config_file)
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("preset is not a mapping")
    return data


def delete_preset(name: str, config_file: str | None = None) -> bool:
    """Remove one preset file. Does not touch config.yaml. False if missing."""
    path = preset_path(name, config_file)
    if not os.path.isfile(path):
        return False
    os.remove(path)
    return True
