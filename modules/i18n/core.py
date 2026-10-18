"""Lookup UI strings. English msgids are the keys; missing entries fall back."""

from __future__ import annotations

from typing import Any

from modules.i18n.languages import DEFAULT_LANGUAGE, normalize_code

_current = DEFAULT_LANGUAGE
_catalog: dict[str, dict[str, str]] = {}


def _ensure_catalog() -> None:
    global _catalog
    if _catalog:
        return
    from modules.i18n import catalog as cat

    _catalog = dict(cat.TRANSLATIONS)


def current_language() -> str:
    return _current


def set_language(code: str | None) -> str:
    """Activate a UI language. Returns the code actually used."""
    global _current
    _ensure_catalog()
    _current = normalize_code(code)
    return _current


def load_from_config(config: dict[str, Any] | None) -> str:
    ui = (config or {}).get("ui") or {}
    return set_language(ui.get("language"))


def t(msgid: str) -> str:
    """Translate a UI string. ``msgid`` must be the English source text.

    Padding is not part of the text: "  Audio" is looked up as "Audio" and
    comes back padded the same way, so a label indented with spaces still
    matches its catalog entry."""
    if not msgid:
        return msgid
    _ensure_catalog()
    if _current == DEFAULT_LANGUAGE:
        return msgid
    table = _catalog.get(_current) or {}
    # An empty entry is a gap, not a translation: returned as-is it blanked
    # the widget (Korean's "⏸ Pause" button had no text at all).
    exact = table.get(msgid)
    if exact:
        return exact
    core = msgid.strip()
    found = table.get(core) if core != msgid else None
    if not found:
        return msgid
    lead = msgid[:len(msgid) - len(msgid.lstrip())]
    trail = msgid[len(msgid.rstrip()):]
    return f"{lead}{found}{trail}"
