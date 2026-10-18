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
    """Translate a UI string. ``msgid`` must be the English source text."""
    if not msgid:
        return msgid
    _ensure_catalog()
    if _current == DEFAULT_LANGUAGE:
        return msgid
    table = _catalog.get(_current) or {}
    return table.get(msgid, msgid)
