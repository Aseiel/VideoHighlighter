"""What models this machine already has, so nothing has to be typed from memory.

The report's model dialog asked for an Ollama model as free text. The server it
is about to talk to holds the authoritative list, the chat panel has been
reading that list for years, and the dialog next to it made the user remember a
tag and spell it right — a model name typed with a typo is indistinguishable
from one that is not pulled yet, and the failure arrives minutes later at the
end of a generation the user waited for.

So this is the "ask the machine" half, kept out of :mod:`modules.llm_models`
because that module is deliberately pure — parsing and naming what was stored,
no I/O — and out of the dialog because a widget that reaches the network in its
constructor cannot be opened in a test.

Two sources, both already existing
----------------------------------

**The Ollama server**, via :func:`llm.llm_module.get_ollama_models`. Cached for
the life of the process: the list changes when somebody pulls a model, which is
not something that happens between two openings of a dialog, and the call costs
a three-second timeout every time the server is *not* running — which is exactly
the case where a user opens the dialog repeatedly wondering what is wrong.
``refresh=True`` is the Refresh button and bypasses the cache.

**The GGUF files the chat panel has been used with.** Deliberately the same
QSettings list the panel itself writes, not a second one of the report's own. A
file browsed to in either place shows up in both, which is what somebody who
has already picked that model once expects, and two private lists of the same
thing is how they end up disagreeing about which model is "recent".

Nothing here raises. Every function answers with an empty list when the server
is down, the package is missing, or Qt is not available — the dialog stays
usable and says so, because the fallback for all of this is the typing that
already worked.
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# The chat panel's own store, matched exactly: QSettings(organisation,
# application) with these two strings, and this key. If the panel ever moves
# them, this moves with it -- a silently separate list is worse than no list.
SETTINGS_ORG = "VideoHighlighter/LLMChat"
SETTINGS_APP = "LLMChat"
RECENT_GGUF_KEY = "recent_gguf_paths"

# Matches the chat panel's own cap. Longer than this is not a shortlist.
MAX_RECENT_GGUF = 10

# {base_url: [names]} for this process. Populated on the first ask.
_cache: dict = {}


def ollama_models(base_url: str = DEFAULT_OLLAMA_URL,
                  refresh: bool = False) -> list:
    """The models the local Ollama server holds, or ``[]`` if it cannot be asked.

    An empty list is not an error and must not be shown as one: a user who has
    not started Ollama yet, or who only uses GGUF files, is in a perfectly
    ordinary state. The caller says "not reachable" and leaves the field
    typable.
    """
    if not refresh and base_url in _cache:
        return list(_cache[base_url])
    try:
        from llm.llm_module import get_ollama_models
        found = [str(name) for name in (get_ollama_models(base_url) or [])]
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Could not list Ollama models: {exc}")
        found = []
    _cache[base_url] = found
    return list(found)


def forget_ollama_models() -> None:
    """Drop the cache. For the Refresh button's sake, and for tests."""
    _cache.clear()


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def _stored(settings) -> list:
    """The raw list, with the shape QSettings actually returns it in."""
    raw = settings.value(RECENT_GGUF_KEY, [])
    # QSettings hands back a bare string when the list holds one item, which is
    # the same quirk `modules.llm_models.parse` documents for its own value.
    if isinstance(raw, str):
        return [raw] if raw else []
    return list(raw or [])


def recent_gguf(settings=None) -> list:
    """GGUF files the chat panel has connected to, newest first.

    Paths that no longer exist are dropped from what is *shown* and left in what
    is stored. A file on a disconnected drive is not a mistake to clean up
    behind the user's back, and the list is short enough that it costs nothing
    to keep.

    ``settings`` exists so a test can be given a store of its own. Without it
    the tests would write into the real one — this is the user's actual chat
    panel history, and a test suite is not allowed to edit it.
    """
    try:
        stored = _stored(settings if settings is not None else _settings())
    except Exception:                              # pragma: no cover - defensive
        return []
    # QSettings hands back a bare string when the list holds one item, which is
    # the same quirk `modules.llm_models.parse` documents for its own value.
    out = []
    for path in stored:
        path = str(path)
        if path and path not in out and os.path.isfile(path):
            out.append(path)
    return out[:MAX_RECENT_GGUF]


def remember_gguf(path: str, settings=None) -> None:
    """Put a GGUF at the top of that same list, so the chat panel sees it too."""
    path = str(path or "").strip()
    if not path:
        return
    try:
        settings = settings if settings is not None else _settings()
        stored = [str(p) for p in _stored(settings) if str(p) != path]
        settings.setValue(RECENT_GGUF_KEY, [path] + stored[:MAX_RECENT_GGUF - 1])
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Could not remember the GGUF path: {exc}")
