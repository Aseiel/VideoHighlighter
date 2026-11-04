"""What models this machine already has, so nothing has to be typed from memory.

The report's model dialog asked for an Ollama model as free text. The server it
is about to talk to holds the authoritative list, the chat panel has been
reading that list for years, and the dialog next to it made the user remember a
tag and spell it right — a model name typed with a typo is indistinguishable
from one that is not pulled yet, and the failure arrives minutes later at the
end of a generation the user waited for.

So this is the "ask the machine" half, kept out of :mod:`modules.narration.llm_models`
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

from llm.ollama_host import resolve as resolve_ollama_host

# Kept as a name because callers and tests refer to it; the *decision* about
# which server to ask lives in :mod:`llm.ollama_host`, so a user who moved
# Ollama to another machine moves this list with it.
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


def ollama_models(base_url: Optional[str] = None,
                  refresh: bool = False) -> list:
    """The models the configured Ollama server holds, or ``[]`` if it cannot be asked.

    An empty list is not an error and must not be shown as one: a user who has
    not started Ollama yet, or who only uses GGUF files, is in a perfectly
    ordinary state. The caller says "not reachable" and leaves the field
    typable.

    ``base_url=None`` is the server the user configured (localhost unless they
    said otherwise). The cache is keyed by the resolved URL, so pointing the app
    at a different machine asks that machine rather than replaying the answer
    the old one gave.
    """
    base_url = resolve_ollama_host(base_url)
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
    _lmstudio_cache.clear()


# Kept apart from `_cache` because both are keyed by URL, and a user can point
# both servers at the same machine on different ports without either list
# replaying the other's.
_lmstudio_cache: dict = {}


def lmstudio_models(base_url: Optional[str] = None,
                    refresh: bool = False) -> list:
    """The chat models the configured LM Studio server offers, or ``[]``.

    Same contract as :func:`ollama_models`: cached per URL for the process,
    ``refresh=True`` asks again, and nothing here raises.
    """
    from llm import lmstudio
    base_url = lmstudio.resolve(base_url)
    if not refresh and base_url in _lmstudio_cache:
        return list(_lmstudio_cache[base_url])
    found = lmstudio.list_models(base_url)
    _lmstudio_cache[base_url] = found
    return list(found)


def models_for(backend: str, refresh: bool = False) -> list:
    """The model list for a server backend; ``[]`` for anything else."""
    if backend == "ollama":
        return ollama_models(refresh=refresh)
    if backend == "lmstudio":
        return lmstudio_models(refresh=refresh)
    return []


# The servers worth looking for, in the order a menu offers them.
SERVER_BACKENDS = ("ollama", "lmstudio")


def _probe(backend: str, timeout: float):
    """The server's model list, or ``None`` when nothing answered."""
    import requests
    try:
        if backend == "ollama":
            resp = requests.get(f"{resolve_ollama_host()}/api/tags",
                                timeout=timeout)
            resp.raise_for_status()
            return [str(m["name"]) for m in resp.json().get("models", [])]
        from llm import lmstudio
        return [str(m["id"]) for m in
                lmstudio.server_models(timeout=timeout)]
    except Exception:
        return None


def running_servers(timeout: float = 1.0) -> dict:
    """``{backend: [models] or None}`` for each server, asked at once.

    ``None`` is "nothing answered"; ``[]`` is "running, but no models", which
    is a different thing to tell the user. The probes run in parallel so the
    whole answer costs one timeout, not one per server, and the lists are
    stored in the caches above so the menu that asked does not ask again.
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(SERVER_BACKENDS)) as pool:
        futures = {name: pool.submit(_probe, name, timeout)
                   for name in SERVER_BACKENDS}
        found = {name: future.result() for name, future in futures.items()}
    if found.get("ollama") is not None:
        _cache[resolve_ollama_host()] = list(found["ollama"])
    if found.get("lmstudio") is not None:
        from llm import lmstudio
        _lmstudio_cache[lmstudio.resolve()] = list(found["lmstudio"])
    return found


def preferred_server(found: dict, current: Optional[str] = None) -> Optional[str]:
    """Which server to select: ``current`` if it answered, else the first that did.

    A server that answered with models beats one that answered empty, so a
    stopped-but-installed LM Studio does not win over an Ollama that can run.
    """
    if current in found and found[current]:
        return current
    for name in SERVER_BACKENDS:
        if found.get(name):
            return name
    if current in found and found[current] is not None:
        return current
    for name in SERVER_BACKENDS:
        if found.get(name) is not None:
            return name
    return None


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def _stored(settings) -> list:
    """The raw list, with the shape QSettings actually returns it in."""
    raw = settings.value(RECENT_GGUF_KEY, [])
    # QSettings hands back a bare string when the list holds one item, which is
    # the same quirk `modules.narration.llm_models.parse` documents for its own value.
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
    # the same quirk `modules.narration.llm_models.parse` documents for its own value.
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
