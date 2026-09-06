"""Which Ollama server to talk to, decided in one place.

Every Ollama call in the app used to carry its own
``base_url="http://localhost:11434"`` default — the backend, the model list, the
capability probe, the report's dialog. That is fine right up until somebody runs
the server on a different machine, at which point there is no single thing to
change: the chat panel would reach the remote box and the report would still ask
localhost, or the other way round, and the mismatch shows up as "model not
found" for a model that is plainly there.

So the default is not written at the call sites any more. They pass ``None`` and
this module answers, from — in order —

1. what the caller explicitly asked for (a test, or a call that genuinely means
   one specific server),
2. what the user set in the app (the LLM panel's host field, stored in the chat
   panel's own QSettings so there is one store rather than two that disagree),
3. ``VH_OLLAMA_HOST`` or ``OLLAMA_HOST`` in the environment — the second is the
   variable Ollama's own tooling uses, so a machine already configured for a
   remote server needs nothing said twice,
4. ``http://localhost:11434``, which is what it always was.

The environment sits *below* the stored setting on purpose. A field the user
filled in is a decision they can see; an inherited variable silently overruling
it would make that field lie about where the run went.

Nothing here imports Qt at module level — the settings store is reached lazily
inside the one function that needs it, so this module is importable in a test
process with no PySide6 and no display.
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_PORT = 11434

# The chat panel's store, matched exactly — same pair of strings
# `modules.llm_discovery` matches, and for the same reason: a second private
# list of the same setting is how two parts of the app end up pointed at two
# different servers.
SETTINGS_ORG = "VideoHighlighter/LLMChat"
SETTINGS_APP = "LLMChat"
HOST_KEY = "ollama_base_url"

# `VH_` first so this app can be pointed somewhere without disturbing an
# `OLLAMA_HOST` that the machine's own `ollama` CLI is already using.
ENV_VARS = ("VH_OLLAMA_HOST", "OLLAMA_HOST")

# What a server binds to is not what a client connects to. `OLLAMA_HOST` on the
# *server* is routinely `0.0.0.0` (that is how it is made reachable at all), and
# inheriting that literally would have the client dial the any-address.
_BIND_ANY = {"0.0.0.0", "::", "[::]", ""}


def normalise(value) -> str:
    """Turn whatever a user typed into a base URL, or ``""`` if it was nothing.

    Accepts the shapes people actually enter — ``192.168.1.118``,
    ``192.168.1.118:11434``, ``http://box.lan:11434/``, and the whole endpoint
    pasted out of a terminal, ``http://box.lan:11434/api/generate``.

    Two rules are worth knowing because they are guesses:

    * a missing scheme becomes ``http://``, since a bare LAN address is
      overwhelmingly a plain Ollama server rather than a TLS one;
    * a missing port becomes 11434 **only under http**. Under https it is left
      alone, because a name with no port there means a reverse proxy on 443 and
      appending Ollama's port would break the one setup that took effort.

    Both are shown back to the user in the field they typed into, so a wrong
    guess is visible and editable rather than silent.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if "//" not in text:
        text = "http://" + text.lstrip("/")

    parts = urlsplit(text)
    scheme = parts.scheme or "http"
    host = parts.hostname or ""
    if host in _BIND_ANY:
        host = "localhost"
    if not host:
        return ""
    if ":" in host and not host.startswith("["):        # bare IPv6 literal
        host = f"[{host}]"

    port = None
    try:
        port = parts.port
    except ValueError:                                  # unparseable port
        port = None
    if port is None and scheme == "http":
        port = DEFAULT_PORT

    netloc = host if port is None else f"{host}:{port}"

    # A pasted endpoint keeps its prefix but loses the API path: callers append
    # `/api/...` themselves, and a proxy mounted at `/ollama` still works.
    path = parts.path.rstrip("/")
    cut = path.find("/api")
    if cut >= 0:
        path = path[:cut]

    return urlunsplit((scheme, netloc, path, "", ""))


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def stored(settings=None) -> str:
    """The host the user set in the app, or ``""`` — never raises.

    ``settings`` exists so a test can be handed a store of its own; without it
    this reads the user's real one, which a test suite has no business writing.
    """
    try:
        settings = settings if settings is not None else _settings()
        return normalise(settings.value(HOST_KEY, ""))
    except Exception:                                   # pragma: no cover - defensive
        return ""


def remember(value, settings=None) -> str:
    """Store a host (normalised) and hand back what was stored.

    An empty value clears the setting rather than storing an empty string, so
    "I want the default back" is expressible by emptying the field.
    """
    url = normalise(value)
    try:
        settings = settings if settings is not None else _settings()
        if url:
            settings.setValue(HOST_KEY, url)
        else:
            settings.remove(HOST_KEY)
    except Exception as exc:                            # pragma: no cover - defensive
        print(f"⚠️ Could not store the Ollama host: {exc}")
    return url


def from_env() -> str:
    """The first of :data:`ENV_VARS` that is set to something usable."""
    for name in ENV_VARS:
        url = normalise(os.environ.get(name, ""))
        if url:
            return url
    return ""


def resolve(explicit=None, settings=None) -> str:
    """The server this call should talk to. Always a usable base URL."""
    return (normalise(explicit) or stored(settings) or from_env()
            or DEFAULT_BASE_URL)


def is_remote(base_url=None) -> bool:
    """Whether the resolved server is on another machine.

    Only used to say so in a message: a failure against a remote box has a
    different first thing to check (the server was started bound to the LAN, and
    the firewall lets 11434 through) than a failure against localhost.
    """
    host = urlsplit(resolve(base_url)).hostname or ""
    return host.lower() not in {"localhost", "127.0.0.1", "::1", "[::1]"}
