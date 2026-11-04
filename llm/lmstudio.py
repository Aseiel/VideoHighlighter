"""LM Studio — and any other OpenAI-compatible local server — as an LLM backend.

Why a second server beside Ollama
---------------------------------

Some users already run LM Studio: it has a model browser, per-model settings and
a GPU picker, and a model downloaded there should not have to be pulled again
into Ollama to be used here. Its server speaks the OpenAI chat-completions
protocol on ``http://localhost:1234/v1``. So do llama.cpp's own ``llama-server``,
Jan, vLLM and LocalAI, which is why this talks the protocol rather than
anything LM Studio-only. The one LM Studio extra it reads, ``/api/v0/models``
(model type, loaded state, context length), is asked best-effort and a server
without it still works.

What does not carry over from Ollama, measured against LM Studio 0.4 with
``google/gemma-4-e4b``:

* **The context is fixed when the model loads.** Ollama takes ``num_ctx`` per
  request, so its backend grows the context when a prompt does not fit. Here
  the context is a load setting in LM Studio, so a prompt that does not fit
  ends in a message saying where to raise it rather than a retry.
* **Reasoning cannot be switched off from here.** ``reasoning_effort: none``,
  ``chat_template_kwargs``, ``reasoning: off`` were each tried, and the same
  question reasoned on one call and not on the next. The reasoning arrives in
  ``reasoning_content`` (or inline in ``<think>`` tags when LM Studio is set
  not to separate it), so this does what the Ollama backend does: a reply that
  is reasoning only and stopped at the cap is retried uncapped, and the model
  stays uncapped for the rest of the run.
* **Models load on first use.** With just-in-time loading on, which is LM
  Studio's default, a model that is downloaded but not loaded is listed and
  the first request loads it. That took 40 s for a 4B model, so the read
  timeout is per chunk and long.

Host handling mirrors :mod:`llm.ollama_host` (stored setting, then
``VH_LMSTUDIO_HOST``, then localhost) and shares its settings store, so the
chat panel, the report and narration all follow one field.

Nothing here imports Qt or :mod:`llm.llm_module` at module level, so the
model dialog can list models without loading the local-model stack.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional
from urllib.parse import urlsplit, urlunsplit

DEFAULT_PORT = 1234
DEFAULT_BASE_URL = f"http://localhost:{DEFAULT_PORT}"

# Same store as `llm.ollama_host`, so both servers' hosts live side by side.
SETTINGS_ORG = "VideoHighlighter/LLMChat"
SETTINGS_APP = "LLMChat"
HOST_KEY = "lmstudio_base_url"

# LM Studio defines no environment variable of its own, so only ours.
ENV_VARS = ("VH_LMSTUDIO_HOST",)

_BIND_ANY = {"0.0.0.0", "::", "[::]", ""}

# (connect, read). Read is per chunk: the first one waits for a just-in-time
# load plus the prompt, and with frames in it on a CPU that is minutes.
TIMEOUT = (10, 600)

# Model types LM Studio reports that cannot answer a chat request.
_NOT_CHAT = ("embeddings", "embedding")

# A chat model's id never says "embed"; an embedding model's nearly always does.
# Only used when the server does not report a type.
_EMBED_MARKER = "embed"


# --------------------------------------------------------------------------- #
# Where the server is
# --------------------------------------------------------------------------- #
def normalise(value) -> str:
    """Turn whatever a user typed into a base URL, or ``""`` if it was nothing.

    The same rules as :func:`llm.ollama_host.normalise` with this server's port:
    a missing scheme is ``http://``, a missing port under http is 1234, and a
    pasted endpoint (``http://box:1234/v1/chat/completions``) loses its API path
    because callers append ``/v1/...`` themselves.
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

    try:
        port = parts.port
    except ValueError:                                  # unparseable port
        port = None
    if port is None and scheme == "http":
        port = DEFAULT_PORT
    netloc = host if port is None else f"{host}:{port}"

    path = parts.path.rstrip("/")
    for marker in ("/v1", "/api"):
        cut = path.find(marker)
        if cut >= 0:
            path = path[:cut]
    return urlunsplit((scheme, netloc, path, "", ""))


def _settings():
    from PySide6.QtCore import QSettings
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def stored(settings=None) -> str:
    """The host the user set in the app, or ``""`` — never raises."""
    try:
        settings = settings if settings is not None else _settings()
        return normalise(settings.value(HOST_KEY, ""))
    except Exception:                                   # pragma: no cover - defensive
        return ""


def remember(value, settings=None) -> str:
    """Store a host (normalised) and hand back what was stored; empty clears it."""
    url = normalise(value)
    try:
        settings = settings if settings is not None else _settings()
        if url:
            settings.setValue(HOST_KEY, url)
        else:
            settings.remove(HOST_KEY)
    except Exception as exc:                            # pragma: no cover - defensive
        print(f"⚠️ Could not store the LM Studio host: {exc}")
    return url


def from_env() -> str:
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
    host = (urlsplit(resolve(base_url)).hostname or "").lower()
    return host not in ("localhost", "127.0.0.1", "::1")


# --------------------------------------------------------------------------- #
# What it has
# --------------------------------------------------------------------------- #
def _is_chat(entry: dict) -> bool:
    kind = str(entry.get("type") or "").lower()
    if kind:
        return kind not in _NOT_CHAT
    return _EMBED_MARKER not in str(entry.get("id") or "").lower()


def server_models(base_url: Optional[str] = None, timeout: float = 3) -> list:
    """The chat models the server offers, as dicts with at least ``id``.

    Raises ``requests.RequestException`` when nothing answers, so a caller can
    tell "not running" from "running with nothing loaded". LM Studio's own
    endpoint is asked first because it says which models see images and how
    much context each one has; any other OpenAI-compatible server answers the
    second.
    """
    import requests
    base = resolve(base_url)
    try:
        resp = requests.get(f"{base}/api/v0/models", timeout=timeout)
        if resp.ok:
            data = resp.json().get("data")
            if isinstance(data, list):
                return [m for m in data if isinstance(m, dict) and m.get("id")
                        and _is_chat(m)]
    except requests.ConnectionError:
        raise
    except (requests.RequestException, ValueError):
        pass
    resp = requests.get(f"{base}/v1/models", timeout=timeout)
    resp.raise_for_status()
    data = resp.json().get("data") or []
    return [m for m in data if isinstance(m, dict) and m.get("id") and _is_chat(m)]


def list_models(base_url: Optional[str] = None, timeout: float = 3) -> list:
    """Model ids on the server, or ``[]`` when it cannot be asked."""
    try:
        return [str(m["id"]) for m in server_models(base_url, timeout=timeout)]
    except Exception:
        return []


def match_model(wanted: str, available: list) -> Optional[str]:
    """The server's id for ``wanted``: exact, then case-blind, then a unique part.

    LM Studio ids carry a publisher (``google/gemma-4-e4b``) that a user typing
    from memory leaves off, so ``gemma-4-e4b`` is accepted when exactly one id
    contains it. Two that do is a question only the user can answer.
    """
    wanted = (wanted or "").strip()
    if not wanted:
        return None
    if wanted in available:
        return wanted
    lowered = wanted.lower()
    for name in available:
        if name.lower() == lowered:
            return name
    partial = [name for name in available if lowered in name.lower()]
    return partial[0] if len(partial) == 1 else None


# --------------------------------------------------------------------------- #
# The backend
# --------------------------------------------------------------------------- #
def _data_url(image_b64: str) -> str:
    """A base64 frame as the data URL the protocol wants; JPEG unless it says PNG."""
    if image_b64.startswith("data:"):
        return image_b64
    mime = "image/png" if image_b64.startswith("iVBOR") else "image/jpeg"
    return f"data:{mime};base64,{image_b64}"


def _error_detail(resp) -> str:
    """The reason the server gave, not just its status."""
    try:
        parsed = resp.json()
    except ValueError:
        return (resp.text or f"HTTP {resp.status_code}").strip()
    err = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    if isinstance(err, str):
        return err
    return (resp.text or f"HTTP {resp.status_code}").strip()


def _context_overflow(detail: str) -> bool:
    text = (detail or "").lower()
    return "context" in text and any(
        word in text for word in ("length", "overflow", "exceed", "too long",
                                  "greater than"))


class _ThinkSplitter:
    """Splits streamed content into answer and inline ``<think>`` reasoning.

    LM Studio puts reasoning in ``reasoning_content`` by default, but a user
    who turned that off gets it inline, and llama-server does the same. The
    tags arrive as tokens of their own in practice; one split across two
    chunks would leak into the answer, which the sanitiser then has to cope
    with like any other stray text.
    """
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.inside = False

    def feed(self, text: str) -> tuple:
        answer, thought = [], []
        while text:
            if self.inside:
                cut = text.find(self.CLOSE)
                if cut < 0:
                    thought.append(text)
                    break
                thought.append(text[:cut])
                text = text[cut + len(self.CLOSE):]
                self.inside = False
            else:
                cut = text.find(self.OPEN)
                if cut < 0:
                    answer.append(text)
                    break
                answer.append(text[:cut])
                text = text[cut + len(self.OPEN):]
                self.inside = True
        return "".join(answer), "".join(thought)


class LMStudioBackend:
    """Talks to LM Studio, or any OpenAI-compatible server, at ``base_url``.

    Duck-typed to ``llm.llm_module._LLMBackend`` rather than subclassing it,
    so importing this module does not import that one.
    """

    def __init__(self, model: str = "", base_url: Optional[str] = None):
        self.model = model
        self.base_url = resolve(base_url)
        self._loaded = False
        self._thinks = False
        # What `/api/v0/models` said about this model; empty for other servers.
        self._info: dict = {}

    @staticmethod
    def available() -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def _where(self) -> str:
        return f"LM Studio at {self.base_url}"

    def load(self, **kwargs):
        """Check the server answers and has the model, and learn what it is."""
        import requests
        try:
            models = server_models(self.base_url, timeout=5)
        except requests.RequestException:
            hint = ("On that machine, turn on \"Serve on Local Network\" in LM "
                    "Studio's Developer tab and let port 1234 through its "
                    "firewall." if is_remote(self.base_url) else
                    "In LM Studio open the Developer tab and start the server, "
                    "or run: lms server start")
            raise RuntimeError(f"Cannot connect to {self._where()}. Is its "
                               f"server running?\n{hint}")
        ids = [str(m["id"]) for m in models]
        if not ids:
            raise RuntimeError(f"{self._where()} has no chat models. Download "
                               f"one in LM Studio, or load one if just-in-time "
                               f"loading is off.")
        found = match_model(self.model, ids)
        if not found:
            raise RuntimeError(f"Model '{self.model}' not found on "
                               f"{self._where()}. Available: {ids}")
        self.model = found
        self._info = next((m for m in models if m.get("id") == found), {})
        self._loaded = True

        kind = self._info.get("type")
        state = self._info.get("state")
        ctx = (self._info.get("loaded_context_length")
               or self._info.get("max_context_length"))
        print(f"[lmstudio] {found} at {self.base_url}"
              + (f" · {kind}" if kind else "")
              + (f" · {state}" if state else "")
              + (f" · context {ctx}" if ctx else ""))
        if state == "not-loaded":
            print("[lmstudio] not loaded yet; the first request loads it, "
                  "which can take a minute.")

    def is_loaded(self) -> bool:
        return self._loaded

    def accepts_images(self) -> bool:
        """Whether frames can be sent. Unknown (not LM Studio) means try."""
        kind = str(self._info.get("type") or "").lower()
        return kind != "llm"

    def unload(self):
        # LM Studio unloads idle models on its own timer; ending our use of the
        # model is not a reason to evict it from a server others may be using.
        self._loaded = False

    def _payload(self, prompt, system, max_tokens, temperature, images):
        from llm.llm_module import STOP_SEQUENCES
        content = prompt
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": _data_url(img)}}
                for img in images]
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": content})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "stop": STOP_SEQUENCES,
            # Not in OpenAI's protocol; LM Studio and llama-server read it,
            # others ignore it. Same value the Ollama backend sends.
            "repeat_penalty": 1.3,
            "stream_options": {"include_usage": True},
        }
        # No cap is expressed by leaving it out: "-1" is LM Studio's spelling
        # and not every compatible server accepts it.
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        return payload

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                 images: list | None = None, cancellation_token=None) -> str:
        import requests
        from llm.llm_module import (THINKING_BUDGET, _SELF_TALK_PATTERNS,
                                    _check_cancel, sanitize_response)

        if self._thinks:
            max_tokens = THINKING_BUDGET

        payload = self._payload(prompt, system, max_tokens, temperature, images)
        answer, thought = [], []
        splitter = _ThinkSplitter()
        usage, finish = None, None
        t_start = time.perf_counter()
        t_first = None

        with requests.post(f"{self.base_url}/v1/chat/completions",
                           json=payload, stream=True, timeout=TIMEOUT) as resp:
            if resp.status_code >= 400:
                detail = _error_detail(resp)
                print(f"[lmstudio] {self.base_url} refused the request "
                      f"({resp.status_code}): {detail}")
                if _context_overflow(detail):
                    ctx = self._info.get("loaded_context_length")
                    raise RuntimeError(
                        f"The request does not fit the context {self.model} is "
                        f"loaded with{f' ({ctx} tokens)' if ctx else ''}. Raise "
                        f"its Context Length in LM Studio and load it again.")
                raise RuntimeError(f"LM Studio ({self.model}) refused the "
                                   f"request: {detail}")
            for raw in resp.iter_lines():
                if not raw or not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    break
                _check_cancel(cancellation_token)
                if t_first is None:
                    t_first = time.perf_counter()
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                stop_now = False
                for choice in chunk.get("choices") or ():
                    finish = choice.get("finish_reason") or finish
                    delta = choice.get("delta") or {}
                    reasoning = (delta.get("reasoning_content")
                                 or delta.get("reasoning") or "")
                    if reasoning:
                        thought.append(reasoning)
                    text, inline = splitter.feed(delta.get("content") or "")
                    if inline:
                        thought.append(inline)
                    if text:
                        answer.append(text)
                        if _SELF_TALK_PATTERNS.search("".join(answer)):
                            stop_now = True
                            break
                        if stream_callback:
                            stream_callback(text)
                if stop_now:
                    break

        wall = time.perf_counter() - t_start
        if usage:
            out_n = usage.get("completion_tokens") or 0
            gen = wall - ((t_first - t_start) if t_first else 0)
            print(f"   ⏱  lmstudio: ttfb={((t_first or t_start) - t_start) * 1000:5.0f}ms  "
                  f"wall={wall * 1000:5.0f}ms  prompt={usage.get('prompt_tokens', 0)}t  "
                  f"out={out_n}t @ {out_n / gen if gen > 0 else 0:5.1f}t/s")

        raw_text = "".join(answer)
        if not raw_text.strip() and thought:
            if max_tokens != THINKING_BUDGET:
                self._thinks = True
                print(f"[thinking] '{self.model}' spent all {max_tokens} "
                      f"tokens reasoning; retrying with no cap, and leaving it "
                      f"uncapped for the rest of this run.")
                return self.generate(
                    prompt, system=system, max_tokens=THINKING_BUDGET,
                    temperature=temperature, stream_callback=stream_callback,
                    images=images, cancellation_token=cancellation_token)
            raise RuntimeError(
                f"'{self.model}' reasoned for {len(''.join(thought))} characters "
                f"without an answer (stopped: {finish or 'unknown'}). A shorter "
                f"brief, or a larger context in LM Studio, leaves room for the "
                f"reply.")
        return sanitize_response(raw_text)
