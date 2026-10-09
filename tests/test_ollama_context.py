"""A request bigger than the context grows the context; any refusal says why.

The context starts at 2048 so an 11B vision model fits an 8 GB card. Newer
Ollama servers no longer quietly cut the front off a prompt that does not fit -
they refuse it with a 400, and the chat panel turned that into "400 Client
Error: Bad Request", which names neither the cause nor the fix. The server
reports how many tokens the request took, so the fix is exact: retry once at a
size that holds it, and keep that size so the model is not reloaded per call.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from llm.llm_module import DEFAULT_NUM_CTX, _ollama_error, _OllamaBackend

# Ollama 0.40's real body: the runner's JSON error nested as a string.
TOO_BIG = json.dumps({"error": json.dumps({"error": {
    "code": 400,
    "message": "request (4105 tokens) exceeds the available context size "
               "(2048 tokens), try increasing it",
    "type": "exceed_context_size_error"}})})


class _Response:
    def __init__(self, status=200, text="", chunks=()):
        self.status_code = status
        self.text = text
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        for chunk in self._chunks:
            yield json.dumps(chunk).encode("utf-8")


def _server(monkeypatch, *responses):
    """Answer each POST with the next response; record the num_ctx asked for."""
    asked = []
    queue = list(responses)

    def post(url, data=None, **kw):
        asked.append(json.loads(data)["options"]["num_ctx"])
        return queue.pop(0)

    stub = types.ModuleType("requests")
    stub.post = post
    monkeypatch.setitem(sys.modules, "requests", stub)
    return asked


def _ok(text="fine"):
    return _Response(chunks=[{"response": text}, {"response": "", "done": True}])


def test_the_nested_message_is_what_the_user_sees():
    assert _ollama_error(_Response(400, TOO_BIG)).startswith("request (4105 tokens)")
    assert _ollama_error(_Response(400, '{"error":"model not loaded"}')) == "model not loaded"
    assert _ollama_error(_Response(502, "")) == "HTTP 502"


def test_a_prompt_that_does_not_fit_is_retried_once_at_a_size_that_does(monkeypatch):
    asked = _server(monkeypatch, _Response(400, TOO_BIG), _ok())
    backend = _OllamaBackend(model="m", base_url="http://box:11434")
    assert backend.generate("long", max_tokens=512) == "fine"
    # 4105 + 512 rounds up to the next power of two.
    assert asked == [DEFAULT_NUM_CTX, 8192]


def test_the_grown_context_is_kept_so_the_model_is_not_reloaded(monkeypatch):
    asked = _server(monkeypatch, _Response(400, TOO_BIG), _ok(), _ok())
    backend = _OllamaBackend(model="m", base_url="http://box:11434")
    backend.generate("long", max_tokens=512)
    backend.generate("short", max_tokens=512)
    assert asked[-1] == 8192


def test_the_models_own_ceiling_is_respected(monkeypatch):
    _server(monkeypatch, _Response(400, TOO_BIG))
    backend = _OllamaBackend(model="m", base_url="http://box:11434")
    backend._max_ctx = 4096                     # smaller than 4105 + reply
    with pytest.raises(RuntimeError, match="exceeds the available context"):
        backend.generate("long", max_tokens=512)


def test_any_other_refusal_carries_the_servers_reason(monkeypatch):
    _server(monkeypatch, _Response(400, '{"error":"invalid image input"}'))
    backend = _OllamaBackend(model="m", base_url="http://box:11434")
    with pytest.raises(RuntimeError, match="invalid image input"):
        backend.generate("hi")
