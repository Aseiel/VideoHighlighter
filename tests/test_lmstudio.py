"""LM Studio, and any OpenAI-compatible server, as a backend beside Ollama.

What is pinned here is what differs from Ollama and would fail quietly if it
regressed: the model list skipping embedding models, a publisher-less name
finding its model, reasoning never becoming the answer (in either of the two
places a server puts it), frames going out as data URLs, and a prompt that does
not fit the loaded context saying where to fix it instead of retrying.

`requests` is stubbed in `sys.modules`, as in test_ollama_thinking.py: it is not
installed on CI, and every function under test imports it inside the call.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from llm import lmstudio
from llm.llm_module import THINKING_BUDGET


# --------------------------------------------------------------------------- #
# A fake server
# --------------------------------------------------------------------------- #
class _RequestException(Exception):
    pass


class _ConnectionError(_RequestException):
    pass


class _Resp:
    def __init__(self, status=200, body=None, lines=()):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self._lines = lines
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body

    def raise_for_status(self):
        if not self.ok:
            raise _RequestException(f"HTTP {self.status_code}")

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _requests(monkeypatch, get=None, post=None):
    stub = types.ModuleType("requests")
    stub.RequestException = _RequestException
    stub.ConnectionError = _ConnectionError
    stub.get = get or (lambda *a, **kw: (_ for _ in ()).throw(_ConnectionError()))
    stub.post = post or (lambda *a, **kw: (_ for _ in ()).throw(_ConnectionError()))
    monkeypatch.setitem(sys.modules, "requests", stub)
    return stub


def _sse(*deltas, finish="stop", usage=None):
    """The lines a streaming chat completion sends, one delta per chunk."""
    lines = [b": keep-alive", b""]
    for delta in deltas:
        chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        lines.append(b"data: " + json.dumps(chunk).encode())
    lines.append(b"data: " + json.dumps(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}).encode())
    if usage:
        lines.append(b"data: " + json.dumps({"choices": [], "usage": usage}).encode())
    lines.append(b"data: [DONE]")
    return lines


V0_MODELS = {"data": [
    {"id": "google/gemma-4-e4b", "type": "vlm", "state": "not-loaded",
     "max_context_length": 131072},
    {"id": "qwen/qwen3-8b", "type": "llm", "state": "loaded",
     "loaded_context_length": 4096},
    {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings"},
]}


def _lmstudio_get(url, **kw):
    if url.endswith("/api/v0/models"):
        return _Resp(body=V0_MODELS)
    raise AssertionError(f"unexpected GET {url}")


# --------------------------------------------------------------------------- #
# Where the server is
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("typed, expected", [
    ("", ""),
    ("localhost", "http://localhost:1234"),
    ("192.168.1.50", "http://192.168.1.50:1234"),
    ("box.lan:5000", "http://box.lan:5000"),
    ("http://box.lan:1234/v1/chat/completions", "http://box.lan:1234"),
    ("http://0.0.0.0:1234/", "http://localhost:1234"),
    ("https://llm.example.org", "https://llm.example.org"),
])
def test_normalise_accepts_what_people_type(typed, expected):
    assert lmstudio.normalise(typed) == expected


def test_resolve_prefers_the_stored_setting_over_the_environment(monkeypatch):
    monkeypatch.setenv("VH_LMSTUDIO_HOST", "envbox")

    class Store(dict):
        def value(self, key, default=None):
            return self.get(key, default)

    assert lmstudio.resolve(settings=Store()) == "http://envbox:1234"
    stored = Store({lmstudio.HOST_KEY: "storedbox"})
    assert lmstudio.resolve(settings=stored) == "http://storedbox:1234"
    assert lmstudio.resolve("explicit", settings=stored) == "http://explicit:1234"


# --------------------------------------------------------------------------- #
# What it has
# --------------------------------------------------------------------------- #
def test_model_list_leaves_out_embedding_models(monkeypatch):
    _requests(monkeypatch, get=_lmstudio_get)
    assert lmstudio.list_models("http://x:1234") == [
        "google/gemma-4-e4b", "qwen/qwen3-8b"]


def test_a_plain_openai_server_is_asked_the_standard_way(monkeypatch):
    def get(url, **kw):
        if url.endswith("/api/v0/models"):
            return _Resp(status=404, body={"error": "not found"})
        return _Resp(body={"data": [{"id": "llama-3.2-3b"},
                                    {"id": "nomic-embed-text"}]})
    _requests(monkeypatch, get=get)
    assert lmstudio.list_models("http://x:8080") == ["llama-3.2-3b"]


def test_nothing_running_is_an_empty_list_not_an_error(monkeypatch):
    _requests(monkeypatch)
    assert lmstudio.list_models() == []


@pytest.mark.parametrize("wanted, found", [
    ("google/gemma-4-e4b", "google/gemma-4-e4b"),
    ("GOOGLE/Gemma-4-E4B", "google/gemma-4-e4b"),
    ("gemma-4-e4b", "google/gemma-4-e4b"),
    ("gemma", None),            # two candidates: the user has to say which
    ("mistral", None),
    ("", None),
])
def test_match_model(wanted, found):
    ids = ["google/gemma-4-e4b", "google/gemma-3-12b", "qwen/qwen3-8b"]
    assert lmstudio.match_model(wanted, ids) == found


# --------------------------------------------------------------------------- #
# load()
# --------------------------------------------------------------------------- #
def test_load_resolves_the_full_id_and_learns_it_sees_images(monkeypatch):
    _requests(monkeypatch, get=_lmstudio_get)
    backend = lmstudio.LMStudioBackend("gemma-4-e4b", base_url="http://x:1234")
    backend.load()
    assert backend.is_loaded()
    assert backend.model == "google/gemma-4-e4b"
    assert backend.accepts_images()


def test_a_text_model_says_it_cannot_see(monkeypatch):
    _requests(monkeypatch, get=_lmstudio_get)
    backend = lmstudio.LMStudioBackend("qwen/qwen3-8b", base_url="http://x:1234")
    backend.load()
    assert not backend.accepts_images()


def test_load_names_the_server_when_nothing_answers(monkeypatch):
    _requests(monkeypatch)
    backend = lmstudio.LMStudioBackend("any", base_url="http://localhost:1234")
    with pytest.raises(RuntimeError, match="lms server start"):
        backend.load()


def test_load_lists_what_is_there_when_the_model_is_not(monkeypatch):
    _requests(monkeypatch, get=_lmstudio_get)
    backend = lmstudio.LMStudioBackend("mistral", base_url="http://x:1234")
    with pytest.raises(RuntimeError, match="qwen/qwen3-8b"):
        backend.load()


# --------------------------------------------------------------------------- #
# generate()
# --------------------------------------------------------------------------- #
def _backend_with(monkeypatch, *replies):
    """A backend whose server answers each call with the next canned stream."""
    sent = []
    queue = list(replies)

    def post(url, json=None, **kw):
        sent.append(json)
        return queue.pop(0)
    _requests(monkeypatch, post=post)
    backend = lmstudio.LMStudioBackend("google/gemma-4-e4b",
                                       base_url="http://x:1234")
    return backend, sent


def test_streamed_answer_is_assembled_and_streamed_on(monkeypatch):
    backend, sent = _backend_with(monkeypatch, _Resp(lines=_sse(
        {"role": "assistant", "content": "Two people"},
        {"content": " talk."},
        usage={"prompt_tokens": 10, "completion_tokens": 3})))
    seen = []
    assert backend.generate("what?", system="be brief", max_tokens=50,
                            stream_callback=seen.append) == "Two people talk."
    assert seen == ["Two people", " talk."]
    body = sent[0]
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["max_tokens"] == 50 and body["stream"] is True


def test_frames_go_out_as_data_urls(monkeypatch):
    backend, sent = _backend_with(monkeypatch, _Resp(lines=_sse({"content": "ok"})))
    backend.generate("look", images=["/9j/AAAA", "iVBORw0KGgo"])
    parts = sent[0]["messages"][-1]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/")
    assert parts[2]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")


def test_no_cap_is_sent_as_no_max_tokens(monkeypatch):
    backend, sent = _backend_with(monkeypatch, _Resp(lines=_sse({"content": "ok"})))
    backend.generate("q", max_tokens=THINKING_BUDGET)
    assert "max_tokens" not in sent[0]


def test_reasoning_only_reply_is_retried_uncapped_and_stays_uncapped(monkeypatch):
    backend, sent = _backend_with(
        monkeypatch,
        _Resp(lines=_sse({"reasoning_content": "Let me think..."}, finish="length")),
        _Resp(lines=_sse({"reasoning_content": "Done thinking."},
                         {"content": "391"})),
        _Resp(lines=_sse({"content": "next"})))
    assert backend.generate("17*23?", max_tokens=60) == "391"
    assert sent[0]["max_tokens"] == 60
    assert "max_tokens" not in sent[1]
    backend.generate("again", max_tokens=60)
    assert "max_tokens" not in sent[2]


def test_inline_think_tags_never_reach_the_answer(monkeypatch):
    backend, _ = _backend_with(monkeypatch, _Resp(lines=_sse(
        {"content": "<think>"}, {"content": "hmm, red"}, {"content": "</think>"},
        {"content": "It is red."})))
    seen = []
    assert backend.generate("colour?", stream_callback=seen.append) == "It is red."
    assert "".join(seen) == "It is red."


def test_reasoning_with_no_answer_even_uncapped_is_an_error(monkeypatch):
    backend, _ = _backend_with(monkeypatch, _Resp(
        lines=_sse({"reasoning": "on and on"}, finish="length")))
    with pytest.raises(RuntimeError, match="without an answer"):
        backend.generate("q", max_tokens=THINKING_BUDGET)


def test_a_prompt_too_long_for_the_loaded_context_says_where_to_fix_it(monkeypatch):
    backend, _ = _backend_with(monkeypatch, _Resp(status=400, body={"error": (
        "Trying to keep the first 5000 tokens when context the overflows. "
        "However, the model is loaded with context length of 4096")}))
    backend._info = {"loaded_context_length": 4096}
    with pytest.raises(RuntimeError, match="Context Length in LM Studio"):
        backend.generate("a very long brief")


def test_other_refusals_carry_the_servers_reason(monkeypatch):
    backend, _ = _backend_with(monkeypatch, _Resp(
        status=400, body={"error": {"message": "Model unloaded."}}))
    with pytest.raises(RuntimeError, match="Model unloaded"):
        backend.generate("q")


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_llm_module_builds_it(monkeypatch):
    from llm.llm_module import LLMModule
    module = LLMModule(backend="lmstudio", model="qwen/qwen3-8b",
                       base_url="http://x:1234")
    assert isinstance(module._backend, lmstudio.LMStudioBackend)
    assert module._backend.base_url == "http://x:1234"


def test_stored_model_lists_keep_lmstudio_entries():
    from modules.narration.llm_models import parse
    assert parse([{"backend": "lmstudio", "model": "google/gemma-4-e4b"}]) == [
        {"backend": "lmstudio", "model": "google/gemma-4-e4b"}]


@pytest.mark.parametrize("found, current, expected", [
    ({"ollama": ["llama3"], "lmstudio": ["gemma"]}, "lmstudio", "lmstudio"),
    ({"ollama": None, "lmstudio": ["gemma"]}, "ollama", "lmstudio"),
    ({"ollama": ["llama3"], "lmstudio": None}, "lmstudio", "ollama"),
    # Running with models beats running empty.
    ({"ollama": [], "lmstudio": ["gemma"]}, "ollama", "lmstudio"),
    ({"ollama": [], "lmstudio": None}, "lmstudio", "ollama"),
    ({"ollama": None, "lmstudio": None}, "ollama", None),
])
def test_preferred_server(found, current, expected):
    from modules.narration.llm_discovery import preferred_server
    assert preferred_server(found, current) == expected


def test_running_servers_asks_both_and_tells_off_from_empty(monkeypatch):
    from modules.narration import llm_discovery

    def get(url, **kw):
        if ":11434/" in url:
            raise _ConnectionError()
        if url.endswith("/api/v0/models"):
            return _Resp(body={"data": []})
        raise AssertionError(url)
    _requests(monkeypatch, get=get)
    monkeypatch.setattr(lmstudio, "stored", lambda settings=None: "")
    monkeypatch.setattr(llm_discovery, "resolve_ollama_host",
                        lambda *a, **kw: "http://localhost:11434")
    llm_discovery.forget_ollama_models()
    assert llm_discovery.running_servers() == {"ollama": None, "lmstudio": []}


def test_narration_preflight_for_lmstudio(monkeypatch):
    from modules.narration import story_run
    _requests(monkeypatch, get=_lmstudio_get)
    monkeypatch.setattr(lmstudio, "stored", lambda settings=None: "")

    real_wanted = story_run.wanted

    def warnings(model, clips=True):
        config = {"write_highlight_report": True,
                  "narrate_chapters": True, "narrate_clips": clips,
                  "llm_model": {"backend": "lmstudio", "model": model}}
        monkeypatch.setattr(story_run, "_entry", lambda cfg: cfg["llm_model"])
        # Chapters and clips on; Pro's third pass (the summary) off.
        passes = len(real_wanted(config))
        monkeypatch.setattr(story_run, "wanted",
                            lambda cfg: (True, clips) + (False,) * (passes - 2))
        return story_run.preflight(config)

    assert warnings("gemma-4-e4b") == []
    assert "no vision half" in warnings("qwen/qwen3-8b")[0]
    assert warnings("qwen/qwen3-8b", clips=False) == []
    assert "has no model 'mistral'" in warnings("mistral")[0]

    _requests(monkeypatch)
    assert "not running" in warnings("gemma-4-e4b")[0]
