"""Ollama says whether a model can see, and the backend repeats it.

The backend used to answer "yes" for every Ollama model, on the grounds that
only the server knew. The server does say - `/api/show` lists `vision` among a
model's capabilities - and not asking had a cost: a text-only model was handed
a chapter's frames and refused the whole request ("Multimodal data provided,
but model does not support multimodal requests"). Anything that asks
`accepts_images()` first is spared that call.
"""

from __future__ import annotations

import sys
import types

import pytest

from llm.llm_module import _OllamaBackend


class _Reply:
    def __init__(self, payload, ok=True):
        self._payload, self.ok = payload, ok

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _loaded(monkeypatch, shown, ok=True):
    """A backend loaded against a server whose `/api/show` answers `shown`.

    A stub module, not a patch on the real one: `requests` is not installed on
    CI, and `load()` imports it at call time.
    """
    stub = types.ModuleType("requests")
    stub.ConnectionError = ConnectionError
    stub.get = lambda *_a, **_kw: _Reply({"models": [{"name": "some-model:1b"}]})
    stub.post = lambda *_a, **_kw: _Reply(shown, ok=ok)
    monkeypatch.setitem(sys.modules, "requests", stub)
    backend = _OllamaBackend(model="some-model:1b",
                             base_url="http://localhost:11434")
    backend.load()
    return backend


def test_a_model_listed_with_vision_sees(monkeypatch):
    backend = _loaded(monkeypatch, {"capabilities": ["completion", "vision"]})
    assert backend.accepts_images() is True


def test_a_model_listed_without_vision_does_not(monkeypatch):
    backend = _loaded(monkeypatch, {"capabilities": ["completion"]})
    assert backend.accepts_images() is False


@pytest.mark.parametrize("shown", [{}, {"capabilities": []},
                                   {"capabilities": None}])
def test_a_server_that_does_not_say_is_given_the_benefit_of_the_doubt(
        monkeypatch, shown):
    # An older Ollama has no `capabilities` field. That is "not told", and
    # treating it as "cannot see" would switch frames off for every model on it.
    assert _loaded(monkeypatch, shown).accepts_images() is True


def test_a_failed_question_changes_nothing(monkeypatch):
    assert _loaded(monkeypatch, {}, ok=False).accepts_images() is True


def test_before_load_the_answer_is_still_yes():
    assert _OllamaBackend(model="some-model:1b",
                          base_url="http://localhost:11434").accepts_images() is True
