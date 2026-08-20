"""A thinking model must be given room to finish, not a smaller job.

`think: False` is a request the server forwards and some models ignore. When one
does, its reasoning streams in a separate `thinking` field and `response` stays
empty until the reasoning finishes — so a token budget smaller than the reasoning
returns an empty string and no error at all.

That failure is silent in exactly the place it can least afford to be. The clip
reader treats an empty paragraph as a clip that had none, keeps its measurements
and moves on; a whole run of them produces a report with nothing in it to
summarise, after paying for every call. Measured on `qwen3-vl:8b` against Ollama
0.17.1, where the flag, `/no_think` in the prompt and `/no_think` in the system
message are all ignored: see `docs/INTEL-GPU.md`.

The answer is to take the cap off rather than to trim the reasoning. Such a
model is *chosen* for this page — a narrator is asked to connect what the run
measured to what the frames show, and one that reasons before writing is doing
the thing that was wanted, at a wall-clock cost the user accepts knowingly.

So there are two properties here. Reasoning never becomes the answer, and a
model that reasons is uncapped after the first call that shows it does. What
remains an error is only the case no budget can fix: uncapped, reasoning still
filling the context, and no reply.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from llm.llm_module import THINKING_BUDGET, _OllamaBackend


class _FakeResponse:
    """The slice of `requests.Response` that `generate()` actually uses.

    Carries `raise_for_status` as well as `status_code`: this edition checks the
    first and the other checks the second, and a fake that offers only one of
    them fails on whichever it is not.
    """

    def __init__(self, chunks):
        self.status_code = 200
        self._chunks = chunks

    def raise_for_status(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self):
        for chunk in self._chunks:
            yield json.dumps(chunk).encode("utf-8")


def _stream(monkeypatch, chunks):
    """Point the backend at a canned stream instead of a server.

    A stub module rather than a patched attribute on the real one: `requests`
    is not installed on CI, and importing it here to patch it fails the test
    on the one machine that runs it on every push. `generate()` imports it
    inside the call, so a stub in `sys.modules` is what that import finds —
    and the test then needs nothing installed to exercise the parsing, which
    is all it is about.
    """
    stub = types.ModuleType("requests")
    stub.post = lambda *a, **kw: _FakeResponse(chunks)
    monkeypatch.setitem(sys.modules, "requests", stub)
    return _OllamaBackend(model="a-thinking-model")


def _done(**extra):
    return {"response": "", "done": True, **extra}


def _budget_aware(monkeypatch, *, small, large, enough=THINKING_BUDGET):
    """Stream `large` once the budget is `enough`, and `small` before that.

    `enough` defaults to the uncapped budget, which is the only one that always
    outlasts the reasoning. Models the thing being fixed: the reply exists, and
    whether the caller ever sees it depends only on the budget.
    """
    seen = []

    def _sufficient(budget):
        if budget == THINKING_BUDGET:      # uncapped: nothing can cut it off
            return True
        if enough == THINKING_BUDGET:      # only uncapped will do
            return False
        return budget >= enough

    def post(*args, **kwargs):
        budget = (kwargs.get("data") and json.loads(kwargs["data"])
                  or kwargs.get("json") or {})
        budget = (budget.get("options") or {}).get("num_predict", 0)
        seen.append(budget)
        return _FakeResponse(large if _sufficient(budget) else small)

    stub = types.ModuleType("requests")
    stub.post = post
    monkeypatch.setitem(sys.modules, "requests", stub)
    return _OllamaBackend(model="a-thinking-model"), seen


class TestReasoningOnly:
    def test_a_budget_spent_entirely_on_reasoning_is_an_error(self, monkeypatch):
        llm = _stream(monkeypatch, [
            {"thinking": "Let me look at the frames. ", "response": ""},
            {"thinking": "They appear to show an interface.", "response": ""},
            _done(),
        ])
        with pytest.raises(RuntimeError) as caught:
            llm.generate("describe this", max_tokens=200)
        assert "without an answer" in str(caught.value)

    def test_the_error_names_the_model_and_how_far_it_got(self, monkeypatch):
        """No budget is quoted any more: there is no larger one to move to, and
        naming one would send someone to raise a number that is already off."""
        llm = _stream(monkeypatch, [
            {"thinking": "thinking hard", "response": ""},
            _done(),
        ])
        with pytest.raises(RuntimeError) as caught:
            llm.generate("describe this", max_tokens=THINKING_BUDGET)
        message = str(caught.value)
        assert "a-thinking-model" in message
        assert "13" in message          # characters of reasoning reached


class TestEverythingElseIsUnchanged:
    def test_an_answer_after_reasoning_is_returned_without_the_reasoning(self, monkeypatch):
        """The reasoning is collected to classify the failure, never to answer."""
        llm = _stream(monkeypatch, [
            {"thinking": "First I consider the frames.", "response": ""},
            {"thinking": "", "response": "A person walks "},
            {"thinking": "", "response": "across a room."},
            _done(),
        ])
        assert llm.generate("describe this", max_tokens=200) == \
            "A person walks across a room."

    def test_an_empty_answer_with_no_reasoning_is_still_just_empty(self, monkeypatch):
        """A model that had nothing to say is a different thing, and not an error."""
        llm = _stream(monkeypatch, [
            {"response": "", "done": True},
        ])
        assert llm.generate("describe this", max_tokens=200) == ""

    def test_whitespace_is_not_an_answer(self, monkeypatch):
        """Reasoning plus a stray newline is still a run that produced nothing."""
        llm = _stream(monkeypatch, [
            {"thinking": "considering", "response": ""},
            {"thinking": "", "response": "\n  "},
            _done(),
        ])
        with pytest.raises(RuntimeError):
            llm.generate("describe this", max_tokens=200)


class TestRetryingPastTheReasoning:
    """A reasoning model is given room to finish rather than a smaller job.

    The reasoning is the reason to choose such a model for this page: a narrator
    is asked to connect what the run measured to what the frames show, and one
    that works that through before writing is doing what was wanted. So the
    budget is removed rather than tuned. See `docs/INTEL-GPU.md` for what that
    costs in wall-clock.
    """

    REASONING_ONLY = [{"thinking": "still considering", "response": ""}, _done()]
    ANSWERS = [{"thinking": "considered", "response": ""},
               {"thinking": "", "response": "A person walks across a room."},
               _done()]

    def test_a_reasoning_only_answer_is_retried_with_no_cap(self, monkeypatch):
        llm, seen = _budget_aware(monkeypatch, small=self.REASONING_ONLY,
                                  large=self.ANSWERS)
        answer = llm.generate("describe this", max_tokens=200)
        assert answer == "A person walks across a room."
        assert seen == [200, THINKING_BUDGET]

    def test_later_calls_are_uncapped_from_the_start(self, monkeypatch):
        """Discovered once per run, not re-discovered once per clip."""
        llm, seen = _budget_aware(monkeypatch, small=self.REASONING_ONLY,
                                  large=self.ANSWERS)
        for _ in range(3):
            llm.generate("describe this", max_tokens=200)
        assert seen == [200, THINKING_BUDGET, THINKING_BUDGET, THINKING_BUDGET]

    def test_a_caller_budget_never_shortens_a_thinking_model(self, monkeypatch):
        """Not `max(caller, floor)`: any cap is one the reasoning can hit."""
        llm, seen = _budget_aware(monkeypatch, small=self.REASONING_ONLY,
                                  large=self.ANSWERS)
        llm.generate("describe this", max_tokens=200)
        llm.generate("describe this", max_tokens=8000)
        assert seen[-1] == THINKING_BUDGET

    def test_uncapped_and_still_no_answer_is_an_error(self, monkeypatch):
        """Retried once, not forever. Uncapped means context ran out, not
        budget, so the caller has to be told something different."""
        llm, seen = _budget_aware(monkeypatch, small=self.REASONING_ONLY,
                                  large=self.REASONING_ONLY)
        with pytest.raises(RuntimeError) as caught:
            llm.generate("describe this", max_tokens=200)
        assert "ran out of context" in str(caught.value)
        assert seen == [200, THINKING_BUDGET]

    def test_the_error_points_at_the_brief_rather_than_a_bigger_budget(self, monkeypatch):
        """There is no bigger budget left to suggest."""
        llm, _ = _budget_aware(monkeypatch, small=self.REASONING_ONLY,
                               large=self.REASONING_ONLY)
        with pytest.raises(RuntimeError) as caught:
            llm.generate("describe this", max_tokens=200)
        assert "shorter brief" in str(caught.value)

    def test_an_already_uncapped_call_is_not_retried(self, monkeypatch):
        """Nothing to raise it to, so it fails instead of repeating."""
        llm, seen = _budget_aware(monkeypatch, small=self.REASONING_ONLY,
                                  large=self.REASONING_ONLY)
        with pytest.raises(RuntimeError):
            llm.generate("describe this", max_tokens=THINKING_BUDGET)
        assert seen == [THINKING_BUDGET]

    def test_the_uncapped_budget_is_what_ollama_reads_as_no_limit(self):
        """A positive floor would be a cap wearing a different name."""
        assert THINKING_BUDGET < 0


class TestAskingTheServerInsteadOfGuessing:
    """Ollama lists `thinking` among a model's capabilities, so the first call
    can already be uncapped. The wasted call it replaces is not cheap: a
    reasoning-only reply spends its whole budget, and on a CPU-bound server that
    is over a minute of a run buying only a log line."""

    def _server(self, monkeypatch, capabilities, chunks=None):
        seen = []

        class _Resp:
            def __init__(self, payload, ok=True):
                self._payload = payload
                self.ok = ok
                self.status_code = 200 if ok else 500

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        def get(url, **kw):
            return _Resp({"models": [{"name": "a-thinking-model"}]})

        def post(url, **kw):
            if url.endswith("/api/show"):
                return _Resp({"capabilities": capabilities})
            body = kw.get("json") or json.loads(kw.get("data") or "{}")
            seen.append((body.get("options") or {}).get("num_predict"))
            return _FakeResponse(chunks or [_done()])

        stub = types.ModuleType("requests")
        stub.get, stub.post = get, post
        stub.ConnectionError = type("ConnectionError", (Exception,), {})
        stub.Timeout = type("Timeout", (Exception,), {})
        monkeypatch.setitem(sys.modules, "requests", stub)
        backend = _OllamaBackend(model="a-thinking-model")
        backend.load()
        return backend, seen

    def test_a_thinking_model_is_uncapped_on_the_very_first_call(self, monkeypatch):
        answers = [{"thinking": "considered", "response": ""},
                   {"thinking": "", "response": "A person walks."}, _done()]
        llm, seen = self._server(monkeypatch, ["completion", "vision", "thinking"],
                                 answers)
        assert llm.generate("describe this", max_tokens=200) == "A person walks."
        assert seen == [THINKING_BUDGET]     # no 200-token call was ever made

    def test_a_model_without_the_capability_keeps_the_callers_budget(self, monkeypatch):
        llm, seen = self._server(monkeypatch, ["completion", "vision"],
                                 [{"response": "A person walks."}, _done()])
        assert llm.generate("describe this", max_tokens=200) == "A person walks."
        assert seen == [200]

    def test_a_server_that_cannot_answer_falls_back_to_discovering_it(self, monkeypatch):
        """Older servers, proxies and odd models must not break loading - the
        retry path still covers them."""
        llm, seen = self._server(monkeypatch, None,
                                 [{"response": "A person walks."}, _done()])
        assert llm._thinks is False
        assert llm.generate("describe this", max_tokens=200) == "A person walks."
