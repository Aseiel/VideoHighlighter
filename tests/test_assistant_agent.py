"""The chat assistant that uses the app (`llm/assistant_agent.py`) and the
tool-calling turn it rides on (`AnthropicBackend.converse`). No network: the
model is a script of replies, and HTTP is faked."""
from __future__ import annotations

import json
import os
import threading
import time

import pytest

from llm import assistant_agent as aa


def _text(t):
    return {"type": "text", "text": t}


def _call(i, name, **args):
    return {"type": "tool_use", "id": f"t{i}", "name": name, "input": args}


class Script:
    """A model that replies from a list, recording what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def __call__(self, messages, **kw):
        self.sent.append((json.loads(json.dumps(messages)), kw))
        content = self.replies.pop(0)
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return {"content": content, "stop_reason": stop}


def _tool(name, fn, on_gui=False):
    return aa.Tool(name, f"{name} tool", {"type": "object", "properties": {}}, fn, on_gui)


# ── the loop ─────────────────────────────────────────────────────────────


def test_tools_run_and_their_results_go_back_until_it_answers():
    model = Script([_text("Reading."), _call(1, "read", x=1)], [_text("Done: 3 clips.")])
    seen, steps = [], []
    answer, used = aa.run(model, "sys", "question", [_tool("read", lambda a: seen.append(a) or {"clips": 3})],
                          call_on_gui=lambda fn: fn(), on_step=lambda n, a: steps.append(n))
    # What it said while calling the tool is kept, not only the last turn.
    assert answer == "Reading.\n\nDone: 3 clips." and used == ["read"] and steps == ["read"]
    assert seen == [{"x": 1}]
    result = model.sent[1][0][-1]["content"][0]
    assert result == {"type": "tool_result", "tool_use_id": "t1", "content": '{"clips": 3}'}
    assert model.sent[0][1]["tools"][0]["name"] == "read"


def test_a_failing_or_unknown_tool_is_reported_to_the_model_not_raised():
    def boom(args):
        raise ValueError("video not found")

    model = Script([_call(1, "boom"), _call(2, "nope")], [_text("Sorry.")])
    answer, used = aa.run(model, "", "q", [_tool("boom", boom)], call_on_gui=lambda fn: fn())
    first, second = model.sent[1][0][-1]["content"]
    assert first["is_error"] and "video not found" in first["content"]
    assert second["is_error"] and "No such tool" in second["content"]
    assert answer == "Sorry."


def test_gui_tools_go_through_the_invoker():
    ran_via = []
    model = Script([_call(1, "show")], [_text("ok")])
    aa.run(model, "", "q", [_tool("show", lambda a: "shown", on_gui=True)],
           call_on_gui=lambda fn: ran_via.append("gui") or fn())
    assert ran_via == ["gui"]


def test_stop_and_the_step_budget():
    with pytest.raises(aa.Cancelled):
        aa.run(Script([_text("x")]), "", "q", [], call_on_gui=lambda fn: fn(),
               cancelled=lambda: True)
    looping = Script(*[[_call(i, "read")] for i in range(5)])
    answer, used = aa.run(looping, "", "q", [_tool("read", lambda a: 1)],
                          call_on_gui=lambda fn: fn(), max_steps=3)
    assert "stopped after 3 steps" in answer and len(used) == 3


def test_long_results_are_cut_and_say_so():
    text = aa._as_text("x" * (aa.RESULT_CHARS + 50))
    assert text.endswith("[cut: 50 more characters]")


# ── reports ──────────────────────────────────────────────────────────────


def _write_report(folder, name, video, mtime_offset=0):
    report = {
        "video": {"path": video, "duration": 600},
        "totals": {"duration": 16},
        "settings": {"motion_peak_points": 5, "coverage": 0.0,
                     "detector_activity": {"motion_event": 80}},
        "segments": [{"start": 100, "end": 108, "score": 5,
                      "breakdown": {"motion_peak": 5, "scene": 0}}],
        "advice": [{"id": "unused_motion_event", "title": "Unused", "severity": "high",
                    "detail": "d", "action": {"label": "l",
                                              "settings": {"motion_event_points": 4}}}],
    }
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh)
    t = time.time() + mtime_offset
    os.utime(path, (t, t))
    return path


def test_newest_report_is_the_newest_about_this_video(tmp_path):
    video = str(tmp_path / "a.mp4")
    other = str(tmp_path / "b.mp4")
    _write_report(tmp_path, "old_why.json", video, -100)
    newest = _write_report(tmp_path, "new_why.json", video)
    _write_report(tmp_path, "other_why.json", other, 100)
    assert aa.newest_report(video) == newest


def test_a_report_summary_keeps_clips_settings_and_concrete_suggestions(tmp_path):
    path = _write_report(tmp_path, "r_why.json", str(tmp_path / "a.mp4"))
    s = aa.summarize_report(path)
    assert s["clips"] == [{"start": 100.0, "end": 108.0, "score": 5.0,
                           "signals": {"motion_peak": 5}, "actions": [], "objects": []}]
    assert s["settings"] == {"motion_peak_points": 5, "coverage": 0.0}
    assert s["suggestions"][0]["change"] == {"motion_event_points": 4}


def test_a_run_refuses_settings_it_does_not_know(tmp_path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"\0")
    tools = {t.name: t for t in aa.app_tools(object(), call_on_gui=lambda fn: fn(),
                                             cancelled=lambda: False)}
    with pytest.raises(ValueError, match="Unknown settings"):
        tools["run_highlight"].fn({"video_path": str(video),
                                   "settings": {"rm_rf": True}})


def test_the_prompt_carries_history_video_and_question():
    prompt = aa.compose_prompt("make it shorter", [{"role": "user", "content": "hi"},
                                                  {"role": "assistant", "content": "hello"}],
                               None, "D:/v.mp4")
    assert prompt.index("D:/v.mp4") < prompt.index("make it shorter")
    assert "hello" in prompt


# ── the GUI-thread invoker ───────────────────────────────────────────────


def test_the_invoker_runs_on_the_gui_thread_and_returns(qtbot_app):
    from PySide6.QtCore import QCoreApplication, QThread

    invoker = aa.GuiInvoker()
    gui = QThread.currentThread()
    box = {}

    def work():
        box["result"] = invoker.call(lambda: QThread.currentThread() is gui)
        try:
            invoker.call(lambda: 1 / 0)
        except ZeroDivisionError:
            box["raised"] = True

    t = threading.Thread(target=work)
    t.start()
    deadline = time.monotonic() + 5
    while t.is_alive() and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)
    assert box == {"result": True, "raised": True}


@pytest.fixture(scope="module")
def qtbot_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ── the backend turn ─────────────────────────────────────────────────────


def test_converse_sends_tools_and_returns_blocks(monkeypatch):
    # Only in an edition with the hosted backends; by name, so a build
    # without them has no import to miss.
    AnthropicBackend = pytest.importorskip("llm.cloud_backends").AnthropicBackend
    requests = pytest.importorskip("requests")

    sent = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"content": [_call(1, "read")], "stop_reason": "tool_use"}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, payload=json)
        return Resp()

    monkeypatch.setattr(requests, "post", post)
    backend = AnthropicBackend("claude-test", api_key="k")
    reply = backend.converse([{"role": "user", "content": "q"}], system="s",
                             tools=[{"name": "read"}])
    assert reply == {"content": [_call(1, "read")], "stop_reason": "tool_use"}
    assert sent["url"].endswith("/messages")
    assert sent["payload"]["tools"] == [{"name": "read"}]
    assert sent["payload"]["system"] == "s" and "stream" not in sent["payload"]


def test_converse_never_sends_an_image():
    cloud = pytest.importorskip("llm.cloud_backends")
    AnthropicBackend, CloudError = cloud.AnthropicBackend, cloud.CloudError

    backend = AnthropicBackend("claude-test", api_key="k")
    nested = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                            "content": [{"type": "image", "source": {}}]}]}]
    with pytest.raises(CloudError, match="image"):
        backend.converse(nested)


def test_claude_or_an_lm_studio_model_can_act():
    from llm.lmstudio import LMStudioBackend

    class LLM:
        def __init__(self, backend):
            self._backend = backend

    claude = type("B", (), {"provider": "anthropic", "converse": lambda *a, **k: None})()
    assert aa.can_act(LLM(claude))
    assert not aa.can_act(LLM(type("B", (), {"provider": "openai"})()))
    assert not aa.can_act(object())
    local = LMStudioBackend("google/gemma-4-e4b", base_url="http://localhost:1234")
    turn = aa.converse_for(LLM(local))
    assert turn.__self__.model == "google/gemma-4-e4b"


# ── the OpenAI-compatible turn (LM Studio) ──────────────────────────────


def test_the_conversation_and_tools_convert_to_chat_completions():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [_text("Looking."), _call(1, "read", x=1)]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": '{"clips": 3}'}]},
    ]
    out = aa._openai_messages(messages, "sys")
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "q"}
    assert out[2]["role"] == "assistant" and out[2]["content"] == "Looking."
    assert out[2]["tool_calls"][0]["function"] == {"name": "read", "arguments": '{"x": 1}'}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"clips": 3}'}
    tools = aa._openai_tools([{"name": "read", "description": "d",
                               "input_schema": {"type": "object"}}])
    assert tools == [{"type": "function", "function": {
        "name": "read", "description": "d", "parameters": {"type": "object"}}}]


def test_a_tool_call_comes_back_as_a_tool_use_block(monkeypatch):
    requests = pytest.importorskip("requests")
    sent = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {
                "content": "<think>should read it</think>Reading the report.",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "read_report", "arguments": '{"video_path": "a.mp4"}'}}]}}]}

    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None:
                        sent.update(url=url, payload=json) or Resp())
    reply = aa.OpenAIToolTurn("http://localhost:1234", "m").converse(
        [{"role": "user", "content": "q"}], system="s", tools=[{"name": "read_report"}])
    assert reply["stop_reason"] == "tool_use"
    assert reply["content"] == [
        {"type": "text", "text": "Reading the report."},
        {"type": "tool_use", "id": "c1", "name": "read_report",
         "input": {"video_path": "a.mp4"}}]
    assert sent["url"] == "http://localhost:1234/v1/chat/completions"
    assert sent["payload"]["tools"][0]["function"]["name"] == "read_report"
