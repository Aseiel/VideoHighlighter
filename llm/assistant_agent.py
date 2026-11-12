"""The chat assistant that can use the app: a model with tools.

A model served by LM Studio (any OpenAI-compatible server) through
:class:`OpenAIToolTurn`, or, in an edition that has the hosted backends,
Claude through the Anthropic API; see :func:`converse_for`.

The chat could talk about a run but not do anything: changing a setting,
running again, or putting clips on the timeline was always the user's job after
reading the answer. A hosted Claude model can call tools, so here it is given a
few of the app's own actions and runs them itself. It reads the run's report,
re-runs with different settings, and shows clips on the edit timeline.

What it may do is a short, fixed list (:func:`app_tools`). There is no "run
code" tool. Each tool returns text: paths, times in seconds, scores, findings.
That keeps the hosted-model rule of ``llm/cloud_backends.py``: no frame leaves
the machine, and ``converse`` refuses an image even if one were handed to it.
Everything a tool changes is something the user could undo or redo by hand.
Clips land on the edit timeline as one undoable step, and a run writes a new
reel beside the old one, the way the Run button does.

Pieces:

* :func:`run` is the loop: ask, run the tools Claude asked for, give it the
  results, repeat until it answers or the step budget runs out. No Qt.
* :class:`GuiInvoker` runs a tool on the GUI thread when it touches widgets,
  while the loop itself stays on the chat's worker thread.
* :func:`app_tools` is the tool list, built over the main window.
* :class:`AgentWorker` is the chat's worker for an agent turn: same signals as
  the plain one, so the panel's display and Stop button need nothing new.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

MAX_STEPS = 10
RESULT_CHARS = 20000            # a tool result longer than this is cut, and says so
RUN_POLL_SECONDS = 2.0

SYSTEM_PROMPT = """\
You are the assistant inside VideoHighlighter, a desktop app that finds and cuts \
highlights from the user's own videos. You can use the app through tools: read \
the report of a run, re-run with different settings, and put clips on the edit \
timeline. You never see the video itself: tools give you paths, times in seconds, \
scores, and the report's findings.

How to work:
- Look before you change: read the report (read_report) before suggesting or \
running anything, and base what you say on what it measured.
- Say what you are about to do in one sentence before a run; runs take a while.
- Prefer the report's own suggestions (each has a concrete change) over \
inventing settings.
- When you put clips on the timeline, say that Ctrl+Z in the timeline viewer \
takes them back.
- Tools give times in seconds. Write them for the user as m:ss: 6 seconds is 0:06, 75 seconds is 1:15, 3725 seconds is 1:02:05.
- Keep answers short and concrete.
- If a tool fails, say what failed and what the user can do; do not pretend it \
worked."""


class Cancelled(Exception):
    """The user pressed Stop."""


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[[dict], object]
    on_gui: bool = False                # touches widgets: run on the GUI thread

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description,
                "input_schema": self.schema}


def _as_text(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False,
                                                           default=str)
    if len(text) > RESULT_CHARS:
        text = text[:RESULT_CHARS] + f"\n[cut: {len(text) - RESULT_CHARS} more characters]"
    return text


def run(converse: Callable[..., dict], system: str, prompt: str,
        tools: Sequence[Tool], *, call_on_gui: Callable[[Callable], object],
        on_step: Optional[Callable[[str, dict], None]] = None,
        cancelled: Callable[[], bool] = lambda: False,
        max_steps: int = MAX_STEPS) -> tuple[str, list]:
    """Talk to the model until it answers. Returns ``(answer, tools used)``.

    ``converse(messages, system=, tools=)`` is one model turn
    (``AnthropicBackend.converse``)."""
    by_name = {t.name: t for t in tools}
    specs = [t.spec() for t in tools]
    messages: list = [{"role": "user", "content": prompt}]
    used: list = []
    # Everything it says, not only the last turn: a model often answers the
    # question in the same turn it calls a tool, and the final turn is then
    # only "done", which left the answer the user asked for off the screen.
    said: list = []
    for _step in range(max_steps):
        if cancelled():
            raise Cancelled()
        reply = converse(messages, system=system, tools=specs)
        blocks = reply.get("content") or []
        messages.append({"role": "assistant", "content": blocks})
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        text = "".join(b.get("text", "") for b in blocks
                       if b.get("type") == "text").strip()
        if text:
            said.append(text)
        if reply.get("stop_reason") != "tool_use" or not calls:
            return "\n\n".join(said), used
        results = []
        for call in calls:
            name, args = call.get("name", ""), call.get("input") or {}
            if on_step:
                on_step(name, args)
            tool = by_name.get(name)
            try:
                if tool is None:
                    raise ValueError(f"No such tool: {name}")
                out = (call_on_gui(lambda: tool.fn(args)) if tool.on_gui
                       else tool.fn(args))
                result = {"type": "tool_result", "tool_use_id": call["id"],
                          "content": _as_text(out)}
            except Cancelled:
                raise
            except Exception as e:  # noqa: BLE001 - the model is told, and adapts
                result = {"type": "tool_result", "tool_use_id": call["id"],
                          "content": f"{type(e).__name__}: {e}", "is_error": True}
            used.append(name)
            results.append(result)
        messages.append({"role": "user", "content": results})
    said.append(f"I stopped after {max_steps} steps without finishing. Ask again "
                f"with a narrower request, or tell me which step to take.")
    return "\n\n".join(said), used


# ── running things on the GUI thread ─────────────────────────────────────────

def _make_invoker_class():
    from PySide6.QtCore import QObject, QThread, Signal, Slot

    class GuiInvoker(QObject):
        """Create on the GUI thread; ``call(fn)`` from any thread runs ``fn``
        there and returns its result (or raises its exception)."""

        _request = Signal(object)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._request.connect(self._run)

        @Slot(object)
        def _run(self, job):
            fn, box, done = job
            try:
                box["result"] = fn()
            except BaseException as e:  # noqa: BLE001 - re-raised in call()
                box["error"] = e
            finally:
                done.set()

        def call(self, fn, timeout: Optional[float] = None):
            if QThread.currentThread() is self.thread():
                return fn()
            box: dict = {}
            done = threading.Event()
            self._request.emit((fn, box, done))
            if not done.wait(timeout):
                raise TimeoutError("The app did not get to it in time.")
            if "error" in box:
                raise box["error"]
            return box.get("result")

    return GuiInvoker


_GuiInvoker = None


def GuiInvoker(parent=None):
    global _GuiInvoker
    if _GuiInvoker is None:
        _GuiInvoker = _make_invoker_class()
    return _GuiInvoker(parent)


# ── the app's tools ──────────────────────────────────────────────────────────

# Settings a run may be given, as the pipeline reads them. Anything else is
# refused with this list, so the model can correct itself.
def _run_keys() -> set:
    from modules.report.advice_actions import RERUN_KEYS
    return set(RERUN_KEYS) | {"max_duration", "exact_duration", "interesting_actions",
                              "highlight_objects", "use_transcript",
                              "search_keywords", "export_separate_clips"}


def newest_report(video_path: str) -> Optional[str]:
    """The newest ``*_why.json`` beside ``video_path`` that is about it."""
    folder = os.path.dirname(os.path.abspath(video_path))
    video = os.path.normcase(os.path.abspath(video_path))     # for comparing only
    found = []
    for name in os.listdir(folder):
        if not name.endswith("_why.json"):
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, encoding="utf-8") as fh:
                about = (json.load(fh).get("video") or {}).get("path") or ""
        except Exception:  # noqa: BLE001 - not ours, or not readable
            continue
        if os.path.normcase(os.path.abspath(about)) == video:
            found.append(path)
    return max(found, key=os.path.getmtime) if found else None


def summarize_report(path: str) -> dict:
    """What a model needs from a report, without the bulk."""
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    if any("action" not in f for f in report.get("advice") or [{}]):
        from modules.report.highlight_advice import attach_advice
        attach_advice(report, concern=report.get("advice_concern"))
    settings = report.get("settings") or {}
    clips = []
    for s in (report.get("segments") or [])[:60]:
        breakdown = {k: v for k, v in (s.get("breakdown") or {}).items() if v}
        clips.append({"start": round(float(s["start"]), 1), "end": round(float(s["end"]), 1),
                      "score": round(float(s.get("score") or 0), 2),
                      "signals": breakdown,
                      "actions": [a.get("name") for a in s.get("actions") or []][:3],
                      "objects": (s.get("objects") or [])[:5]})
    return {
        "report": path,
        "video": (report.get("video") or {}).get("path"),
        "duration": (report.get("video") or {}).get("duration"),
        "highlight_seconds": (report.get("totals") or {}).get("duration"),
        "settings": {k: v for k, v in settings.items()
                     if k.endswith("_points") or k in (
                         "coverage", "clip_time", "target_duration", "duration_mode",
                         "min_signals_for_boost", "multi_signal_boost")},
        "detected": settings.get("detector_activity"),
        "clips": clips,
        "suggestions": [{"id": f.get("id"), "severity": f.get("severity"),
                         "title": f.get("title"), "detail": f.get("detail"),
                         "change": (f.get("action") or {}).get("settings") or None}
                        for f in report.get("advice") or []],
    }


def app_tools(window, *, call_on_gui: Callable[[Callable], object],
              cancelled: Callable[[], bool],
              say: Callable[[str], None] = lambda _line: None,
              local: bool = False) -> list:
    """The tools, over the main window. ``say`` gets progress lines from
    the long ones; ``local`` is whether the model runs on this machine (see
    :func:`runs_here`)."""

    def _video(args) -> str:
        path = os.path.abspath(str(args.get("video_path") or ""))
        if not os.path.isfile(path):
            raise ValueError(f"Video not found: {path}")
        return path

    def app_state(args):
        timeline = None
        tw = getattr(window, "timeline_window", None)
        try:
            timeline = tw.video_path if tw else None
        except RuntimeError:
            timeline = None
        worker = getattr(window, "worker", None)
        return {"videos": window.get_file_list(), "timeline_video": timeline,
                "run_in_progress": bool(worker and worker.isRunning())}

    def read_report(args):
        path = newest_report(_video(args))
        if path is None:
            return {"report": None,
                    "note": "No report for this video yet: run it first (run_highlight)."}
        return summarize_report(path)

    def show_clips(args):
        video = _video(args)
        clips = [(float(a), float(b)) for a, b in (args.get("clips") or [])]
        if not clips or any(b - a < 0.5 for a, b in clips):
            raise ValueError("clips must be [[start, end], ...] in seconds, each at "
                             "least half a second long")
        note = str(args.get("note") or "chosen in the chat")
        from modules.media.video_cache import VideoAnalysisCache
        VideoAnalysisCache().save_highlight_segments(
            video, {"source": "assistant", "note": note}, sorted(clips),
            [{"score": 1.0, "signals": {"chosen": 1.0}, "primary_reason": note}
             for _ in clips], score_info={"source": "assistant"})
        show = getattr(window, "_live_show_clips", None)
        if show is None:                        # an edition without the live view
            return {"saved": True, "clips": len(clips), "shown": False,
                    "note": "Saved as the video's newest highlight version. Opening "
                            "the video in the timeline viewer shows them."}
        return show({"video_path": video, "clips": sorted(clips), "note": note})

    def run_highlight(args):
        video = _video(args)
        settings = dict(args.get("settings") or {})
        unknown = sorted(set(settings) - _run_keys())
        if unknown:
            raise ValueError(f"Unknown settings {unknown}. Allowed: {sorted(_run_keys())}")
        report_only = bool(args.get("report_only", False))
        started = time.time()

        def start():
            from modules.media.output_name import highlight_output_path
            worker = getattr(window, "worker", None)
            if worker and worker.isRunning():
                raise RuntimeError("A run is already going in the app.")
            expected = os.path.splitext(highlight_output_path(
                video, window.output_input.text().strip(), multiple=False))[0] + "_why.json"
            previous = window.worker
            window.run_pipeline(report_only=report_only,
                                simple=getattr(window, "_simple_run", False),
                                overrides={**settings, "write_highlight_report": True},
                                video_paths=[video])
            if window.worker is previous or not window.worker.isRunning():
                raise RuntimeError("The run did not start; the app's log says why.")
            return expected

        expected = call_on_gui(start)
        while call_on_gui(lambda: bool(window.worker and window.worker.isRunning())):
            if cancelled():
                raise Cancelled()
            time.sleep(RUN_POLL_SECONDS)
        if not os.path.exists(expected) or os.path.getmtime(expected) < started:
            raise RuntimeError("The run ended without writing a new report; it may have "
                               "failed or been cancelled. The app's log says why.")
        return {"finished": True, "report_only": report_only,
                **summarize_report(expected)}

    video_arg = {"video_path": {"type": "string",
                                "description": "Full path of the video file."}}
    tools = [
        Tool("app_state",
             "The videos in the app's file list, the video open in the timeline "
             "viewer, and whether a run is in progress.",
             {"type": "object", "properties": {}}, app_state, on_gui=True),
        Tool("read_report",
             "The newest highlight report for a video: its clips (seconds, score, "
             "which signals scored), the settings used, how much each detector "
             "found, and the report's suggestions. A suggestion's 'change' is the "
             "exact settings it recommends.",
             {"type": "object", "properties": video_arg, "required": ["video_path"]},
             read_report),
        Tool("show_clips",
             "Put clips on the app's edit timeline for a video (opens the timeline "
             "viewer on it). The user can adjust and export them there; Ctrl+Z "
             "undoes the change.",
             {"type": "object", "properties": {
                 **video_arg,
                 "clips": {"type": "array", "description": "[[start, end], ...] seconds",
                           "items": {"type": "array", "items": {"type": "number"},
                                     "minItems": 2, "maxItems": 2}},
                 "note": {"type": "string", "description": "Why these clips, briefly."}},
              "required": ["video_path", "clips"]},
             show_clips, on_gui=True),
        Tool("run_highlight",
             "Run the highlighter on a video with the app's current settings plus "
             "`settings` for this run only (for example a suggestion's 'change'). "
             "Waits until the run finishes (minutes) and returns the new report. "
             "report_only scores without rendering a video: much faster, use it "
             "to try a change.",
             {"type": "object", "properties": {
                 **video_arg,
                 "settings": {"type": "object",
                              "description": "Pipeline settings for this run only."},
                 "report_only": {"type": "boolean"}},
              "required": ["video_path"]},
             run_highlight),
    ]
    # The dataset loop (llm/dataset_agent.py), where this build has it.
    try:
        from llm.dataset_agent import dataset_tools
        tools += dataset_tools(cancelled=cancelled, say=say, local=local)
    except ImportError as e:
        print(f"ℹ assistant: no dataset tools in this build ({e})")
    return tools


def compose_prompt(message: str, history, analysis_data, video_path: str) -> str:
    """The first user turn: earlier conversation, what the app knows about the
    open video, then the question."""
    from llm import llm_module
    from llm.llm_module import VideoContextBuilder

    def plain_history(turns):                   # an edition with an older chat
        lines = [f"{'Q' if t.get('role') == 'user' else 'A'}: {t.get('content', '')}"
                 for t in (turns or [])[-12:]]
        return "[EARLIER IN THIS CHAT]\n" + "\n".join(lines) if lines else ""

    format_history = getattr(llm_module, "format_history", plain_history)

    parts = []
    block = format_history(history)
    if block:
        parts.append(block)
    if video_path:
        parts.append(f"--- OPEN VIDEO ---\n{video_path}\n--- END ---")
    if analysis_data:
        parts.append("--- VIDEO ANALYSIS DATA ---\n"
                     f"{VideoContextBuilder.build(analysis_data, video_path)}\n--- END ---")
    parts.append(message)
    return "\n".join(parts)


# ── the chat's worker ────────────────────────────────────────────────────────

def _make_worker_class():
    from PySide6.QtCore import QObject, Signal, Slot

    class AgentWorker(QObject):
        """One agent turn off the GUI thread. Same signals as the chat's plain
        worker, so the panel shows it the same way."""

        token_received = Signal(str)
        finished = Signal(str)
        error = Signal(str)

        def __init__(self, converse, prompt: str, window, invoker):
            super().__init__()
            self.converse = converse                 # see converse_for()
            self.prompt = prompt
            self.window = window
            self.invoker = invoker
            self._stop = threading.Event()

        def cancel(self):
            self._stop.set()

        @Slot()
        def run(self):
            labels = {"app_state": "looking at the app",
                      "read_report": "reading the report",
                      "show_clips": "putting clips on the timeline",
                      "run_highlight": "running the highlighter"}
            system = SYSTEM_PROMPT
            try:
                tools = app_tools(
                    self.window, call_on_gui=self.invoker.call,
                    cancelled=self._stop.is_set,
                    say=lambda line: self.token_received.emit(f"   · {line}\n"),
                    local=runs_here(self.converse))
                if any(t.name == "improve_dataset" for t in tools):
                    from llm import dataset_agent
                    labels.update(dataset_agent.LABELS)
                    system += dataset_agent.PROMPT
                answer, used = run(
                    lambda messages, **kw: self.converse(messages, **kw),
                    system, self.prompt, tools,
                    call_on_gui=self.invoker.call,
                    on_step=lambda name, args: self.token_received.emit(
                        f"🔧 {labels.get(name, name)}…\n"),
                    cancelled=self._stop.is_set)
                if used:
                    answer += f"\n\n(used: {', '.join(dict.fromkeys(used))})"
                self.finished.emit(answer or "(no answer)")
            except Cancelled:
                self.finished.emit("[stopped by user]")
            except Exception as e:  # noqa: BLE001 - shown in the chat
                self.error.emit(str(e))

    return AgentWorker


_AgentWorker = None


def AgentWorker(*args, **kwargs):
    global _AgentWorker
    if _AgentWorker is None:
        _AgentWorker = _make_worker_class()
    return _AgentWorker(*args, **kwargs)


# ── models other than Claude ─────────────────────────────────────────────────

def _openai_messages(messages: Sequence[dict], system: str) -> list:
    """The loop's conversation (Messages API shape) in chat-completions shape."""
    out = [{"role": "system", "content": system}] if system else []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        blocks = content or []
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"],
                                   "arguments": json.dumps(b.get("input") or {})}}
                     for b in blocks if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        for b in blocks:                         # a user turn: results, or text
            if b.get("type") == "tool_result":
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                            "content": str(b.get("content", ""))})
            elif b.get("type") == "text":
                out.append({"role": "user", "content": b.get("text", "")})
    return out


def _openai_tools(tools: Optional[Sequence[dict]]) -> list:
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("input_schema") or {"type": "object"}}}
            for t in tools or []]


class OpenAIToolTurn:
    """``converse`` for an OpenAI-compatible server: LM Studio, llama-server.

    Speaks chat-completions function calling to the server and gives the loop
    back the same shape ``AnthropicBackend.converse`` does, so one loop and one
    tool list serve every model that can call tools. Inline ``<think>``
    reasoning (LM Studio sends it that way when its separate field is off) is
    not part of the answer and is dropped."""

    def __init__(self, base_url: str, model: str, timeout=(10, 600)):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def converse(self, messages, *, system: str = "", tools=None, max_tokens: int = 2048,
                 cancellation_token=None) -> dict:
        import requests
        from llm.lmstudio import _ThinkSplitter

        payload = {"model": self.model, "messages": _openai_messages(messages, system),
                   "stream": False, "temperature": 0.2}
        if tools:
            payload["tools"] = _openai_tools(tools)
            payload["tool_choice"] = "auto"
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload,
                             timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"{self.base_url} refused the request "
                               f"({resp.status_code}): {resp.text[:300]}")
        message = ((resp.json().get("choices") or [{}])[0].get("message") or {})
        text, _thought = _ThinkSplitter().feed(message.get("content") or "")
        blocks = [{"type": "text", "text": text.strip()}] if text.strip() else []
        for n, call in enumerate(message.get("tool_calls") or []):
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
            blocks.append({"type": "tool_use", "id": call.get("id") or f"call_{n}",
                           "name": fn.get("name", ""),
                           "input": args if isinstance(args, dict) else {}})
        calls = any(b["type"] == "tool_use" for b in blocks)
        return {"content": blocks, "stop_reason": "tool_use" if calls else "end_turn"}


def converse_for(llm):
    """The connected model's tool-calling turn, or ``None`` if it has none."""
    backend = getattr(llm, "_backend", None)
    if backend is None:
        return None
    if getattr(backend, "provider", "") == "anthropic" and hasattr(backend, "converse"):
        return backend.converse
    try:
        from llm.lmstudio import LMStudioBackend
    except ImportError:
        return None
    if isinstance(backend, LMStudioBackend):
        return OpenAIToolTurn(backend.base_url, backend.model).converse
    return None


def runs_here(converse) -> bool:
    """Whether this turn's model is served on this machine: what it is told
    goes nowhere, so tools may show it the user's own class names."""
    turn = getattr(converse, "__self__", None)
    if not isinstance(turn, OpenAIToolTurn):
        return False
    from llm.lmstudio import is_remote
    return not is_remote(turn.base_url)


def can_act(llm) -> bool:
    """Whether the connected model can use tools here: a hosted Claude, or a
    model served by LM Studio (or another OpenAI-compatible server)."""
    return converse_for(llm) is not None
