"""The live link's ``restart`` command, on the main window's side.

Against a stand-in window: the handler must refuse while work runs, answer
before it closes, start the new process before closing, and hand over the
video and moment only when the viewer is on screen.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("PySide6")
main = pytest.importorskip("main")

from modules.system import app_restart  # noqa: E402


class Job:
    def __init__(self, running):
        self.running = running

    def isRunning(self):
        return self.running


@pytest.fixture
def app(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    window = types.SimpleNamespace(video_path=str(video), current_time=1428.04,
                                   isVisible=lambda: True, _analysis_thread=None)
    events = []
    app = types.SimpleNamespace(timeline_window=window, worker=None, _signal_worker=None,
                                download_worker=None, _scan_worker=None,
                                append_log=lambda m: events.append(("log", m)),
                                close=lambda: events.append(("close",)))
    for name in ("_live_timeline_video", "_live_restart"):
        setattr(app, name, types.MethodType(getattr(main.VideoHighlighterGUI, name), app))
    timers = []
    monkeypatch.setattr(main.QTimer, "singleShot", lambda ms, fn: timers.append(fn))
    monkeypatch.setattr(app_restart, "spawn",
                        lambda video, seconds: events.append(("spawn", video, seconds)))
    return app, window, events, timers, str(video)


def test_answers_first_then_starts_the_new_app_then_closes(app):
    app, window, events, timers, video = app
    out = app._live_restart({})
    assert out == {"restarting": True, "reopen": video, "seconds": 1428.04}
    assert [e[0] for e in events] == ["log"]          # nothing has happened yet
    timers[0]()
    assert [e[0] for e in events] == ["log", "spawn", "close"]
    assert events[1] == ("spawn", video, 1428.04)


def test_refuses_while_work_runs(app):
    app, window, events, timers, _ = app
    app.worker = Job(True)
    with pytest.raises(RuntimeError, match="a highlight run"):
        app._live_restart({})
    assert timers == [] and events == []


def test_a_hidden_viewer_is_not_reopened(app):
    app, window, events, timers, _ = app
    window.isVisible = lambda: False
    assert app._live_restart({})["reopen"] is None
    timers[0]()
    assert events[1] == ("spawn", None, None)


def test_a_failed_start_keeps_the_app_open(app, monkeypatch):
    app, window, events, timers, _ = app

    def boom(video, seconds):
        raise OSError("no such file")
    monkeypatch.setattr(app_restart, "spawn", boom)
    app._live_restart({})
    timers[0]()
    assert ("close",) not in events
    assert "stays open" in events[-1][1]


def test_the_viewer_is_not_opened_twice_at_once(app):
    """A request arriving while the viewer is being built (the build pumps
    events) is turned away, not answered with a second window."""
    app, window, events, timers, video = app
    built = []

    def build(video_path=None):
        built.append(video_path)
        # mid-build, another request arrives (a click, a live-link command)
        assert app.open_timeline_viewer(video_path="other.mp4") is None
    app._open_timeline_viewer = build
    app.open_timeline_viewer = types.MethodType(main.VideoHighlighterGUI.open_timeline_viewer, app)
    app.open_timeline_viewer(video_path=video)
    assert built == [video]
    assert any("still opening" in e[1] for e in events if e[0] == "log")
    app.open_timeline_viewer(video_path=video)      # and it opens again after
    assert built == [video, video]


def test_live_commands_are_refused_while_the_viewer_opens(app):
    app, window, events, timers, video = app
    viewer_on = getattr(main.VideoHighlighterGUI, "_live_viewer_on", None)
    if viewer_on is None:                            # an edition without seek/overlay
        pytest.skip("no viewer-driving live commands in this edition")
    app._timeline_opening = True
    with pytest.raises(RuntimeError, match="still opening"):
        viewer_on(app, {"video_path": video})
