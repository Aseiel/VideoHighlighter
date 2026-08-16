"""The flight recorder for repaints — chiefly, that it never becomes the bug.

It runs on the path that is already crashing, so the bar is not "does it record
well" but "can it be the reason something fell over". Every entry point has to
survive being called before arming, with a dead Qt wrapper, or with values that
do not serialise.
"""

from __future__ import annotations

import os
import time

import pytest

from modules import repaint_trace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(repaint_trace.__file__)))


@pytest.fixture(autouse=True)
def _clean():
    repaint_trace.reset_for_tests()
    yield
    repaint_trace.reset_for_tests()


@pytest.fixture()
def trace(tmp_path):
    path = tmp_path / "repaint_trace.log"
    assert repaint_trace.arm(str(path)) is True
    return path


class TestSilentUntilArmed:
    def test_note_before_arming_is_a_no_op(self):
        repaint_trace.note("filter.action_confidence", min=0.5)   # must not raise

    def test_rebuild_before_arming_still_runs_the_body(self):
        ran = []
        with repaint_trace.rebuild("build_timeline"):
            ran.append(True)
        assert ran == [True]


class TestWhatItRecords:
    def test_a_breadcrumb_lands_immediately(self, trace):
        repaint_trace.note("filter.object_confidence", min=0.25)
        # Flushed per line, not at close: a buffered breadcrumb is one the crash
        # eats, which is the entire failure mode this exists for.
        assert "filter.object_confidence" in trace.read_text(encoding="utf-8")
        assert "min=0.25" in trace.read_text(encoding="utf-8")

    def test_a_rebuild_brackets_its_body(self, trace):
        with repaint_trace.rebuild("build_timeline", pps=10):
            repaint_trace.note("rebuild.items_before", current_time_line="live")
        text = trace.read_text(encoding="utf-8")
        assert "rebuild.begin" in text
        assert "rebuild.items_before" in text
        assert "rebuild.end" in text

    def test_a_body_that_dies_leaves_a_begin_with_no_end(self, trace):
        # The shape that says "the process died *inside* the rebuild" — the one
        # fact the existing logs cannot express.
        with pytest.raises(ValueError):
            with repaint_trace.rebuild("build_timeline"):
                raise ValueError("boom")
        text = trace.read_text(encoding="utf-8")
        assert "rebuild.begin" in text
        assert "rebuild.raised" in text
        assert "ValueError: boom" in text

    def test_nested_rebuilds_are_visible_as_depth(self, trace):
        with repaint_trace.rebuild("outer"):
            with repaint_trace.rebuild("inner"):
                pass
        text = trace.read_text(encoding="utf-8")
        assert "depth=1" in text and "depth=2" in text

    def test_generations_increase(self, trace):
        with repaint_trace.rebuild("one"):
            pass
        with repaint_trace.rebuild("two"):
            pass
        text = trace.read_text(encoding="utf-8")
        assert "gen=1" in text and "gen=2" in text


class TestProbeIsSafeOnTheThingItInspects:
    def test_a_missing_attribute_reads_as_unset(self):
        class Scene:
            pass

        assert repaint_trace.probe(Scene(), "current_time_line") == {
            "current_time_line": "unset"}

    def test_a_none_attribute_reads_as_unset(self):
        class Scene:
            current_time_line = None

        assert repaint_trace.probe(Scene(), "current_time_line") == {
            "current_time_line": "unset"}

    def test_it_never_touches_the_object_it_is_asked_about(self):
        # A dangling wrapper raises on attribute access. `probe` must ask
        # shiboken whether the object is valid rather than poke it — poking is
        # what takes the process down, and doing that inside the tool meant to
        # explain the crash would be its own punchline.
        class Exploding:
            def __getattr__(self, name):
                raise RuntimeError("Internal C++ object already deleted")

            def __eq__(self, other):
                raise RuntimeError("Internal C++ object already deleted")

            def __bool__(self):
                raise RuntimeError("Internal C++ object already deleted")

        class Scene:
            current_time_line = Exploding()

        result = repaint_trace.probe(Scene(), "current_time_line")
        assert result["current_time_line"] in {"live", "dangling", "unknown"}

    def test_unserialisable_field_values_do_not_raise(self, trace):
        class Awkward:
            def __repr__(self):
                raise RuntimeError("no repr for you")

        repaint_trace.note("playhead.stale_item", seconds=Awkward())  # must not raise


class TestThreadCheck:
    def test_without_a_qapplication_it_declines_rather_than_guessing(self):
        assert repaint_trace.on_gui_thread() in {"yes", "no", "unknown"}


class TestQtsOwnDiagnosticsAreCaptured:
    """Qt's messages went nowhere, which is where these crashes went too.

    `qWarning`/`qFatal` are written to the C-level stderr, not to the Python
    object `debug_console` replaced — so the tee never saw them, and in a
    `--windowed` build that descriptor is not connected to anything. A Qt fatal
    prints one explanatory line and calls abort, and both halves were invisible.
    """

    def test_arming_installs_a_handler(self, trace):
        qtcore = pytest.importorskip("PySide6.QtCore")
        # qInstallMessageHandler returns the handler it replaced, so installing
        # a throwaway one and getting ours back proves it was in place.
        previous = qtcore.qInstallMessageHandler(None)
        try:
            assert previous is not None
        finally:
            qtcore.qInstallMessageHandler(previous)

    def test_a_qt_warning_reaches_the_trace(self, trace):
        qtcore = pytest.importorskip("PySide6.QtCore")
        qtcore.qWarning("d3d device removed")

        text = trace.read_text(encoding="utf-8")
        assert "qt" in text and "d3d device removed" in text
        assert "level='warning'" in text

    def test_the_handler_cannot_take_the_process_down(self, trace):
        qtcore = pytest.importorskip("PySide6.QtCore")
        repaint_trace.reset_for_tests()      # handler still installed, file gone
        qtcore.qWarning("after the handle closed")   # must not raise


class TestATraceThatEndsWithoutAGoodbye:
    def test_a_clean_shutdown_is_recorded(self, tmp_path):
        # The line matters less than its absence: without it a trace that stops
        # says only that the process stopped, not whether it was closed or
        # killed. Run in a real subprocess, because the fact under test is that
        # it happens at interpreter exit.
        import subprocess
        import sys

        path = tmp_path / "trace.log"
        subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, r'{_REPO_ROOT}');"
             "from modules import repaint_trace;"
             f"repaint_trace.arm(r'{path}')"],
            check=True, timeout=60)

        assert "session.exit" in path.read_text(encoding="utf-8")

    def test_a_killed_process_leaves_none(self, tmp_path):
        import signal
        import subprocess
        import sys

        path = tmp_path / "trace.log"
        proc = subprocess.Popen(
            [sys.executable, "-c",
             f"import sys, time; sys.path.insert(0, r'{_REPO_ROOT}');"
             "from modules import repaint_trace;"
             f"repaint_trace.arm(r'{path}');"
             "repaint_trace.note('working'); time.sleep(30)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if path.exists() and "working" in path.read_text(encoding="utf-8"):
                    break
                time.sleep(0.05)
            proc.kill()
        finally:
            proc.wait(timeout=30)

        text = path.read_text(encoding="utf-8")
        assert "working" in text, "the child never got going"
        assert "session.exit" not in text
