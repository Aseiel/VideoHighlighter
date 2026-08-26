"""The recorder wired into the real scene, on the path that crashes.

Unit-testing the recorder proves it writes lines. What matters is whether those
lines appear at the moment a user moves the confidence slider, and whether they
say anything a person could act on — so this drives the actual
`SignalTimelineScene` rather than a stand-in.

It also pins the finding the wiring produced, and the fix for it. `clear()`
deletes the C++ objects behind `current_time_line` and `current_time_caret`,
and `build_timeline` repositions the playhead before it returns — so until
those references were dropped alongside the selection ones, every confidence
adjustment reached through a deleted wrapper. That is undefined behaviour
rather than a defined no-op, and it survived often enough to present as an
occasional crash with no traceback instead of a reproducible exception, which
is precisely why it needed a recorder to find.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from modules import repaint_trace
from video_ai_editor.signal_timeline import SignalTimelineScene


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture()
def trace(tmp_path, app):
    repaint_trace.reset_for_tests()
    path = tmp_path / "repaint_trace.log"
    repaint_trace.arm(str(path))
    yield path
    repaint_trace.reset_for_tests()


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


class TestTheConfidenceSliderIsRecorded:
    def test_moving_it_names_the_control_and_the_value(self, trace):
        scene = SignalTimelineScene({}, 60.0)
        scene.set_action_confidence_filter(0.4)

        text = trace.read_text(encoding="utf-8")
        assert "filter.action_confidence" in text
        assert "min=0.4" in text

    def test_the_rebuild_it_triggers_is_bracketed(self, trace):
        scene = SignalTimelineScene({}, 60.0)
        scene.set_object_confidence_filter(0.6)

        events = [ln.split("] ", 1)[1].split(" ", 1)[0]
                  for ln in _lines(trace) if "] " in ln]
        # A begin with a matching end means the rebuild completed. A begin with
        # no end is what a crash inside the rebuild would leave, and telling
        # those two apart is the whole point.
        assert events.count("rebuild.begin") == events.count("rebuild.end")
        assert "filter.object_confidence" in events

    def test_the_rebuild_reports_which_thread_it_ran_on(self, trace):
        scene = SignalTimelineScene({}, 60.0)
        scene.build_timeline()
        assert "gui_thread='yes'" in trace.read_text(encoding="utf-8")


class TestTheStalePlayheadIsCaught:
    def test_clear_leaves_the_playhead_references_dangling(self, trace, app):
        scene = SignalTimelineScene({}, 60.0)
        scene.set_current_time(5.0)
        assert repaint_trace.probe(scene, "current_time_line") == {
            "current_time_line": "live"}

        scene.clear()

        # Not a hypothesis — this is what QGraphicsScene.clear() does, and
        # build_timeline does not reset these two attributes the way it resets
        # the selection ones.
        assert repaint_trace.probe(
            scene, "current_time_line", "current_time_caret") == {
                "current_time_line": "dangling",
                "current_time_caret": "dangling"}

    def test_a_rebuild_leaves_the_playhead_valid_rather_than_dangling(self, trace):
        scene = SignalTimelineScene({}, 60.0)
        scene.set_current_time(5.0)        # playback tick creates the playhead
        scene.build_timeline()

        # The rebuild drops the references clear() invalidated and then draws a
        # new playhead before it returns, so the end state is a live item. What
        # matters is the word that must never appear here: `dangling` would mean
        # the attribute survived the scene it belonged to.
        assert repaint_trace.probe(
            scene, "current_time_line", "current_time_caret") == {
                "current_time_line": "live",
                "current_time_caret": "live"}

    def test_the_crash_trigger_no_longer_touches_a_dead_playhead(self, trace):
        scene = SignalTimelineScene({}, 60.0)
        scene.set_current_time(5.0)         # playback tick creates the playhead
        scene.set_action_confidence_filter(0.4)    # the reported crash trigger

        # The rebuild repositions the playhead before it returns. It used to do
        # that through a wrapper clear() had already deleted — undefined
        # behaviour that happened to survive on some builds, which is why it
        # surfaced as a crash with no traceback rather than an exception.
        assert not any("playhead.stale_item" in ln for ln in _lines(trace)), (
            "a rebuild reached through a deleted playhead reference again")

    def test_the_playhead_still_works_after_a_rebuild(self, trace):
        scene = SignalTimelineScene({}, 60.0)
        scene.set_current_time(5.0)
        scene.set_action_confidence_filter(0.4)
        scene.set_current_time(7.0)

        # Dropping the references must cost nothing visible: the next tick
        # rebuilds the playhead rather than leaving the timeline without one.
        assert repaint_trace.probe(scene, "current_time_line") == {
            "current_time_line": "live"}
        assert scene.current_time_seconds == 7.0

    def test_a_quiet_playhead_move_records_nothing(self, trace):
        scene = SignalTimelineScene({}, 60.0)
        scene.set_current_time(5.0)
        before = len(_lines(trace))
        scene.set_current_time(6.0)      # nothing stale — must stay silent

        # This runs on every playback tick; a line per tick would bury the one
        # line that matters under a log nobody can read.
        assert len(_lines(trace)) == before


class TestTheRulerLetsGoOfWhatTheRebuildDeleted:
    """The same mistake as the playhead's, and this one was caught in the act.

    `repaint_trace.log` holds an access violation whose current frame is
    `draw_time_markers`, on the loop that removes the previous pass's markers —
    a rebuild had just deleted every one of them in `clear()`. The loop guards
    against that with `except RuntimeError`, which only helps when PySide
    noticed the deletion; when it does not, asking a corpse for its `scene()`
    is undefined behaviour, and there it killed the process.
    """

    def test_clear_leaves_the_marker_references_dangling(self, trace, app):
        scene = SignalTimelineScene({}, 60.0)
        scene.build_timeline()
        assert scene._time_marker_items, "the ruler drew no markers to lose"
        scene._marker = scene._time_marker_items[0]

        scene.clear()

        assert repaint_trace.probe(scene, "_marker") == {"_marker": "dangling"}

    def test_a_rebuild_removes_nothing_it_has_already_deleted(self, trace, app):
        scene = SignalTimelineScene({}, 60.0)
        scene.build_timeline()             # first pass leaves a full list

        seen = []
        original = scene.draw_time_markers
        scene.draw_time_markers = lambda: (
            seen.append(list(scene._time_marker_items)), original())[1]
        scene.build_timeline()

        # The list the ruler finds on the first pass after clear() must be
        # empty. Anything in it is an item the rebuild has already destroyed.
        assert seen and seen[0] == []

    def test_the_ruler_still_clears_its_own_markers_on_a_zoom(self, trace, app):
        # Dropping the list is only safe where clear() has run. Redrawing the
        # ruler on its own — which is what a zoom does — must still take the
        # previous markers off the scene rather than stacking a second set.
        scene = SignalTimelineScene({}, 60.0)
        scene.build_timeline()
        before = len(scene.items())

        scene.draw_time_markers()

        assert len(scene.items()) == before
