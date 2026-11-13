"""The time ruler draws what the view can reach, and only when it has to.

It is redrawn on every wheel step, and a mark is a scene item: drawing the
whole video's marks each time cost 210 ms a step on a 76-minute film, for a
view that showed two dozen of them.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QGraphicsTextItem  # noqa: E402

from video_ai_editor.signal_timeline import (  # noqa: E402
    SignalTimelineScene, SignalTimelineView,
)

HOUR = 3600.0


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def timeline(app):
    """An hour of video in an 800px view: far more ruler than fits."""
    scene = SignalTimelineScene(cache_data={}, video_duration=HOUR)
    view = SignalTimelineView(scene)
    view.resize(800, 300)
    scene.build_timeline()
    return scene, view


def label_seconds(scene):
    pps = scene.pixels_per_second
    return sorted(round(i.pos().x() / pps) for i in scene._time_marker_items
                  if isinstance(i, QGraphicsTextItem))


def shown_seconds(scene, view):
    rect = view.mapToScene(view.viewport().rect()).boundingRect()
    pps = scene.pixels_per_second
    return max(0.0, rect.left() / pps), min(HOUR, rect.right() / pps)


def covers(scene, view):
    """Every label the view should be showing is drawn."""
    lo, hi = shown_seconds(scene, view)
    step = scene._ruler_drawn["label_iv"]
    wanted = [s for s in range(0, int(HOUR) + 1, step) if lo <= s <= hi]
    drawn = set(label_seconds(scene))
    return wanted and all(s in drawn for s in wanted)


class TestItDrawsWhatTheViewCanReach:
    def test_not_the_whole_video(self, timeline):
        scene, view = timeline
        lo, hi = shown_seconds(scene, view)
        labels = label_seconds(scene)
        assert labels, "no ruler at all"
        assert labels[-1] < HOUR / 2, "drew marks far beyond the view"
        assert labels[-1] <= hi + (hi - lo) + scene._ruler_drawn["label_iv"]

    def test_but_everything_that_is_on_screen(self, timeline):
        scene, view = timeline
        assert covers(scene, view)

    def test_without_a_view_it_is_the_whole_video(self, app):
        scene = SignalTimelineScene(cache_data={}, video_duration=600.0)
        scene.build_timeline()
        labels = label_seconds(scene)
        assert labels[0] == 0 and labels[-1] >= 590


class TestItFollowsTheView:
    def test_scrolling_far_along_brings_the_marks_there(self, timeline):
        scene, view = timeline
        bar = view.horizontalScrollBar()
        bar.setValue(bar.maximum())                  # the end of the hour
        lo, _ = shown_seconds(scene, view)
        assert lo > HOUR / 2
        assert covers(scene, view)

    def test_and_back(self, timeline):
        scene, view = timeline
        bar = view.horizontalScrollBar()
        bar.setValue(bar.maximum())
        bar.setValue(0)
        assert covers(scene, view)

    def test_zooming_out_to_the_whole_video_marks_all_of_it(self, timeline):
        scene, view = timeline
        for _ in range(40):
            view.scale(1 / 1.15, 1.0)
            scene.draw_time_markers()
        assert covers(scene, view)
        assert label_seconds(scene)[-1] >= HOUR - scene._ruler_drawn["label_iv"]


class TestItDoesNothingWhenNothingChanged:
    def test_a_second_call_leaves_the_marks_alone(self, timeline):
        scene, view = timeline
        before = list(scene._time_marker_items)
        scene.draw_time_markers()
        assert scene._time_marker_items == before
        assert all(a is b for a, b in zip(scene._time_marker_items, before))

    def test_a_small_scroll_leaves_them_alone_too(self, timeline):
        scene, view = timeline
        before = list(scene._time_marker_items)
        view.horizontalScrollBar().setValue(50)
        assert all(a is b for a, b in zip(scene._time_marker_items, before))
        assert covers(scene, view)

    def test_a_zoom_that_changes_the_interval_redraws(self, timeline):
        scene, view = timeline
        step = scene._ruler_drawn["label_iv"]
        for _ in range(12):
            view.scale(1 / 1.15, 1.0)
            scene.draw_time_markers()
        assert scene._ruler_drawn["label_iv"] > step
        assert covers(scene, view)

    def test_a_rebuild_draws_it_afresh(self, timeline):
        scene, view = timeline
        scene.build_timeline()
        assert scene._time_marker_items and covers(scene, view)
