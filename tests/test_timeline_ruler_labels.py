"""
The timeline's time labels, and staying inside the timeline.

The ruler's labels carry `ItemIgnoresTransformations`, so their glyphs are a
fixed size however the scene is scaled — but the position anchoring them is a
scene coordinate, which is not. They were placed a fixed 22 *scene* units above
the bottom, so as the view squashed the scene to fit (many lanes in a short
viewport, which is what a window looks like before its layout settles) that gap
shrank on screen until the text hung over the bottom edge and was drawn cut in
half. Which is how the app opened.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QApplication, QGraphicsTextItem

from video_ai_editor.signal_timeline import SignalTimelineScene, SignalTimelineView


BAND = SignalTimelineScene.LABEL_BAND_PX


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def timeline(app):
    """A scene with a ruler, in a view, with nothing else drawn."""
    scene = SignalTimelineScene(cache_data={}, video_duration=600.0)
    view = SignalTimelineView(scene)
    view.resize(800, 300)
    return scene, view


def labels(scene):
    """The ruler's own labels.

    From the list `draw_time_markers` keeps, not every text item in the scene —
    the lane names are text too, and they are not part of the ruler.
    """
    return [i for i in getattr(scene, "_time_marker_items", [])
            if isinstance(i, QGraphicsTextItem)]


def set_vertical_scale(view, scale_y):
    t = view.transform()
    view.setTransform(QTransform(t.m11(), t.m12(), t.m21(), scale_y, t.dx(), t.dy()))


class TestLabelsSitInsideTheTimeline:
    @pytest.mark.parametrize("scale_y", [0.25, 0.5, 0.8, 1.0, 2.0, 4.8])
    def test_the_gap_above_the_bottom_is_the_same_on_screen(self, timeline, scale_y):
        scene, view = timeline
        set_vertical_scale(view, scale_y)
        scene.draw_time_markers()

        bottom = scene.sceneRect().height()
        label = labels(scene)[0]
        # Scene units between the label and the bottom, converted back to the
        # screen pixels the glyphs are actually drawn in.
        on_screen_gap = (bottom - label.pos().y()) * scale_y
        assert on_screen_gap == pytest.approx(BAND, abs=0.5)

    def test_squashing_the_view_does_not_push_labels_off_the_bottom(self, timeline):
        scene, view = timeline
        set_vertical_scale(view, 0.3)          # a short viewport, many lanes
        scene.draw_time_markers()

        bottom = scene.sceneRect().height()
        for label in labels(scene):
            assert label.pos().y() < bottom

    def test_the_old_fixed_offset_would_have_failed_this(self, timeline):
        """Pins the actual regression: 22 scene units at a small scale is 6px."""
        scene, view = timeline
        set_vertical_scale(view, 0.3)
        scene.draw_time_markers()

        bottom = scene.sceneRect().height()
        label = labels(scene)[0]
        old_style_gap = (bottom - 22) * 0.3    # what the old code produced
        new_gap = (bottom - label.pos().y()) * 0.3
        assert new_gap > (bottom * 0.3 - old_style_gap)

    def test_labels_never_go_above_the_top(self, timeline):
        # An absurdly squashed view must not push them off the other end.
        scene, view = timeline
        set_vertical_scale(view, 0.001)
        scene.draw_time_markers()
        for label in labels(scene):
            assert label.pos().y() >= 0

    def test_labels_are_still_drawn_at_all(self, timeline):
        scene, view = timeline
        scene.draw_time_markers()
        assert labels(scene), "the ruler should have labels"

    def test_they_still_ignore_the_view_transform(self, timeline):
        from PySide6.QtWidgets import QGraphicsItem
        scene, view = timeline
        scene.draw_time_markers()
        flag = QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
        assert all(bool(l.flags() & flag) for l in labels(scene))


class TestRefittingRedrawsTheRuler:
    def test_fit_vertical_re_anchors_the_labels(self, timeline):
        scene, view = timeline
        set_vertical_scale(view, 1.0)
        scene.draw_time_markers()
        before = labels(scene)[0].pos().y()

        # A taller viewport: _fit_vertical scales up, and the labels must move
        # with it rather than keep the spacing of the old size.
        view.resize(800, 900)
        view._fit_vertical()

        after = labels(scene)[0].pos().y()
        assert after != before

        bottom = scene.sceneRect().height()
        gap = (bottom - after) * view.transform().m22()
        assert gap == pytest.approx(BAND, abs=0.5)

    def test_a_fit_that_changes_nothing_does_not_redraw(self, timeline):
        scene, view = timeline
        view._fit_vertical()
        first = labels(scene)
        view._fit_vertical()          # same size, same scale
        assert labels(scene) == first
