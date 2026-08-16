"""The whole-video filmstrip lane.

Two properties matter more than how it looks. It must ask for only the frames
on screen — an hour of video is ~2000 slots and decoding them to scroll one
screen would be unusable — and its slots must be anchored to a fixed grid, so
thumbnails stay put while you scroll instead of resampling to new timestamps
every time the exposed region shifts by a pixel.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import (
    QApplication, QGraphicsItem, QStyleOptionGraphicsItem)

from video_ai_editor.filmstrip_lane import (
    LANE_HEIGHT, MAX_SLOTS_PER_PAINT, FilmstripLane)
from video_ai_editor.signal_timeline import SignalTimelineScene

DURATION = 3826.0          # ~64 minutes, the length that made this necessary
PPS = 60.0
WIDTH = DURATION * PPS


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


class RecordingCache:
    """Stands in for ThumbnailCache, remembering what was asked for."""

    def __init__(self):
        self.times = []

    def request(self, time_seconds, height_px, priority=0):
        self.times.append(float(time_seconds))
        return None            # always a placeholder; nothing to decode


def _paint(lane, exposed):
    image = QImage(400, int(LANE_HEIGHT), QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = exposed
    try:
        lane.paint(painter, option)
    finally:
        painter.end()


class TestItOnlyAsksForWhatIsOnScreen:
    def test_it_opts_into_a_narrowed_exposed_rect(self, app):
        """The flag this asserts is the whole reason the lane is affordable.

        Qt only computes `exposedRect` for items carrying
        ItemUsesExtendedStyleOption; without it every paint is handed the full
        boundingRect, which for an hour of video is ~2400 slots re-requested on
        every repaint — measured, and it makes the window unusable. Asserting
        the flag rather than the behaviour because the tests that fed `paint` a
        hand-built option passed happily while the real app froze: the option
        Qt supplies was never exercised.
        """
        lane = FilmstripLane(WIDTH, DURATION, RecordingCache())
        assert lane.flags() & QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption

    def test_a_viewport_costs_a_screenful_not_a_video(self, app):
        cache = RecordingCache()
        lane = FilmstripLane(WIDTH, DURATION, cache)

        _paint(lane, QRectF(0, 0, 1400, LANE_HEIGHT))

        whole_timeline = WIDTH / lane.slot_width()
        assert len(cache.times) < 40, "asked for far more than one screen"
        assert whole_timeline > 1000, "fixture no longer represents a long video"

    def test_an_oversized_exposed_region_is_capped(self, app):
        # Belt and braces for the failure above: even if the exposed rect is
        # not viewport-shaped, one paint must never enqueue thousands of
        # decodes on the GUI thread.
        cache = RecordingCache()
        lane = FilmstripLane(WIDTH, DURATION, cache)

        _paint(lane, QRectF(0, 0, WIDTH, LANE_HEIGHT))

        assert len(cache.times) <= MAX_SLOTS_PER_PAINT

    def test_scrolling_asks_about_where_it_scrolled_to(self, app):
        cache = RecordingCache()
        lane = FilmstripLane(WIDTH, DURATION, cache)

        _paint(lane, QRectF(WIDTH / 2, 0, 1400, LANE_HEIGHT))

        # Times should sit around the middle of the video, not the start.
        assert cache.times
        assert all(t > DURATION * 0.4 for t in cache.times)

    def test_nothing_is_requested_off_the_end(self, app):
        cache = RecordingCache()
        lane = FilmstripLane(WIDTH, DURATION, cache)

        _paint(lane, QRectF(WIDTH - 200, 0, 1400, LANE_HEIGHT))

        assert cache.times
        assert max(cache.times) <= DURATION + 1e-6


class TestTheSlotGridIsStable:
    def test_the_same_frames_are_asked_for_after_a_one_pixel_scroll(self, app):
        # The reason for snapping to a grid anchored at x=0: without it, every
        # pixel of scroll resamples every slot to a slightly different time and
        # the strip shimmers.
        first, second = RecordingCache(), RecordingCache()
        FilmstripLane(WIDTH, DURATION, first)   # keep refs alive
        lane_a = FilmstripLane(WIDTH, DURATION, first)
        lane_b = FilmstripLane(WIDTH, DURATION, second)

        _paint(lane_a, QRectF(5000, 0, 800, LANE_HEIGHT))
        _paint(lane_b, QRectF(5001, 0, 800, LANE_HEIGHT))

        shared = set(first.times) & set(second.times)
        assert len(shared) >= len(first.times) - 1

    def test_slots_align_to_multiples_of_the_slot_width(self, app):
        cache = RecordingCache()
        lane = FilmstripLane(WIDTH, DURATION, cache)
        slot = lane.slot_width()

        _paint(lane, QRectF(1234, 0, 800, LANE_HEIGHT))

        # Each requested time is a slot midpoint on the global grid.
        for t in cache.times:
            x = t / DURATION * WIDTH
            offset = (x % slot) / slot
            assert offset == pytest.approx(0.5, abs=0.02)


class TestItSurvivesMissingPieces:
    def test_no_thumbnail_source_paints_nothing_rather_than_raising(self, app):
        lane = FilmstripLane(WIDTH, DURATION, None)
        _paint(lane, QRectF(0, 0, 400, LANE_HEIGHT))   # must not raise

    def test_an_exposed_region_outside_the_lane_is_skipped(self, app):
        cache = RecordingCache()
        lane = FilmstripLane(WIDTH, DURATION, cache)
        _paint(lane, QRectF(WIDTH + 500, 0, 400, LANE_HEIGHT))
        assert cache.times == []

    def test_a_zero_length_video_does_not_divide_by_zero(self, app):
        lane = FilmstripLane(100, 0.0, RecordingCache())
        _paint(lane, QRectF(0, 0, 100, LANE_HEIGHT))


class TestTheLaneInTheScene:
    def _scene(self):
        scene = SignalTimelineScene({}, DURATION)
        scene.pixels_per_second = PPS
        scene.build_timeline()
        return scene

    def _lanes(self, scene):
        return [i for i in scene.items() if isinstance(i, FilmstripLane)]

    def test_it_is_on_by_default(self, app):
        assert self._scene().visible_layers.get('filmstrip') is True

    def test_it_is_the_bottom_row(self, app):
        # Closest to the edit timeline underneath, which is the strip you are
        # comparing it against.
        scene = self._scene()
        (lane,) = self._lanes(scene)
        assert lane.pos().y() == max(y for _, y in scene.row_labels)

    def test_it_spans_the_whole_scene(self, app):
        scene = self._scene()
        (lane,) = self._lanes(scene)
        assert lane.boundingRect().width() == pytest.approx(
            scene.sceneRect().width(), abs=1.0)

    def test_turning_it_off_removes_it_and_gives_the_height_back(self, app):
        scene = self._scene()
        tall = scene.sceneRect().height()

        scene.visible_layers['filmstrip'] = False
        scene.build_timeline()

        assert self._lanes(scene) == []
        assert scene.sceneRect().height() < tall

    def test_turning_it_back_on_restores_it(self, app):
        scene = self._scene()
        scene.visible_layers['filmstrip'] = False
        scene.build_timeline()
        scene.visible_layers['filmstrip'] = True
        scene.build_timeline()
        assert len(self._lanes(scene)) == 1

    def test_a_rebuild_lets_go_of_the_old_lane(self, app):
        # Thumbnails arrive asynchronously; one landing after a rebuild must not
        # repaint through the item that clear() deleted.
        scene = self._scene()
        scene.build_timeline()
        scene._filmstrip_item = None
        scene._on_thumbnail_ready(1.0, 54, None)   # must not raise

    def test_arriving_frames_are_coalesced_into_one_repaint(self, app):
        # A screenful of thumbnails lands one at a time. Repainting per frame
        # means dozens of full-lane repaints a second, each re-walking every
        # visible slot, which competes with whatever the user is doing.
        scene = self._scene()
        repaints = []
        scene._repaint_filmstrip = lambda: repaints.append(1)

        for _ in range(30):
            scene._on_thumbnail_ready(1.0, 54, None)

        assert scene._filmstrip_repaint_pending is True
        assert repaints == [], "repainted synchronously per frame"

    def test_zooming_in_samples_more_finely(self, app):
        # "Re-sample to fill the view": at higher zoom the scene is wider, so
        # the same screen covers less time and the slots land closer together.
        scene = self._scene()
        (wide,) = self._lanes(scene)
        coarse = wide.boundingRect().width() / wide.slot_width()

        scene.pixels_per_second = PPS * 4
        scene.build_timeline()
        (zoomed,) = self._lanes(scene)
        fine = zoomed.boundingRect().width() / zoomed.slot_width()

        assert fine > coarse * 3
