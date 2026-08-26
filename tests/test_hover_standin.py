"""Not making the user read the word "loading".

The hover popup asks for a 180px frame; the filmstrips ask for ~54px ones. The
cache is keyed on (time, height), so those never share an entry — the moment
under the cursor has almost always been decoded already, just at another size,
and asking for it again meant a fresh decode and a placeholder every time the
cursor moved to a new 100ms bucket.

So a miss now puts up the nearest frame already in memory, whatever size it was
decoded at, and the sharp one replaces it when it lands. Entering a clip also
starts warming hover-sized frames along it, so there is usually nothing to
stand in for.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from collections import OrderedDict

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from video_ai_editor.edit_timeline import (
    HOVER_PREFETCH_SLOTS, HOVER_PREVIEW_HEIGHT, EditClipItem)
from video_ai_editor.thumbnail_cache import ThumbnailCache


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _pix(w=96, h=54):
    pix = QPixmap(w, h)
    pix.fill()
    return pix


def _cache(entries, vr_mode=False):
    """A ThumbnailCache with only its memory dict populated."""
    cache = ThumbnailCache.__new__(ThumbnailCache)
    cache._mem = OrderedDict()
    cache.mem_limit = 300
    cache._vr_mode = vr_mode
    for entry in entries:
        t_ms, height = entry[0], entry[1]
        vr = entry[2] if len(entry) > 2 else False
        cache._mem[(t_ms, height, vr)] = _pix(h=height)
    return cache


class TestPeekNearest:
    def test_the_same_moment_at_another_size_is_found(self, app):
        # The case that matters: the filmstrip decoded this exact frame at
        # 54px and the hover wants 180px.
        cache = _cache([(12300, 54)])
        assert cache.peek_nearest(12.3456) is not None

    def test_an_exact_moment_beats_a_nearer_size(self, app):
        # Soft but right beats sharp but elsewhere: showing a different part of
        # the video is a much bigger lie than showing this part upscaled.
        cache = _cache([(12300, 54), (12900, HOVER_PREVIEW_HEIGHT)])
        got = cache.peek_nearest(12.30)
        assert got.height() == 54

    def test_a_taller_frame_wins_a_tie(self, app):
        cache = _cache([(12300, 54), (12300, 120)])
        assert cache.peek_nearest(12.30).height() == 120

    def test_nothing_within_the_window_returns_none(self, app):
        cache = _cache([(1000, 54)])
        assert cache.peek_nearest(60.0) is None

    def test_the_window_is_honoured(self, app):
        cache = _cache([(12300, 54)])          # 0.6s away from 12.9
        assert cache.peek_nearest(12.9, max_delta=0.05) is None
        assert cache.peek_nearest(12.9, max_delta=5.0) is not None

    def test_an_empty_cache_is_fine(self, app):
        assert _cache([]).peek_nearest(1.0) is None

    def test_the_other_crop_is_not_a_stand_in(self, app):
        # A whole side-by-side frame is not a lower-resolution version of the
        # left eye, it is a different picture — standing one in for the other
        # would put the seam between the eyes under the cursor.
        cache = _cache([(12300, 54, False)], vr_mode=True)
        assert cache.peek_nearest(12.30) is None

    def test_it_never_queues_work(self, app):
        # A stand-in is for showing *now*. If this could enqueue, moving the
        # mouse would flood the queue ahead of the frame actually wanted.
        cache = _cache([(12300, 54)])
        cache._pending = {}
        cache.peek_nearest(12.3)
        assert cache._pending == {}


class TestEnteringAClipWarmsIt:
    class RecordingCache:
        def __init__(self):
            self.prefetched = []

        def prefetch_range(self, start, end, height, n_slots):
            self.prefetched.append((start, end, height, n_slots))

    def test_hovering_a_clip_prefetches_hover_sized_frames(self, app):
        from PySide6.QtWidgets import QGraphicsScene

        scene = QGraphicsScene()
        cache = self.RecordingCache()
        scene.thumb_cache = cache
        item = EditClipItem(10.0, 40.0, 0, 60, QColor("#3388ff"), 0)
        scene.addItem(item)

        item.hoverEnterEvent(_HoverEvent())

        assert cache.prefetched == [(10.0, 40.0, HOVER_PREVIEW_HEIGHT,
                                     HOVER_PREFETCH_SLOTS)]

    def test_a_long_clip_costs_the_same_as_a_short_one(self, app):
        from PySide6.QtWidgets import QGraphicsScene

        scene = QGraphicsScene()
        cache = self.RecordingCache()
        scene.thumb_cache = cache
        # An hour-long clip must not queue a decode per second.
        item = EditClipItem(0.0, 3600.0, 0, 60, QColor("#3388ff"), 0)
        scene.addItem(item)

        item.hoverEnterEvent(_HoverEvent())

        assert cache.prefetched[0][3] == HOVER_PREFETCH_SLOTS

    def test_no_cache_is_not_fatal(self, app):
        from PySide6.QtWidgets import QGraphicsScene

        scene = QGraphicsScene()
        scene.thumb_cache = None
        item = EditClipItem(10.0, 40.0, 0, 60, QColor("#3388ff"), 0)
        scene.addItem(item)
        item.hoverEnterEvent(_HoverEvent())      # must not raise


def _HoverEvent():
    """A real hover event — the base class rejects a stand-in object."""
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGraphicsSceneHoverEvent
    return QGraphicsSceneHoverEvent(QEvent.Type.GraphicsSceneHoverEnter)
