"""The hover popup, and the frame it is waiting for.

The popup opens before its thumbnail exists, painting "loading…". Nothing used
to replace that: `show_at` runs only on mouse movement, so a cursor held still
sat on "loading…" indefinitely — long after the frame had decoded. Hovering
away and back looked like a cure, but only because the second request found the
frame in the cache, which is also why it looked intermittent rather than broken.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from video_ai_editor.hover_preview import HoverPreview


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _frame(w=320, h=180):
    pix = QPixmap(w, h)
    pix.fill()
    return pix


class TestALateFrameStillReachesThePopup:
    def test_a_frame_arriving_after_the_popup_opened_is_shown(self, app):
        popup = HoverPreview()
        popup.show_at(QPoint(400, 400), None, caption="12:34")
        assert popup._pixmap is None          # "loading…"

        popup.set_pixmap(_frame())

        assert popup._pixmap is not None
        popup.hide_preview()

    def test_the_caption_survives_the_swap(self, app):
        popup = HoverPreview()
        popup.show_at(QPoint(400, 400), None, caption="12:34")
        popup.set_pixmap(_frame())
        assert popup._caption == "12:34"
        popup.hide_preview()

    def test_the_popup_is_resized_for_the_real_aspect(self, app):
        # The placeholder is drawn at a 16:9 guess. Footage that is not 16:9
        # would be letterboxed into the wrong box if the swap kept that size.
        popup = HoverPreview()
        popup.show_at(QPoint(400, 400), None, caption="")
        guessed = popup.width()

        popup.set_pixmap(_frame(180, 320))    # vertical video

        assert popup.width() < guessed
        popup.hide_preview()

    def test_a_hidden_popup_ignores_a_late_frame(self, app):
        # The mouse has already left; repainting it would flash a popup back on.
        popup = HoverPreview()
        popup.show_at(QPoint(400, 400), None)
        popup.hide_preview()

        popup.set_pixmap(_frame())

        assert not popup.isVisible()

    def test_a_null_frame_leaves_the_placeholder_alone(self, app):
        popup = HoverPreview()
        popup.show_at(QPoint(400, 400), None)
        popup.set_pixmap(QPixmap())
        assert popup._pixmap is None
        popup.hide_preview()


class TestDeliveryFromTheCache:
    """The scene half: matching an arriving frame to what the popup wants.

    Driven through `_on_thumb_ready`, the slot actually connected to the cache,
    rather than the helper it delegates to. The bug was a missing call, not
    faulty logic, and a test that invoked the helper directly would have passed
    against the broken code.
    """

    class FakePopup:
        def __init__(self):
            self.pix = None
            self.visible = True

        def isVisible(self):
            return self.visible

        def set_pixmap(self, pix):
            self.pix = pix

    class FakeCache:
        """Buckets times to 100ms like the real one, and answers from memory."""

        def __init__(self, ready=()):
            self._ready = {int(t * 10) for t in ready}
            self.asked = []

        def request(self, t, height, priority=0):
            self.asked.append((t, height))
            return _frame() if int(t * 10) in self._ready else None

    def _scene(self, cache, wanted):
        from video_ai_editor.edit_timeline import EditTimelineScene

        scene = EditTimelineScene.__new__(EditTimelineScene)
        scene.clip_items = []
        scene.thumb_cache = cache
        scene._hover_preview = self.FakePopup()
        scene._hover_wanted = wanted
        return scene

    def test_the_waiting_frame_is_delivered_when_it_arrives(self, app):
        from video_ai_editor.edit_timeline import HOVER_PREVIEW_HEIGHT

        cache = self.FakeCache(ready=[12.3])
        scene = self._scene(cache, wanted=12.3456)

        scene._on_thumb_ready(12.3, HOVER_PREVIEW_HEIGHT, None)

        assert scene._hover_preview.pix is not None
        assert scene._hover_wanted is None

    def test_a_filmstrip_frame_does_not_disturb_the_popup(self, app):
        from video_ai_editor.edit_timeline import HOVER_PREVIEW_HEIGHT

        cache = self.FakeCache(ready=[12.3])
        scene = self._scene(cache, wanted=12.3456)

        # Filmstrip thumbs are a different height and cannot be what the popup
        # is waiting for; asking the cache for each of them would be wasted.
        scene._on_thumb_ready(12.3, HOVER_PREVIEW_HEIGHT // 3, None)

        assert scene._hover_preview.pix is None
        assert cache.asked == []

    def test_nothing_happens_when_the_popup_wants_nothing(self, app):
        from video_ai_editor.edit_timeline import HOVER_PREVIEW_HEIGHT

        cache = self.FakeCache(ready=[12.3])
        scene = self._scene(cache, wanted=None)

        scene._on_thumb_ready(12.3, HOVER_PREVIEW_HEIGHT, None)

        assert cache.asked == []

    def test_a_still_missing_frame_leaves_the_want_standing(self, app):
        from video_ai_editor.edit_timeline import HOVER_PREVIEW_HEIGHT

        cache = self.FakeCache(ready=[])          # a different frame arrived
        scene = self._scene(cache, wanted=12.3456)

        scene._on_thumb_ready(12.3, HOVER_PREVIEW_HEIGHT, None)

        assert scene._hover_preview.pix is None
        assert scene._hover_wanted == 12.3456     # still waiting, not dropped

    def test_a_hidden_popup_is_not_fed(self, app):
        from video_ai_editor.edit_timeline import HOVER_PREVIEW_HEIGHT

        cache = self.FakeCache(ready=[12.3])
        scene = self._scene(cache, wanted=12.3456)
        scene._hover_preview.visible = False

        scene._on_thumb_ready(12.3, HOVER_PREVIEW_HEIGHT, None)

        assert cache.asked == []
