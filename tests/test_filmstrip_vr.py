"""The filmstrip on footage that holds two eyes per frame.

Side-by-side video was the one case where the strip was actively misleading.
Two things had to be wrong at once for that, and fixing either alone leaves it
wrong:

* the cache has to crop to the left eye, or every thumbnail is the same moment
  twice at half the size;
* the slots have to be reshaped to match, because a slot sized for 16:9 that is
  handed a square frame either pads it or, when the frame is the full
  side-by-side one, centre-crops onto the seam between the eyes — the single
  strip of the picture guaranteed to show nothing.

So the shape is asked of the cache, which is the thing that knows both the
source size and whether it is cropping, rather than remembered separately by
each view that draws thumbnails.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from collections import OrderedDict

from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from video_ai_editor.filmstrip_lane import FilmstripLane, LANE_HEIGHT
from video_ai_editor.filmstrip_painter import DEFAULT_ASPECT
from video_ai_editor.signal_timeline import SignalTimelineScene
from video_ai_editor.thumbnail_cache import ThumbnailCache

DURATION = 600.0


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


class FakeCache:
    """A ThumbnailCache stand-in with a source size and a crop flag."""

    def __init__(self, width=5760, height=2880):
        self._width, self._height = width, height
        self.vr_mode = False
        self.crops = []

    def frame_aspect(self):
        width = self._width / 2 if self.vr_mode else self._width
        return width / self._height

    def set_vr_mode(self, enabled):
        self.vr_mode = enabled
        self.crops.append(enabled)

    def request(self, time_seconds, height_px, priority=0):
        return None


def _scene(cache):
    scene = SignalTimelineScene({}, DURATION)
    scene._thumb_cache = cache      # stands in for the one _ensure would open
    scene.build_timeline()
    return scene


def _lane(scene):
    lanes = [i for i in scene.items() if isinstance(i, FilmstripLane)]
    assert len(lanes) == 1
    return lanes[0]


class TestTheStripFollowsTheCrop:
    def test_the_crop_reaches_the_cache(self, app):
        cache = FakeCache()
        scene = _scene(cache)

        scene.set_vr_mode(True)

        assert cache.crops == [True]

    def test_the_slots_are_reshaped_to_the_frame_that_will_arrive(self, app):
        # 5760×2880 is 2:1 whole and square per eye, so the same lane height
        # holds twice as many thumbnails once it is cropping.
        cache = FakeCache(5760, 2880)
        scene = _scene(cache)
        lane = _lane(scene)
        uncropped = lane.slot_width()

        scene.set_vr_mode(True)

        assert uncropped == pytest.approx(LANE_HEIGHT * 2.0, abs=1.0)
        assert lane.slot_width() == pytest.approx(LANE_HEIGHT * 1.0, abs=1.0)

    def test_a_rebuild_keeps_the_shape(self, app):
        # The lane is recreated by every rebuild, so the shape cannot live only
        # on the item that was there when VR was switched on.
        cache = FakeCache(5760, 2880)
        scene = _scene(cache)
        scene.set_vr_mode(True)

        scene.build_timeline()

        assert _lane(scene).slot_width() == pytest.approx(LANE_HEIGHT, abs=1.0)

    def test_switching_back_restores_the_whole_frame(self, app):
        cache = FakeCache(5760, 2880)
        scene = _scene(cache)
        scene.set_vr_mode(True)

        scene.set_vr_mode(False)

        assert cache.crops == [True, False]
        assert _lane(scene).slot_width() == pytest.approx(LANE_HEIGHT * 2.0,
                                                          abs=1.0)

    def test_setting_what_is_already_set_does_nothing(self, app):
        # set_vr_mode on the cache throws away its memory and wipes the frames
        # on disk, so a redundant call costs a full re-extraction.
        cache = FakeCache()
        scene = _scene(cache)

        scene.set_vr_mode(False)

        assert cache.crops == []


class TestTheShapeComesFromTheCache:
    def test_an_ordinary_video_keeps_its_own_aspect(self, app):
        scene = _scene(FakeCache(1920, 1080))
        assert _lane(scene).slot_width() == pytest.approx(
            LANE_HEIGHT * (16 / 9), abs=1.0)

    def test_a_vertical_video_gets_narrow_slots(self, app):
        scene = _scene(FakeCache(1080, 1920))
        assert _lane(scene).slot_width() < LANE_HEIGHT

    def test_an_unknown_size_falls_back_to_16_9(self, app):
        class Unknowing(FakeCache):
            def frame_aspect(self):
                return None            # the source could not be probed

        scene = _scene(Unknowing())
        assert _lane(scene).slot_width() == pytest.approx(
            LANE_HEIGHT * DEFAULT_ASPECT, abs=1.0)


class TestTogglingDoesNotThrowTheFramesAway:
    """Both crops are cached at once, so the checkbox is free after the first pass.

    It used to clear memory and delete every thumbnail on disk on each toggle,
    so ticking the box re-extracted the whole visible strip and unticking it
    re-extracted the same frames again — on 8K side-by-side footage, the single
    most expensive thing the strip can be asked to do.
    """

    def _cache(self, tmp_path):
        cache = ThumbnailCache.__new__(ThumbnailCache)
        cache._mem = OrderedDict()
        cache.mem_limit = 300
        cache._vr_mode = False
        cache._ffmpeg = None
        cache._src_w, cache._src_h = 5760, 2880
        cache.disk_dir = tmp_path
        return cache

    def _remember(self, cache, time_ms, height):
        pix = QPixmap(10, 10)
        pix.fill()
        cache._mem[(time_ms, height, cache._vr_mode)] = pix
        cache._disk_path((time_ms, height, cache._vr_mode)).write_bytes(b"jpeg")

    def test_the_frames_for_the_other_crop_stay_on_disk(self, app, tmp_path):
        cache = self._cache(tmp_path)
        self._remember(cache, 1000, 54)

        cache.set_vr_mode(True)

        assert list(tmp_path.glob("*.jpg")), "deleted the frames it had already"

    def test_switching_back_finds_what_was_there(self, app, tmp_path):
        cache = self._cache(tmp_path)
        self._remember(cache, 1000, 54)          # a full-frame thumb
        cache.set_vr_mode(True)
        self._remember(cache, 1000, 54)          # and a cropped one

        cache.set_vr_mode(False)

        assert cache.request(1.0, 54) is not None
        cache.set_vr_mode(True)
        assert cache.request(1.0, 54) is not None

    def test_the_two_crops_do_not_share_a_file(self, app, tmp_path):
        cache = self._cache(tmp_path)
        whole = cache._disk_path((1000, 54, False))
        left = cache._disk_path((1000, 54, True))
        assert whole != left

    def test_a_full_frame_thumb_keeps_the_name_it_always_had(self, app, tmp_path):
        # Existing disk caches were written before the crop was part of the
        # name; they must stay usable rather than silently re-extracting.
        cache = self._cache(tmp_path)
        assert cache._disk_path((1000, 54, False)).name == "1000_54.jpg"


class TestTheExpensivePathLeavesATrace:
    """What the filmstrip was doing when the process died.

    The VR crashes reported here happen while the strip is loading and leave
    nothing behind — not a Python traceback, and not a faulthandler dump. So
    each extraction on the expensive path writes a line before it starts and
    one after it finishes: a `thumb.begin` that is never followed by its
    `thumb.end` names the frame, the decoder and the worker that the process
    stopped inside.
    """

    def _cache(self, prefer_ffmpeg):
        cache = ThumbnailCache.__new__(ThumbnailCache)
        cache._prefer_ffmpeg = prefer_ffmpeg
        cache._src_w, cache._src_h = 8192, 4096
        cache._hwaccels = ["d3d11va"]
        cache._extract_one_frame = lambda key, out_path, cap=None, priority=0: cap
        cache._disk_path = lambda key: None
        return cache

    def test_a_vr_extraction_is_bracketed(self, app, tmp_path):
        from modules import repaint_trace

        repaint_trace.reset_for_tests()
        path = tmp_path / "trace.log"
        repaint_trace.arm(str(path))
        try:
            self._cache(True)._extract_one((12300, 54, True))
        finally:
            repaint_trace.reset_for_tests()

        text = path.read_text(encoding="utf-8")
        assert "thumb.begin" in text and "thumb.end" in text
        assert "t_ms=12300" in text and "vr=True" in text
        assert "hw='d3d11va'" in text          # which decoder was in use
        assert "src='8192x4096'" in text

    def test_ordinary_footage_stays_out_of_the_trace(self, app, tmp_path):
        # Normal video decodes thousands of thumbnails through the cheap path
        # and is not where this crash lives; logging them would bury the lines
        # that matter.
        from modules import repaint_trace

        repaint_trace.reset_for_tests()
        path = tmp_path / "trace.log"
        repaint_trace.arm(str(path))
        try:
            self._cache(False)._extract_one((12300, 54, False))
        finally:
            repaint_trace.reset_for_tests()

        assert "thumb.begin" not in path.read_text(encoding="utf-8")


class TestItSurvivesMissingPieces:
    def test_a_scene_with_no_thumbnail_source_can_still_be_toggled(self, app):
        scene = SignalTimelineScene({}, DURATION)
        scene.build_timeline()
        scene.set_vr_mode(True)          # must not raise
        assert scene._vr_mode is True

    def test_a_lane_deleted_by_a_rebuild_is_let_go_of(self, app):
        # The toggle reshapes the lane it can see; if a rebuild has deleted it
        # since, reaching through the wrapper is undefined behaviour.
        cache = FakeCache()
        scene = _scene(cache)
        scene.clear()                     # deletes the item behind the ref

        scene.set_vr_mode(True)

        assert scene._filmstrip_item is None
        assert cache.crops == [True]
