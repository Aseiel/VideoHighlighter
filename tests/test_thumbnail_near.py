"""Filmstrip slots answered from the nearest keyframe: a request may say how
far off its frame may be, and one that does can be served about eight times
cheaper than the exact frame - but never to a request that wanted it exact."""
from collections import OrderedDict

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from video_ai_editor import thumbnail_decoder  # noqa: E402
from video_ai_editor.thumbnail_cache import ThumbnailCache  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Queue(list):
    def put(self, item):
        self.append(item)


def _cache(tmp_path):
    """A cache with no workers and no video: only the bookkeeping under test."""
    import itertools
    import threading

    cache = ThumbnailCache.__new__(ThumbnailCache)
    cache._mem = OrderedDict()
    cache.mem_limit = 50
    cache._vr_mode = ThumbnailCache._make_key(_Probe(), 0.0, 1)[2]
    cache.disk_dir = tmp_path
    cache._near, cache._mem_near, cache._within = {}, set(), {}
    cache._pending, cache._active = {}, set()
    cache._queue, cache._seq = _Queue(), itertools.count()
    cache._lock = threading.Lock()
    return cache


class _Probe:
    """Just enough of a cache to ask what an unset view is called."""
    _vr_mode = False


def _picture(path):
    pix = QPixmap(8, 8)
    pix.fill()
    assert pix.save(str(path), "JPG")


class TestTheChildSaysHowFarTheKeyframeWas:
    def test_a_plain_request_gets_the_reply_it_always_got(self):
        assert thumbnail_decoder._answer("software", 1200.0, None) == ("ok", "software")

    def test_a_keyframe_inside_the_limit_is_ok_with_its_distance(self):
        assert thumbnail_decoder._answer("d3d11va", 400.0, 500.0) == ("ok", "d3d11va", 400.0)

    def test_a_keyframe_outside_the_limit_is_far(self):
        assert thumbnail_decoder._answer("d3d11va", 900.0, 500.0) == ("far", "d3d11va", 900.0)

    def test_an_unknown_distance_is_never_taken_on_trust(self):
        assert thumbnail_decoder._answer("d3d11va", None, 500.0)[0] == "far"


class TestARequestSaysHowFarOffItsFrameMayBe:
    def _with_a_keyframe_thumb(self, tmp_path, distance):
        cache = _cache(tmp_path)
        key = cache._make_key(10.0, 54)
        _picture(cache._near_path(key, distance))
        cache._near[key] = distance
        return cache, key

    def test_a_slot_wide_enough_takes_the_keyframe_thumb(self, app, tmp_path):
        cache, key = self._with_a_keyframe_thumb(tmp_path, 3.0)
        assert cache.request(10.0, 54, tolerance=5.0) is not None
        assert key in cache._mem_near
        assert cache._queue == []                 # nothing left to decode

    def test_an_exact_request_does_not_take_it(self, app, tmp_path):
        cache, key = self._with_a_keyframe_thumb(tmp_path, 3.0)
        assert cache.request(10.0, 54) is None
        assert [k for _, _, k in cache._queue] == [key]
        assert cache._within[key] == 0.0          # and is decoded exactly

    def test_a_narrower_slot_does_not_take_it_either(self, app, tmp_path):
        cache, key = self._with_a_keyframe_thumb(tmp_path, 3.0)
        assert cache.request(10.0, 54, tolerance=1.0) is None
        assert cache._within[key] == 1.0

    def test_it_stands_in_while_the_exact_frame_is_decoded(self, app, tmp_path):
        cache, key = self._with_a_keyframe_thumb(tmp_path, 3.0)
        shown = cache.request(10.0, 54, tolerance=5.0)
        assert cache.request(10.0, 54) is shown   # better than an empty slot
        assert [k for _, _, k in cache._queue] == [key]

    def test_the_strictest_request_for_a_key_decides(self, app, tmp_path):
        cache = _cache(tmp_path)
        key = cache._make_key(10.0, 54)
        cache.request(10.0, 54, tolerance=5.0)
        cache.request(10.0, 54, tolerance=0.0)
        assert cache._within[key] == 0.0

    def test_the_exact_frame_wins_once_it_is_there(self, app, tmp_path):
        cache, key = self._with_a_keyframe_thumb(tmp_path, 3.0)
        cache.request(10.0, 54, tolerance=5.0)
        _picture(cache._disk_path(key))
        assert cache.request(10.0, 54) is not None
        assert key not in cache._mem_near


class TestKeyframeThumbsAreFoundAgainNextSession:
    def test_they_are_read_back_with_their_distance(self, app, tmp_path):
        cache = _cache(tmp_path)
        key = cache._make_key(10.0, 54)
        _picture(cache._near_path(key, 2.5))
        _picture(cache._disk_path(cache._make_key(20.0, 54)))     # an exact one

        assert cache._read_near_from_disk() == {key: 2.5}

    def test_they_never_sit_where_an_exact_thumb_would(self, app, tmp_path):
        cache = _cache(tmp_path)
        key = cache._make_key(10.0, 54)
        assert cache._near_path(key, 2.5) != cache._disk_path(key)
