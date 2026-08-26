"""The decoder that keeps the file open, and keeps its own process.

Speed is why it exists — a thumbnail costs 54ms through a decoder that stays
open against 356ms through one started for the occasion, because opening a 5 GB
file and creating a GPU device dwarf decoding a frame. But speed is not what
these tests are about. The design deliberately spends a whole child process to
keep one property the thing it replaces had for free: **when the decoder dies,
only the decoder dies**. That is the property worth pinning, since it is the
one that would quietly disappear the day somebody decides the extra process
looks wasteful.

So: a child that cannot start, a child that dies mid-life, and a decoder shared
by both timelines rather than opened twice for one video.
"""

from __future__ import annotations

import os
import sys
import threading
import time

import pytest

# conftest is loaded by pytest rather than imported, so its helpers are not on
# the path by default.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv          # noqa: E402

from video_ai_editor import thumbnail_decoder as td


def cache_priorities():
    """The cache's priority constants → (hover, visible, prefetch).

    Fetched inside the tests that need them rather than imported up here:
    `thumbnail_cache` pulls in PySide6, which the fast CI environment does not
    install (it is not among conftest's shimmed heavy deps either — a missing
    Qt is an ImportError, not a MagicMock). The decoder itself needs no Qt at
    all, and neither does most of what is tested here, so importing it at
    module level would take the whole file down with it.
    """
    pytest.importorskip("PySide6")
    from video_ai_editor.thumbnail_cache import (
        PRIORITY_HOVER, PRIORITY_PREFETCH, PRIORITY_VISIBLE)
    return PRIORITY_HOVER, PRIORITY_VISIBLE, PRIORITY_PREFETCH


class TestTheShapeItProduces:
    """Matching `scale=-2:h`, so the two extraction paths are interchangeable."""

    def test_a_16_9_source(self):
        assert td.thumb_width(1920, 1080, 54, vr=False) == 96

    def test_side_by_side_gives_one_eye(self):
        # 5760×2880 is 2:1 whole, square per eye — so at 54 high the eye is 54
        # wide, not 108.
        assert td.thumb_width(5760, 2880, 54, vr=True) == 54
        assert td.thumb_width(5760, 2880, 54, vr=False) == 108

    def test_widths_are_even(self):
        # Odd widths are a swscale trap, and `-2:h` in the ffmpeg path rounds
        # the same way — the two must not disagree about the size of a frame.
        for src_w in range(1000, 1030):
            assert td.thumb_width(src_w, 1080, 54, vr=False) % 2 == 0

    @pytest.mark.parametrize("args", [(0, 0, 54, False), (1920, 0, 54, False),
                                      (1920, 1080, 0, False)])
    def test_nonsense_in_does_not_divide_by_zero(self, args):
        assert td.thumb_width(*args) >= 2


class TestWhenTheDecoderCannotRun:
    def test_a_file_it_cannot_open_is_reported_once_and_dropped(self, tmp_path):
        # The child opens the video before answering anything. If that fails
        # there is no point starting another one for every frame of a video
        # that is not going to become readable.
        decoder = td.PersistentDecoder(str(tmp_path / "not-a-video.mp4"))
        try:
            assert decoder.extract(1000, 54, False, tmp_path / "out.jpg") is False
            assert decoder.available is False
            assert decoder.extract(2000, 54, False, tmp_path / "out.jpg") is False
            assert decoder.restarts == 1, "kept restarting a hopeless child"
        finally:
            decoder.stop()

    def test_a_death_between_requests_is_still_counted(self, tmp_path):
        # A child that dies while nothing is being asked of it is noticed at
        # the next request, when it is simply no longer alive. Counting only
        # the deaths caught mid-request would report zero for a decoder that
        # dies every single time -- and that count is the log line somebody
        # reads when they are trying to explain a crash.
        decoder = td.PersistentDecoder("whatever.mp4")

        class Corpse:
            exitcode = -1

            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        decoder._process = Corpse()
        decoder.available = False       # stop before it spawns a real child
        decoder._start()

        assert decoder.restarts == 1

    def test_an_unavailable_decoder_answers_without_starting_anything(self, tmp_path):
        decoder = td.PersistentDecoder("whatever.mp4")
        decoder.available = False
        assert decoder.extract(0, 54, False, tmp_path / "out.jpg") is False
        assert decoder._process is None

    def test_stopping_one_that_never_ran_is_fine(self):
        td.PersistentDecoder("whatever.mp4").stop()


class TestTheFrameUnderTheCursorGoesFirst:
    """Measured: a hover costs 70ms idle and 300ms while the strip loads.

    The child decodes one frame at a time, so whoever is waiting when it comes
    free decides what you feel. A plain lock hands out its turn in arrival
    order, which puts the frame under the cursor behind however many filmstrip
    slots were queued first — and the queue is deepest exactly when someone is
    scrubbing across a strip that is still filling in.
    """

    def _order(self, priorities):
        """The order `priorities` are let through when they all wait at once."""
        turnstile = td._Turnstile()
        turnstile.acquire(0)                  # somebody is already decoding
        started, done = threading.Barrier(len(priorities) + 1), []

        def wait(priority):
            started.wait()
            turnstile.acquire(priority)
            done.append(priority)
            turnstile.release()

        threads = [threading.Thread(target=wait, args=(p,), daemon=True)
                   for p in priorities]
        for t in threads:
            t.start()
        started.wait()
        time.sleep(0.15)                      # let them all queue up
        turnstile.release()                   # the in-flight frame finishes
        for t in threads:
            t.join(timeout=10)
        return done

    def test_a_hover_overtakes_queued_filmstrip_work(self):
        hover, visible, prefetch = cache_priorities()

        order = self._order([prefetch, visible, prefetch, hover])

        assert order[0] == hover
        assert order == sorted(order)

    def test_the_ordering_matches_what_the_cache_queues_with(self):
        # The decoder orders by the same numbers the cache's own queue uses;
        # if those stopped meaning "hover first" this would be sorting by
        # something arbitrary.
        hover, visible, prefetch = cache_priorities()
        assert hover < visible < prefetch

    def test_equal_priorities_keep_their_place_in_the_queue(self):
        # Without a tiebreaker the heap would compare whatever came next and
        # reorder same-priority work arbitrarily, so a filmstrip would fill in
        # out of order for no reason.
        turnstile = td._Turnstile()
        for _ in range(3):
            turnstile.acquire(0)
            turnstile.release()

    def test_it_lets_go_even_when_the_request_fails(self, tmp_path):
        # A decoder that cannot run must not leave the turnstile shut, or every
        # later request waits forever on a child that is never coming.
        decoder = td.PersistentDecoder("nothing.mp4")
        decoder.available = False
        for _ in range(3):
            assert decoder.extract(0, 54, False, tmp_path / "x.jpg") is False
        decoder.available = True
        decoder._closed = True
        assert decoder.extract(0, 54, False, tmp_path / "x.jpg") is False


class TestOneDecoderPerVideo:
    """Both timelines cache thumbnails of the same file.

    Two children would mean the same video open twice and, on the hardware
    path, two GPU devices for one filmstrip.
    """

    def setup_method(self):
        td._decoders.clear()
        td._users.clear()

    def test_the_second_caller_gets_the_same_one(self):
        assert td.acquire("a.mp4") is td.acquire("a.mp4")

    def test_different_videos_get_their_own(self):
        assert td.acquire("a.mp4") is not td.acquire("b.mp4")

    def test_it_survives_the_first_caller_letting_go(self):
        first = td.acquire("a.mp4")
        td.acquire("a.mp4")
        td.release("a.mp4")
        assert td.acquire("a.mp4") is first, "closed a decoder still in use"

    def test_the_last_one_out_closes_it(self):
        td.acquire("a.mp4")
        td.acquire("a.mp4")
        td.release("a.mp4")
        td.release("a.mp4")
        assert "a.mp4" not in td._decoders

    def test_releasing_something_never_acquired_is_ignored(self):
        td.release("never-had-it.mp4")


class TestItActuallyDecodes:
    """One real round trip, because everything above is about the plumbing."""

    def test_a_frame_comes_back_the_size_it_was_asked_for(self, tmp_path):
        av = pytest.importorskip("av")
        cv2 = real_opencv()
        if cv2 is None:
            pytest.skip("needs a real OpenCV to build the fixture")

        import numpy as np
        path = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10.0, (640, 320))
        if not writer.isOpened():
            pytest.skip("no encoder available for the fixture")
        rng = np.random.default_rng(4)
        try:
            for _ in range(30):
                writer.write(rng.integers(0, 255, (320, 640, 3), dtype=np.uint8))
        finally:
            writer.release()

        decoder = td.PersistentDecoder(str(path))
        out = tmp_path / "thumb.jpg"
        try:
            assert decoder.extract(500, 54, False, out) is True, "no frame came back"
        finally:
            decoder.stop()

        image = cv2.imread(str(out))
        assert image is not None
        assert image.shape[0] == 54
        assert image.shape[1] == td.thumb_width(640, 320, 54, vr=False)

    def test_the_left_eye_is_half_as_wide(self, tmp_path):
        pytest.importorskip("av")
        cv2 = real_opencv()
        if cv2 is None:
            pytest.skip("needs a real OpenCV to build the fixture")

        import numpy as np
        path = tmp_path / "sbs.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10.0, (640, 320))
        if not writer.isOpened():
            pytest.skip("no encoder available for the fixture")
        rng = np.random.default_rng(5)
        try:
            for _ in range(30):
                writer.write(rng.integers(0, 255, (320, 640, 3), dtype=np.uint8))
        finally:
            writer.release()

        decoder = td.PersistentDecoder(str(path))
        out = tmp_path / "eye.jpg"
        try:
            assert decoder.extract(500, 54, True, out) is True
        finally:
            decoder.stop()

        assert cv2.imread(str(out)).shape[1] == td.thumb_width(640, 320, 54, vr=True)
