"""'More like these': ranking a video by confirmed matches, and grouping a
ranking into moments. Pure arithmetic on a frame index - no model, no video."""
import numpy as np
import pytest

from llm.clip_index import ClipFrameIndex, l2_normalize
from llm.more_like_these import (
    example_rows, moments, more_like, score_like,
)


def _index(seed=0):
    """60 frames, one a second. Every frame is mostly one shared direction (as
    frames of one video are); 10-14s and 40-44s add the same extra direction,
    25-29s adds a different one."""
    rng = np.random.default_rng(seed)
    dim = 32
    common, kind_a, kind_b = (l2_normalize(rng.normal(size=dim)) for _ in range(3))
    emb = np.tile(common * 5.0, (60, 1)) + rng.normal(scale=0.05, size=(60, dim))
    for t in list(range(10, 15)) + list(range(40, 45)):
        emb[t] += kind_a
    for t in range(25, 30):
        emb[t] += kind_b
    return ClipFrameIndex(np.arange(60, dtype=np.float32),
                          l2_normalize(emb.astype(np.float32)))


class TestScoreLike:
    def test_the_other_occurrence_outranks_everything_else(self):
        index = _index()
        scores = score_like(index, [12.0])
        elsewhere = np.delete(scores, list(range(10, 15)) + list(range(40, 45)))
        assert scores[40:45].min() > elsewhere.max()

    def test_raw_cosine_alone_could_not_have_told_them_apart(self):
        # Why the centring is there: uncentred, every frame is ~1.0 to the example.
        index = _index()
        raw = index.embeddings @ index.embeddings[12]
        assert raw.min() > 0.9

    def test_scores_are_between_zero_and_one(self):
        scores = score_like(_index(), [12.0, 41.0])
        assert scores.min() >= 0.0 and scores.max() <= 1.0

    def test_neighbouring_frames_join_the_example(self):
        index = _index()
        assert example_rows(index, [12.0]) == [11, 12, 13]
        assert example_rows(index, [12.0], spread=0.0) == [12]

    def test_row_order_of_the_index_does_not_matter(self):
        index = _index()
        shuffled = np.random.default_rng(1).permutation(60)
        other = ClipFrameIndex(index.timestamps[shuffled], index.embeddings[shuffled])
        np.testing.assert_allclose(score_like(other, [12.0]),
                                   score_like(index, [12.0])[shuffled], atol=1e-5)

    def test_an_example_outside_the_index_is_an_error(self):
        with pytest.raises(ValueError):
            score_like(_index(), [500.0])


class TestMoments:
    def test_neighbouring_frames_are_one_moment(self):
        ts = np.arange(20.0)
        scores = np.zeros(20)
        scores[3:7] = [0.9, 0.95, 0.8, 0.7]
        scores[15] = 0.6
        found = moments(ts, scores, top_k=2, gap=2.0)
        assert [(m.start, m.end, m.timestamp, m.frames) for m in found] == [
            (3.0, 6.0, 4.0, 4), (15.0, 15.0, 15.0, 1)]

    def test_top_k_counts_moments_not_frames(self):
        ts = np.arange(100.0)
        scores = np.zeros(100)
        scores[10:20] = 0.9                    # ten strong frames, one place
        scores[50], scores[80] = 0.5, 0.4
        found = moments(ts, scores, top_k=3, gap=2.0)
        assert [m.timestamp for m in found] == [10.0, 50.0, 80.0]

    def test_best_first(self):
        found = moments([0, 10, 20], [0.2, 0.9, 0.5], top_k=3, gap=2.0)
        assert [m.timestamp for m in found] == [10.0, 20.0, 0.0]

    def test_a_frame_between_two_moments_joins_them(self):
        found = moments([0, 4, 8], [0.9, 0.1, 0.8], top_k=3, gap=4.0)
        assert len(found) == 1
        assert (found[0].start, found[0].end, found[0].frames) == (0.0, 8.0, 3)

    def test_min_score_drops_weak_frames(self):
        found = moments([0, 10, 20], [0.2, 0.9, 0.5], top_k=3, gap=2.0, min_score=0.4)
        assert [m.timestamp for m in found] == [10.0, 20.0]

    def test_nothing_in_nothing_out(self):
        assert moments([], [], top_k=5) == []
        assert moments([1.0], [0.5], top_k=0) == []

    def test_lengths_must_agree(self):
        with pytest.raises(ValueError):
            moments([1.0, 2.0], [0.5], top_k=1)


class TestMoreLike:
    def test_finds_the_example_and_its_twin_first(self):
        found = more_like(_index(), [12.0], top_k=2)
        assert sorted((m.start, m.end) for m in found) == [(10.0, 14.0), (40.0, 44.0)]

    def test_since_leaves_out_the_part_before_it(self):
        found = more_like(_index(), [12.0], top_k=1, since=30.0)
        assert (found[0].start, found[0].end) == (40.0, 44.0)


class TestTextSearchReportsMoments:
    """The CLIP / SigLIP2 search hands back top_k places, not top_k frames."""

    def _worker(self, top_k, interval=1.0, margins=None):
        pytest.importorskip("PySide6")
        from types import SimpleNamespace
        from llm.llm_chat_widget import _VisualSearchWorker

        stub = SimpleNamespace(top_k=top_k, interval=interval,
                               _clip_margins=margins or {})
        return lambda scored: _VisualSearchWorker._best_moments(stub, scored)

    def test_a_long_match_takes_one_slot(self):
        scored = [(float(t), 0.9) for t in range(10, 20)] + [(50.0, 0.6), (80.0, 0.5)]
        best = self._worker(top_k=3)(scored)
        assert [ts for ts, _ in best] == [10.0, 50.0, 80.0]

    def test_margin_orders_frames_whose_score_rounded_to_one(self):
        scored = [(0.0, 1.0), (20.0, 1.0), (40.0, 1.0)]
        best = self._worker(top_k=2, margins={0.0: 5.0, 20.0: 9.0, 40.0: 7.0})(scored)
        assert best == [(20.0, 1.0), (40.0, 1.0)]

    def test_samples_further_apart_than_the_gap_still_join_when_adjacent(self):
        scored = [(0.0, 0.9), (10.0, 0.8), (50.0, 0.7)]
        best = self._worker(top_k=5, interval=10.0)(scored)
        assert [ts for ts, _ in best] == [0.0, 50.0]

    def test_no_frames_no_moments(self):
        assert self._worker(top_k=5)([]) == []


class TestRightClickOnTheVideo:
    """The video's own menu hands the playback time to the search panel, and
    says what happened where the user is looking."""

    def _window(self, chat):
        pytest.importorskip("PySide6")
        from types import SimpleNamespace

        said = []
        bar = SimpleNamespace(showMessage=lambda text, ms=0: said.append(("status", text)))
        return SimpleNamespace(llm_chat=chat, statusBar=lambda: bar), said

    def _ask(self, window, seconds, monkeypatch, said):
        import signal_timeline_viewer as stv

        monkeypatch.setattr(stv.QMessageBox, "information",
                            lambda parent, title, text: said.append(("box", text)))
        stv.SignalTimelineWindow._on_more_like_requested(window, seconds)

    def test_the_moment_goes_to_the_search_panel(self, monkeypatch):
        from types import SimpleNamespace

        asked = []
        chat = SimpleNamespace(
            more_like_moment=lambda ts: asked.append(ts) or (True, "found some"))
        window, said = self._window(chat)
        self._ask(window, 12.5, monkeypatch, said)
        assert asked == [12.5]
        assert said == [("status", "found some")]

    def test_a_refusal_is_shown_in_a_box_not_left_in_the_chat(self, monkeypatch):
        from types import SimpleNamespace

        chat = SimpleNamespace(more_like_moment=lambda ts: (False, "search first"))
        window, said = self._window(chat)
        self._ask(window, 12.5, monkeypatch, said)
        assert said == [("box", "search first")]

    def test_without_a_search_panel_it_says_so(self, monkeypatch):
        window, said = self._window(None)
        self._ask(window, 12.5, monkeypatch, said)
        assert [kind for kind, _ in said] == ["status"]

    def test_the_menu_entry_has_its_wording(self):
        from llm.more_like_these import MENU_TEXT

        assert "More like this" in MENU_TEXT


class TestItReadsTheVideoItself:
    """'More like this' does not ask for a search to be run first: where the
    frames are not all saved, it scans, and ranks when that is done."""

    def _widget(self, duration, interval=1.0):
        pytest.importorskip("PySide6")
        from types import SimpleNamespace

        return SimpleNamespace(
            _analyzer=SimpleNamespace(duration=duration),
            search_interval=SimpleNamespace(value=lambda: interval))

    def _covers(self, widget, index):
        from llm.llm_chat_widget import LLMChatWidget

        return LLMChatWidget._index_covers_video(widget, index)

    def test_no_index_does_not_cover_the_video(self):
        assert self._covers(self._widget(60.0), None) is False

    def test_an_index_of_the_last_part_only_does_not_cover_it(self):
        index = _index()                       # 0..59 s
        assert self._covers(self._widget(240.0), index) is False

    def test_a_whole_index_covers_it_even_short_of_the_last_frame(self):
        index = _index()                       # 0..59 s of a 60.4 s video
        assert self._covers(self._widget(60.4), index) is True

    def test_the_scan_can_be_asked_for_the_frames_alone(self):
        pytest.importorskip("PySide6")
        from types import SimpleNamespace
        from llm.llm_chat_widget import _VisualSearchWorker

        sent = []
        worker = SimpleNamespace(
            index_only=True, stop_on_first_match=False,
            _clip_scan=lambda: [(1.0, 0.9), (2.0, 0.8)],
            finished=SimpleNamespace(emit=sent.append),
            found=SimpleNamespace(emit=lambda *a: sent.append("found")),
            frame_analyzed=SimpleNamespace(emit=lambda *a: sent.append("frame")))
        _VisualSearchWorker._run_clip(worker)
        assert sent == [[]]                    # no matches reported, just done


class TestWholeVideoScansReadInOrder:
    """A scan that wants most of a stretch reads the video front to back
    instead of seeking to each sample; a few scattered misses still seek."""

    def _worker(self, memo, interval=1.0):
        pytest.importorskip("PySide6")
        from types import SimpleNamespace
        from llm.llm_chat_widget import _VisualSearchWorker

        class Encoder:
            def embed_frames_bgr(self, frames):
                return np.stack([np.eye(32, dtype=np.float32)[int(f[0, 0, 0]) % 32]
                                 for f in frames])

        said = []
        worker = SimpleNamespace(
            _clip_memo=memo, _clip_memo_added=0, _clip=Encoder(), interval=interval,
            _engine_name="CLIP", analyzer=SimpleNamespace(video_path="video.mp4"),
            _cancel_token=SimpleNamespace(is_cancelled=False),
            progress=SimpleNamespace(emit=lambda *a: said.append(a)),
            IN_ORDER_MIN_SHARE=_VisualSearchWorker.IN_ORDER_MIN_SHARE,
            IN_ORDER_MIN_FRAMES=_VisualSearchWorker.IN_ORDER_MIN_FRAMES)
        return worker, lambda ts: _VisualSearchWorker._fill_memo_in_order(worker, ts)

    def _video(self, monkeypatch, fps=10.0, frames=1000):
        """A stand-in video whose frame number is its pixel value."""
        import cv2
        from modules.media import video_decode

        opened = []

        class Cap:
            at = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return fps if prop == cv2.CAP_PROP_FPS else float(self.at)

            def set(self, prop, value):
                self.at = int(round(value / 1000.0 * fps))

            def grab(self):
                self.at += 1
                return self.at <= frames

            def read(self):
                frame = np.full((2, 2, 3), self.at // int(fps), dtype=np.uint8)
                self.at += 1
                return self.at <= frames, frame

            def release(self):
                pass

        def open_sparse(path, choice=None):
            opened.append(path)
            return Cap(), "cpu"

        monkeypatch.setattr(video_decode, "open_sparse", open_sparse)
        return opened

    def _empty(self):
        return ClipFrameIndex(np.zeros(0, dtype=np.float32),
                              np.zeros((0, 32), dtype=np.float32))

    def test_every_wanted_frame_is_read_and_filed_under_its_own_second(self, monkeypatch):
        opened = self._video(monkeypatch)
        memo = self._empty()
        worker, fill = self._worker(memo)
        wanted = [float(t) for t in range(60)]
        fill(wanted)
        assert opened == ["video.mp4"]
        assert sorted(memo.timestamps.tolist()) == wanted
        assert worker._clip_memo_added == 60
        # The frame filed at second t is the frame of second t.
        for t in (0.0, 7.0, 59.0):
            row = memo.lookup([t])[0][0]
            assert int(np.argmax(memo.embeddings[row])) == int(t) % 32

    def test_only_the_gap_is_read(self, monkeypatch):
        self._video(monkeypatch)
        memo = self._empty()
        worker, fill = self._worker(memo)
        memo.extend([float(t) for t in range(50, 100)],
                    np.zeros((50, 32), dtype=np.float32))
        fill([float(t) for t in range(100)])
        assert worker._clip_memo_added == 50
        assert len(memo) == 100

    def test_a_few_scattered_misses_are_left_to_the_seek(self, monkeypatch):
        opened = self._video(monkeypatch)
        memo = self._empty()
        _, fill = self._worker(memo)
        fill([float(t) for t in range(0, 4000, 100)])     # 40 frames, far apart
        assert opened == [] and len(memo) == 0

    def test_a_short_scan_is_left_to_the_seek(self, monkeypatch):
        opened = self._video(monkeypatch)
        _, fill = self._worker(self._empty())
        fill([float(t) for t in range(10)])
        assert opened == []
