"""
Cancelling a transcript stops it now, not eventually.

Whisper decodes a chunk of up to ten minutes in a single call and knows nothing
about cancelling, and `get_transcript_segments` took no cancel at all — so
Cancel set a flag that was next looked at once the whole transcript had
finished. On a feature-length video that is the entire wait, with the button
already pressed and the log already saying "cancelling…".

The per-window hook the progress bar rides on is the place a cancel can land,
so these check it lands there, that nothing keeps running afterwards, and that
the copied audio chunks are cleaned up on the way out.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from modules import transcript as tr
from modules import analysis_ondemand as aod


SEGMENT = {"start": 0.0, "end": 2.0, "text": "a real sentence here"}


@pytest.fixture
def fake_whisper_transcribe(monkeypatch):
    mod = types.ModuleType("whisper.transcribe")
    mod.tqdm = "the-real-tqdm"
    monkeypatch.setitem(sys.modules, "whisper.transcribe", mod)

    def shadowing_function(*a, **k):
        raise AssertionError("not the module")

    monkeypatch.setattr(tr.whisper, "transcribe", shadowing_function, raising=False)
    return mod


class _Model:
    """Drives the patched bar the way Whisper's decode loop does."""

    def __init__(self, steps=6, on_step=None):
        self.steps = steps
        self.on_step = on_step
        self.windows_decoded = 0

    def transcribe(self, chunk, **params):
        wt = sys.modules["whisper.transcribe"]
        with wt.tqdm.tqdm(total=self.steps, unit="frames", disable=False) as bar:
            for _ in range(self.steps):
                self.windows_decoded += 1
                if self.on_step:
                    self.on_step()
                bar.update(1)
        return {"language": "en", "segments": [SEGMENT]}


@pytest.fixture
def run(monkeypatch, tmp_path, fake_whisper_transcribe):
    """`get_transcript_segments` over three real (empty) chunk files."""
    state = {"chunks": [], "model": None}

    def fake_split(video_file, chunk_length=600, should_cancel=None):
        paths = []
        for i in range(3):
            p = tmp_path / f"movie_chunk_{i:03d}.wav"
            p.write_bytes(b"RIFF")
            paths.append(str(p))
        state["chunks"] = paths
        return paths

    monkeypatch.setattr(tr, "split_audio", fake_split)
    monkeypatch.setattr(tr.torch.cuda, "is_available", lambda: False, raising=False)

    def go(should_cancel=None, steps=6, on_step=None, **kwargs):
        model = _Model(steps=steps, on_step=on_step)
        state["model"] = model
        monkeypatch.setattr(tr.whisper, "load_model", lambda *a, **k: model,
                            raising=False)
        return tr.get_transcript_segments(
            str(tmp_path / "movie.mp4"), log_fn=lambda *a: None,
            should_cancel=should_cancel, **kwargs)

    state["go"] = go
    return state


class TestCancelLandsInsideAChunk:
    def test_raises_part_way_through_the_first_chunk(self, run):
        cancelled = {"now": False}

        def on_step():
            cancelled["now"] = True        # pressed during window 1

        with pytest.raises(tr.TranscriptionCancelled):
            run["go"](should_cancel=lambda: cancelled["now"], on_step=on_step)

    def test_stops_decoding_rather_than_finishing_the_chunk(self, run):
        cancelled = {"now": False}

        def on_step():
            cancelled["now"] = True

        with pytest.raises(tr.TranscriptionCancelled):
            run["go"](should_cancel=lambda: cancelled["now"], steps=50,
                      on_step=on_step)
        # One window decoded, then the cancel was seen — not all 50, and not
        # the two chunks that would have followed.
        assert run["model"].windows_decoded == 1

    def test_a_flag_already_set_never_reaches_whisper(self, run):
        # Cancelled before the run started: it should not even load or split.
        with pytest.raises(tr.TranscriptionCancelled):
            run["go"](should_cancel=lambda: True)
        assert run["model"].windows_decoded == 0
        assert run["chunks"] == [], "nothing should have been split"

    def test_a_cancel_between_chunks_stops_at_that_boundary(self, run):
        # False until the audio is split, then true: the loop aborts on entry.
        with pytest.raises(tr.TranscriptionCancelled):
            run["go"](should_cancel=lambda: bool(run["chunks"]))
        assert run["model"].windows_decoded == 0

    def test_cancelled_is_a_runtime_error(self):
        # pipeline.py's transcript step catches RuntimeError to abort the run;
        # anything else there is treated as "failed, carry on with no
        # transcript", which is not what cancelling means.
        assert issubclass(tr.TranscriptionCancelled, RuntimeError)


class TestChunkFilesAreNotLeftBehind:
    def test_cleaned_up_after_a_cancel(self, run):
        # Cancelled once the chunks are on disk — gigabytes of .wav next to the
        # video is not what pressing Cancel should leave behind.
        with pytest.raises(tr.TranscriptionCancelled):
            run["go"](should_cancel=lambda: bool(run["chunks"]))
        assert run["chunks"], "the fixture should have made chunk files"
        assert not any(os.path.exists(p) for p in run["chunks"])

    def test_cleaned_up_after_a_normal_run(self, run):
        run["go"]()
        assert not any(os.path.exists(p) for p in run["chunks"])

    def test_kept_when_cleanup_is_off(self, run):
        run["go"](cleanup=False)
        assert all(os.path.exists(p) for p in run["chunks"])


class TestUncancelledRunsAreUnchanged:
    def test_no_predicate_transcribes_everything(self, run):
        segments = run["go"]()
        assert len(segments) == 3          # one per chunk

    def test_a_predicate_that_never_fires_changes_nothing(self, run):
        segments = run["go"](should_cancel=lambda: False)
        assert len(segments) == 3
        assert run["model"].windows_decoded == 18   # 6 windows x 3 chunks


class TestTheOnDemandRunnerTranslatesIt:
    def test_run_transcript_reports_a_cancel_as_cancelled(self, monkeypatch, tmp_path):
        """The viewer and main window both key off aod._Cancelled."""
        import threading

        def fake_get(*a, **k):
            raise tr.TranscriptionCancelled("stopped")

        monkeypatch.setattr(tr, "get_transcript_segments", fake_get)
        monkeypatch.setattr(aod, "analysis_defaults",
                            lambda: {"whisper_model": "base", "language": "en"})
        event = threading.Event()
        event.set()
        with pytest.raises(aod._Cancelled):
            aod.run_transcript(str(tmp_path / "movie.mp4"), cancel=event,
                               log=lambda *a: None)

    def test_the_flag_is_handed_down(self, monkeypatch, tmp_path):
        import threading
        seen = {}

        def fake_get(*a, should_cancel=None, **k):
            seen["predicate"] = should_cancel
            return []

        monkeypatch.setattr(tr, "get_transcript_segments", fake_get)
        monkeypatch.setattr(aod, "analysis_defaults",
                            lambda: {"whisper_model": "base", "language": "en"})
        monkeypatch.setattr(aod, "_write_transcript_sidecar", lambda *a, **k: None)
        event = threading.Event()
        aod.run_transcript(str(tmp_path / "movie.mp4"), cancel=event,
                           log=lambda *a: None)
        assert seen["predicate"] is not None
        assert seen["predicate"]() is False
        event.set()
        assert seen["predicate"]() is True
