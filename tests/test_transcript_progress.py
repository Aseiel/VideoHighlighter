"""
Progress reporting out of `modules.transcript`.

Transcription is the longest single step in a run, and in the packaged
`--windowed` build Whisper's own tqdm bar writes to a stderr that goes
nowhere. The GUI's bar is therefore the only thing the user has, so what
matters here is that `progress_fn` keeps being called *while a chunk decodes*
— not just at chunk boundaries, which for a sub-10-minute video means once.

These tests never load Whisper: they stand in a fake `whisper.transcribe`
module and a fake model that drives the patched bar the way the real decode
loop does.
"""

from __future__ import annotations

import sys
import types

import pytest

from modules import transcript as tr


# --------------------------------------------------------------------------- #
# The tqdm stand-in
# --------------------------------------------------------------------------- #
class TestForwardingBar:
    def test_updates_become_fractions_of_total(self):
        seen = []
        bar = tr._ForwardingBar(total=200, on_frac=seen.append)
        bar.update(50)
        bar.update(50)
        assert seen == [0.25, 0.5]

    def test_accepts_the_kwargs_whisper_passes(self):
        # whisper calls tqdm.tqdm(total=..., unit="frames", disable=...)
        bar = tr._ForwardingBar(total=10, unit="frames", on_frac=lambda f: None)
        assert bar.total == 10

    def test_fraction_is_clamped(self):
        seen = []
        bar = tr._ForwardingBar(total=10, on_frac=seen.append)
        bar.update(999)
        assert seen == [1.0]

    def test_no_total_reports_nothing(self):
        seen = []
        bar = tr._ForwardingBar(total=0, on_frac=seen.append)
        bar.update(5)
        assert seen == []

    def test_works_as_a_context_manager(self):
        seen = []
        with tr._ForwardingBar(total=4, on_frac=seen.append) as bar:
            bar.update(2)
        assert seen == [0.5]


# --------------------------------------------------------------------------- #
# Patching whisper's namespace
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_whisper_transcribe(monkeypatch):
    """A stand-in `whisper.transcribe` module holding a sentinel `tqdm`.

    Shaped like the real thing, shadowing included: `whisper/__init__.py` does
    `from .transcribe import transcribe`, so the *attribute* `whisper.transcribe`
    is the function while the module lives only in `sys.modules`. Code that
    reaches for the attribute finds a function with no `tqdm` on it and silently
    does nothing — which is what shipped the first time.
    """
    mod = types.ModuleType("whisper.transcribe")
    mod.tqdm = "the-real-tqdm"
    monkeypatch.setitem(sys.modules, "whisper.transcribe", mod)

    def transcribe(*args, **kwargs):        # the shadowing function
        raise AssertionError("not the module")

    monkeypatch.setattr(tr.whisper, "transcribe", transcribe, raising=False)
    return mod


class TestWhisperProgressPatch:
    def test_bar_is_swapped_in_and_restored(self, fake_whisper_transcribe):
        with tr._whisper_progress(lambda f: None):
            assert fake_whisper_transcribe.tqdm != "the-real-tqdm"
        assert fake_whisper_transcribe.tqdm == "the-real-tqdm"

    def test_restored_even_when_transcription_raises(self, fake_whisper_transcribe):
        with pytest.raises(RuntimeError):
            with tr._whisper_progress(lambda f: None):
                raise RuntimeError("decode blew up")
        assert fake_whisper_transcribe.tqdm == "the-real-tqdm"

    def test_shim_builds_a_forwarding_bar(self, fake_whisper_transcribe):
        seen = []
        with tr._whisper_progress(seen.append):
            bar = fake_whisper_transcribe.tqdm.tqdm(
                total=10, unit="frames", disable=False)
            bar.update(5)
        assert seen == [0.5]

    def test_no_callback_leaves_whisper_alone(self, fake_whisper_transcribe):
        with tr._whisper_progress(None):
            assert fake_whisper_transcribe.tqdm == "the-real-tqdm"

    def test_reaches_the_module_the_function_shadows(self, fake_whisper_transcribe):
        """The bug this whole mechanism shipped with, pinned.

        `whisper.transcribe` the attribute is a function; the module holding
        the `tqdm` name is only in `sys.modules`. Resolving the attribute finds
        no `tqdm`, skips the swap, and leaves the bar exactly as coarse as it
        was — with every test passing, because a fixture that hands back a
        module cannot see the difference.
        """
        assert callable(getattr(tr.whisper, "transcribe"))  # the shadow is in place
        with tr._whisper_progress(lambda f: None):
            assert fake_whisper_transcribe.tqdm != "the-real-tqdm"

    def test_missing_whisper_internals_are_survivable(self, monkeypatch):
        """A future Whisper that no longer looks like this must still transcribe."""
        bare = types.ModuleType("whisper.transcribe")  # no tqdm
        monkeypatch.setitem(sys.modules, "whisper.transcribe", bare)
        with tr._whisper_progress(lambda f: None):
            pass  # no raise


# --------------------------------------------------------------------------- #
# What the GUI actually receives
# --------------------------------------------------------------------------- #
class _FakeModel:
    """Drives the patched bar the way Whisper's decode loop does."""

    def __init__(self, steps=4):
        self.steps = steps

    def transcribe(self, chunk, **params):
        # Whisper's decode loop reads `tqdm` out of its own module globals,
        # which is the sys.modules entry — not the shadowed attribute.
        wt = sys.modules["whisper.transcribe"]
        with wt.tqdm.tqdm(total=self.steps, unit="frames", disable=False) as bar:
            for _ in range(self.steps):
                bar.update(1)
        return {"language": "en", "segments": [
            {"start": 0.0, "end": 2.0, "text": "a real sentence here"}]}


@pytest.fixture
def two_chunk_run(monkeypatch, fake_whisper_transcribe):
    """`get_transcript_segments` over two fake chunks, collecting progress."""
    monkeypatch.setattr(tr.whisper, "load_model", lambda *a, **k: _FakeModel(),
                        raising=False)
    monkeypatch.setattr(tr, "split_audio", lambda *a, **k: ["c0.wav", "c1.wav"])
    monkeypatch.setattr(tr.torch.cuda, "is_available", lambda: False,
                        raising=False)

    calls = []

    def run(**kwargs):
        tr.get_transcript_segments(
            "movie.mp4",
            progress_fn=lambda c, t, task, det: calls.append((c, t, task, det)),
            log_fn=lambda *a: None,
            cleanup=False,
            **kwargs,
        )
        return calls

    return run


class TestProgressDuringTranscription:
    def test_reports_while_a_chunk_decodes(self, two_chunk_run):
        calls = two_chunk_run()
        inner = [c for c in calls if "%" in c[3]]
        # Four decode steps per chunk, two chunks: the bar moves during a chunk,
        # which is the whole point — chunk boundaries alone would give 2 updates.
        assert len(inner) == 8

    def test_percentages_never_go_backwards(self, two_chunk_run):
        pcts = [c[0] for c in two_chunk_run()]
        assert pcts == sorted(pcts)

    def test_stays_within_the_bar(self, two_chunk_run):
        for current, total, _task, _det in two_chunk_run():
            assert total == 100
            assert 0 <= current <= 100

    def test_second_chunk_starts_where_the_first_ended(self, two_chunk_run):
        calls = two_chunk_run()
        first = [c[0] for c in calls if c[3].startswith("Chunk 1/")]
        second = [c[0] for c in calls if c[3].startswith("Chunk 2/")]
        assert first and second
        assert max(first) <= min(second)

    def test_names_the_chunk_it_is_on(self, two_chunk_run):
        details = [c[3] for c in two_chunk_run()]
        assert any(d.startswith("Chunk 1/2") for d in details)
        assert any(d.startswith("Chunk 2/2") for d in details)

    def test_says_something_before_the_first_chunk(self, two_chunk_run):
        # Model load and audio split happen before any chunk exists, and on a
        # long video they are not instant.
        details = [c[3] for c in two_chunk_run()]
        assert "Splitting audio..." in details
        assert any("Loading Whisper" in d for d in details)

    def test_ends_complete(self, two_chunk_run):
        calls = two_chunk_run()
        assert calls[-1][3] == "Complete"

    def test_no_progress_fn_is_fine(self, monkeypatch, fake_whisper_transcribe):
        monkeypatch.setattr(tr.whisper, "load_model", lambda *a, **k: _FakeModel(),
                            raising=False)
        monkeypatch.setattr(tr, "split_audio", lambda *a, **k: ["c0.wav"])
        monkeypatch.setattr(tr.torch.cuda, "is_available", lambda: False,
                            raising=False)
        segs = tr.get_transcript_segments(
            "movie.mp4", progress_fn=None, log_fn=lambda *a: None, cleanup=False)
        assert len(segs) == 1
