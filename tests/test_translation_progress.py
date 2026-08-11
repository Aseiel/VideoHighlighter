"""
Progress out of the subtitle translation pass.

Translating is the *second* long pass a subtitle run makes: an hour of speech
is hundreds of LLM batches, each a separate ollama call. The batches were only
ever printed to the debug log, so the GUI bar sat still through all of it. These
pin that a caller-supplied callback is driven, and that the two passes divide
the bar between them instead of each sweeping it 0-100 in turn.
"""

from __future__ import annotations

import pytest

from modules import analysis_ondemand as aod
import modules.transcript_srt as srt


def _texts(n):
    return [f"line {i}" for i in range(n)]


@pytest.fixture
def ollama(monkeypatch):
    """A stand-in `ollama run` that echoes numbered lines back."""
    class _Result:
        returncode = 0

        def __init__(self, n):
            self.stdout = "\n".join(f"{i+1}. translated {i}" for i in range(n))

    def fake_run(cmd, **kwargs):
        prompt = cmd[-1]
        n = sum(1 for line in prompt.splitlines() if line.strip()[:1].isdigit())
        return _Result(n)

    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)


class TestBatchTranslationReports:
    def test_one_call_per_batch(self, ollama):
        seen = []
        srt.translate_batch_with_llm(
            _texts(25), batch_size=10,
            progress_fn=lambda c, t, task, det: seen.append((c, t, det)))
        # 3 batches + a closing call
        assert len(seen) == 4

    def test_names_the_batch(self, ollama):
        seen = []
        srt.translate_batch_with_llm(
            _texts(25), batch_size=10,
            progress_fn=lambda c, t, task, det: seen.append(det))
        assert seen[0] == "Batch 1/3"
        assert seen[2] == "Batch 3/3"

    def test_counts_segments_not_batches(self, ollama):
        seen = []
        srt.translate_batch_with_llm(
            _texts(25), batch_size=10,
            progress_fn=lambda c, t, task, det: seen.append((c, t)))
        assert seen[0] == (0, 25)
        assert seen[1] == (10, 25)
        assert seen[-1] == (25, 25)

    def test_task_is_named_for_the_bar(self, ollama):
        seen = []
        srt.translate_batch_with_llm(
            _texts(5), batch_size=10,
            progress_fn=lambda c, t, task, det: seen.append(task))
        assert set(seen) == {"Translation"}

    def test_no_callback_still_translates(self, ollama):
        out = srt.translate_batch_with_llm(_texts(12), batch_size=10)
        assert len(out) == 12


class TestReachesTranslateSegments:
    def test_passed_down_from_translate_segments(self, ollama, monkeypatch):
        monkeypatch.setattr(srt, "get_llm_translator", lambda: "ollama")
        seen = []
        segments = [{"start": float(i), "end": i + 1.0, "text": f"line {i}"}
                    for i in range(12)]
        srt.translate_segments(segments, "en", "pl",
                               progress_fn=lambda c, t, task, det: seen.append(det))
        assert any(d.startswith("Batch") for d in seen)

    def test_passed_down_from_create_srt_file(self, ollama, monkeypatch, tmp_path):
        monkeypatch.setattr(srt, "get_llm_translator", lambda: "ollama")
        seen = []
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        srt.create_srt_file(segments, str(tmp_path / "out.srt"),
                            source_lang="en", target_lang="pl",
                            progress_fn=lambda c, t, task, det: seen.append(det))
        assert seen

    def test_untranslated_write_reports_nothing(self, ollama, tmp_path):
        seen = []
        segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        srt.create_srt_file(segments, str(tmp_path / "out.srt"),
                            source_lang="en", target_lang=None,
                            progress_fn=lambda c, t, task, det: seen.append(det))
        assert seen == []


class TestBand:
    """Two long passes, one bar."""

    def test_maps_a_sub_step_into_its_slice(self):
        seen = []
        band = aod._band(lambda c, t, task, det: seen.append(c), 0, 60)
        band(0, 100, "x", "")
        band(50, 100, "x", "")
        band(100, 100, "x", "")
        assert seen == [0, 30, 60]

    def test_second_pass_starts_where_the_first_ended(self):
        seen = []
        first = aod._band(lambda c, t, task, det: seen.append(c), 0, 60)
        second = aod._band(lambda c, t, task, det: seen.append(c), 60, 98)
        first(100, 100, "x", "")
        second(0, 100, "x", "")
        assert seen == [60, 60]

    def test_total_is_normalized_to_100(self):
        seen = []
        band = aod._band(lambda c, t, task, det: seen.append((c, t)), 60, 98)
        band(3, 56, "Translation", "Batch 3/56")
        assert seen[0][1] == 100

    def test_details_survive(self):
        seen = []
        band = aod._band(lambda c, t, task, det: seen.append((task, det)), 60, 98)
        band(1, 10, "Translation", "Batch 1/10")
        assert seen == [("Translation", "Batch 1/10")]

    def test_no_progress_means_no_wrapper(self):
        assert aod._band(None, 0, 60) is None

    def test_zero_total_does_not_divide_by_zero(self):
        seen = []
        band = aod._band(lambda c, t, task, det: seen.append(c), 10, 90)
        band(0, 0, "x", "")
        assert seen == [10]


class TestSubtitleRunSharesTheBar:
    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        calls = {"progress": []}

        monkeypatch.setattr(aod, "read_caches", lambda _p: [])
        monkeypatch.setattr(aod, "analysis_defaults",
                            lambda: {"whisper_model": "base", "language": "en"})
        monkeypatch.setattr(aod, "_write_transcript_sidecar", lambda *a, **k: None)

        def fake_run_transcript(video_path, progress=None, **kwargs):
            # what get_transcript_segments emits: its own 0-100
            if progress:
                progress(5, 100, "Transcription", "Chunk 1/1")
                progress(95, 100, "Transcription", "Complete")
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
                    "language": "en"}

        def fake_create_srt(segments, path, progress_fn=None, **kw):
            if progress_fn:
                progress_fn(0, 56, "Translation", "Batch 1/56")
                progress_fn(56, 56, "Translation", "56 segments")

        monkeypatch.setattr(aod, "run_transcript", fake_run_transcript)
        import modules.transcript_srt as srt_mod
        monkeypatch.setattr(srt_mod, "create_srt_file", fake_create_srt)

        def go(**kwargs):
            aod.run_subtitles(
                str(tmp_path / "movie.mp4"), log=lambda *a: None,
                progress=lambda c, t, task, det: calls["progress"].append((c, task)),
                **kwargs)
            return calls["progress"]

        return go

    def test_never_goes_backwards(self, run):
        pcts = [c for c, _task in run(language="en", target_lang="pl")]
        assert pcts == sorted(pcts)

    def test_transcription_stays_in_the_first_stretch(self, run):
        seen = run(language="en", target_lang="pl")
        transcribing = [c for c, task in seen if task == "Transcription"]
        assert transcribing and max(transcribing) <= 60

    def test_translation_takes_the_rest(self, run):
        seen = run(language="en", target_lang="pl")
        translating = [c for c, task in seen if task == "Translation"]
        assert translating and min(translating) >= 60 and max(translating) <= 98
