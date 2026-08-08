"""
`run_subtitles` reusing a transcript this video already has.

Subtitles are *derived*: the deliverable is an `.srt` written from segments,
and a transcript of the same video may already be in the cache from an earlier
run or a full pipeline pass. Transcribing again to produce it costs minutes to
hours; writing the file costs seconds. But reuse is only sound when the cached
transcript is the whole video and in the right language, which is what these
pin down.
"""

from __future__ import annotations

import pytest

from modules import analysis_ondemand as aod


SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "first line"},
    {"start": 2.0, "end": 4.0, "text": "second line"},
]


@pytest.fixture
def cache(monkeypatch):
    """Control what the on-disk caches report for the video under test.

    `data` is the single-cache case; `files` sets several, newest first.
    """
    state = {"data": {}, "files": None}
    monkeypatch.setattr(
        aod, "read_caches",
        lambda _p: state["files"] if state["files"] is not None else [state["data"]])
    monkeypatch.setattr(aod, "read_cache", lambda _p: state["data"])
    monkeypatch.setattr(aod, "analysis_defaults",
                        lambda: {"whisper_model": "base", "language": "en"})
    return state


@pytest.fixture
def subtitles(monkeypatch, tmp_path, cache):
    """Run `run_subtitles` with transcription and SRT writing recorded."""
    calls = {"transcribed": 0, "srt": [], "log": []}

    def fake_run_transcript(video_path, **kwargs):
        calls["transcribed"] += 1
        return {"segments": SEGMENTS, "language": kwargs.get("language") or "en",
                "cached_full_transcript": True, "keyword_filtered": False}

    monkeypatch.setattr(aod, "run_transcript", fake_run_transcript)
    monkeypatch.setattr(aod, "_write_transcript_sidecar",
                        lambda *a, **k: None)

    import modules.transcript_srt as srt_mod
    monkeypatch.setattr(
        srt_mod, "create_srt_file",
        lambda segments, path, **kw: calls["srt"].append((path, len(segments), kw)),
        raising=False)

    video = str(tmp_path / "movie.mp4")

    def run(**kwargs):
        kwargs.setdefault("log", calls["log"].append)
        return aod.run_subtitles(video, **kwargs)

    calls["run"] = run
    calls["video"] = video
    return calls


class TestReusesWhatIsAlreadyThere:
    def test_cached_transcript_is_not_re_transcribed(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en",
                                        "cached_full_transcript": True,
                                        "keyword_filtered": False}}
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 0

    def test_srt_is_still_written_from_the_cached_segments(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en"}}
        subtitles["run"](language="en")
        assert len(subtitles["srt"]) == 1
        _path, count, _kw = subtitles["srt"][0]
        assert count == len(SEGMENTS)

    def test_says_so_in_the_log(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en"}}
        subtitles["run"](language="en")
        assert any("already in this video's cache" in m for m in subtitles["log"])

    def test_returned_dict_is_still_cache_shaped(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en",
                                        "cached_full_transcript": True,
                                        "keyword_filtered": False}}
        result = subtitles["run"](language="en")
        assert result["segments"] == SEGMENTS

    def test_translation_still_happens_on_a_reused_transcript(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en"}}
        subtitles["run"](language="en", source_lang="en", target_lang="pl")
        path, _count, kw = subtitles["srt"][0]
        assert kw.get("target_lang") == "pl"
        assert path.endswith("_pl.srt")

    def test_progress_reports_the_reuse(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en"}}
        seen = []
        subtitles["run"](language="en",
                         progress=lambda c, t, task, det: seen.append(det))
        assert any("Reusing cached transcript" in d for d in seen)


class TestTranscribesWhenItMust:
    def test_no_cache_at_all(self, cache, subtitles):
        cache["data"] = {}
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 1

    def test_cache_without_a_transcript(self, cache, subtitles):
        cache["data"] = {"objects": [], "actions": []}
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 1

    def test_empty_segment_list(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": [], "language": "en"}}
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 1

    def test_keyword_filtered_transcript_is_only_part_of_the_video(self, cache, subtitles):
        # Subtitles from these would cover the keyword hits and nothing else.
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en",
                                        "cached_full_transcript": False,
                                        "keyword_filtered": True}}
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 1
        assert any("keyword matches" in m for m in subtitles["log"])

    def test_different_language(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "pl"}}
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 1
        assert any("asked for" in m for m in subtitles["log"])

    def test_reuse_can_be_turned_off(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en"}}
        subtitles["run"](language="en", reuse_cached=False)
        assert subtitles["transcribed"] == 1


class TestLanguageMatching:
    def test_auto_accepts_whatever_was_cached(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "pl"}}
        subtitles["run"](language="auto")
        assert subtitles["transcribed"] == 0

    def test_cache_that_does_not_record_a_language_is_taken_as_matching(
            self, cache, subtitles):
        # Legacy caches predate the language key; re-transcribing every one of
        # them to find out would cost exactly what this is here to avoid.
        cache["data"] = {"transcript": {"segments": SEGMENTS}}
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 0

    def test_falls_back_to_the_configured_language(self, cache, subtitles):
        # No language passed: what the main GUI is set to decides the match.
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "pl"}}
        subtitles["run"]()
        assert subtitles["transcribed"] == 1


class TestSeveralCacheFiles:
    """A video can own more than one cache file, and the newest is not
    necessarily the one holding the transcript — observed on this machine: 1362
    segments in one file, zero in its sibling."""

    def test_finds_a_transcript_the_newest_file_does_not_have(self, cache, subtitles):
        cache["files"] = [
            {"transcript": {"segments": [], "language": "en"}},          # newest
            {"transcript": {"segments": SEGMENTS, "language": "en"}},    # older
        ]
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 0

    def test_prefers_the_fuller_transcript(self, cache, subtitles):
        longer = SEGMENTS + [{"start": 4.0, "end": 6.0, "text": "third line"}]
        cache["files"] = [
            {"transcript": {"segments": SEGMENTS, "language": "en"}},
            {"transcript": {"segments": longer, "language": "en"}},
        ]
        subtitles["run"](language="en")
        _path, count, _kw = subtitles["srt"][0]
        assert count == len(longer)

    def test_a_usable_sibling_beats_an_unusable_newest(self, cache, subtitles):
        cache["files"] = [
            {"transcript": {"segments": SEGMENTS, "language": "en",
                            "keyword_filtered": True}},
            {"transcript": {"segments": SEGMENTS, "language": "en"}},
        ]
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 0

    def test_no_cache_files_at_all(self, cache, subtitles):
        cache["files"] = []
        subtitles["run"](language="en")
        assert subtitles["transcribed"] == 1


class TestTheSpokenLanguageHasOneHome:
    """The `.srt` is labelled with the language of the transcript it was written
    from. The app used to ask "what is spoken?" twice — once in Transcript
    Settings (what Whisper is told) and again in Subtitle Settings — and the
    second answer could not change the audio, only mislabel the result."""

    def test_srt_named_for_the_reused_transcripts_language(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "ru"}}
        subtitles["run"](language="auto")
        path, _count, _kw = subtitles["srt"][0]
        assert path.endswith("_ru.srt")

    def test_transcript_language_beats_the_source_lang_argument(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "ru"}}
        subtitles["run"](language="auto", source_lang="en")
        path, _count, kw = subtitles["srt"][0]
        assert kw.get("source_lang") == "ru"
        assert path.endswith("_ru.srt")

    def test_source_lang_is_still_a_fallback(self, cache, subtitles):
        # A transcript that records no language at all.
        cache["data"] = {"transcript": {"segments": SEGMENTS}}
        subtitles["run"](language=None, source_lang="pl")
        _path, _count, kw = subtitles["srt"][0]
        assert kw.get("source_lang") == "pl"

    def test_translating_out_of_the_transcripts_language(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "ru"}}
        subtitles["run"](language="auto", target_lang="pl")
        path, _count, kw = subtitles["srt"][0]
        assert kw.get("source_lang") == "ru"
        assert kw.get("target_lang") == "pl"
        assert path.endswith("_pl.srt")

    def test_auto_does_not_name_the_file(self, cache, subtitles):
        # `movie_auto.srt` names nothing: "auto" is a request to Whisper.
        cache["data"] = {}
        subtitles["run"](language="auto")
        path, _count, _kw = subtitles["srt"][0]
        assert path.endswith("movie.srt")

    def test_no_translation_when_target_matches_what_was_spoken(self, cache, subtitles):
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "pl"}}
        subtitles["run"](language="auto", target_lang="pl")
        _path, _count, kw = subtitles["srt"][0]
        assert kw.get("target_lang") is None


class TestCachedTranscriptHelper:
    def test_returns_the_cache_entry(self, cache):
        entry = {"segments": SEGMENTS, "language": "en"}
        cache["data"] = {"transcript": entry}
        assert aod.cached_transcript("movie.mp4", language="en") is entry

    def test_none_when_there_is_nothing_to_reuse(self, cache):
        cache["data"] = {}
        assert aod.cached_transcript("movie.mp4", language="en") is None

    def test_run_transcript_is_left_alone(self, cache, monkeypatch):
        """The Transcript button means "do it again" — only derived runs reuse."""
        cache["data"] = {"transcript": {"segments": SEGMENTS, "language": "en"}}
        called = []
        monkeypatch.setattr(aod, "_write_transcript_sidecar", lambda *a, **k: None)

        import modules.transcript as tr_mod
        monkeypatch.setattr(tr_mod, "get_transcript_segments",
                            lambda *a, **k: called.append(1) or SEGMENTS)
        aod.run_transcript("movie.mp4", language="en", log=lambda *a: None)
        assert called == [1]
