"""Tests for taking Whisper's repetition loops out of a transcript.

The shapes are the ones Whisper produces: one segment holding a phrase dozens
of times over, a run of segments that all say the same thing (the same short
line once per 30-second window during a pause included), and a segment
Whisper itself scored as not speech. The words are filler: the rules look at
the shape of the text and know no words, in any language. What's pinned down: each collapses to one
occurrence, ordinary repetition people really say is left alone, and cleaning
twice changes nothing.
"""
from __future__ import annotations

import pytest

from modules.audio import transcript_cleanup as tc


def _seg(start, text, end=None, **extra):
    return dict(start=float(start), end=float(end if end is not None else start + 1), text=text,
                **extra)


class TestCollapse:
    @pytest.mark.parametrize("text, expected", [
        # A whole window of one phrase, ending mid-repeat.
        ("one two three, one two three, one two three, one two three, one two three, one",
         "one two three."),
        # The loop after real words: those stay.
        ("first words then alpha beta alpha beta alpha beta alpha beta alpha beta...",
         "first words then alpha beta"),
        # Punctuation and case differ between the repeats.
        ("Red green blue. Red green blue,. red green blue. Red green blue.",
         "Red green blue."),
        ("Wörter für alle. Wörter für alle, wörter für alle. Wörter für alle.",
         "Wörter für alle."),
        ("No, no, no, no, no, no, no.", "No."),
    ])
    def test_a_loop_is_said_once(self, text, expected):
        assert tc.collapse_repeats(text) == expected

    @pytest.mark.parametrize("text", [
        "no, no, no",
        "Wait, wait, wait, wait.",            # a single word four times is speech
        "Go on, go on, go on.",               # a phrase three times is speech
        "Hi, how are you doing today? Good.",
        "",
    ])
    def test_ordinary_repetition_is_left_alone(self, text):
        assert tc.collapse_repeats(text) == text.strip()


class TestCleanSegments:
    def test_a_collapsed_loop_no_longer_spans_the_window(self):
        out = tc.clean_segments([_seg(333.4, "one two three, " * 40, end=363.4)])
        assert out[0]["text"] == "one two three."
        assert out[0]["start"] == 333.4
        assert out[0]["end"] < 336

    def test_a_run_of_the_same_line_keeps_the_first(self):
        segs = [_seg(i, "Same line.") for i in range(29)] + [_seg(40, "Next.")]
        out = tc.clean_segments(segs)
        assert [(s["start"], s["text"]) for s in out] == [(0, "Same line."), (40, "Next.")]

    def test_the_same_ghost_every_window_through_a_pause(self):
        segs = [_seg(t, "Short line.") for t in (0, 30, 60, 90)] + [_seg(120, "Something else.")]
        assert [s["start"] for s in tc.clean_segments(segs)] == [0, 120]

    def test_the_same_line_minutes_apart_is_not_a_loop(self):
        segs = [_seg(t, "Short line.") for t in (0, 600, 1200)]
        assert len(tc.clean_segments(segs)) == 3

    def test_a_line_said_twice_stays(self):
        segs = [_seg(0, "Yeah."), _seg(1, "yeah"), _seg(2, "Then we go.")]
        assert len(tc.clean_segments(segs)) == 3

    def test_whisper_numbers_drop_what_it_thought_was_not_speech(self):
        segs = [_seg(0, "Short line.", no_speech_prob=0.9, avg_logprob=-0.8),
                _seg(5, "Real words here.", no_speech_prob=0.9, avg_logprob=-0.2),
                _seg(9, "More words.", no_speech_prob=0.1, avg_logprob=-1.5)]
        assert [s["text"] for s in tc.clean_segments(segs)] == ["Real words here.", "More words."]

    def test_other_keys_survive_and_the_input_is_untouched(self):
        segs = [_seg(0, "go " * 10, speaker_label="A")]
        out = tc.clean_segments(segs)
        assert out[0]["speaker_label"] == "A" and out[0]["text"] == "go"
        assert segs[0]["text"] == "go " * 10

    def test_cleaning_twice_changes_nothing(self):
        segs = ([_seg(i, "Again and again.") for i in range(5)]
                + [_seg(10, "word " * 12, end=40)] + [_seg(41, "End.")])
        once = tc.clean_segments(segs)
        assert tc.clean_segments(once) == once

    def test_nothing_in_nothing_out(self):
        assert tc.clean_segments([]) == [] and tc.clean_segments(None) == []


def test_the_subtitle_writers_clean_what_they_are_given(tmp_path):
    from modules.audio import transcript_srt

    segs = [_seg(i * 30, "Short line.") for i in range(5)] + [_seg(200, "a b c " * 20, end=230)]
    srt = tmp_path / "out.srt"
    transcript_srt.create_srt_file(segs, str(srt))
    body = srt.read_text(encoding="utf-8-sig")
    assert body.count("Short line.") == 1
    assert body.lower().count("a b c") == 1
    assert transcript_srt.create_enhanced_transcript(segs).count("Short line") == 1


def test_a_cached_transcript_comes_back_clean():
    from modules.media.video_cache import CachedAnalysisData

    data = CachedAnalysisData({"transcript": {"segments": [_seg(i, "Same.") for i in range(6)],
                                              "language": "en"}})
    assert len(data.transcript["segments"]) == 1
    assert data.transcript["language"] == "en"
