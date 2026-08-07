"""Tests for the transcript layer over the chapter breakdown.

Two things here are worth pinning down, and they are the two that would fail
silently.

The first is the denominator. Speech share divides merged speech seconds by the
chapter's own length, and Whisper segments genuinely overlap each other after
chunk offsets are applied — a version that sums durations instead of merging
them reports 140% speech on a chapter that is half silent, and every other test
still passes. `test_overlapping_segments_counted_once` is that guard.

The second is what makes a word "distinctive", and this one has already failed
once in the field: a sixteen-chapter video produced the title *being, was,
today*. Document frequency was the culprit — those words are not in every
chapter, so a binary presence test called them rare while their high counts
carried them to the top. The ranking therefore compares *rates*, and
`test_ordinary_words_are_never_distinctive` builds exactly that case: a common
word missing from one chapter of several, which any tf-idf will rank and keyness
will not.

Fixture speech is deliberately mundane — a workshop, a kitchen, a car journey.
Nothing here needs interesting content to exercise the arithmetic, and the repo
does not carry any.
"""
from __future__ import annotations

import pytest

from modules.chapter_speech import (
    MIN_KEYNESS,
    MIN_QUOTE_GAP,
    MIN_WORD_COUNT,
    MIN_WORDS_FOR_TITLE,
    SPEECH_CHANGE_POINTS,
    clip_speech,
    distinctive_words,
    keyness,
    quotes_for,
    speakers_in,
    speech_seconds,
    summarise_speech,
    tokenize,
    video_speech,
)


def _chapter(number, start, end):
    return {"number": number, "start": float(start), "end": float(end),
            "duration": float(end - start), "timestamp": "0:00:00",
            "title": f"Chapter {number}", "shots": 10, "pace": "steady"}


def _seg(start, end, text, **extra):
    return dict(start=float(start), end=float(end), text=text, **extra)


def _talk(start, end, phrase, every=5.0):
    """A run of identical lines, one every ``every`` seconds."""
    out, t = [], float(start)
    while t + 2.0 <= end:
        out.append(_seg(t, t + 2.0, phrase))
        t += every
    return out


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
class TestTokenize:
    def test_lowercases_and_drops_punctuation(self):
        assert tokenize("The Bench, again!") == ["the", "bench", "again"]

    def test_drops_digits_and_single_letters(self):
        assert tokenize("I cut 20 mm off it") == ["cut", "mm", "off", "it"]

    def test_keeps_non_ascii_letters(self):
        # A Polish transcript has to tokenise like an English one; a byte-wise
        # \w+ over a non-Unicode pattern would split these words apart.
        assert tokenize("Śruba jest krótka") == ["śruba", "jest", "krótka"]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
class TestSpeechSeconds:
    def test_sums_disjoint_segments(self):
        segs = [_seg(0, 2, "one"), _seg(10, 13, "two")]
        assert speech_seconds(segs, 0, 60) == pytest.approx(5.0)

    def test_overlapping_segments_counted_once(self):
        # 0–4 and 3–6 cover six seconds between them, not seven.
        segs = [_seg(0, 4, "one"), _seg(3, 6, "two")]
        assert speech_seconds(segs, 0, 60) == pytest.approx(6.0)

    def test_clips_to_the_window(self):
        # A line running across the chapter's end contributes only the part
        # inside it, or a chapter can report more speech than it has seconds.
        segs = [_seg(55, 65, "spanning the boundary")]
        assert speech_seconds(segs, 0, 60) == pytest.approx(5.0)

    def test_no_speech_is_zero_not_an_error(self):
        assert speech_seconds([], 0, 60) == 0.0


# ---------------------------------------------------------------------------
# Distinctiveness
# ---------------------------------------------------------------------------
class TestKeyness:
    def test_evenly_spread_word_scores_about_one(self):
        docs = [["the", "bench"] * 5, ["the", "kettle"] * 5]
        weights = keyness(docs)
        assert weights[0]["the"] == pytest.approx(1.0, abs=0.2)
        assert weights[0]["bench"] > MIN_KEYNESS

    def test_single_document_yields_no_weights(self):
        # Nothing to be distinctive against. Returning frequencies here would
        # make the one-chapter case rank function words first.
        assert keyness([["bench", "bench"]]) == [{}]

    def test_empty_documents_get_empty_maps(self):
        assert keyness([[], ["bench"], []]) == [{}, {}, {}]


class TestDistinctiveWords:
    def test_ordinary_words_are_never_distinctive(self):
        # The failure this replaced. "was" is missing from the last chapter, so
        # document frequency calls it rare and its high count wins; its *rate*
        # is the same everywhere it appears, so keyness rejects it.
        docs = [["was"] * 10 + ["bench"] * 4,
                ["was"] * 10 + ["kettle"] * 4,
                ["was"] * 10 + ["ladder"] * 4,
                ["gutter"] * 14]
        weights = keyness(docs)
        assert [f["word"] for f in distinctive_words(docs[0], weights[0])] \
            == ["bench"]

    def test_a_single_mention_is_not_enough(self):
        assert MIN_WORD_COUNT > 1
        docs = [["shared"] * 4 + ["hapax"], ["shared"] * 4 + ["other"] * 3]
        weights = keyness(docs)
        assert distinctive_words(docs[0], weights[0]) == []

    def test_ranked_by_weight_and_capped(self):
        docs = [["alpha"] * 8 + ["beta"] * 4 + ["gamma"] * 2 + ["delta"] * 2,
                ["shared"] * 6]
        weights = keyness(docs)
        found = [f["word"] for f in distinctive_words(docs[0], weights[0])]
        assert found[:2] == ["alpha", "beta"]
        assert len(found) <= 3

    def test_a_frequent_shared_word_loses_to_a_rarer_exclusive_one(self):
        # The exact regression. "common" is said eight times here and often
        # elsewhere; "busy" four times and nowhere else. Any ranking weighted by
        # share of the chapter puts "common" first, which is how a stretch ended
        # up titled with ordinary words.
        docs = [["common"] * 8 + ["busy"] * 4 + ["filler"] * 30,
                ["common"] * 6 + ["filler"] * 36,
                ["common"] * 6 + ["filler"] * 36]
        weights = keyness(docs)
        assert distinctive_words(docs[0], weights[0])[0]["word"] == "busy"

    def test_reports_how_many_times_more_often(self):
        docs = [["bench"] * 6 + ["filler"] * 4, ["filler"] * 10]
        weights = keyness(docs)
        found = distinctive_words(docs[0], weights[0])
        assert found[0]["word"] == "bench" and found[0]["times"] > MIN_KEYNESS


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------
class TestQuotes:
    def test_picks_lines_carrying_distinctive_words(self):
        segs = [_seg(0, 2, "yes"), _seg(30, 33, "the bench needs a new bench vice"),
                _seg(60, 62, "yes")]
        idf = {"bench": 2.0, "vice": 2.0}
        quotes = quotes_for(segs, 0, 120, idf, limit=1)
        assert "bench" in quotes[0]["text"]

    def test_quotes_are_spaced_across_the_chapter(self):
        # Three consecutive lines of one exchange describe ten seconds of a
        # long chapter. The spacing rule is what stops that being the answer.
        segs = [_seg(t, t + 2, "the bench vice again") for t in (0, 3, 6)]
        segs.append(_seg(200, 202, "the bench vice again"))
        quotes = quotes_for(segs, 0, 300, {"bench": 2.0, "vice": 2.0}, limit=3)
        stamps = [q["start"] for q in quotes]
        assert all(b - a >= MIN_QUOTE_GAP
                   for a, b in zip(stamps, stamps[1:]))

    def test_returns_something_when_spacing_rejects_everything(self):
        segs = [_seg(0, 2, "one line"), _seg(3, 5, "another line")]
        assert len(quotes_for(segs, 0, 10, {}, limit=3)) == 1

    def test_carries_speaker_but_not_the_unknown_placeholder(self):
        segs = [_seg(0, 2, "over here", speaker="SPEAKER_01"),
                _seg(50, 52, "and here", speaker="UNKNOWN")]
        quotes = quotes_for(segs, 0, 120, {}, limit=2)
        assert quotes[0]["speaker"] == "SPEAKER_01"
        assert "speaker" not in quotes[1]

    def test_long_lines_are_trimmed(self):
        segs = [_seg(0, 40, "sanding " * 60)]
        text = quotes_for(segs, 0, 60, {})[0]["text"]
        assert len(text) <= 181 and text.endswith("…")

    def test_empty_transcript_yields_no_quotes(self):
        assert quotes_for([], 0, 60, {}) == []


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------
class TestSpeakers:
    def test_tallies_turns_and_seconds_ranked_by_time(self):
        segs = [_seg(0, 10, "a long turn", speaker="SPEAKER_00"),
                _seg(12, 14, "a short one", speaker="SPEAKER_01"),
                _seg(20, 22, "and another", speaker="SPEAKER_01")]
        rows = speakers_in(segs, 0, 60)
        assert [r["speaker"] for r in rows] == ["SPEAKER_00", "SPEAKER_01"]
        assert rows[1]["turns"] == 2

    def test_undiarized_transcript_reports_no_speakers(self):
        assert speakers_in([_seg(0, 2, "no tags here")], 0, 60) == []


# ---------------------------------------------------------------------------
# The whole thing
# ---------------------------------------------------------------------------
class TestSummariseSpeech:
    def test_shares_are_measured_against_the_chapter_not_the_video(self):
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 200)]
        # 50 of the first chapter's 100 seconds are speech; none of the second.
        segs = [_seg(t, t + 5, "the bench is level") for t in range(0, 50, 5)]
        rows = summarise_speech(chapters, segs, video_duration=200)
        assert rows[0]["speech_share_pct"] == pytest.approx(50.0)
        assert rows[1]["speech_share_pct"] == 0.0

    def test_input_chapters_are_not_mutated(self):
        chapters = [_chapter(1, 0, 100)]
        summarise_speech(chapters, [_seg(0, 5, "a line")], video_duration=100)
        assert "speech_share_pct" not in chapters[0]

    def test_titles_come_from_words_the_others_did_not_use(self):
        chapters = [_chapter(1, 0, 300), _chapter(2, 300, 600)]
        segs = (_talk(0, 300, "pass me the chisel and the mallet")
                + _talk(300, 600, "pass me the kettle and the mugs"))
        rows = summarise_speech(chapters, segs, video_duration=600)
        first = rows[0]["speech_title"]
        assert "chisel" in first or "mallet" in first
        assert "pass" not in first and "the" not in first

    def test_no_title_below_the_word_floor(self):
        chapters = [_chapter(1, 0, 300), _chapter(2, 300, 600)]
        segs = [_seg(10, 12, "just this")] + _talk(300, 600, "kettle and mugs")
        rows = summarise_speech(chapters, segs, video_duration=600)
        assert "speech_title" not in rows[0]
        assert rows[0]["words"] < MIN_WORDS_FOR_TITLE

    def test_speech_lift_is_against_the_video_rate(self):
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 200)]
        segs = [_seg(t, t + 5, "the bench is level") for t in range(0, 50, 5)]
        rows = summarise_speech(chapters, segs, video_duration=200)
        # 50% here against 25% overall.
        assert rows[0]["speech_lift"] == pytest.approx(2.0, abs=0.05)

    def test_boundary_where_the_talking_stops_is_marked(self):
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 200)]
        segs = [_seg(t, t + 5, "still talking") for t in range(0, 100, 5)]
        rows = summarise_speech(chapters, segs, video_duration=200)
        assert "speech_change" not in rows[0]      # nothing precedes it
        assert rows[1]["speech_change"]["direction"] == "fell"
        drop = (rows[1]["speech_change"]["from_pct"]
                - rows[1]["speech_change"]["to_pct"])
        assert drop >= SPEECH_CHANGE_POINTS

    def test_silent_chapter_keeps_a_zero_rather_than_no_key(self):
        # "Nothing was said here" is a finding. A missing key reads as "the
        # transcript did not run", which is a different report.
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 200)]
        segs = [_seg(10, 15, "only in the first")]
        rows = summarise_speech(chapters, segs, video_duration=200)
        assert rows[1]["speech_share_pct"] == 0.0

    def test_without_a_transcript_nothing_is_added(self):
        chapters = [_chapter(1, 0, 100)]
        rows = summarise_speech(chapters, [], video_duration=100)
        assert "speech_share_pct" not in rows[0]

    def test_blank_lines_do_not_count_as_speech(self):
        chapters = [_chapter(1, 0, 100)]
        rows = summarise_speech(chapters, [_seg(0, 50, "   ")],
                                video_duration=100)
        assert "speech_share_pct" not in rows[0]


class TestVideoSpeech:
    def test_reports_share_and_rate(self):
        segs = [_seg(t, t + 5, "two words") for t in range(0, 50, 5)]
        out = video_speech(segs, video_duration=100)
        assert out["speech_share_pct"] == pytest.approx(50.0)
        assert out["words"] == 20
        assert out["words_per_minute"] == pytest.approx(12.0)

    def test_empty_transcript_is_empty_not_zeroed(self):
        assert video_speech([], 100) == {}


class TestInTheReport:
    """The wiring, not the arithmetic — that a transcript reaches the page.

    Each of these fails on a different broken link: the kwarg, the chapter
    enrichment, the clip enrichment, the renderer. Without them the module above
    can be perfectly correct and the user still sees the report they had before.
    """

    def _report(self, transcript=None):
        import numpy as np

        from modules.highlight_report import build_report

        score = np.zeros(600)
        score[100] = 10.0
        score[400] = 8.0
        return build_report(
            video_path="a.mp4", video_duration=600,
            score=score, signals={"object": score},
            segments=[(95, 125), (395, 425)],
            chapters=[_chapter(1, 0, 300), _chapter(2, 300, 600)],
            transcript=transcript,
        )

    def _transcript(self):
        return (_talk(0, 300, "pass me the chisel and the mallet")
                + _talk(300, 600, "pass me the kettle and the mugs"))

    def test_without_a_transcript_the_report_is_unchanged(self):
        rep = self._report()
        assert rep["speech"] == {}
        assert "speech_share_pct" not in rep["chapters"][0]
        assert "speech" not in rep["segments"][0]

    def test_chapters_gain_words_and_quotes(self):
        rep = self._report(self._transcript())
        first = rep["chapters"][0]
        assert first["quotes"]
        assert "chisel" in first["speech_title"] or "mallet" in first["speech_title"]

    def test_clips_gain_the_lines_spoken_during_them(self):
        rep = self._report(self._transcript())
        assert rep["segments"][0]["speech"]["lines"]

    def test_the_same_clips_are_chosen_either_way(self):
        # The transcript describes; it must not select. A user comparing a run
        # with it against a run without has to be comparing the same cut.
        without = [s["range"] for s in self._report()["segments"]]
        with_it = [s["range"] for s in self._report(self._transcript())["segments"]]
        assert without == with_it

    def test_quotes_reach_the_html_and_the_text(self):
        from modules.highlight_report import render_html, render_text

        rep = self._report(self._transcript())
        quote = rep["chapters"][0]["quotes"][0]["text"]
        assert quote in render_html(rep)
        assert quote in render_text(rep)

    def test_quotes_reach_the_advisor_prompt(self):
        from modules.advisor import build_prompt

        rep = self._report(self._transcript())
        prompt = build_prompt(rep, [])
        assert rep["chapters"][0]["quotes"][0]["text"] in prompt

    def test_a_report_without_speech_says_so_on_the_page(self):
        from modules.highlight_report import render_html

        page = render_html(self._report())
        assert "run the transcript" in page


class TestClipSpeech:
    def test_returns_every_line_in_order(self):
        segs = [_seg(12, 14, "second"), _seg(10, 11, "first")]
        out = clip_speech(segs, 5, 20)
        assert [l["text"] for l in out["lines"]] == ["first", "second"]

    def test_counts_lines_beyond_the_limit(self):
        segs = [_seg(t, t + 1, f"line {t}") for t in range(10, 20)]
        out = clip_speech(segs, 0, 30, limit=3)
        assert len(out["lines"]) == 3 and out["total"] == 10

    def test_silent_clip_returns_nothing(self):
        assert clip_speech([_seg(0, 2, "elsewhere")], 100, 130) == {}
