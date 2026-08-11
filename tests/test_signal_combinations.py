"""Tests for `modules.signal_combinations` — is this combination actually unusual?

The property worth protecting is that the module is willing to deflate a
finding. Four signals agreeing inside one clip looks like a discovery, and on
footage that produces the same four marks in every stretch it is the norm
wearing a discovery's clothes. A base rate is the only thing that can tell those
apart, so the tests that matter most here are the ones where the combination is
everywhere and the module has to say so.
"""

from __future__ import annotations

from modules.highlight_prose import describe_combination
from modules.signal_combinations import (
    MIN_WINDOWS,
    marks_of,
    rate,
    survey,
)


def _levels(n, loud_at=(), quiet=-40.0, loud=-3.0):
    out = [quiet] * n
    for sec in loud_at:
        out[sec] = loud
    return out


def _detections(spans):
    out = {}
    for name, start, end in spans:
        for sec in range(start, end):
            out.setdefault(sec, []).append(name)
    return out


class TestSurvey:
    def test_the_video_is_walked_in_equal_non_overlapping_stretches(self):
        """Overlapping windows would count one event several times."""
        found = survey(600.0, window=30)
        assert found["count"] == 20
        assert found["window"] == 30

    def test_a_video_too_short_to_have_a_rate_says_so(self):
        found = survey(60.0, window=30)
        assert found["count"] == 2
        assert found["enough"] is False
        assert rate(found, ["movement"]) == {}

    def test_a_stretch_records_the_marks_it_carries(self):
        peaks = [45.0]
        found = survey(600.0, window=30, motion_peaks=peaks)
        assert "movement" in found["stretches"][1]
        assert "movement" not in found["stretches"][0]

    def test_loudness_is_judged_against_this_video_not_a_fixed_level(self):
        """A quiet recording has loud moments too, and they are its highlights."""
        found = survey(600.0, window=30, levels=_levels(600, loud_at=[100]))
        assert "loud" in found["stretches"][3]
        assert sum(1 for s in found["stretches"] if "loud" in s) < 5


class TestRate:
    def _found(self, marks_per_window):
        return {"window": 30, "count": len(marks_per_window),
                "stretches": [frozenset(m) for m in marks_per_window],
                "enough": len(marks_per_window) >= MIN_WINDOWS}

    def test_a_stretch_carrying_more_marks_still_counts(self):
        """The question is whether the combination turns up, not only it."""
        found = self._found([{"movement", "loud"}, {"movement", "loud", "reading"}]
                            + [set()] * 8)
        assert rate(found, ["movement", "loud"])["matching"] == 2

    def test_the_share_is_of_every_stretch_including_the_kept_ones(self):
        found = self._found([{"movement"}] * 5 + [set()] * 5)
        assert rate(found, ["movement"])["pct"] == 50.0
        assert rate(found, ["movement"])["windows"] == 10

    def test_a_mark_nothing_carries_reports_zero_rather_than_nothing(self):
        found = self._found([set()] * 10)
        assert rate(found, ["movement"])["matching"] == 0

    def test_marks_the_module_does_not_know_are_ignored(self):
        found = self._found([{"movement"}] * 10)
        assert rate(found, ["movement", "vibes"])["marks"] == ["movement"]


class TestMarksOfAClip:
    def test_the_marks_are_read_off_what_the_report_already_found(self):
        entry = {"event_onset": {"second": 10}, "motion_peak": {"second": 12},
                 "expression_peak": {"second": 14},
                 "loudest": {"second": 16, "level_dbfs": -4.0}}
        assert sorted(marks_of(entry, -10.0)) == ["arrival", "loud", "movement",
                                                  "reading"]

    def test_every_clip_has_a_loudest_second_but_not_every_one_is_loud(self):
        entry = {"loudest": {"second": 16, "level_dbfs": -35.0}}
        assert marks_of(entry, -10.0) == []


class TestProse:
    def _entry(self, pct, marks=("arrival", "loud", "movement")):
        return {"combination": {"marks": sorted(marks), "matching": 1,
                                "windows": 100, "window_seconds": 30,
                                "pct": pct}}

    def test_a_combination_the_video_rarely_produces_is_called_rare(self):
        assert "Hardly anywhere else in this video" in describe_combination(self._entry(2.0))

    def test_a_combination_the_video_produces_constantly_is_deflated(self):
        """The finding that four signals agreed is not one if they always do."""
        said = describe_combination(self._entry(85.0))
        assert "Most of the video does the same thing" in said
        assert "says little about this clip in particular" in said

    def test_the_middle_is_not_dressed_up_as_either(self):
        assert "Not many other stretches" in describe_combination(self._entry(18.0))
        assert "A fair part of the video" in describe_combination(self._entry(40.0))

    def test_one_mark_is_not_a_combination(self):
        assert describe_combination(self._entry(2.0, marks=["loud"])) == ""

    def test_a_clip_without_a_rate_says_nothing(self):
        assert describe_combination({}) == ""

    def test_the_sentence_never_reads_the_combination(self):
        """Rarity is where to look, not what happened there."""
        said = describe_combination(self._entry(2.0))
        for invented in ("means", "because", "caused", "suggests", "probably",
                         "likely", "reaction", "felt"):
            assert invented not in said.lower()
