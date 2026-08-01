"""Tests for `modules.expression_arc` — the shape of the expression reading.

Two properties are worth protecting, and only one of them is about arithmetic.

The first is that the module finds real structure: a file that reads one way and
then another has a split, and the split is where the change actually is.

The second is that it argues against itself when it should. A five-class
classifier let loose on footage it was not trained for produces a confident,
uniform, completely wrong distribution, and every number downstream inherits
that. The dispersion, coverage and stability checks exist to make that case
visible rather than authoritative — so the tests that matter most here are the
ones where the module is handed convincing-looking rubbish and has to say so.
"""

from __future__ import annotations

from modules.expression_arc import (
    DOMINANT_SHARE,
    GOOD_FIT,
    analyse,
    valence_of,
)


def _scan(spec):
    """``{second: (label, confidence)}`` from ``[(label, start, end, conf)]``."""
    out = {}
    for label, start, end, confidence in spec:
        for sec in range(start, end):
            out[sec] = (label, confidence)
    return out


class TestValence:
    def test_the_negative_classes_share_a_sign(self):
        assert valence_of("sad") < 0 and valence_of("anger") < 0

    def test_confidence_scales_the_reading(self):
        assert valence_of("happy", 0.5) == 0.5 * valence_of("happy", 1.0)

    def test_surprise_carries_no_direction(self):
        """Delight and alarm produce it alike, so it cannot be signed.

        Folding it into either side would let the biggest swing in most files
        come from the one label that cannot support one.
        """
        assert valence_of("surprise") == 0.0

    def test_an_unknown_label_is_not_guessed_at(self):
        assert valence_of("bewildered") == 0.0


class TestCoverage:
    def test_shares_are_of_what_was_read_and_coverage_converts_them_back(self):
        scan = _scan([("happy", 0, 100, 0.9)])
        result = analyse(scan, duration=1000)
        assert result["coverage"]["pct"] == 10.0
        assert result["labels"]["happy"]["share_pct"] == 100.0

    def test_a_thin_reading_is_flagged(self):
        result = analyse(_scan([("happy", 0, 50, 0.9)]), duration=1000)
        assert result["reliability"]["level"] != "unflagged"
        assert any("readable face" in r for r in result["reliability"]["reasons"])

    def test_nothing_read_is_not_an_error(self):
        assert analyse({}, duration=100) == {}
        assert analyse({0: ("happy", 0.9)}, duration=0) == {}


class TestShift:
    def test_a_file_that_turns_reports_where(self):
        """The question the whole feature exists for: did it start one way?"""
        scan = _scan([("sad", 0, 300, 0.9), ("happy", 300, 600, 0.9)])
        shift = analyse(scan, duration=600)["shift"]
        assert shift["direction"] == "toward positive"
        assert 250 <= shift["at"] <= 350
        assert shift["before"] < 0 < shift["after"]

    def test_a_file_that_does_not_turn_reports_no_split(self):
        assert analyse(_scan([("sad", 0, 600, 0.9)]), duration=600)["shift"] == {}

    def test_the_split_is_weighted_by_how_much_was_read(self):
        """A stretch with six readable seconds must not outvote one with three
        hundred, or every scan with a sparse tail invents an ending."""
        scan = _scan([("neutral", 0, 580, 0.9), ("happy", 580, 586, 0.99)])
        assert analyse(scan, duration=600)["shift"] == {}


class TestArc:
    def test_a_steady_slide_is_called_a_direction(self):
        scan = {}
        for sec in range(600):
            # A genuine ramp, not a step: the share of sad seconds rises evenly
            # from none at the start to all of them at the end.
            scan[sec] = ("sad" if (sec % 10) < (sec / 60.0) else "happy", 0.9)
        arc = analyse(scan, duration=600)["arc"]
        assert arc["direction"] == "toward negative"
        assert arc["fit"] >= GOOD_FIT
        assert arc["confident"] is True

    def test_a_slope_through_a_scatter_is_not_stated_flatly(self):
        """The honesty test. A line can be fitted through anything; whether it
        describes the points is a separate question, and the one that decides
        whether the direction may be asserted."""
        scan = {}
        for sec in range(600):
            scan[sec] = (("sad" if (sec // 20) % 2 else "happy"), 0.9)
        arc = analyse(scan, duration=600)["arc"]
        assert arc["confident"] is False

    def test_a_flat_file_is_called_flat(self):
        assert analyse(_scan([("neutral", 0, 600, 0.9)]),
                       duration=600)["arc"]["direction"] == "flat"


class TestDispersion:
    """The measurement that separates a finding from a broken classifier."""

    def test_a_label_smeared_evenly_is_called_uniform(self):
        scan = {}
        for sec in range(600):
            scan[sec] = ("sad" if sec % 2 else "neutral", 0.9)
        assert analyse(scan, duration=600)["dispersion"]["sad"]["uniform"] is True

    def test_a_label_that_clusters_is_not(self):
        scan = _scan([("neutral", 0, 400, 0.9), ("sad", 400, 600, 0.9)])
        assert analyse(scan, duration=600)["dispersion"]["sad"]["uniform"] is False

    def test_a_dominant_uniform_label_is_named_as_a_likely_misread(self):
        """The failure this module exists to catch.

        A classifier that reads a kind of footage wrong produces its favourite
        label everywhere, confidently. In a total that is indistinguishable from
        a video that really is that way; spread across the file it is obvious.
        """
        scan = {}
        for sec in range(600):
            scan[sec] = ("sad" if sec % 4 else "neutral", 0.95)
        result = analyse(scan, duration=600)
        assert result["labels"]["sad"]["share_pct"] >= DOMINANT_SHARE
        assert any("spread evenly" in r
                   for r in result["reliability"]["reasons"])

    def test_the_same_share_arriving_in_stretches_is_not_flagged(self):
        scan = _scan([("sad", 0, 450, 0.95), ("neutral", 450, 600, 0.95)])
        result = analyse(scan, duration=600)
        assert result["labels"]["sad"]["share_pct"] >= DOMINANT_SHARE
        assert not any("spread evenly"
                       in r for r in result["reliability"]["reasons"])


class TestStability:
    def test_a_label_flipping_every_second_is_flagged(self):
        scan = {}
        for sec in range(600):
            scan[sec] = (("sad", "happy", "neutral")[sec % 3], 0.9)
        result = analyse(scan, duration=600)
        assert result["stability"]["mean_run_seconds"] < 2.0
        assert any("changed every" in r for r in result["reliability"]["reasons"])

    def test_runs_are_counted_in_read_seconds_not_clock_seconds(self):
        """A scan taken every other second is not an unstable one.

        Counting gaps as breaks would call every sparse scan maximally unstable
        and bury the real signal under a caveat that is not true.
        """
        scan = {sec: ("sad", 0.9) for sec in range(0, 600, 2)}
        assert analyse(scan, duration=600)["stability"]["runs"] == 1


class TestEpisodes:
    def test_the_stretches_that_read_one_way_are_listed_longest_first(self):
        scan = _scan([("neutral", 0, 100, 0.9), ("sad", 100, 160, 0.9),
                      ("neutral", 160, 300, 0.9), ("sad", 300, 420, 0.9)])
        episodes = analyse(scan, duration=600)["episodes"]
        assert episodes[0]["seconds"] > episodes[1]["seconds"]
        assert episodes[0]["start"] == 300.0

    def test_the_two_negative_classes_form_one_episode(self):
        """A stretch alternating between them is one thing, not six."""
        scan = {}
        for sec in range(100, 200):
            scan[sec] = ("sad" if sec % 2 else "anger", 0.9)
        episodes = analyse(scan, duration=600)["episodes"]
        assert len([e for e in episodes if e["sign"] < 0]) == 1

    def test_a_brief_flicker_is_not_an_episode(self):
        scan = _scan([("neutral", 0, 300, 0.9), ("happy", 300, 303, 0.9)])
        assert [e for e in analyse(scan, duration=600)["episodes"]
                if e["sign"] > 0] == []


class TestSegmentReadings:
    def test_a_clip_is_reported_against_the_video_not_in_isolation(self):
        """In a file that reads negative throughout, every clip reads negative.

        The delta is the only part that carries information, and a clip matching
        the video's own baseline has to come out at roughly zero.
        """
        scan = _scan([("sad", 0, 600, 0.9)])
        rows = analyse(scan, duration=600, segments=[(100, 130)])["segments"]
        assert rows[0]["valence"] < 0
        assert abs(rows[0]["delta"]) < 0.05

    def test_a_clip_that_runs_against_the_file_shows_it(self):
        scan = _scan([("sad", 0, 600, 0.9)])
        scan.update(_scan([("happy", 100, 130, 0.9)]))
        rows = analyse(scan, duration=600, segments=[(100, 130)])["segments"]
        assert rows[0]["delta"] > 0.5

    def test_a_clip_with_no_readable_face_makes_no_claim(self):
        scan = _scan([("sad", 0, 100, 0.9)])
        rows = analyse(scan, duration=600, segments=[(300, 330)])["segments"]
        assert rows[0]["read_seconds"] == 0
        assert "valence" not in rows[0]


class TestClassReadings:
    """Whether the reading differs while a given class is on screen.

    The question people arrive with, and the one where a wrong answer is most
    tempting: a file that reads negative throughout hands every class in it a
    negative figure, and quoting that figure beside a class name invites the
    reader to connect the two. The delta is the only part that carries
    information, and "no different from the rest of the file" is a result.
    """

    def _detections(self, spans):
        out = {}
        for name, start, end in spans:
            for sec in range(start, end):
                out.setdefault(sec, []).append(name)
        return out

    def test_a_class_matching_the_baseline_is_reported_as_matching_it(self):
        scan = _scan([("sad", 0, 600, 0.9)])
        rows = analyse(scan, duration=600,
                       detections=self._detections([("dog", 100, 400)]))["by_class"]
        assert rows[0]["name"] == "dog"
        assert rows[0]["distinguishable"] is False
        assert abs(rows[0]["delta"]) < 0.05

    def test_a_class_whose_reading_really_differs_is_flagged(self):
        scan = _scan([("sad", 0, 600, 0.9)])
        scan.update(_scan([("happy", 100, 400, 0.9)]))
        rows = analyse(scan, duration=600,
                       detections=self._detections([("dog", 100, 400)]))["by_class"]
        assert rows[0]["distinguishable"] is True
        assert rows[0]["delta"] > 0.5

    def test_a_class_with_too_little_read_is_left_out_entirely(self):
        scan = _scan([("sad", 0, 600, 0.9)])
        rows = analyse(scan, duration=600,
                       detections=self._detections([("dog", 100, 110)]))["by_class"]
        assert rows == []

    def test_rows_are_ordered_by_how_far_they_sit_from_the_baseline(self):
        scan = _scan([("neutral", 0, 600, 0.9)])
        scan.update(_scan([("happy", 0, 200, 0.9)]))
        detections = self._detections([("dog", 0, 200), ("guitar", 300, 500)])
        rows = analyse(scan, duration=600, detections=detections)["by_class"]
        assert rows[0]["name"] == "dog"
        assert abs(rows[0]["delta"]) > abs(rows[1]["delta"])

    def test_no_detections_means_no_class_breakdown(self):
        assert "by_class" not in analyse(_scan([("sad", 0, 600, 0.9)]),
                                         duration=600)
