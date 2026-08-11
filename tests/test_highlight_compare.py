"""Tests for `modules.highlight_compare` — subjects ranked against their video.

The property worth protecting is that a size claim survives the camera moving.
Frame share cannot tell a larger subject from a nearer one, so the module's
whole reason to exist is the ratio against a second box in the same frame — and
the test that matters is the one where those two measurements disagree.

Class names here are ordinary nouns chosen to be uninteresting. The module has
no vocabulary of its own and compares a class only against itself, so what the
detector was taught to find never reaches this code.
"""

from __future__ import annotations

import numpy as np

from modules.highlight_compare import (
    MIN_STRETCHES,
    build_distributions,
    compare_segment,
)


def _record(second, entries):
    """One bbox-cache record: ``entries`` is ``[(name, x, y, w, h, conf), ...]``."""
    return {
        "timestamp": float(second),
        "objects": [e[0] for e in entries],
        "bboxes": [[e[1], e[2], e[3], e[4]] for e in entries],
        "confidences": [e[5] if len(e) > 5 else 0.9 for e in entries],
    }


# Camera distance, as a factor both boxes are drawn at. Every value here — and
# every box dimension below — is an exact binary fraction on purpose: the point
# of the fixture is that a steady subject produces *identical* ratios, and
# rounding noise at the fifteenth decimal would spread them across a percentile
# range and quietly turn the assertions into tests of float error.
SCALES = (0.5, 1.0, 1.5)

PERSON = (0.25, 0.5)                                   # w, h at scale 1
DOG = (0.125, 0.25)                                    # a quarter of the person


def _steady_video(seconds=60):
    """A video where the camera moves but the dog is always the same size.

    ``scale`` stands in for camera distance: both boxes grow and shrink with it
    together, so frame share varies threefold while the ratio between them never
    moves at all. That is exactly the confound the relative measure exists to
    remove, and this fixture is the clean case where only one of the two
    measurements should stay quiet.
    """
    cache = []
    for sec in range(seconds):
        scale = SCALES[sec % len(SCALES)]
        cache.append(_record(sec, [
            ("person", 0.1, 0.1, PERSON[0] * scale, PERSON[1] * scale, 0.95),
            ("dog", 0.5, 0.5, DOG[0] * scale, DOG[1] * scale, 0.88),
        ]))
    return cache


class TestDistributions:
    def test_prevalence_is_a_share_of_detected_seconds(self):
        cache = _steady_video(seconds=20)
        cache += [_record(50 + i, [("guitar", 0.1, 0.1, 0.2, 0.2, 0.7)])
                  for i in range(5)]
        d = build_distributions(cache)
        assert d["detected_seconds"] == 25
        assert d["seconds_with"]["person"] == 20
        assert d["seconds_with"]["guitar"] == 5

    def test_a_class_is_represented_by_its_largest_box_each_second(self):
        cache = [_record(0, [("dog", 0, 0, 0.1, 0.1, 0.9),
                             ("dog", 0.5, 0.5, 0.4, 0.4, 0.9)])]
        d = build_distributions(cache)
        assert d["largest"][0]["dog"][0] == 0.4 * 0.4

    def test_co_occurrence_is_counted_in_both_directions(self):
        d = build_distributions(_steady_video(seconds=12))
        assert d["pair_seconds"][("dog", "person")] == 12
        assert d["pair_seconds"][("person", "dog")] == 12


class TestRelativeScale:
    def test_a_bigger_subject_is_found_even_when_the_camera_pulled_back(self):
        """The test the whole module is for.

        The dog at second 30 is three times its usual size *relative to the
        person beside it*, but the camera is at its furthest, so in frame share
        it is unremarkable. Frame share alone would report nothing.
        """
        cache = _steady_video(seconds=60)
        far = min(SCALES)                             # camera at its furthest
        cache[30] = _record(30, [
            ("person", 0.1, 0.1, PERSON[0] * far, PERSON[1] * far, 0.95),
            # Three times its usual share of the person: 0.75 rather than 0.25.
            ("dog", 0.5, 0.5, DOG[0] * 2 * far, DOG[1] * 1.5 * far, 0.88),
        ])

        d = build_distributions(cache)
        result = compare_segment(d, 30.0, 31.0)
        dog = next(s for s in result["subjects"] if s["name"] == "dog")

        assert dog["relative"]["reference"] == "person"
        assert dog["relative"]["ratio"] == 0.75
        assert dog["relative"]["percentile"] >= 99.0
        # ...and the measurement that cannot see it, for contrast: shrunk by the
        # distance, this dog covers less of the frame than most of its own kind.
        assert dog["frame_share_percentile"] < 50.0

    def test_the_best_ratio_wins_the_clip_not_the_biggest_box(self):
        """A clip holds both seconds; only one of them is the finding.

        Second 30 has the oversized dog at the furthest camera position, second
        29 an ordinary dog at the nearest. The bigger *box* is at 29. Reporting
        that one would be picking the second the camera moved in — the confound,
        restated as the answer.
        """
        cache = _steady_video(seconds=60)
        far = min(SCALES)
        cache[30] = _record(30, [
            ("person", 0.1, 0.1, PERSON[0] * far, PERSON[1] * far, 0.95),
            ("dog", 0.5, 0.5, DOG[0] * 2 * far, DOG[1] * 1.5 * far, 0.88),
        ])

        d = build_distributions(cache)
        result = compare_segment(d, 28.0, 33.0)
        dog = next(s for s in result["subjects"] if s["name"] == "dog")

        assert dog["relative"]["at"] == 30
        assert dog["relative"]["ratio"] == 0.75
        assert dog["at"] != 30                        # the biggest box is not

    def test_a_steady_subject_is_not_called_unusual(self):
        d = build_distributions(_steady_video(seconds=60))
        result = compare_segment(d, 10.0, 15.0)
        dog = next(s for s in result["subjects"] if s["name"] == "dog")
        # Every ratio in the file is identical, so the midrank puts it at 50 —
        # no better or worse than the rest, which is the situation.
        assert dog["relative"]["percentile"] == 50.0

    def test_a_reference_seen_too_rarely_is_not_used(self):
        cache = _steady_video(seconds=10)
        cache.append(_record(99, [("dog", 0.1, 0.1, 0.5, 0.5, 0.9),
                                  ("guitar", 0.6, 0.6, 0.1, 0.1, 0.9)]))
        d = build_distributions(cache)
        result = compare_segment(d, 99.0, 100.0)
        dog = next(s for s in result["subjects"] if s["name"] == "dog")
        # The person is the well-observed pairing, but is not in this frame; the
        # guitar is, and one co-occurrence ranks nothing. So: no claim at all.
        assert "relative" not in dog


class TestSampleHonesty:
    """A rank is only as good as the number of stretches it beat.

    Windows slide a second at a time, so counting them as evidence would claim
    fifty-eight independent observations from a minute of video. The count that
    survives into the record is divided back down into whole stretches, which is
    the number the reader would have counted themselves.
    """

    def _dog_video(self, seconds):
        return [_record(i, [("dog", 0.1, 0.1, 0.2 + i / 100.0, 0.2, 0.9)])
                for i in range(seconds)]

    def test_a_video_with_too_few_comparable_stretches_is_flagged(self):
        # 18 seconds holds sixteen 3s windows — five stretches, not six.
        d = build_distributions(self._dog_video(18))
        subject = compare_segment(d, 0.0, 3.0)["subjects"][0]
        assert subject["stretches"] < MIN_STRETCHES
        assert subject["enough_samples"] is False

    def test_a_video_with_room_to_compare_clears_the_flag(self):
        d = build_distributions(self._dog_video(60))
        subject = compare_segment(d, 0.0, 3.0)["subjects"][0]
        assert subject["stretches"] >= MIN_STRETCHES
        assert subject["enough_samples"] is True

    def test_the_clip_is_ranked_against_stretches_of_its_own_length(self):
        """The bias this replaced: a long clip beating short ones on length.

        Every second here is identical, so no clip is unusual at any length —
        and taking a clip's maximum against the video's individual seconds would
        still have put the longer one further up the scale.
        """
        cache = [_record(i, [("dog", 0.1, 0.1, 0.3, 0.2, 0.9)]) for i in range(120)]
        d = build_distributions(cache)
        short = compare_segment(d, 0.0, 4.0)["subjects"][0]
        long = compare_segment(d, 0.0, 40.0)["subjects"][0]
        assert short["stretch_seconds"] == 4 and long["stretch_seconds"] == 40
        assert short["frame_share_percentile"] == long["frame_share_percentile"]


class TestPresence:
    def test_a_flicker_reports_the_share_of_the_clip_it_held(self):
        cache = [_record(i, [("person", 0.1, 0.1, 0.3, 0.4, 0.9)])
                 for i in range(20)]
        cache[5]["objects"].append("dog")
        cache[5]["bboxes"].append([0.5, 0.5, 0.4, 0.4])
        cache[5]["confidences"].append(0.4)
        d = build_distributions(cache)
        result = compare_segment(d, 0.0, 20.0)
        dog = next(s for s in result["subjects"] if s["name"] == "dog")
        assert dog["clip_presence_pct"] == 5.0

    def test_the_largest_instance_is_taken_from_the_whole_clip(self):
        cache = [_record(i, [("dog", 0.1, 0.1, 0.1, 0.1, 0.9)]) for i in range(10)]
        cache[7] = _record(7, [("dog", 0.1, 0.1, 0.5, 0.5, 0.9)])
        d = build_distributions(cache)
        result = compare_segment(d, 0.0, 10.0)
        assert result["subjects"][0]["at"] == 7


def _expression_video(surprise_seconds, total=100, surprise_confidence=0.9):
    seconds = {sec: ("neutral", 0.7) for sec in range(total)}
    for sec in surprise_seconds:
        seconds[sec] = ("surprise", surprise_confidence)
    return seconds


class TestExpression:
    def test_lift_measures_the_clip_against_the_video(self):
        expressions = _expression_video(list(range(40, 48)))
        d = build_distributions([], expressions)
        result = compare_segment(d, 40.0, 50.0)
        e = result["expression"]
        assert e["label"] == "surprise"
        assert e["clip_share_pct"] == 80.0
        assert e["video_share_pct"] == 8.0
        assert e["lift"] == 10.0
        assert e["video_dominant"] == "neutral"

    def test_dominance_is_decided_on_strength_not_on_a_head_count(self):
        expressions = {0: ("neutral", 0.51), 1: ("neutral", 0.52),
                       2: ("surprise", 0.99), 3: ("surprise", 0.98)}
        expressions.update({sec: ("neutral", 0.7) for sec in range(4, 30)})
        d = build_distributions([], expressions)
        result = compare_segment(d, 0.0, 4.0)
        assert result["expression"]["label"] == "surprise"

    def test_confidence_is_ranked_against_that_label_only(self):
        expressions = _expression_video(list(range(40, 50)),
                                        surprise_confidence=0.6)
        expressions[45] = ("surprise", 0.98)
        d = build_distributions([], expressions)
        result = compare_segment(d, 44.0, 47.0)
        e = result["expression"]
        assert e["confidence"] == 0.98
        # High, but short of the ceiling: it is competing with the other ten
        # surprise seconds, not with ninety neutral ones it would trivially beat.
        assert 80.0 <= e["confidence_percentile"] < 100.0

    def test_a_label_the_video_barely_shows_is_flagged(self):
        expressions = _expression_video([40, 41])
        d = build_distributions([], expressions)
        result = compare_segment(d, 40.0, 42.0)
        assert result["expression"]["enough_samples"] is False

    def test_a_scan_shaped_as_dicts_is_accepted_too(self):
        expressions = {sec: {"label": "happy", "confidence": 0.8}
                       for sec in range(20)}
        d = build_distributions([], expressions)
        assert compare_segment(d, 0.0, 5.0)["expression"]["label"] == "happy"


class TestNothingToSay:
    def test_no_detections_and_no_faces_produce_no_comparison(self):
        assert compare_segment(build_distributions([], {}), 0.0, 10.0) == {}
