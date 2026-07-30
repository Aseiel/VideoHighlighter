"""
Tests for `modules.highlight_report` — the "why was this moment chosen" record.

Pure numpy and stdlib, no Qt, no cv2, no video file: thumbnails are injected
through `thumbnail_fn`, which is the reason that parameter exists.
"""

from __future__ import annotations

import json

import numpy as np

from modules.highlight_report import (
    build_report,
    format_timestamp,
    peak_second,
    render_html,
    render_text,
    write_report,
)


def _signals(n=60, **overrides):
    """Zeroed per-second arrays, with named seconds given points."""
    keys = ("scene", "motion_event", "motion_peak", "audio",
            "keyword", "object", "action")
    sig = {k: np.zeros(n) for k in keys}
    for key, points in overrides.items():
        for sec, value in points.items():
            sig[key][sec] = value
    return sig


def _score(signals, n=60, boost_at=None, boost_points=0.0):
    total = sum(signals.values())
    if boost_at is not None:
        total = total.copy()
        total[boost_at] += boost_points
    return total


class TestPeakSecond:
    def test_finds_the_maximum_inside_the_window(self):
        score = np.zeros(60)
        score[12] = 5.0
        score[17] = 9.0
        assert peak_second(score, 10, 20) == 17

    def test_ignores_higher_scores_outside_the_window(self):
        score = np.zeros(60)
        score[3] = 99.0
        score[17] = 9.0
        assert peak_second(score, 10, 20) == 17

    def test_window_past_the_end_is_clamped(self):
        score = np.zeros(10)
        score[9] = 1.0
        assert peak_second(score, 5, 500) == 9


class TestFormatTimestamp:
    def test_under_an_hour(self):
        assert format_timestamp(75) == "1:15"

    def test_over_an_hour_gains_a_field(self):
        assert format_timestamp(3725) == "1:02:05"


class TestAttribution:
    def test_segment_is_explained_by_its_peak_second(self):
        sig = _signals(object={17: 10.0}, audio={17: 5.0}, motion_peak={12: 2.0})
        score = _score(sig)
        rep = build_report(
            video_path="a.mp4", video_duration=60, score=score, signals=sig,
            segments=[(10, 20)],
            object_detections={17: ["person", "dog"]},
        )
        seg = rep["segments"][0]
        assert seg["second"] == 17
        assert seg["breakdown"]["object"] == 10.0
        assert seg["breakdown"]["audio"] == 5.0
        # A signal that fired elsewhere in the window is not credited to the peak.
        assert seg["breakdown"]["motion_peak"] == 0.0
        assert seg["objects"] == ["person", "dog"]
        assert seg["score"] == 15.0

    def test_missing_signal_arrays_report_zero(self):
        """A caller that never ran a detector should not have to fake one."""
        sig = {"object": np.full(60, 3.0)}
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=np.full(60, 3.0), signals=sig,
                           segments=[(0, 10)])
        assert rep["segments"][0]["breakdown"]["audio"] == 0.0
        assert rep["segments"][0]["breakdown"]["object"] == 3.0

    def test_segments_are_indexed_and_sorted_by_time(self):
        sig = _signals(object={5: 1.0, 45: 1.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig,
                           segments=[(40, 50), (0, 10)])
        assert [s["index"] for s in rep["segments"]] == [1, 2]
        assert rep["segments"][0]["start"] == 0.0

    def test_totals(self):
        sig = _signals(object={5: 1.0})
        rep = build_report(video_path="a.mp4", video_duration=100,
                           score=_score(sig), signals=sig,
                           segments=[(0, 10), (20, 30)])
        assert rep["totals"]["segments"] == 2
        assert rep["totals"]["duration"] == 20.0
        assert rep["totals"]["coverage_pct"] == 20.0


class TestActions:
    def test_confidence_tiers(self):
        sig = _signals(action={17: 12.0})
        rep = build_report(
            video_path="a.mp4", video_duration=60, score=_score(sig), signals=sig,
            segments=[(10, 20)],
            actions_by_sec={17: [("jumping", 0.9)]},
            action_percentiles={"jumping": {"50th": 0.4, "90th": 0.8}},
        )
        assert rep["segments"][0]["actions"][0]["tier"] == "bonus"

    def test_tier_is_none_without_percentiles(self):
        sig = _signals(action={17: 12.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           actions_by_sec={17: [("jumping", 0.9)]})
        assert rep["segments"][0]["actions"][0]["tier"] is None

    def test_detected_action_counts_as_a_signal_even_at_zero_points(self):
        """'Require objects' can suppress an action's points while it was still
        detected — it must still count toward the multi-signal total."""
        sig = _signals(audio={17: 5.0})          # action array stays zero
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           actions_by_sec={17: [("jumping", 0.9)]})
        assert "action" in rep["segments"][0]["signals_present"]


class TestBoost:
    def test_boost_reported_when_score_exceeds_the_sum(self):
        sig = _signals(audio={17: 5.0}, object={17: 10.0}, keyword={17: 3.0})
        score = _score(sig, boost_at=17, boost_points=3.6)
        rep = build_report(video_path="a.mp4", video_duration=60, score=score,
                           signals=sig, segments=[(10, 20)],
                           boost_multiplier=1.2, min_signals_for_boost=2)
        boost = rep["segments"][0]["boost"]
        assert boost["applied"] is True
        assert boost["signal_count"] == 3
        assert round(boost["points"], 2) == 3.6

    def test_no_boost_when_only_one_signal_fired(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           boost_multiplier=1.2, min_signals_for_boost=2)
        assert rep["segments"][0]["boost"]["applied"] is False


class TestNearMisses:
    def test_reports_high_scorers_outside_every_segment(self):
        sig = _signals(object={5: 10.0, 40: 8.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(0, 10)])
        seconds = [n["second"] for n in rep["near_misses"]]
        assert 40 in seconds
        assert 5 not in seconds, "a second inside a kept segment is not a near miss"

    def test_adjacent_seconds_collapse_to_one_row(self):
        sig = _signals(object={40: 9.0, 41: 8.5, 42: 8.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(0, 10)])
        assert len(rep["near_misses"]) == 1

    def test_zero_scoring_seconds_are_not_near_misses(self):
        sig = _signals(object={5: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(0, 10)])
        assert rep["near_misses"] == []

    def test_can_be_disabled(self):
        sig = _signals(object={5: 10.0, 40: 8.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(0, 10)],
                           near_miss_count=0)
        assert rep["near_misses"] == []


class TestThumbnails:
    def test_injected_bytes_become_a_data_uri(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           thumbnail_fn=lambda sec: b"\xff\xd8jpegbytes")
        assert rep["segments"][0]["thumbnail"].startswith("data:image/jpeg;base64,")

    def test_a_failing_extractor_does_not_break_the_report(self):
        def boom(sec):
            raise RuntimeError("no such frame")

        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           thumbnail_fn=boom)
        assert "thumbnail" not in rep["segments"][0]


class TestSerialisable:
    def test_report_round_trips_through_json(self):
        """numpy scalars are not JSON-serialisable; the record must be plain."""
        sig = _signals(object={17: 10.0}, audio={17: 5.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           actions_by_sec={17: [("jumping", np.float32(0.9))]},
                           settings={"clip_time": 10})
        assert json.loads(json.dumps(rep))["segments"][0]["score"] == 15.0


class TestRenderers:
    def _report(self):
        sig = _signals(object={17: 10.0}, audio={17: 5.0}, action={45: 4.0})
        score = _score(sig)
        return build_report(
            video_path=r"D:\clips\my video.mp4", video_duration=60,
            score=score, signals=sig, segments=[(10, 20)],
            object_detections={17: ["person"]},
            actions_by_sec={45: [("jumping", 0.9)]},
            settings={"clip_time": 10},
        )

    def test_text_lists_each_segment_and_its_signals(self):
        text = render_text(self._report())
        assert "1 segment(s)" in text
        assert "Objects: 10.0" in text
        assert "person" in text

    def test_html_is_self_contained(self):
        page = render_html(self._report())
        assert page.startswith("<!doctype html>")
        # Nothing may be fetched when the file is opened.
        for token in ("http://", "https://", "<script", "src=\"/"):
            assert token not in page, f"external reference in report: {token}"
        assert "<style>" in page

    def test_html_escapes_untrusted_names(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           object_detections={17: ['<img onerror=alert(1)>']})
        page = render_html(rep)
        assert "<img onerror" not in page
        assert "&lt;img onerror" in page

    def test_near_miss_table_only_rendered_when_there_are_some(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           near_miss_count=0)
        assert "Scored well, but not included" not in render_html(rep)

    def test_write_report_emits_both_files(self, tmp_path):
        rep = self._report()
        html_path = tmp_path / "r.html"
        json_path = tmp_path / "r.json"
        write_report(rep, str(html_path), str(json_path))

        assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
        assert json.loads(json_path.read_text(encoding="utf-8"))["schema"] == 1
