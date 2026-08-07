"""
Tests for `modules.highlight_report` — the "why was this moment chosen" record.

Pure numpy and stdlib, no Qt, no cv2, no video file: thumbnails are injected
through `thumbnail_fn`, which is the reason that parameter exists.
"""

from __future__ import annotations

import json
import os
import re

import pytest

import numpy as np

from modules.highlight_report import (
    SILENCE_DBFS,
    boxes_by_second,
    build_report,
    downsample,
    format_timestamp,
    peak_second,
    render_html,
    percentile_rank,
    render_text,
    score_from_report,
    segments_from_report,
    split_tags,
    to_dbfs,
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
        assert json.loads(json_path.read_text(encoding="utf-8"))["schema"] == 3


class TestTagGrouping:
    """Detections and composed events arrive in one list; the report splits them."""

    def test_split_tags_separates_composed_events(self):
        objects, events = split_tags(["cup", "table", "table_set"],
                                     composed_event_names=["table_set"])
        assert objects == ["cup", "table"]
        assert events == ["table_set"]

    def test_split_tags_without_a_composed_list_calls_everything_an_object(self):
        objects, events = split_tags(["cup", "table_set"])
        assert objects == ["cup", "table_set"]
        assert events == []

    def test_report_reports_the_two_kinds_separately(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           object_detections={17: ["cup", "table", "table_set"]},
                           composed_event_names=["table_set"])
        entry = rep["segments"][0]
        assert entry["objects"] == ["cup", "table"]
        assert entry["events"] == ["table_set"]

    def test_the_page_labels_each_group(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           object_detections={17: ["cup", "table_set"]},
                           composed_event_names=["table_set"])
        page = render_html(rep)
        assert '<span class="kind">objects</span>' in page
        assert '<span class="kind">events</span>' in page

    def test_a_group_with_nothing_in_it_is_not_rendered(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           object_detections={17: ["cup"]})
        page = render_html(rep)
        assert '<span class="kind">objects</span>' in page
        assert '<span class="kind">events</span>' not in page

    def test_event_names_are_escaped_like_everything_else(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           object_detections={17: ["<b>x</b>"]},
                           composed_event_names=["<b>x</b>"])
        page = render_html(rep)
        assert "<b>x</b>" not in page
        assert "&lt;b&gt;x&lt;/b&gt;" in page


class TestCurves:
    def test_downsample_keeps_peaks_rather_than_averaging_them_away(self):
        values = np.zeros(1000)
        values[500] = 9.0
        assert max(downsample(values, points=10)) == 9.0

    def test_downsample_leaves_short_arrays_alone(self):
        assert downsample([1.0, 2.0, 3.0], points=100) == [1.0, 2.0, 3.0]

    def test_downsample_of_nothing_is_nothing(self):
        assert downsample([]) == []

    def test_report_carries_a_score_curve(self):
        sig = _signals(n=600, object={300: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=600,
                           score=_score(sig, n=600), signals=sig,
                           segments=[(295, 305)])
        assert max(rep["curves"]["score"]) == 10.0

    def test_waveform_triples_are_reduced_to_their_rms(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           waveform=[(-0.5, 0.5, 0.25)] * 60)
        assert rep["curves"]["audio"] and max(rep["curves"]["audio"]) == 0.25

    def test_bare_amplitudes_work_too(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           waveform=[0.1] * 30 + [0.9] * 30)
        assert max(rep["curves"]["audio"]) == 0.9

    def test_no_waveform_means_no_audio_curve_and_no_crash(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)])
        assert rep["curves"]["audio"] == []
        assert "Loudness" not in render_html(rep)


class TestOverviewAndSummary:
    def _rep(self, **kw):
        sig = _signals(n=600, object={100: 10.0}, audio={400: 4.0})
        return build_report(video_path="a.mp4", video_duration=600,
                            score=_score(sig, n=600), signals=sig,
                            segments=[(95, 105), (395, 405)], **kw)

    def test_overview_draws_a_block_per_kept_segment(self):
        page = render_html(self._rep())
        assert page.count('fill="#5ac8b0"') >= 2

    def test_overview_is_plain_svg_with_nothing_fetched(self):
        page = render_html(self._rep(waveform=[0.4] * 600))
        assert "<svg" in page
        for token in ("http://", "https://", "<script"):
            assert token not in page

    def test_signal_totals_add_up_across_the_whole_cut(self):
        rep = self._rep()
        assert rep["signal_totals"]["object"] == 10.0
        assert rep["signal_totals"]["audio"] == 4.0

    def test_signals_that_scored_nothing_are_left_out(self):
        assert "keyword" not in self._rep()["signal_totals"]

    def test_a_single_source_of_points_is_called_out(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)])
        page = render_html(rep)
        assert "Every point in this highlight came from" in page

    def test_no_such_warning_when_several_signals_contributed(self):
        assert "Every point in this highlight came from" not in render_html(self._rep())

    def test_summary_is_skipped_when_nothing_scored(self):
        sig = _signals(n=60)
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=np.zeros(60), signals=sig, segments=[(10, 20)])
        assert "What decided the cut" not in render_html(rep)

    def test_a_zero_length_video_does_not_divide_by_zero(self):
        sig = _signals(object={5: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=0,
                           score=_score(sig), signals=sig, segments=[(0, 10)])
        assert render_html(rep).startswith("<!doctype html>")


class TestBoxes:
    _THUMB = "data:image/jpeg;base64,AAAA"

    def _cache(self, sec=17):
        return [{"timestamp": float(sec),
                 "objects": ["cup", "table_set"],
                 "bboxes": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 0.2]],
                 "confidences": [0.9, 0.7]}]

    def _report(self, **kw):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           object_detections={17: ["cup", "table_set"]},
                           composed_event_names=["table_set"],
                           thumbnail_fn=lambda sec: b"\x00\x00\x00",
                           **kw)
        return rep

    def test_cache_is_indexed_by_second(self):
        indexed = boxes_by_second(self._cache())
        assert list(indexed) == [17]
        assert indexed[17][0]["name"] == "cup"
        assert indexed[17][0]["box"] == [0.1, 0.2, 0.3, 0.4]

    def test_a_record_with_no_boxes_is_skipped(self):
        assert boxes_by_second([{"timestamp": 3.0, "objects": [], "bboxes": []}]) == {}

    def test_malformed_boxes_do_not_crash_the_report(self):
        indexed = boxes_by_second([{"timestamp": 1.0, "objects": ["x"],
                                    "bboxes": [[0.1, 0.2]], "confidences": [0.5]}])
        assert indexed == {}

    def test_no_cache_at_all_is_fine(self):
        assert boxes_by_second(None) == {}

    def test_boxes_land_on_the_segment_for_their_second(self):
        rep = self._report(bbox_cache=self._cache())
        assert len(rep["segments"][0]["boxes"]) == 2

    def test_boxes_are_absent_when_no_cache_was_given(self):
        assert "boxes" not in self._report()["segments"][0]

    def test_boxes_render_as_percentage_positioned_overlays(self):
        page = render_html(self._report(bbox_cache=self._cache()))
        assert 'class="bx"' in page
        assert "left:10.0%;top:20.0%;width:30.0%;height:40.0%" in page

    def test_a_composed_event_box_is_styled_apart_from_a_detection(self):
        page = render_html(self._report(bbox_cache=self._cache()))
        assert 'class="bx evt"' in page          # table_set
        assert 'class="bx"' in page              # cup

    def test_box_labels_are_escaped(self):
        cache = self._cache()
        cache[0]["objects"] = ["<script>x</script>", "table_set"]
        page = render_html(self._report(bbox_cache=cache))
        assert "<script>" not in page
        assert "&lt;script&gt;" in page

    def test_no_thumbnail_means_no_overlay_and_no_crash(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           bbox_cache=self._cache())
        page = render_html(rep)
        assert 'class="bx' not in page
        assert page.startswith("<!doctype html>")


class TestSegmentAudio:
    """A clip's loudness strip must be its own slice, at its own resolution.

    Sliced out of the page-wide curve instead, a 30-second clip in a long video
    gets a handful of samples and draws as a straight ramp — which reads as a
    real trend when it is only interpolation.
    """

    def _long_report(self, **kw):
        n = 3600
        sig = _signals(n=n, object={1800: 10.0})
        # 4 samples/sec, as the pipeline extracts: one sharp burst mid-clip.
        wave = [0.05] * (n * 4)
        for i in range(1800 * 4, 1800 * 4 + 8):
            wave[i] = 0.9
        return build_report(video_path="a.mp4", video_duration=n,
                            score=_score(sig, n=n), signals=sig,
                            segments=[(1790, 1820)], waveform=wave, **kw)

    def test_each_clip_carries_its_own_envelope(self):
        entry = self._long_report()["segments"][0]
        assert len(entry["audio"]) > 50

    def test_a_short_burst_survives_into_the_clip_strip(self):
        """The whole point: the page-wide curve would flatten this away."""
        entry = self._long_report()["segments"][0]
        assert max(entry["audio"]) == 0.9
        quiet = [v for v in entry["audio"] if v < 0.5]
        assert quiet, "a strip that is loud everywhere has lost the burst"

    def test_the_global_peak_is_recorded_for_shared_scaling(self):
        assert self._long_report()["curves"]["audio_peak"] == 0.9

    def test_a_quiet_clip_is_not_stretched_to_full_height(self):
        n = 600
        sig = _signals(n=n, object={100: 10.0, 400: 10.0})
        wave = [0.02] * (n * 4)
        for i in range(400 * 4, 400 * 4 + 20):
            wave[i] = 1.0
        rep = build_report(video_path="a.mp4", video_duration=n,
                           score=_score(sig, n=n), signals=sig,
                           segments=[(95, 105)], waveform=wave)
        page = render_html(rep)
        # Only the clip's own strip, not the page-wide curves above it.
        strip = re.search(r'viewBox="0 0 400 26".*?<path d="([^"]+)"', page)
        assert strip, "no per-clip strip rendered"
        ys = [float(m) for m in re.findall(r",(\d+\.\d\d)", strip.group(1))]
        # Drawn against the video's peak (1.0), the quiet clip must stay near
        # its baseline of 24 rather than reaching the top of the 22px band.
        assert ys and min(ys) > 20.0

    def test_the_strip_explains_that_volume_did_not_decide_the_pick(self):
        page = render_html(self._long_report())
        assert "contributed no points to this pick" in page

    def test_the_strip_says_so_when_audio_did_score(self):
        n = 600
        sig = _signals(n=n, object={100: 10.0}, audio={100: 6.0})
        rep = build_report(video_path="a.mp4", video_duration=n,
                           score=_score(sig, n=n), signals=sig,
                           segments=[(95, 105)], waveform=[0.3] * (n * 4))
        assert "contributed 6 points" in render_html(rep)

    def test_an_older_record_without_per_clip_audio_still_renders(self):
        rep = self._long_report()
        for entry in rep["segments"]:
            entry.pop("audio", None)
        assert 'class="wave"' in render_html(rep)


class TestReportAsSwapInput:
    """A saved report must be enough to re-choose a clip, with no video."""

    def _report(self):
        n = 400
        sig = _signals(n=n, object={60: 10.0, 150: 9.0, 250: 8.0, 340: 7.0})
        return build_report(video_path="a.mp4", video_duration=n,
                            score=_score(sig, n=n), signals=sig,
                            segments=[(55, 65), (145, 155)])

    def test_the_full_score_survives_the_round_trip(self, tmp_path):
        path = tmp_path / "r.json"
        write_report(self._report(), str(tmp_path / "r.html"), str(path))
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        score = score_from_report(reloaded)
        assert len(score) == 400
        assert score[60] == 10.0

    def test_segments_come_back_as_ranges(self):
        assert segments_from_report(self._report()) == [(55.0, 65.0), (145.0, 155.0)]

    def test_a_report_can_be_swapped_without_the_video(self, tmp_path):
        from modules.highlight_select import swap_segment

        path = tmp_path / "r.json"
        write_report(self._report(), str(tmp_path / "r.html"), str(path))
        reloaded = json.loads(path.read_text(encoding="utf-8"))

        swapped = swap_segment(
            score_from_report(reloaded),
            segments=segments_from_report(reloaded),
            index=0, video_duration=400, clip_time=10,
        )
        assert swapped is not None
        assert (55.0, 65.0) not in swapped
        assert (145.0, 155.0) in swapped
        assert sum(e - s for s, e in swapped) == 20.0

    def test_an_older_report_without_the_curve_yields_an_empty_score(self):
        assert len(score_from_report({"segments": []})) == 0


class TestAdviceSection:
    """Findings attached to the record must reach the page."""

    def _rep(self):
        sig = _signals(n=600, object={100: 10.0})
        return build_report(video_path="a.mp4", video_duration=600,
                            score=_score(sig, n=600), signals=sig,
                            segments=[(95, 105)],
                            settings={"clip_time": 10, "object_points": 5})

    def test_no_advice_means_no_section(self):
        assert "What to try next" not in render_html(self._rep())

    def test_findings_are_rendered_with_their_remedy(self):
        rep = self._rep()
        rep["advice"] = [{
            "id": "single_signal", "severity": "high",
            "title": "Only one kind of evidence decided this highlight",
            "detail": "All 10 points came from objects.",
            "remedy": "Give another signal a weight.", "topic": "weights",
        }]
        page = render_html(rep)
        assert "What to try next" in page
        assert "Give another signal a weight." in page
        assert 'class="find sev-high"' in page

    def test_a_narration_is_shown_above_the_findings(self):
        rep = self._rep()
        rep["advice"] = [{"id": "x", "severity": "low", "title": "T",
                          "detail": "D", "remedy": "REMEDY-MARKER",
                          "topic": "weights"}]
        rep["advice_narration"] = "Start by enabling audio peaks."
        page = render_html(rep)
        assert "Start by enabling audio peaks." in page
        assert page.index("Start by enabling") < page.index("REMEDY-MARKER")

    def test_advice_text_is_escaped(self):
        rep = self._rep()
        rep["advice"] = [{"id": "x", "severity": "low",
                          "title": "<script>x</script>", "detail": "d",
                          "remedy": "r", "topic": "weights"}]
        page = render_html(rep)
        assert "<script>x</script>" not in page
        assert "&lt;script&gt;" in page


class TestMeasurements:
    """Physical facts, not points — the substrate for explaining a moment."""

    def _rep(self, n=600, segments=None, **kw):
        sig = _signals(n=n, **kw.pop("signals", {}))
        return build_report(video_path="a.mp4", video_duration=n,
                            score=_score(sig, n=n), signals=sig,
                            segments=segments or [(95, 105)], **kw)

    def test_percentile_rank_is_a_position_not_a_value(self):
        values = np.sort(np.array([1.0, 2.0, 3.0, 4.0]))
        assert percentile_rank(values, 1.0) == 12.5
        assert percentile_rank(values, 3.0) == 62.5
        assert percentile_rank(values, 99.0) == 100.0

    def test_ties_share_the_middle_rather_than_the_bottom(self):
        """A run where every kept moment scored the same is not a run of
        bottom-ranked moments — they are simply all alike."""
        values = np.full(20, 15.0)
        assert percentile_rank(values, 15.0) == 50.0

    def test_percentile_of_nothing_is_zero_not_a_crash(self):
        assert percentile_rank(np.array([]), 5.0) == 0.0

    def test_dbfs_conversion(self):
        assert to_dbfs(1.0) == 0.0
        assert to_dbfs(0.5) == -6.0
        assert to_dbfs(0.1) == -20.0

    def test_silence_reports_a_floor_not_negative_infinity(self):
        assert to_dbfs(0.0) == SILENCE_DBFS
        assert np.isfinite(to_dbfs(0.0))

    def test_a_top_moment_ranks_near_the_top(self):
        """Against a real spread of activity, not against three data points."""
        points = {sec: 1.0 + (sec % 5) for sec in range(200, 500, 5)}
        points[100] = 20.0                     # the clip's peak, highest of all
        rep = self._rep(signals={"object": points}, segments=[(95, 105)])
        assert rep["segments"][0]["measured"]["score_percentile"] > 95

    def test_loudness_is_reported_in_dbfs_with_its_rank(self):
        n = 600
        wave = [0.01] * (n * 4)
        for i in range(100 * 4, 100 * 4 + 8):
            wave[i] = 0.5                      # -6 dBFS burst inside the clip
        rep = self._rep(signals={"object": {100: 10.0}}, waveform=wave)
        measured = rep["segments"][0]["measured"]
        assert measured["loudness_dbfs"] == -6.0
        assert measured["loudness_percentile"] > 90

    def test_no_waveform_means_no_loudness_claim(self):
        measured = self._rep(signals={"object": {100: 10.0}})["segments"][0]["measured"]
        assert "loudness_dbfs" not in measured

    def test_each_contributing_signal_gets_a_rank(self):
        rep = self._rep(signals={"object": {100: 10.0, 300: 1.0},
                                 "audio": {100: 8.0, 400: 1.0}})
        signals = rep["segments"][0]["measured"]["signals"]
        assert set(signals) == {"object", "audio"}
        assert signals["object"]["percentile"] == 75.0   # top of two, midranked
        assert signals["object"]["at"] == 100

    def test_signals_that_never_fired_in_the_clip_are_absent(self):
        rep = self._rep(signals={"object": {100: 10.0}, "audio": {400: 8.0}})
        assert "audio" not in rep["segments"][0]["measured"]["signals"]

    def test_signals_firing_together_are_reported_as_coinciding(self):
        rep = self._rep(signals={"object": {100: 10.0}, "audio": {100: 8.0}})
        measured = rep["segments"][0]["measured"]
        assert measured["signal_spread_seconds"] == 0.0
        assert measured["signals_coincide"] is True

    def test_signals_far_apart_are_not_a_coincidence(self):
        rep = self._rep(signals={"object": {96: 10.0}, "audio": {104: 8.0}},
                        segments=[(95, 105)])
        measured = rep["segments"][0]["measured"]
        assert measured["signal_spread_seconds"] == 8.0
        assert measured["signals_coincide"] is False

    def test_one_signal_alone_makes_no_coincidence_claim(self):
        measured = self._rep(signals={"object": {100: 10.0}})["segments"][0]["measured"]
        assert "signals_coincide" not in measured

    def test_detection_confidence_comes_from_the_boxes(self):
        rep = self._rep(signals={"object": {100: 10.0}},
                        bbox_cache=[{"timestamp": 100.0, "objects": ["a", "b"],
                                     "bboxes": [[0, 0, 1, 1], [0, 0, 1, 1]],
                                     "confidences": [0.42, 0.91]}])
        assert rep["segments"][0]["measured"]["detection_confidence"] == 0.91

    def test_measurements_are_json_serialisable(self):
        json.dumps(self._rep(signals={"object": {100: 10.0}})["segments"][0]["measured"])

    def test_the_page_states_them_in_real_units(self):
        n = 600
        wave = [0.01] * (n * 4)
        for i in range(100 * 4, 100 * 4 + 8):
            wave[i] = 0.5
        rep = self._rep(signals={"object": {100: 10.0}, "audio": {100: 8.0}},
                        waveform=wave)
        page = render_html(rep)
        assert "dBFS" in page
        assert "scored above" in page
        assert "signals landed together" in page

    def test_percentiles_ignore_the_silence_between_events(self):
        """Ranking against zeros would make any detection look exceptional."""
        points = {sec: 1.0 + (sec % 7) for sec in range(0, 600, 5)}
        points[100] = 3.0                      # unremarkable among its peers
        sig = _signals(n=600, object=points)
        rep = build_report(video_path="a.mp4", video_duration=600,
                           score=_score(sig, n=600), signals=sig,
                           segments=[(95, 105)])
        assert rep["segments"][0]["measured"]["score_percentile"] < 90


class TestConfidenceExcludesRules:
    """A rule outcome is not a detector's certainty."""

    def _rep(self, **kw):
        sig = _signals(object={17: 10.0})
        return build_report(video_path="a.mp4", video_duration=60,
                            score=_score(sig), signals=sig, segments=[(10, 20)],
                            bbox_cache=[{"timestamp": 17.0,
                                         "objects": ["cup", "table_set"],
                                         "bboxes": [[0, 0, 1, 1], [0, 0, 1, 1]],
                                         "confidences": [0.62, 1.0]}], **kw)

    def test_a_composed_event_does_not_become_the_confidence(self):
        rep = self._rep(composed_event_names=["table_set"])
        assert rep["segments"][0]["measured"]["detection_confidence"] == 0.62

    def test_without_the_composed_list_the_rule_still_dominates(self):
        """Documents why composed_event_names has to reach the measurement."""
        rep = self._rep()
        assert rep["segments"][0]["measured"]["detection_confidence"] == 1.0

    def test_only_rules_at_a_second_means_no_confidence_claim(self):
        sig = _signals(object={17: 10.0})
        rep = build_report(video_path="a.mp4", video_duration=60,
                           score=_score(sig), signals=sig, segments=[(10, 20)],
                           composed_event_names=["table_set"],
                           bbox_cache=[{"timestamp": 17.0,
                                        "objects": ["table_set"],
                                        "bboxes": [[0, 0, 1, 1]],
                                        "confidences": [1.0]}])
        assert "detection_confidence" not in rep["segments"][0]["measured"]


class TestSubjectComparison:
    """The comparative reading reaches the record and both renderers.

    `modules.highlight_compare` owns the arithmetic and is tested there; what is
    checked here is the wiring — that a caller passing the detector's boxes and
    the expression scan gets findings in the JSON, on the page and in the debug
    log, and that a caller passing neither is not penalised for it.
    """

    N = 120

    def _cache(self):
        """A steady video whose one clip holds a subject three times its usual
        size relative to the person beside it."""
        cache = []
        for sec in range(self.N):
            scale = (0.5, 1.0, 1.5)[sec % 3]
            ratio = 3.0 if 58 <= sec <= 66 else 1.0
            cache.append({
                "timestamp": float(sec),
                "objects": ["person", "dog"],
                "bboxes": [[0.1, 0.1, 0.25 * scale, 0.5 * scale],
                           [0.5, 0.5, 0.125 * scale * ratio, 0.25 * scale]],
                "confidences": [0.95, 0.88],
            })
        return cache

    def _built(self, **over):
        signals = _signals(n=self.N, object={60: 9.0})
        kwargs = dict(video_path="v.mp4", video_duration=float(self.N),
                      score=_score(signals, n=self.N), signals=signals,
                      segments=[(58.0, 68.0)], near_miss_count=0)
        kwargs.update(over)
        return build_report(**kwargs)

    def test_the_comparison_lands_in_the_record(self):
        report = self._built(bbox_cache=self._cache())
        comparison = report["segments"][0]["measured"]["comparison"]
        dog = next(s for s in comparison["subjects"] if s["name"] == "dog")
        assert dog["relative"]["reference"] == "person"
        assert dog["relative"]["percentile"] > 90.0

    def test_the_finding_reaches_the_page(self):
        page = render_html(self._built(bbox_cache=self._cache()))
        assert '<ul class="why">' in page
        assert "the person beside it" in page

    def test_the_finding_reaches_the_debug_log(self):
        assert "the person beside it" in render_text(
            self._built(bbox_cache=self._cache()))

    def test_the_expression_scan_is_accepted_in_either_shape(self):
        tuples = {sec: ("neutral", 0.7) for sec in range(self.N)}
        tuples.update({sec: ("surprise", 0.95) for sec in range(58, 68)})
        dicts = {sec: {"label": label, "confidence": conf}
                 for sec, (label, conf) in tuples.items()}
        for scan in (tuples, dicts):
            comparison = self._built(expressions=scan)["segments"][0][
                "measured"]["comparison"]
            assert comparison["expression"]["label"] == "surprise"

    def _turning_scan(self):
        scan = {sec: ("neutral", 0.9) for sec in range(self.N)}
        scan.update({sec: ("surprise", 0.9) for sec in range(63, 68)})
        return scan

    def test_the_marked_second_reaches_the_record_and_both_renderings(self):
        report = self._built(expressions=self._turning_scan(),
                             loudness_levels=[-30.0] * self.N)
        reading = report["segments"][0]["expression_peak"]
        assert reading["label"] == "surprise" and reading["second"] == 63
        assert reading["turned"] is True
        for page in (render_text(report), render_html(report)):
            assert "turns from neutral to surprise" in page

    def test_the_marked_second_is_ordered_against_the_loudest_one(self):
        # Loudest at second 60, the reading settles at 63.
        levels = [-30.0] * self.N
        levels[60] = -5.0
        report = self._built(expressions=self._turning_scan(),
                             loudness_levels=levels)
        assert "3s after the loudest point" in render_text(report)

    def test_the_marked_second_is_reachable_from_the_player(self):
        report = self._built(expressions=self._turning_scan(),
                             loudness_levels=[-30.0] * self.N)
        page = render_html(report, media_src="v.mp4")
        assert 'data-t="63"' in page and "reads surprise" in page

    def test_the_signed_comparison_reaches_both_renderings(self):
        levels = [-40.0] * self.N
        levels[60] = -5.0
        report = self._built(expressions=self._turning_scan(),
                             loudness_levels=levels)
        assert "vs the video: + loudness" in render_text(report)
        page = render_html(report)
        assert '<div class="vs">' in page and 'class="ax up"' in page

    def test_a_run_with_no_detector_and_no_faces_reports_nothing_extra(self):
        report = self._built()
        assert "comparison" not in report["segments"][0]["measured"]
        # Named sections rather than the bullet markup they share: the clip
        # still summarises itself, it just has nothing comparative to say.
        page = render_html(report)
        assert "<b>On screen</b>" not in page
        assert "<b>Face expression</b>" not in page


# --- clip players -----------------------------------------------------------

MEDIA_DIR = os.path.join(os.sep + "media", "movies")
REPORT_ELSEWHERE = os.path.join(os.sep + "media", "reports")
VIDEO_NAME = "clip test [x].mp4"


def _player_report():
    return build_report(
        video_path=os.path.join(MEDIA_DIR, VIDEO_NAME), video_duration=100.0,
        score=np.ones(101), signals={}, segments=[(10, 20), (40, 50)],
        object_detections={15: ["a"]}, actions_by_sec={},
        loudness_levels=[-30.0] * 101, settings={},
    )


def test_media_src_is_relative_and_percent_encoded():
    from modules.highlight_report import media_src_for
    rep = _player_report()
    src = media_src_for(rep, os.path.join(MEDIA_DIR, "out_why.html"))
    assert src == "clip%20test%20%5Bx%5D.mp4"
    # Brackets and spaces must not survive raw: a bare '#' or '?' in a real
    # filename would truncate the URL before the media fragment.
    assert " " not in src and "[" not in src


def test_media_src_walks_up_when_the_report_is_written_elsewhere(tmp_path):
    """Real absolute dirs, so this exercises relpath rather than one OS's spelling."""
    from modules.highlight_report import media_src_for
    movies, reports = tmp_path / "movies", tmp_path / "reports"
    movies.mkdir()
    reports.mkdir()
    rep = build_report(
        video_path=str(movies / VIDEO_NAME), video_duration=100.0,
        score=np.ones(101), signals={}, segments=[(10, 20)],
        object_detections={}, actions_by_sec={}, settings={})
    src = media_src_for(rep, str(reports / "out_why.html"))
    assert src.startswith("../movies/")
    assert src.endswith("clip%20test%20%5Bx%5D.mp4")
    assert "\\" not in src          # URLs use forward slashes on every platform


@pytest.mark.skipif(os.name != "nt",
                    reason="only Windows has volumes with no relative path between them")
def test_media_src_declines_rather_than_emitting_an_absolute_path():
    """A different drive has no relative path; a file:// URL would only work here."""
    from modules.highlight_report import media_src_for
    rep = build_report(
        video_path=r"D:\movies\clip.mp4", video_duration=10.0,
        score=np.ones(11), signals={}, segments=[(0, 10)],
        object_detections={}, actions_by_sec={}, settings={})
    assert media_src_for(rep, r"C:\elsewhere\out_why.html") is None
    assert media_src_for({"video": {}},
                         os.path.join(MEDIA_DIR, "out_why.html")) is None


def test_each_clip_gets_a_player_seeked_to_its_own_range():
    from modules.highlight_report import media_src_for
    rep = _player_report()
    page = render_html(rep, media_src=media_src_for(rep, os.path.join(MEDIA_DIR, "o.html")))
    assert page.count("<video") == 2
    assert "#t=10,20" in page and "#t=40,50" in page
    # preload matters: twelve autoloading players would open twelve connections.
    assert 'preload="none"' in page


def test_the_seek_button_targets_the_loudest_second():
    from modules.highlight_report import media_src_for
    rep = _player_report()
    page = render_html(rep, media_src=media_src_for(rep, os.path.join(MEDIA_DIR, "o.html")))
    assert re.findall(r'data-t="([0-9.]+)"', page) == ["10", "40"]


def test_the_page_is_standalone_when_no_source_is_linked():
    """Without media_src the report is exactly what it was before players."""
    page = render_html(_player_report())
    assert "<video" not in page
    assert "<script>" not in page


def test_a_missing_source_is_explained_rather_than_silent():
    from modules.highlight_report import media_src_for
    rep = _player_report()
    page = render_html(rep, media_src=media_src_for(rep, os.path.join(MEDIA_DIR, "o.html")))
    assert "Source video not found" in page
    assert "every measurement above is unaffected" in page


def test_write_report_can_opt_out_of_linking(tmp_path):
    from modules.highlight_report import write_report
    html_path = tmp_path / "r.html"
    write_report(_player_report(), str(html_path), link_media=False)
    assert "<video" not in html_path.read_text(encoding="utf-8")


# --- the cut's own timeline -------------------------------------------------

def _cut_report():
    n = 3600
    score = np.zeros(n + 1)
    score[100:120] = 20
    score[1500:1510] = 12
    score[3000:3040] = 30
    return build_report(
        video_path="x.mp4", video_duration=float(n), score=score,
        signals={"object": score.copy()},
        segments=[(100, 120), (1500, 1510), (3000, 3040)],
        object_detections={}, actions_by_sec={}, settings={},
    )


def test_output_positions_run_on_the_highlights_clock_not_the_sources():
    rep = _cut_report()
    starts = [e["output_start"] for e in rep["segments"]]
    ends = [e["output_end"] for e in rep["segments"]]
    assert starts == [0.0, 20.0, 30.0]
    assert ends == [20.0, 30.0, 70.0]
    # Each clip begins exactly where the previous ended: no gaps, because the
    # gaps are what the cut removed.
    assert starts[1:] == ends[:-1]


def test_output_length_matches_the_sum_of_the_clips():
    rep = _cut_report()
    total = sum(e["duration"] for e in rep["segments"])
    assert rep["segments"][-1]["output_end"] == total


def test_source_range_is_still_the_default_everywhere_else():
    """Output positions are an addition; every other timestamp stays source."""
    rep = _cut_report()
    first = rep["segments"][0]
    assert first["range"].startswith("1:40")        # source
    assert first["output_range"].startswith("0:00")  # output


def test_the_cut_timeline_gives_every_clip_visible_width():
    """A 10s clip in an hour is two pixels on the full-video strip; not here."""
    from modules.highlight_report import _cut_timeline
    svg = _cut_timeline(_cut_report())
    assert "The cut, end to end" in svg
    widths = [float(w) for w in re.findall(r'<rect[^>]*width="([0-9.]+)"', svg)]
    assert len(widths) == 3
    assert min(widths) > 100        # the 10s clip is 1/7 of a 1000-unit strip
    # Widths are proportional to duration: 20s, 10s, 40s.
    assert widths[2] > widths[0] > widths[1]


def test_the_cut_timeline_carries_both_clocks_for_each_clip():
    from modules.highlight_report import _cut_timeline
    svg = _cut_timeline(_cut_report())
    assert "in the output" in svg          # tooltip
    assert "In the highlight" in svg       # table heading
    assert "In the source" in svg


def test_no_clips_means_no_cut_timeline():
    from modules.highlight_report import _cut_timeline
    assert _cut_timeline({"segments": []}) == ""


# --- motion peaks -----------------------------------------------------------

def _motion_report(peaks, points=5.0):
    n = 300
    score = np.zeros(n + 1)
    score[100:130] = 20
    score[120] = 40          # unambiguous scoring second, so "nearest" has meaning
    return build_report(
        video_path=os.path.join(MEDIA_DIR, "m.mp4"), video_duration=float(n), score=score,
        signals={"motion_peak": score.copy() * (points / 20.0)},
        segments=[(100, 130)], object_detections={}, actions_by_sec={},
        motion_peaks=peaks, settings={},
    )


def test_the_peak_nearest_the_scoring_second_is_the_one_quoted():
    """A clip can contain several; the one being offered as evidence is the
    one beside the second that actually scored."""
    rep = _motion_report([102.0, 118.0, 128.0])
    peak = rep["segments"][0]["motion_peak"]
    assert peak["count"] == 3
    assert rep["segments"][0]["second"] == 120     # the second that scored
    assert peak["second"] == 118                   # the peak beside it, not 102


def test_peaks_outside_the_clip_are_not_counted():
    rep = _motion_report([50.0, 105.0, 250.0])
    assert rep["segments"][0]["motion_peak"]["count"] == 1


def test_a_clip_with_no_peak_carries_no_claim():
    rep = _motion_report([50.0, 250.0])
    assert "motion_peak" not in rep["segments"][0]


def test_the_sentence_names_the_shape_not_the_word_action():
    from modules.highlight_prose import describe_motion_peak
    said = describe_motion_peak({"motion_peak": {"second": 118,
                                                "timestamp": "1:58", "count": 1}})
    assert "spiked at 1:58" in said
    assert "dropped away after" in said
    # "action" is what the number gets misread as; the rule measures stillness.
    assert "action" not in said.lower()


def test_a_peak_that_scored_nothing_is_not_narrated():
    """It is in the breakdown already; a sentence would imply it drove the pick."""
    from modules.highlight_report import _measurements
    rep = _motion_report([118.0], points=0.0)
    entry = rep["segments"][0]
    assert "spiked at" not in _measurements(entry, [20.0])


def test_both_marked_seconds_get_a_seek_button():
    from modules.highlight_report import media_src_for, render_html
    rep = _motion_report([118.0])
    rep["segments"][0]["loudest"] = {"second": 110, "timestamp": "1:50",
                                     "level_dbfs": -12.0, "classes": []}
    page = render_html(rep, media_src=media_src_for(rep, os.path.join(MEDIA_DIR, "o.html")))
    assert "loudest second" in page and "motion peak" in page
    assert sorted(re.findall(r'data-t="([0-9.]+)"', page)) == ["110", "118"]


def test_seek_buttons_follow_the_clock_not_the_computation_order():
    """The row is a miniature timeline; it must not run backwards."""
    from modules.highlight_report import media_src_for, render_html
    rep = _motion_report([105.0])
    rep["segments"][0]["loudest"] = {"second": 125, "timestamp": "2:05",
                                     "level_dbfs": -12.0, "classes": []}
    page = render_html(rep, media_src=media_src_for(rep, os.path.join(MEDIA_DIR, "o.html")))
    assert re.findall(r'data-t="([0-9.]+)"', page) == ["105", "125"]
    # ...and the labels travel with their seconds.
    assert page.index("motion peak") < page.index("loudest second")


def test_the_later_signal_comes_second():
    from modules.highlight_report import media_src_for, render_html
    rep = _motion_report([128.0])
    rep["segments"][0]["loudest"] = {"second": 104, "timestamp": "1:44",
                                     "level_dbfs": -12.0, "classes": []}
    page = render_html(rep, media_src=media_src_for(rep, os.path.join(MEDIA_DIR, "o.html")))
    assert re.findall(r'data-t="([0-9.]+)"', page) == ["104", "128"]
    assert page.index("loudest second") < page.index("motion peak")


# --- what arrived on screen, as the mark the others order around ------------

class TestEventOnset:
    """The category channel: named by the user's own rules, not by this repo.

    Only *arrivals* count. A class on screen from a clip's first second is
    scenery — it was there before the clip started and marks nothing — and
    without that rule the mark would be "a person is present", which is true of
    almost every clip in almost every video.
    """

    def _built(self, detections, composed=None, **over):
        n = 200
        score = np.zeros(n)
        score[100:130] = 5.0
        kwargs = dict(video_path="v.mp4", video_duration=float(n), score=score,
                      signals={}, segments=[(100.0, 130.0)], near_miss_count=0,
                      object_detections=detections, settings={},
                      composed_event_names=composed)
        kwargs.update(over)
        return build_report(**kwargs)

    def _detections(self, spans):
        out = {}
        for name, start, end in spans:
            for sec in range(start, end):
                out.setdefault(sec, []).append(name)
        return out

    def test_something_arriving_mid_clip_is_the_mark(self):
        report = self._built(self._detections([("dog", 108, 125)]))
        onset = report["segments"][0]["event_onset"]
        assert onset["second"] == 108 and onset["name"] == "dog"

    def test_scenery_present_from_the_first_second_marks_nothing(self):
        report = self._built(self._detections([("person", 90, 130)]))
        assert "event_onset" not in report["segments"][0]

    def test_a_single_frame_is_a_flicker_not_an_arrival(self):
        report = self._built(self._detections([("dog", 110, 111)]))
        assert "event_onset" not in report["segments"][0]

    def test_a_user_composed_event_outranks_a_raw_detection(self):
        report = self._built(
            self._detections([("dog", 105, 128), ("routine A", 112, 120)]),
            composed=["routine A"])
        onset = report["segments"][0]["event_onset"]
        assert onset["name"] == "routine A" and onset["composed"] is True

    def test_the_arrival_leads_the_clip_sequence(self):
        levels = [-30.0] * 200
        levels[115] = -4.0
        report = self._built(self._detections([("dog", 108, 125)]),
                             loudness_levels=levels)
        said = render_text(report)
        assert "In order: dog comes on screen at 1:48" in said
        assert "the loudest point arrives 7s later" in said

    def test_the_run_level_count_reaches_both_renderings(self):
        n = 600
        score = np.zeros(n)
        segments = [(60.0, 90.0), (160.0, 190.0), (260.0, 290.0), (360.0, 390.0)]
        spans, levels = [], [-30.0] * n
        for start, _end in segments:
            start = int(start)
            score[start:start + 30] = 5.0
            spans.append(("dog", start + 5, start + 25))
            levels[start + 12] = -4.0
        report = build_report(
            video_path="v.mp4", video_duration=float(n), score=score, signals={},
            segments=segments, near_miss_count=0, settings={},
            object_detections=self._detections(spans), loudness_levels=levels)
        assert len(report["event_relations"]) == 1
        said = report["event_relations"][0]
        assert "dog arrives in 4 of the kept clips" in said
        assert said in render_text(report)
        assert "dog arrives in 4 of the kept clips" in render_html(report)


def test_a_model_summary_reaches_a_page_with_no_findings_on_it():
    """A clean run wrote the summary into the record and showed nothing.

    `_advice` returned early when there were no findings, taking the narration
    with it — so the model ran, the page was re-rendered, and the report looked
    untouched.
    """
    report = _player_report()
    report["advice"] = []
    report["advice_narration"] = "The cut is drawn from two stretches."
    page = render_html(report)
    assert "The cut is drawn from two stretches." in page
    assert "Nothing in this run was diagnosed as a problem" in page


def test_a_page_with_neither_findings_nor_a_summary_says_nothing():
    report = _player_report()
    assert "What to try next" not in render_html(report)


class TestConversation:
    """Questions asked of a report, kept with it.

    Each answer used to replace the last, so a second question destroyed the
    first — the wrong shape for the thing people do with it, which is ask again.
    """

    def _asked(self, *turns):
        report = _player_report()
        report["conversation"] = list(turns)
        return report

    def test_every_answer_is_kept_with_the_question_that_got_it(self):
        report = self._asked(
            {"asked": "What is going on here?", "answer": "A first answer.",
             "model": "some-model", "at": "2026-08-07T10:00:00"},
            {"asked": "And in the last clip?", "answer": "A second answer."})
        page = render_html(report)
        for text in ("What is going on here?", "A first answer.",
                     "And in the last clip?", "A second answer."):
            assert text in page

    def test_the_thread_names_the_model_that_answered(self):
        page = render_html(self._asked(
            {"asked": "q", "answer": "a", "model": "some-model",
             "at": "2026-08-07T10:00:00"}))
        assert "some-model" in page

    def test_it_is_never_presented_as_measured(self):
        page = render_html(self._asked({"asked": "q", "answer": "a"}))
        assert "not measured" in page

    def test_an_older_report_with_a_single_reading_still_renders(self):
        report = _player_report()
        report["reading"] = "An answer from before the thread existed."
        assert "An answer from before the thread existed." in render_html(report)

    def test_the_debug_view_carries_the_thread_too(self):
        said = render_text(self._asked({"asked": "What is going on?",
                                        "answer": "An answer."}))
        assert "Q: What is going on?" in said
        assert "A: An answer." in said

    def test_a_report_nobody_asked_anything_shows_no_thread(self):
        assert "Asked of this report" not in render_html(_player_report())
