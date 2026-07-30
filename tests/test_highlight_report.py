"""
Tests for `modules.highlight_report` — the "why was this moment chosen" record.

Pure numpy and stdlib, no Qt, no cv2, no video file: thumbnails are injected
through `thumbnail_fn`, which is the reason that parameter exists.
"""

from __future__ import annotations

import json
import re

import numpy as np

from modules.highlight_report import (
    boxes_by_second,
    build_report,
    downsample,
    format_timestamp,
    peak_second,
    render_html,
    render_text,
    score_from_report,
    segments_from_report,
    split_tags,
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
        assert json.loads(json_path.read_text(encoding="utf-8"))["schema"] == 2


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
