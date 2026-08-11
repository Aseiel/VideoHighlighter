"""Tests for `modules.highlight_swap` — "give me a different clip".

No Qt and no video: a session is built from a report record, which is the only
thing the real UI has to hand after a run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from modules.highlight_report import build_report, write_report
from modules.highlight_swap import REPORT_SUFFIX, SwapSession, report_path_for


def _report(n=400, clip_time=10, **kw):
    keys = ("scene", "motion_event", "motion_peak", "audio",
            "keyword", "object", "action")
    sig = {k: np.zeros(n) for k in keys}
    for sec, points in ((60, 10.0), (150, 9.0), (250, 8.0), (340, 7.0), (30, 6.0)):
        if sec < n:
            sig["object"][sec] = points
    score = sum(sig.values())
    return build_report(video_path="a.mp4", video_duration=n, score=score,
                        signals=sig, segments=[(55, 65), (145, 155)],
                        settings={"clip_time": clip_time, "duration_mode": "MAX"},
                        **kw)


def _total(segments):
    return sum(end - start for start, end in segments)


class TestReportPath:
    def test_prefers_the_output_name_over_the_source_name(self, tmp_path):
        out = tmp_path / "cut.mp4"
        (tmp_path / f"cut{REPORT_SUFFIX}").write_text("{}", encoding="utf-8")
        (tmp_path / f"src{REPORT_SUFFIX}").write_text("{}", encoding="utf-8")
        found = report_path_for(str(tmp_path / "src.mp4"), str(out))
        assert found.endswith(f"cut{REPORT_SUFFIX}")

    def test_falls_back_to_the_source_name(self, tmp_path):
        (tmp_path / f"src{REPORT_SUFFIX}").write_text("{}", encoding="utf-8")
        found = report_path_for(str(tmp_path / "src.mp4"), str(tmp_path / "cut.mp4"))
        assert found.endswith(f"src{REPORT_SUFFIX}")

    def test_none_when_no_report_was_written(self, tmp_path):
        assert report_path_for(str(tmp_path / "src.mp4")) is None


class TestSession:
    def test_built_from_a_record(self):
        session = SwapSession.from_report(_report())
        assert session.usable
        assert session.segments == [(55.0, 65.0), (145.0, 155.0)]
        assert session.clip_time == 10

    def test_built_from_a_file_on_disk(self, tmp_path):
        path = tmp_path / "r.json"
        write_report(_report(), str(tmp_path / "r.html"), str(path))
        session = SwapSession.from_report(str(path))
        assert session.usable
        assert len(session.score) == 400

    def test_auto_segmentation_falls_back_to_the_average_clip_length(self):
        """clip_time 0 means variable-length clips; there is no window to reuse."""
        session = SwapSession.from_report(_report(clip_time=0))
        assert session.clip_time == 10

    def test_a_report_without_a_score_is_not_usable(self):
        record = _report()
        record["curves"]["score_per_second"] = []
        assert SwapSession.from_report(record).usable is False

    def test_swap_replaces_only_that_clip(self):
        session = SwapSession.from_report(_report())
        assert session.swap(0) is True
        assert (55.0, 65.0) not in session.segments
        assert (145.0, 155.0) in session.segments

    def test_swap_keeps_the_cut_the_same_length(self):
        session = SwapSession.from_report(_report())
        before = _total(session.segments)
        session.swap(1)
        assert _total(session.segments) == before

    def test_repeated_swaps_never_repeat_a_moment(self):
        session = SwapSession.from_report(_report())
        seen = {tuple(session.segments)}
        for _ in range(3):
            if not session.swap(0):
                break
            assert tuple(session.segments) not in seen
            seen.add(tuple(session.segments))

    def test_swap_reports_failure_when_nothing_is_left(self):
        record = _report(n=120)
        record["curves"]["score_per_second"] = [0.0] * 120
        record["curves"]["score_per_second"][60] = 10.0
        session = SwapSession.from_report(record)
        session.segments = [(55.0, 65.0)]
        assert session.swap(0) is False
        assert session.segments == [(55.0, 65.0)], "a failed swap changes nothing"

    def test_a_rejected_moment_is_remembered(self):
        session = SwapSession.from_report(_report())
        session.swap(0)
        assert (55.0, 65.0) in session.rejected

    def test_undo_restores_the_previous_clip(self):
        session = SwapSession.from_report(_report())
        before = list(session.segments)
        session.swap(0)
        assert session.undo() is True
        assert session.segments == before

    def test_undo_with_nothing_to_undo_says_so(self):
        assert SwapSession.from_report(_report()).undo() is False

    def test_undo_keeps_the_cut_the_same_length(self):
        session = SwapSession.from_report(_report())
        before = _total(session.segments)
        session.swap(0)
        session.undo()
        assert _total(session.segments) == before

    def test_an_out_of_range_index_raises(self):
        session = SwapSession.from_report(_report())
        with pytest.raises(IndexError):
            session.swap(99)

    def test_the_session_survives_a_json_round_trip(self, tmp_path):
        path = tmp_path / "r.json"
        write_report(_report(), str(tmp_path / "r.html"), str(path))
        reloaded = json.loads(path.read_text(encoding="utf-8"))
        session = SwapSession.from_report(reloaded)
        assert session.swap(0) is True
