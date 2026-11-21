"""Shots run on their own, frame-accurate, and are kept by a later motion run;
sudden stops are a scored signal of their own."""
from __future__ import annotations

import pytest

from modules.report import analysis_ondemand as ao


class _Found:
    shots = [(0.0, 1.2), (1.2, 4.0)]
    method = "test"


def test_run_shots_is_a_patch_of_frame_accurate_scenes(monkeypatch, tmp_path):
    from modules.segments import shot_boundaries
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "film.mp4"
    video.write_bytes(b"x")
    calls = []
    monkeypatch.setattr(shot_boundaries, "detect_shots",
                        lambda path, **kw: calls.append(path) or _Found())
    seen = []
    patch = ao.run_shots(str(video), progress=lambda *a: seen.append(a), log=lambda *a: None)
    assert patch == {"scenes": [{"start": 0.0, "end": 1.2}, {"start": 1.2, "end": 4.0}],
                     "scenes_source": ao.SHOTS_FRAME_ACCURATE}
    assert seen
    # Cached: the second run reads the file and does not detect again.
    assert ao.run_shots(str(video), log=lambda *a: None) == patch and len(calls) == 1


def test_a_cancelled_shot_run_raises_cancelled(monkeypatch, tmp_path):
    from modules.segments import shot_boundaries
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "film.mp4"
    video.write_bytes(b"x")

    def stop(path, **kw):
        raise shot_boundaries.ShotDetectionCancelled()
    monkeypatch.setattr(shot_boundaries, "detect_shots", stop)
    with pytest.raises(ao._Cancelled):
        ao.run_shots(str(video), log=lambda *a: None)


def test_a_motion_patch_keeps_frame_accurate_shots():
    motion = {"scenes": [{"start": 0, "end": 9}], "motion_peaks": [3.0]}
    kept = ao.keep_frame_accurate_shots(motion, {"scenes_source": ao.SHOTS_FRAME_ACCURATE})
    assert "scenes" not in kept and kept["motion_peaks"] == [3.0]
    assert ao.keep_frame_accurate_shots(motion, {}) == motion
    assert ao.keep_frame_accurate_shots(motion, None) == motion


def test_sudden_stops_are_named_weighed_and_gated_like_peaks():
    from modules.report import analysis_plan, highlight_advice, highlight_prose, highlight_report
    assert ("motion_stop", "Sudden stop") in highlight_report.SIGNAL_LABELS
    assert "motion_stop" in highlight_report.BOOST_SIGNALS
    assert highlight_advice.SIGNAL_WEIGHTS["motion_stop"] == "motion_stop_points"
    assert highlight_advice.SIGNAL_NAMES["motion_stop"] == "sudden stops"
    assert "motion_stop_points" in analysis_plan.GATED_ARTIFACTS["motion"]
    assert "motion_stop" in highlight_prose.SIGNAL_PROSE


def test_the_simple_run_does_not_score_stops():
    from modules.segments.simple_run import SIMPLE_SCORING
    assert SIMPLE_SCORING["motion_stop_points"] == 0
