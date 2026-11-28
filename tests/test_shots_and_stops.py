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


# --- shots and motion as separate scans (main window, 2026-11-28) ------------

def _cache_dir_with(tmp_path, monkeypatch, cache: dict):
    import json
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache").mkdir()
    path = tmp_path / "cache" / "abc.cache.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setattr(ao, "_cache_files", lambda vp: ("abc", [path]))
    return path


def test_the_main_window_s_motion_scan_keeps_frame_accurate_shots(tmp_path, monkeypatch):
    """Only the viewer checked before folding: Scan motion in the main window
    replaced 2044 frame-accurate cuts with the motion pass's 1129."""
    import json
    path = _cache_dir_with(tmp_path, monkeypatch, {
        "scenes": [{"start": 0.0, "end": 0.4}, {"start": 0.4, "end": 9.0}],
        "scenes_source": ao.SHOTS_FRAME_ACCURATE})
    ao.merge_into_cache("film.mp4", {"scenes": [{"start": 0.0, "end": 9.0}],
                                     "motion_peaks": [3.0]}, log=lambda *a: None)
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert len(disk["scenes"]) == 2 and disk["motion_peaks"] == [3.0]


def test_a_shots_scan_replaces_earlier_frame_accurate_shots(tmp_path, monkeypatch):
    import json
    path = _cache_dir_with(tmp_path, monkeypatch, {
        "scenes": [{"start": 0.0, "end": 9.0}], "scenes_source": ao.SHOTS_FRAME_ACCURATE})
    ao.merge_into_cache("film.mp4", {"scenes": [{"start": 0.0, "end": 4.0},
                                                {"start": 4.0, "end": 9.0}],
                                     "scenes_source": ao.SHOTS_FRAME_ACCURATE},
                        log=lambda *a: None)
    assert len(json.loads(path.read_text(encoding="utf-8"))["scenes"]) == 2


def test_saved_status_tells_shots_from_motion():
    from modules.report.scan_state import KINDS, kinds_in
    assert "shots" in KINDS and KINDS.index("shots") < KINDS.index("motion")
    accurate = {"scenes": [{"start": 0, "end": 1}],
                "scenes_source": ao.SHOTS_FRAME_ACCURATE}
    assert kinds_in(accurate) == {"shots"}                       # no motion yet
    assert kinds_in({**accurate, "motion_curve": [0.1]}) == {"shots", "motion"}
    # the motion pass's own cuts are motion's result, not a shots scan
    assert kinds_in({"scenes": [{"start": 0, "end": 1}]}) == {"motion"}


def test_the_main_window_has_a_scan_button_for_each():
    import re
    from pathlib import Path
    src = Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
    assert re.search(r'spin_scene_points, "shots", "Scan shots"', src)
    assert re.search(r'spin_motion_event_points, "motion", "Scan motion"', src)
    assert 'if self.kind == "shots":' in src
