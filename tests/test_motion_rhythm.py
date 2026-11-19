"""Tests for `modules.vision.motion_rhythm`: steady beat vs. motion without one.

Every case here is a failure the first versions had on a real film: the
"strongest autocorrelation" picked the shortest lag for any smooth signal, and
a regular pattern at the edge of the band (the codec's frame-type pulse) read
as a beat. Synthetic flow curves pin both without a video file.
"""
from __future__ import annotations

import numpy as np

from modules.vision import motion_rhythm as mr

FPS = 25.0


def _seconds(n):
    return np.arange(int(n * FPS)) / FPS


def test_a_steady_beat_scores_high_and_reports_its_rate():
    t = _seconds(12)
    flow = 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    out = mr.per_second(flow, FPS)
    mid = out["rhythm"][3:9]
    assert min(mid) > 0.8
    assert all(abs(hz - 2.0) < 0.2 for hz in out["beat_hz"][3:9])


def test_smooth_motion_without_a_beat_scores_low():
    # A slow drift plus noise: the old "largest autocorrelation" read this as
    # a beat at the shortest lag, because smooth signals resemble themselves.
    rng = np.random.default_rng(0)
    t = _seconds(12)
    flow = 1.0 + 0.4 * np.sin(2 * np.pi * 0.1 * t) + 0.05 * rng.standard_normal(len(t))
    out = mr.per_second(flow, FPS)
    assert max(out["rhythm"][2:10]) < 0.4


def test_a_still_frame_has_no_rhythm():
    flow = np.full(int(10 * FPS), 0.01)
    out = mr.per_second(flow, FPS)
    assert out["rhythm"] == [0.0] * 10
    assert out["beat_hz"] == [0.0] * 10


def test_beat_and_then_none_is_told_apart_second_by_second():
    rng = np.random.default_rng(1)
    t = _seconds(20)
    beat = 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)
    irregular = 1.0 + 0.5 * rng.standard_normal(len(t))
    flow = np.where(t < 10, beat, irregular)
    r = mr.per_second(flow, FPS)["rhythm"]
    assert min(r[2:7]) > 0.7
    assert max(r[13:18]) < 0.4


def test_one_value_per_whole_second():
    out = mr.per_second(np.ones(int(7.5 * FPS)), FPS)
    assert len(out["level"]) == len(out["rhythm"]) == len(out["beat_hz"]) == 7


def test_beats_outside_the_band_are_not_reported():
    # 6.25 Hz is a P frame every 4 frames at 25 fps: codec structure, not motion.
    t = _seconds(12)
    flow = 1.0 + 0.5 * np.sin(2 * np.pi * 6.25 * t)
    out = mr.per_second(flow, FPS)
    assert all(hz == 0.0 or hz <= mr.MAX_HZ + 0.5 for hz in out["beat_hz"])


def test_analyse_reuses_its_cache(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    calls = []

    def fake_flow(frames):
        calls.append(1)
        t = _seconds(6)
        return 1.0 + 0.5 * np.sin(2 * np.pi * 2.0 * t)

    monkeypatch.setattr(mr, "_frame_rate", lambda p: FPS)
    monkeypatch.setattr(mr, "_decode", lambda *a, **k: iter(()))
    monkeypatch.setattr(mr, "flow_magnitudes", fake_flow)
    first = mr.analyse(str(video), cache_dir=str(tmp_path), log_fn=lambda *_: None)
    second = mr.analyse(str(video), cache_dir=str(tmp_path), log_fn=lambda *_: None)
    assert first == second and len(first["rhythm"]) == 6
    assert len(calls) == 1


def test_analyse_returns_empty_when_the_rate_is_unreadable(tmp_path, monkeypatch):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(mr, "_frame_rate", lambda p: 0.0)
    assert mr.analyse(str(video), cache_dir=str(tmp_path), log_fn=lambda *_: None) == {}


def test_gather_offers_the_motion_signals_only_when_asked(tmp_path, monkeypatch):
    from modules.rules import composition_signals as cs

    seen = []

    def fake_analyse(path, **kw):
        seen.append(path)
        return {"level": [1.0], "rhythm": [0.9], "beat_hz": [2.0]}

    monkeypatch.setattr(mr, "analyse", fake_analyse)
    got = cs.gather("v.mp4", cache_dir=str(tmp_path), needed={"motion_rhythm"},
                    log_fn=lambda *_: None)
    assert got["motion_rhythm"] == [0.9] and got["motion_beat_hz"] == [2.0]
    cs.gather("v.mp4", cache_dir=str(tmp_path), needed={"audio_level"},
              log_fn=lambda *_: None)
    assert len(seen) == 1
