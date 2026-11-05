"""Render mode "auto" and the Main-tier hardware HEVC args.

"auto" sends VR footage (wider than 4096 px, or side-by-side) to the CPU
encoder, which headset players accept, and everything else to the hardware
encoder. Hardware HEVC is asked for Main tier explicitly: ffmpeg's hevc_qsv
defaults to High tier, which the VR sources and libx265 never use.
"""

from __future__ import annotations

import pytest

from modules.system import encoder_select as es


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for cache in (es._size_cache, es._fps_cache, es._chain_cache, es._vr_cache):
        cache.clear()
    monkeypatch.setattr(es, "_available_encoders",
                        lambda ffmpeg=None: "hevc_qsv h264_qsv hevc_nvenc h264_nvenc")
    monkeypatch.setattr(es, "preferred_gpu_vendor", lambda: "intel")
    yield
    for cache in (es._size_cache, es._fps_cache, es._chain_cache, es._vr_cache):
        cache.clear()


def _size(monkeypatch, w, h, fps=60.0):
    monkeypatch.setattr(es, "probe_video_size", lambda path, ffmpeg=None: (w, h))
    monkeypatch.setattr(es, "probe_video_fps", lambda path: fps)


@pytest.mark.parametrize("w,h,fps,level", [
    (1920, 1080, 30, 40),
    (1920, 1080, 60, 41),
    (3840, 2160, 30, 50),
    (3840, 2160, 60, 51),
    (5760, 2880, 60, 60),     # 16.6 Mpx is over level 5's picture size
    (8192, 4096, 60, 61),
    (8192, 4096, 30, 60),
    (8192, 4096, 120, 62),
    (0, 0, 0, 52),
])
def test_hevc_level(w, h, fps, level):
    assert es.hevc_level(w, h, fps) == level


def test_qsv_hevc_is_main_tier_with_a_level(monkeypatch):
    _size(monkeypatch, 8192, 4096, 59.94)
    chain = dict(es.encoder_chain("vr.mp4", mode="gpu"))
    args = chain["hevc_qsv"]
    assert args[args.index("-tier") + 1] == "main"
    assert args[args.index("-level") + 1] == "61"


def test_nvenc_hevc_is_main_tier(monkeypatch):
    _size(monkeypatch, 8192, 4096)
    monkeypatch.setattr(es, "preferred_gpu_vendor", lambda: "nvidia")
    args = dict(es.encoder_chain("vr.mp4", mode="gpu"))["hevc_nvenc"]
    assert args[args.index("-tier") + 1] == "main"


def test_auto_sends_wide_footage_to_the_cpu(monkeypatch):
    _size(monkeypatch, 8192, 4096)
    assert es.resolve_mode("vr.mp4", "auto") == "cpu"
    assert [n for n, _ in es.encoder_chain("vr.mp4", mode="auto")] == ["libx265"]


def test_auto_sends_side_by_side_footage_to_the_cpu(monkeypatch):
    _size(monkeypatch, 3840, 1920)
    from modules.media import vr_detect
    monkeypatch.setattr(vr_detect, "probe",
                        lambda p: vr_detect.Layout(True, 3840, 1920, "halves match"))
    assert es.resolve_mode("sbs.mp4", "auto") == "cpu"
    assert [n for n, _ in es.encoder_chain("sbs.mp4", mode="auto")] == ["libx264"]


def test_auto_sends_flat_footage_to_the_gpu(monkeypatch):
    _size(monkeypatch, 1920, 1080)
    from modules.media import vr_detect
    monkeypatch.setattr(vr_detect, "probe",
                        lambda p: vr_detect.Layout(False, 1920, 1080, "too narrow"))
    assert es.resolve_mode("flat.mp4", "auto") == "gpu"
    assert [n for n, _ in es.encoder_chain("flat.mp4", mode="auto")] == ["h264_qsv", "libx264"]


def test_explicit_modes_are_kept(monkeypatch):
    _size(monkeypatch, 8192, 4096)
    assert es.resolve_mode("vr.mp4", "gpu") == "gpu"
    assert es.resolve_mode("vr.mp4", "cpu") == "cpu"


def test_unknown_mode_is_auto(monkeypatch):
    _size(monkeypatch, 8192, 4096)
    assert es.resolve_mode("vr.mp4", "fast") == "cpu"
