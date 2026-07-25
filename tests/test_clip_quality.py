"""
Tests for clip-quality sharpness scoring.

These tests decode real video, so they need the real cv2 and a working ffmpeg
(resolved the same way the engine resolves it, via modules.app_paths). Both
fixtures are generated on the fly: testsrc2 has fine detail, and piping the
same source through boxblur=10 destroys it — the Laplacian variance of the
two must be far apart. The is_blurry threshold is placed between the two
measured scores rather than hardcoded, because absolute Laplacian values
shift with codec, resolution, and ffmpeg build.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def _real_cv2():
    """conftest shims cv2 so logic-only CI can run without it; these tests
    need the real library. Swap the shim out, restoring it if cv2 is absent
    so later-collected modules still import cleanly."""
    mod = sys.modules.get("cv2")
    if mod is not None and not isinstance(mod, MagicMock):
        return mod
    sys.modules.pop("cv2", None)
    try:
        import cv2
        return cv2
    except ImportError:
        if mod is not None:
            sys.modules["cv2"] = mod
        return None


cv2 = _real_cv2()
if cv2 is None:
    pytest.skip("real cv2 required to decode video fixtures", allow_module_level=True)

from modules import clip_quality
from modules.app_paths import ffmpeg_exe


def _make_video(tmp_path_factory, name: str, vf: str | None) -> str:
    out = tmp_path_factory.mktemp("clips") / name
    cmd = [ffmpeg_exe(), "-v", "error", "-y",
           "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:duration=2"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-pix_fmt", "yuv420p", str(out)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"ffmpeg unavailable or cannot generate fixtures: {e}")
    return str(out)


@pytest.fixture(scope="session")
def sharp_video(tmp_path_factory):
    return _make_video(tmp_path_factory, "sharp.mp4", None)


@pytest.fixture(scope="session")
def blurry_video(tmp_path_factory):
    return _make_video(tmp_path_factory, "blurry.mp4", "boxblur=10")


def test_laplacian_sharpness_orders_synthetic_frames():
    flat = np.full((100, 100), 128, dtype=np.uint8)
    checker = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)
    assert clip_quality.laplacian_sharpness(flat) == 0.0
    assert clip_quality.laplacian_sharpness(checker) > 0.0


def test_sharp_scores_well_above_blurry(sharp_video, blurry_video):
    sharp = clip_quality.sample_sharpness(sharp_video, 0.0, 2.0)
    blurry = clip_quality.sample_sharpness(blurry_video, 0.0, 2.0)
    assert sharp is not None
    assert blurry is not None
    # boxblur=10 flattens testsrc2's fine detail; anything less than a wide
    # gap here means the sampling or grayscale path is broken.
    assert sharp > blurry * 5


def test_is_blurry_with_threshold_between_measured_scores(sharp_video, blurry_video):
    sharp = clip_quality.sample_sharpness(sharp_video, 0.0, 2.0)
    blurry = clip_quality.sample_sharpness(blurry_video, 0.0, 2.0)
    threshold = (sharp + blurry) / 2.0
    assert clip_quality.is_blurry(blurry, threshold)
    assert not clip_quality.is_blurry(sharp, threshold)


def test_downscale_still_produces_a_score(sharp_video):
    assert clip_quality.sample_sharpness(sharp_video, 0.0, 2.0, max_dim=160) is not None


def test_nonexistent_file_returns_none(tmp_path):
    assert clip_quality.sample_sharpness(str(tmp_path / "missing.mp4"), 0.0, 1.0) is None


def test_none_score_is_never_blurry():
    # Missing data must not penalize a clip.
    assert clip_quality.is_blurry(None, 60.0) is False
