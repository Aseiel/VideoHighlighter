"""
Tests for `modules.music_track`.

Real end-to-end runs against ffmpeg: a synthetic 4 s video (testsrc2 + 440 Hz
sine) gets a 2 s 220 Hz music bed applied in every mode. The pinned
guarantees: the video stream is bit-copied (codec unchanged), duration is
preserved, and exactly one audio stream comes out. A silent video exercises
the mix/duck → replace degradation path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest

from modules.app_paths import ffmpeg_exe
from modules import music_track
from modules.music_track import apply_music

try:
    from modules import video_probe  # noqa: F401
except ImportError:
    # Parallel-dev shim: modules/video_probe.py is a pinned sibling contract
    # that may not have landed yet. Satisfy the one call music_track makes
    # (duration, for the replace-mode fade-out) with an ffprobe stand-in.
    _stub = types.ModuleType("modules.video_probe")

    def _probe_video(path):
        out = subprocess.run(
            [music_track._ffprobe_exe(), "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True, timeout=30)
        return {"duration": float(out.stdout.strip()), "width": 0,
                "height": 0, "fps": 0.0, "rotation": 0}

    _stub.probe_video = _probe_video
    sys.modules["modules.video_probe"] = _stub


def _ffmpeg_ok() -> bool:
    try:
        subprocess.run([ffmpeg_exe(), "-version"],
                       capture_output=True, check=True, timeout=15)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ffmpeg_ok(), reason="ffmpeg not available on this machine")


# ---------------------------------------------------------------------------
# Fixtures: synthetic media, built once per session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def media(tmp_path_factory):
    root = tmp_path_factory.mktemp("music_track")
    video = str(root / "input.mp4")
    silent = str(root / "silent.mp4")
    music = str(root / "bed.wav")
    subprocess.run(
        [ffmpeg_exe(), "-y",
         "-f", "lavfi", "-i", "testsrc2=duration=4:size=320x240:rate=30",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", video],
        capture_output=True, check=True, timeout=120)
    subprocess.run(
        [ffmpeg_exe(), "-y",
         "-f", "lavfi", "-i", "testsrc2=duration=4:size=320x240:rate=30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", silent],
        capture_output=True, check=True, timeout=120)
    subprocess.run(
        [ffmpeg_exe(), "-y",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=2", music],
        capture_output=True, check=True, timeout=120)
    return {"video": video, "silent": silent, "music": music}


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------
def _probe(*args) -> str:
    out = subprocess.run(
        [music_track._ffprobe_exe(), "-v", "error", *args],
        capture_output=True, text=True, check=True, timeout=30)
    return out.stdout.strip()


def _duration(path) -> float:
    return float(_probe("-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", path))


def _audio_stream_count(path) -> int:
    out = _probe("-select_streams", "a", "-show_entries", "stream=index",
                 "-of", "csv=p=0", path)
    return len([ln for ln in out.splitlines() if ln.strip()])


def _video_codec(path) -> str:
    out = _probe("-select_streams", "v:0", "-show_entries",
                 "stream=codec_name",
                 "-of", "default=noprint_wrappers=1:nokey=1", path)
    return next(ln.strip() for ln in out.splitlines() if ln.strip())


# ---------------------------------------------------------------------------
# Every mode: valid output, duration kept, one audio stream, video untouched
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["replace", "mix", "duck"])
def test_mode_produces_valid_output(media, tmp_path, mode):
    out = str(tmp_path / f"out_{mode}.mp4")
    result = apply_music(media["video"], media["music"], out, mode=mode,
                         log_fn=lambda *_: None)
    assert result == out
    assert os.path.exists(out)
    assert abs(_duration(out) - 4.0) <= 0.5
    assert _audio_stream_count(out) == 1
    assert _video_codec(out) == _video_codec(media["video"])


# ---------------------------------------------------------------------------
# Silent source: mix/duck degrade to replace instead of failing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["mix", "duck"])
def test_no_audio_video_degrades_to_replace(media, tmp_path, mode):
    logs = []
    out = str(tmp_path / f"fallback_{mode}.mp4")
    result = apply_music(media["silent"], media["music"], out, mode=mode,
                         log_fn=logs.append)
    assert result == out
    assert os.path.exists(out)
    assert abs(_duration(out) - 4.0) <= 0.5
    assert _audio_stream_count(out) == 1
    assert _video_codec(out) == _video_codec(media["silent"])
    assert any("replace" in msg for msg in logs)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def test_unknown_mode_raises(media, tmp_path):
    with pytest.raises(ValueError, match="mode"):
        apply_music(media["video"], media["music"],
                    str(tmp_path / "x.mp4"), mode="karaoke")


def test_missing_video_raises(media, tmp_path):
    with pytest.raises(ValueError, match="video"):
        apply_music(str(tmp_path / "nope.mp4"), media["music"],
                    str(tmp_path / "x.mp4"))


def test_missing_music_raises(media, tmp_path):
    with pytest.raises(ValueError, match="music"):
        apply_music(media["video"], str(tmp_path / "nope.wav"),
                    str(tmp_path / "x.mp4"))


def test_in_place_output_raises(media):
    with pytest.raises(ValueError, match="in.?place|overwrite"):
        apply_music(media["video"], media["music"], media["video"])
