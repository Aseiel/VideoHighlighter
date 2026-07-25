"""
Tests for modules.video_probe — the single-call ffprobe wrapper.

The rotation sign convention is the part worth pinning hard: ffprobe reports
the displaymatrix angle counter-clockwise (a sideways phone recording shows
-90), while probe_video speaks player-clockwise. Fixtures are generated with
the engine's own ffmpeg; ``-display_rotation -90`` writes the same matrix a
rotate-90-CW phone clip carries, so that fixture must probe as rotation == 90.
``-display_rotation 90`` is the opposite direction (ffprobe reports +90,
player-clockwise 270) and gets its own test so the sign can never silently
flip. On ffmpeg builds that write neither a displaymatrix nor the legacy
rotate tag, the file-based rotation tests skip rather than lie.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from modules import video_probe
from modules.app_paths import ffmpeg_exe


def _ffmpeg(*args):
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True, check=True, timeout=120,
    )


def _raw_rotation(path):
    """What ffprobe itself reports, bypassing the module under test.

    Returns the displaymatrix rotation, else the rotate tag, else None —
    used to confirm fixture metadata actually stuck before asserting on it.
    """
    out = subprocess.run(
        [video_probe.ffprobe_exe(), "-v", "error", "-print_format", "json",
         "-show_streams", path],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout
    stream = json.loads(out)["streams"][0]
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            return float(side["rotation"])
    rotate = (stream.get("tags") or {}).get("rotate")
    return None if rotate is None else float(rotate)


@pytest.fixture(scope="session")
def plain_clip(tmp_path_factory):
    """A 2s 320x240 30fps testsrc2 clip with no rotation metadata."""
    path = str(tmp_path_factory.mktemp("probe") / "plain.mp4")
    try:
        _ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
                "-pix_fmt", "yuv420p", path)
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"ffmpeg cannot generate fixtures: {e}")
    return path


@pytest.fixture(scope="session")
def rotated_clip(plain_clip, tmp_path_factory):
    """plain_clip plus sideways-recording metadata (player must rotate 90 CW).

    ffmpeg 5+'s -display_rotation takes counter-clockwise degrees, so the
    matrix a rotate-90-CW clip carries is written with -90 (ffprobe then
    reports rotation: -90, matching real phone footage). Builds without the
    option fall back to the legacy rotate tag, which is already clockwise.
    """
    out = str(tmp_path_factory.mktemp("probe_rot") / "rotated.mp4")
    try:
        _ffmpeg("-display_rotation", "-90", "-i", plain_clip, "-c", "copy", out)
        if _raw_rotation(out) is not None:
            return out
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        _ffmpeg("-i", plain_clip, "-metadata:s:v:0", "rotate=90", "-c", "copy", out)
        if _raw_rotation(out) is not None:
            return out
    except (OSError, subprocess.CalledProcessError):
        pass
    pytest.skip("ffmpeg writes no detectable rotation metadata "
                "(-display_rotation and the rotate tag are both ignored)")


# ---------------------------------------------------------------------------
# Real files
# ---------------------------------------------------------------------------
def test_plain_clip_metadata(plain_clip):
    info = video_probe.probe_video(plain_clip)
    assert info["rotation"] == 0
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["duration"] == pytest.approx(2.0, abs=0.2)
    assert info["fps"] == pytest.approx(30.0)


def test_sideways_clip_probes_as_90_clockwise(rotated_clip):
    assert video_probe.probe_video(rotated_clip)["rotation"] == 90


def test_rotation_leaves_stored_dimensions_alone(rotated_clip):
    info = video_probe.probe_video(rotated_clip)
    assert (info["width"], info["height"]) == (320, 240)
    assert info["duration"] == pytest.approx(2.0, abs=0.2)


def test_get_rotation_convenience(plain_clip, rotated_clip):
    assert video_probe.get_rotation(plain_clip) == 0
    assert video_probe.get_rotation(rotated_clip) == 90


def test_ccw_matrix_maps_to_270_clockwise(plain_clip, tmp_path):
    """-display_rotation +90 is the OTHER direction: ffprobe reports +90
    (counter-clockwise), which under the player-clockwise convention is 270."""
    out = str(tmp_path / "ccw.mp4")
    try:
        _ffmpeg("-display_rotation", "90", "-i", plain_clip, "-c", "copy", out)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("-display_rotation unsupported by this ffmpeg")
    if _raw_rotation(out) is None:
        pytest.skip("-display_rotation wrote no detectable metadata")
    assert video_probe.probe_video(out)["rotation"] == 270


# ---------------------------------------------------------------------------
# Parsing convention (no ffmpeg needed) — real fixtures can only exercise the
# metadata the local ffmpeg agrees to write, so the sign and normalization
# rules are pinned on fabricated ffprobe stream dicts as well.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    (0, 0),
    (-90, 90),    # the classic portrait phone clip
    (90, 270),
    (180, 180),
    (-180, 180),
    (-270, 270),
    (270, 90),
    (-450, 90),   # over-360 stays mod 360
    (630, 90),
])
def test_displaymatrix_is_ccw_result_is_player_cw(raw, expected):
    stream = {"side_data_list": [{"side_data_type": "Display Matrix",
                                  "rotation": raw}]}
    assert video_probe._rotation_from_stream(stream) == expected


@pytest.mark.parametrize("raw,expected", [
    ("90", 90),
    ("270", 270),
    ("-90", 270),  # negative tag normalizes mod 360
    ("450", 90),
])
def test_legacy_rotate_tag_is_already_clockwise(raw, expected):
    assert video_probe._rotation_from_stream({"tags": {"rotate": raw}}) == expected


def test_missing_rotation_is_zero():
    assert video_probe._rotation_from_stream({}) == 0
    assert video_probe._rotation_from_stream(
        {"tags": {}, "side_data_list": []}) == 0


def test_displaymatrix_wins_over_stale_rotate_tag():
    stream = {"side_data_list": [{"rotation": -90}], "tags": {"rotate": "180"}}
    assert video_probe._rotation_from_stream(stream) == 90


def test_garbage_metadata_is_zero():
    assert video_probe._rotation_from_stream(
        {"tags": {"rotate": "sideways"}}) == 0
    assert video_probe._rotation_from_stream(
        {"side_data_list": [{"rotation": "??"}]}) == 0


def test_fps_fraction_parsing():
    assert video_probe._fps({"r_frame_rate": "30000/1001"}) == pytest.approx(29.97, abs=0.01)
    assert video_probe._fps({"r_frame_rate": "0/0", "avg_frame_rate": "25/1"}) == 25.0
    assert video_probe._fps({}) == 0.0
