"""
End-to-end rotation test for modules.video_cutter.cut_video.

The trap this pins: a sideways-shot clip carries rotation metadata (a portrait
phone recording stores 320x240 landscape frames plus a "rotate 90 CW" matrix).
cut_video re-encodes (encoder chain + `-vf format=yuv420p`), and ffmpeg
autorotates by default during a re-encode — so the cut clip should come out
upright, with the rotation BAKED into the pixels: rotation metadata gone (0)
and the displayed dimensions swapped (240x320) relative to the stored input.

The only genuinely broken outcome is rotation==0 with UNSWAPPED dimensions:
that means the rotation was thrown away without being applied, and the clip
plays sideways. Preserving the metadata (rotation==90, dims unchanged) is also
acceptable — a downstream player would still display it correctly.

Fixture generation mirrors tests/test_video_probe.py: `-display_rotation -90`
writes the same displaymatrix a rotate-90-CW phone clip carries, which
probe_video reports as rotation==90. If the local ffmpeg refuses to write
detectable rotation metadata, the test skips rather than lying.
"""

from __future__ import annotations

import subprocess

import pytest

from modules import video_probe
from modules.app_paths import ffmpeg_exe
from modules.video_cutter import cut_video


def _ffmpeg(*args):
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True, text=True, check=True, timeout=120,
    )


@pytest.fixture()
def rotated_source(tmp_path):
    """A 2s 320x240 clip carrying a rotate-90-CW displaymatrix (rotation==90)."""
    plain = str(tmp_path / "plain.mp4")
    rot = str(tmp_path / "rot.mp4")
    try:
        _ffmpeg("-f", "lavfi",
                "-i", "testsrc2=size=320x240:rate=30:duration=2",
                "-pix_fmt", "yuv420p", plain)
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"ffmpeg cannot generate fixtures: {e}")

    try:
        _ffmpeg("-display_rotation", "-90", "-i", plain, "-c", "copy", rot)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("-display_rotation unsupported by this ffmpeg")

    if video_probe.probe_video(rot)["rotation"] != 90:
        pytest.skip("rotation metadata did not stick on this ffmpeg build")
    return rot


def test_cut_video_handles_rotation(rotated_source):
    """A rotated source cut must not come out playing sideways."""
    out = rotated_source.replace("rot.mp4", "cut.mp4")
    try:
        cut_video(rotated_source, 0.2, 1.2, out)
    except RuntimeError as e:
        pytest.skip(f"no working encoder available for cut_video: {e}")

    src = video_probe.probe_video(rotated_source)
    res = video_probe.probe_video(out)

    baked = (
        res["rotation"] == 0
        and (res["width"], res["height"]) == (src["height"], src["width"])
    )
    preserved = res["rotation"] == 90

    assert baked or preserved, (
        "cut_video produced a sideways clip: rotation was dropped (0) without "
        f"swapping the display dimensions. source={src}, result={res}"
    )
