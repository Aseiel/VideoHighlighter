"""
Tests for modules.motion — making the cut move rather than dissolve.

The claims worth pinning are the ones a render would not reveal. A filter chain
that is syntactically fine and moves nothing produces a reel indistinguishable
from one with the feature off, and a chain that quietly resamples the frame
rate produces one that drifts out of sync with the music. Both happened while
this was being written, so both are measured here through real ffmpeg.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from modules.app_paths import ffmpeg_exe
from modules.motion import (
    MOTION_LABELS,
    MOTIONS,
    apply_motion,
    motion_filter,
    normalise_motion,
)

FFMPEG = ffmpeg_exe()


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([FFMPEG, "-version"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")


@pytest.fixture(scope="module")
def marked(tmp_path_factory):
    """Two seconds of a white square on black, at 30fps.

    A square is the measurement: a zoom widens it, a shake moves its centre,
    a roll widens its bounding box without moving it.
    """
    path = str(tmp_path_factory.mktemp("motion") / "square.mp4")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-f", "lavfi",
         "-i", "color=c=black:s=320x240:r=30:d=2,"
               "drawbox=x=140:y=100:w=40:h=40:color=white:t=fill",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         path], check=True, capture_output=True)
    return path


def _box(path, t):
    """(width, centre x, centre y) of the white square at ``t``."""
    import numpy as np

    raw = path + f".{t}.gray"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", raw],
        check=True, capture_output=True)
    frame = np.frombuffer(open(raw, "rb").read(), dtype=np.uint8).reshape(240, 320)
    ys, xs = np.where(frame > 110)
    if not len(xs):
        return (0, 0.0, 0.0)
    return (int(xs.max() - xs.min() + 1), float(xs.mean()), float(ys.mean()))


def _moved(path, name, tmp_path, **kw):
    out = str(tmp_path / f"{name}-{kw.get('head', False)}.mp4")
    options = dict(duration=2.0, width=320, height=240, fps=30,
                   head=False, tail=True)
    options.update(kw)
    apply_motion(path, out, name, log_fn=lambda *_: None, **options)
    return out


def test_names_are_accepted_loosely_or_rejected_clearly():
    assert normalise_motion("Punch") == "punch"
    assert normalise_motion("  SHAKE ") == "shake"
    assert normalise_motion("") == "none"
    with pytest.raises(ValueError, match="unknown motion"):
        normalise_motion("wobble")


def test_every_motion_has_a_label():
    assert set(MOTION_LABELS) == set(MOTIONS)


def test_nothing_is_done_when_nothing_was_asked():
    """No motion, or no end to apply it at, must produce no filter at all —
    otherwise every clip pays for an extra encode that changes nothing."""
    assert motion_filter("none", duration=2, width=320, height=240, tail=True) == ""
    assert motion_filter("punch", duration=2, width=320, height=240) == ""
    assert motion_filter("punch", duration=2, width=320, height=240,
                         tail=True, strength=0) == ""


def test_a_still_clip_is_copied_through_untouched(marked, tmp_path):
    assert _box(_moved(marked, "none", tmp_path), 1.9) == _box(marked, 1.9)


def test_the_punch_grows_towards_the_cut(marked, tmp_path):
    """The whole point: the frame is largest where the cut lands."""
    out = _moved(marked, "punch", tmp_path)

    middle, end = _box(out, 1.0), _box(out, 1.95)

    assert middle[0] == pytest.approx(40, abs=1), "the middle should be untouched"
    assert end[0] > middle[0] + 3, f"no zoom: {middle[0]} -> {end[0]}"


def test_the_pull_does_the_opposite(marked, tmp_path):
    out = _moved(marked, "pull", tmp_path)

    middle, end = _box(out, 1.0), _box(out, 1.95)

    assert end[0] < middle[0] - 3, f"no pull back: {middle[0]} -> {end[0]}"


@pytest.mark.parametrize("name", ["shake", "glitch"])
def test_a_shake_moves_the_frame_without_resizing_it(marked, tmp_path, name):
    out = _moved(marked, name, tmp_path)

    middle, end = _box(out, 1.0), _box(out, 1.95)
    travel = abs(end[1] - middle[1]) + abs(end[2] - middle[2])

    assert travel > 2, f"{name} did not move the picture"


def test_a_roll_turns_the_frame(marked, tmp_path):
    """A rotated square has a wider bounding box while its centre stays put —
    which is how a roll is told apart from a shake."""
    out = _moved(marked, "roll", tmp_path)

    middle, end = _box(out, 1.0), _box(out, 1.95)

    assert end[0] > middle[0] + 1, "the frame did not turn"
    assert abs(end[1] - middle[1]) < 3, "a roll should not shift the centre"


def test_the_motion_happens_at_the_end_it_was_asked_for(marked, tmp_path):
    head = _moved(marked, "punch", tmp_path, head=True, tail=False)
    assert _box(head, 0.02)[0] > _box(head, 1.0)[0] + 3


@pytest.mark.parametrize("name", [m for m in MOTIONS if m != "none"])
def test_no_motion_changes_the_frame_rate(marked, tmp_path, name):
    """zoompan sets its own output rate and defaults to 25. Left alone it
    silently resamples every clip it touches, which puts a reel cut to music
    progressively out of time with it."""
    out = _moved(marked, name, tmp_path)
    from modules.video_probe import ffprobe_exe

    probe = subprocess.run(
        [ffprobe_exe(), "-v", "error",
         "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate",
         "-of", "default=nw=1:nk=1", out],
        capture_output=True, text=True)
    rate = (probe.stdout or "0/1").strip().splitlines()[0]
    numerator, _, denominator = rate.partition("/")
    fps = float(numerator) / float(denominator or 1)

    assert fps == pytest.approx(30, abs=1), f"{name} rendered at {fps:.1f}fps"


def test_an_unusable_clip_costs_the_motion_and_not_the_reel(tmp_path):
    src = tmp_path / "broken.mp4"
    src.write_bytes(b"not a video")
    dst = tmp_path / "out.mp4"

    apply_motion(str(src), str(dst), "punch", duration=2.0, width=320,
                 height=240, tail=True, log_fn=lambda *_: None)

    assert dst.read_bytes() == b"not a video"
