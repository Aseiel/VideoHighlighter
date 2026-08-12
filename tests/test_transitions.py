"""
Tests for modules.transitions against real ffmpeg.

Two things can go wrong here and only one of them is visible in a duration
check. The offsets can be wrong, which shows up as a reel of the wrong length;
or the filtergraph can be built correctly and blend nothing, which does not.
So the transition tests sample actual pixels through the join: a crossfade from
red to blue must be purple in the middle, and a dip to black must be black
there. A reel that is exactly the right length and cuts hard would pass every
timing assertion in this file and fail those.

Clips are generated at 160x120 so a whole reel builds in a second or two.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from modules.app_paths import ffmpeg_exe
from modules.transitions import (
    DEFAULT_DURATION,
    MIN_DURATION,
    TRANSITIONS,
    ReelCancelled,
    Transition,
    build_reel,
    duration_for_bars,
    normalise_kind,
    plan_transitions,
)

FFMPEG = ffmpeg_exe()


def _ffmpeg_ok() -> bool:
    try:
        return subprocess.run([FFMPEG, "-version"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _ffmpeg_ok(), reason="ffmpeg not available")


def _clip(path, colour="red", duration=3.0, size="160x120", rate=30):
    """One solid-colour clip with a silent audio track."""
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c={colour}:size={size}:rate={rate}:duration={duration}",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
         "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         str(path)],
        check=True, capture_output=True)
    return str(path)


def _duration(path) -> float:
    from modules.video_probe import probe_video
    return probe_video(path)["duration"]


def _rgb_at(path, t, tmp_path):
    """Average RGB of the frame at ``t``, for asserting a blend really happened."""
    raw = str(tmp_path / "probe.rawvideo")
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-ss", str(t), "-i", str(path),
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", raw],
        check=True, capture_output=True)
    data = open(raw, "rb").read()
    pixels = [data[i:i + 3] for i in range(0, len(data), 3)]
    count = len(pixels) or 1
    return tuple(round(sum(p[c] for p in pixels) / count) for c in range(3))


@pytest.fixture
def two_clips(tmp_path):
    return [_clip(tmp_path / "a.mp4", "red"), _clip(tmp_path / "b.mp4", "blue")]


@pytest.fixture
def three_clips(tmp_path):
    return [_clip(tmp_path / f"{n}.mp4", c)
            for n, c in (("a", "red"), ("b", "green"), ("c", "blue"))]


# ---------------------------------------------------------------------------
# The part a duration check cannot see
# ---------------------------------------------------------------------------

def test_crossfade_actually_blends_the_pictures(two_clips, tmp_path):
    """Midway through a red-to-blue crossfade the frame must be neither."""
    out = str(tmp_path / "reel.mp4")
    build_reel(two_clips, out, transitions=[Transition(0, "crossfade", 1.0)],
               log_fn=lambda *_: None)

    start = _rgb_at(out, 0.5, tmp_path)
    middle = _rgb_at(out, 2.5, tmp_path)
    end = _rgb_at(out, 4.5, tmp_path)

    assert start[0] > 200 and start[2] < 50, "first clip is not red"
    assert end[2] > 200 and end[0] < 50, "last clip is not blue"
    assert middle[0] > 60 and middle[2] > 60, f"midpoint {middle} is not a blend"


def test_dip_to_black_passes_through_black(two_clips, tmp_path):
    """A dip is not a dissolve: the midpoint must be dark, not purple."""
    out = str(tmp_path / "reel.mp4")
    build_reel(two_clips, out, transitions=[Transition(0, "dip_to_black", 1.0)],
               log_fn=lambda *_: None)

    middle = _rgb_at(out, 2.5, tmp_path)

    assert sum(middle) < 150, f"midpoint {middle} is not near black"


def test_a_cut_does_not_blend(two_clips, tmp_path):
    """The control for the two tests above — with no transition, every frame
    belongs to exactly one clip."""
    out = str(tmp_path / "reel.mp4")
    build_reel(two_clips, out, kind="cut", log_fn=lambda *_: None)

    before = _rgb_at(out, 2.8, tmp_path)
    after = _rgb_at(out, 3.2, tmp_path)

    assert before[0] > 200 and before[2] < 50
    assert after[2] > 200 and after[0] < 50


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def test_transitions_shorten_the_reel_by_their_own_length(three_clips, tmp_path):
    """Nine seconds of clips joined by two 1s crossfades is a 7s reel — the
    overlap is time the reel does not spend."""
    out = str(tmp_path / "reel.mp4")
    build_reel(three_clips, out, kind="crossfade", duration=1.0,
               log_fn=lambda *_: None)

    assert _duration(out) == pytest.approx(7.0, abs=0.3)


def test_a_reel_of_cuts_keeps_the_full_length(three_clips, tmp_path):
    out = str(tmp_path / "reel.mp4")
    build_reel(three_clips, out, kind="cut", log_fn=lambda *_: None)

    assert _duration(out) == pytest.approx(9.0, abs=0.3)


def test_a_hard_cut_before_a_blend_still_renders(three_clips, tmp_path):
    """The ordering that shipped broken.

    The concat *filter* cannot feed xfade — ffmpeg reports "Could not open
    encoder before EOF" and writes nothing — so a graph that expressed cuts
    inline worked only while no reel happened to start with one. Runs of cuts
    are joined by the demuxer first, outside the graph, so order stops
    mattering.
    """
    out = str(tmp_path / "reel.mp4")

    build_reel(three_clips, out,
               transitions=[Transition(0, "cut", 0.0),
                            Transition(1, "crossfade", 0.6)],
               log_fn=lambda *_: None)

    assert _duration(out) == pytest.approx(8.4, abs=0.3)


def test_cuts_and_blends_alternating_render(tmp_path):
    """Several runs of different lengths, to exercise the partitioning rather
    than one lucky arrangement."""
    clips = [_clip(tmp_path / f"{i}.mp4", c, duration=2.0)
             for i, c in enumerate(["red", "green", "blue", "yellow", "white"])]
    out = str(tmp_path / "reel.mp4")

    build_reel(clips, out,
               transitions=[Transition(0, "cut", 0.0),
                            Transition(1, "crossfade", 0.4),
                            Transition(2, "cut", 0.0),
                            Transition(3, "dip_to_black", 0.4)],
               log_fn=lambda *_: None)

    assert _duration(out) == pytest.approx(10.0 - 0.8, abs=0.4)


def test_mixed_joins_only_lose_the_blended_ones(three_clips, tmp_path):
    out = str(tmp_path / "reel.mp4")
    build_reel(three_clips, out,
               transitions=[Transition(0, "dip_to_black", 0.6),
                            Transition(1, "cut", 0.0)],
               log_fn=lambda *_: None)

    assert _duration(out) == pytest.approx(8.4, abs=0.3)


def test_a_transition_longer_than_its_clips_is_clamped(tmp_path):
    """ffmpeg fails outright rather than clamping, and it fails *after* the
    normalise pass has already run — so an over-long request must be shrunk
    before it gets there, not turned into a late error."""
    clips = [_clip(tmp_path / "a.mp4", "red", duration=1.0),
             _clip(tmp_path / "b.mp4", "blue", duration=1.0)]
    out = str(tmp_path / "reel.mp4")

    build_reel(clips, out, kind="crossfade", duration=5.0, log_fn=lambda *_: None)

    # Clamped to a third of the shorter clip, so barely any time is lost.
    assert _duration(out) == pytest.approx(1.67, abs=0.3)


# ---------------------------------------------------------------------------
# Delivery size
# ---------------------------------------------------------------------------

def test_the_canvas_can_be_overridden_for_delivery(two_clips, tmp_path):
    """Camera footage arrives at whatever it was shot at; the reel is allowed
    to be a sane size."""
    from modules.video_probe import probe_video

    out = str(tmp_path / "reel.mp4")
    build_reel(two_clips, out, kind="crossfade", duration=0.5,
               width=64, height=48, log_fn=lambda *_: None)

    info = probe_video(out)
    assert (info["width"], info["height"]) == (64, 48)


def test_odd_dimensions_are_made_even(two_clips, tmp_path):
    """yuv420p cannot represent an odd width, and ffmpeg's error for it is
    obscure."""
    from modules.video_probe import probe_video

    out = str(tmp_path / "reel.mp4")
    build_reel(two_clips, out, kind="crossfade", duration=0.5,
               width=65, height=49, log_fn=lambda *_: None)

    info = probe_video(out)
    assert info["width"] % 2 == 0 and info["height"] % 2 == 0


# ---------------------------------------------------------------------------
# Planning and validation
# ---------------------------------------------------------------------------

def test_plan_makes_one_transition_per_join():
    assert len(plan_transitions(5)) == 4
    assert len(plan_transitions(1)) == 0
    assert len(plan_transitions(0)) == 0


def test_plan_can_place_a_transition_every_nth_join():
    """So a reel gets a dip at each section change without dissolving through
    every single cut."""
    plan = plan_transitions(7, kind="dip_to_black", every=3, other="cut")

    assert [t.kind for t in plan] == [
        "dip_to_black", "cut", "cut", "dip_to_black", "cut", "cut"]


def test_unknown_transition_is_refused_not_silently_cut():
    """Rendering twenty minutes of hard cuts because of a typo gives the user
    no way to discover the typo."""
    with pytest.raises(ValueError, match="unknown transition"):
        normalise_kind("wipe_lefy")
    with pytest.raises(ValueError, match="unknown transition"):
        plan_transitions(3, kind="nope")


def test_transition_names_are_accepted_loosely():
    assert normalise_kind("Dip-To-Black") == "dip_to_black"
    assert normalise_kind(" crossfade ") == "crossfade"


def test_every_named_transition_maps_to_something_ffmpeg_knows():
    assert TRANSITIONS["cut"] == ""
    assert all(v for k, v in TRANSITIONS.items() if k != "cut")


def test_no_valid_inputs_raises(tmp_path):
    with pytest.raises(ValueError, match="No valid input"):
        build_reel([str(tmp_path / "nope.mp4")], str(tmp_path / "out.mp4"),
                   log_fn=lambda *_: None)


def test_cancel_is_reported_as_reel_cancelled(three_clips, tmp_path):
    with pytest.raises(ReelCancelled):
        build_reel(three_clips, str(tmp_path / "reel.mp4"), kind="crossfade",
                   duration=0.5, log_fn=lambda *_: None,
                   cancel_check=lambda: True)


def test_cancelling_leaves_the_previous_output_alone(three_clips, tmp_path):
    """The reel is staged in a temp dir and only moved at the end, so a
    cancelled re-render cannot destroy the reel from last time."""
    out = tmp_path / "reel.mp4"
    out.write_bytes(b"previous reel")

    with pytest.raises(ReelCancelled):
        build_reel(three_clips, str(out), kind="crossfade", duration=0.5,
                   log_fn=lambda *_: None, cancel_check=lambda: True)

    assert out.read_bytes() == b"previous reel"


# ---------------------------------------------------------------------------
# Beat timing
# ---------------------------------------------------------------------------

class _Analysis:
    def __init__(self, bpm, meter=4):
        self.beat_interval = 60.0 / bpm if bpm else 0.0
        self.meter = meter


def test_transition_length_can_come_from_the_bar():
    """At 120 BPM a bar is 2 seconds, so half a bar is 1."""
    assert duration_for_bars(_Analysis(120)) == pytest.approx(1.0)
    assert duration_for_bars(_Analysis(120), bars=1) == pytest.approx(2.0)
    assert duration_for_bars(_Analysis(60), bars=0.25) == pytest.approx(1.0)


def test_bar_timing_falls_back_without_a_tempo():
    """A track that could not be analysed must still produce a sane reel."""
    assert duration_for_bars(_Analysis(0)) == DEFAULT_DURATION
    assert duration_for_bars(None) == DEFAULT_DURATION
    assert duration_for_bars(_Analysis(120), bars=0.0) == MIN_DURATION
