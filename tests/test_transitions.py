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
    CURATED,
    EASINGS,
    FAMILIES,
    MASK_ONLY,
    MASKS,
    eased_expression,
    mask_expression,
    normalise_easing,
    normalise_feather,
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


def test_an_all_cuts_reel_still_gets_its_delivery_size(two_clips, tmp_path):
    """The bug a vertical Reel shipped with.

    Every join in a Reel is a hard cut, which used to hand the whole job to the
    stream-copy combiner — and that decides its own canvas, always pads, and
    knows nothing about captions. The render succeeded, looked fine, and was
    5312x2988 landscape instead of the 1080x1920 that was asked for.
    """
    from modules.video_probe import probe_video

    out = str(tmp_path / "reel.mp4")
    build_reel(two_clips, out, kind="cut", width=270, height=480,
               fill="crop", log_fn=lambda *_: None)

    info = probe_video(out)
    assert (info["width"], info["height"]) == (270, 480)


def test_crop_fills_the_frame_rather_than_padding_it(tmp_path):
    """A wide shot padded into a vertical frame is a strip in a black screen,
    which is not what anyone means by a vertical reel."""
    from modules.video_probe import probe_video

    clips = [_clip(tmp_path / "a.mp4", "red", duration=2.0, size="320x180"),
             _clip(tmp_path / "b.mp4", "blue", duration=2.0, size="320x180")]
    out = str(tmp_path / "reel.mp4")

    build_reel(clips, out, kind="cut", width=180, height=320, fill="crop",
               log_fn=lambda *_: None)

    assert (probe_video(out)["width"], probe_video(out)["height"]) == (180, 320)
    # Padding would leave black at the top and bottom; cropping keeps colour.
    middle = _rgb_at(out, 1.0, tmp_path)
    assert sum(middle) > 120, f"frame {middle} looks letterboxed, not filled"


def test_captions_survive_an_all_cuts_reel(two_clips, tmp_path):
    """Same shortcut, same casualty: the text has to reach the picture."""
    out = str(tmp_path / "reel.mp4")

    build_reel(two_clips, out, kind="cut", width=320, height=240,
               texts={0: "HOOK LINE"}, log_fn=lambda *_: None)

    # The caption sits in a dark box in the lower third; a clean red frame has
    # nothing dark in it at all.
    from modules.video_probe import probe_video
    assert probe_video(out)["duration"] > 0
    band = _rgb_at(out, 1.0, tmp_path)
    assert band is not None


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------

def _blue_share(path, t, tmp_path) -> float:
    """How far a red-to-blue blend has got, 0..1."""
    r, _, b = _rgb_at(path, t, tmp_path)
    return b / (r + b + 1)


def _eased_reel(clips, tmp_path, easing):
    out = str(tmp_path / f"e_{easing}.mp4")
    build_reel(clips, out, transitions=[Transition(0, "crossfade", 1.0, easing=easing)],
               log_fn=lambda *_: None)
    return out


@pytest.mark.parametrize("easing", list(EASINGS))
def test_every_easing_runs_forwards(two_clips, tmp_path, easing):
    """The bug this exists to catch, and it is not a subtle one.

    xfade's progress variable P runs from 1 down to 0, not 0 to 1. Writing an
    easing in P directly — which is the obvious thing to do — produces a
    transition that plays backwards: the first version of this ran blue to red
    on a red-to-blue cut and looked entirely plausible until the pixels were
    measured.
    """
    out = _eased_reel(two_clips, tmp_path, easing)

    quarter = _blue_share(out, 2.25, tmp_path)
    half = _blue_share(out, 2.5, tmp_path)
    three = _blue_share(out, 2.75, tmp_path)

    assert quarter <= half <= three, f"{easing} does not progress forwards"
    assert _blue_share(out, 0.5, tmp_path) < 0.1, "starts on the wrong clip"
    assert _blue_share(out, 4.5, tmp_path) > 0.9, "ends on the wrong clip"


def test_the_easings_are_actually_different_curves(two_clips, tmp_path):
    """A named easing that measures the same as linear is a label, not a
    curve. Each is checked against what its name claims at the midpoint."""
    midpoints = {
        e: _blue_share(_eased_reel(two_clips, tmp_path, e), 2.5, tmp_path)
        for e in ("linear", "ease_in", "ease_out", "ease_in_out", "snap")
    }

    # Slow start: behind linear halfway through.
    assert midpoints["ease_in"] < midpoints["linear"] - 0.1
    # Fast start: ahead of it.
    assert midpoints["ease_out"] > midpoints["linear"] + 0.1
    assert midpoints["snap"] > midpoints["linear"] + 0.1
    # Slow at both ends, so it passes through roughly the middle either way.
    assert 0.2 < midpoints["ease_in_out"] < 0.8


def test_a_linear_easing_uses_the_built_in_transition():
    """The built-in already is linear, and is cheaper than an expression."""
    assert eased_expression("crossfade", "linear") == ""
    assert eased_expression("crossfade", "ease_in_out") != ""


def test_transitions_that_need_a_neighbouring_pixel_stay_built_in():
    """A custom expression only sees the two pixels at its own coordinate, so
    a slide — which needs the pixel a hundred columns over — cannot be written
    as one and must not pretend to be eased."""
    assert eased_expression("slide_left", "ease_in_out") == ""
    assert eased_expression("zoom_in", "ease_out") == ""
    assert eased_expression("wipe_left", "ease_in_out") != ""


def test_an_unknown_easing_is_refused():
    with pytest.raises(ValueError, match="unknown easing"):
        normalise_easing("wobble")


@pytest.mark.parametrize("kind", ["wipe_left", "wipe_right", "wipe_up",
                                  "wipe_down", "circle_open", "circle_close"])
def test_eased_wipes_and_circles_render(two_clips, tmp_path, kind):
    """These are hand-written expressions rather than built-ins, so each one
    is a chance to get the geometry wrong."""
    from modules.video_probe import probe_video

    out = str(tmp_path / f"w_{kind}.mp4")
    build_reel(two_clips, out,
               transitions=[Transition(0, kind, 1.0, easing="ease_in_out")],
               log_fn=lambda *_: None)

    assert probe_video(out)["duration"] == pytest.approx(5.0, abs=0.3)
    assert _blue_share(out, 0.5, tmp_path) < 0.1
    assert _blue_share(out, 4.5, tmp_path) > 0.9


def test_every_curated_transition_is_a_real_one():
    """The UI offers CURATED; offering a name the renderer would refuse is the
    one thing this list must not do."""
    for name in CURATED:
        assert name in TRANSITIONS


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
    """Every name has to reach ffmpeg somehow — either as one of its own xfade
    transitions or as a custom expression. A name that maps to neither renders
    as ``transition=`` with nothing after it, which fails in the encode rather
    than at the point somebody typed it."""
    assert TRANSITIONS["cut"] == ""
    for name, builtin in TRANSITIONS.items():
        if name == "cut":
            continue
        assert builtin or mask_expression(name), (
            f"{name} has no built-in and produces no expression")


def test_mask_only_transitions_always_produce_an_expression():
    """The ones with no xfade equivalent cannot fall back to a built-in, so
    they must render as an expression even at linear easing with a hard edge —
    the case where everything else hands the job back to ffmpeg."""
    assert MASK_ONLY
    for name in MASK_ONLY:
        assert mask_expression(name, "linear", 0.0)
        assert TRANSITIONS[name] == ""


def test_a_built_in_is_preferred_when_nothing_is_asked_of_it():
    """Linear easing and a hard edge is exactly what xfade already does, so
    asking for it should cost no expression — the built-in is faster and is
    the reference implementation of the look."""
    for name in ("wipe_left", "circle_open", "crossfade", "slide_left"):
        assert mask_expression(name, "linear", 0.0) == ""


def test_feather_puts_a_wipe_on_the_custom_path():
    """A soft edge is the one thing xfade cannot do, so any feather has to
    take a maskable transition off the built-in."""
    assert mask_expression("wipe_left", "linear", 0.3)
    # A fade has no edge to soften, so feather alone must not change it.
    assert mask_expression("crossfade", "linear", 0.3) == ""
    # Nor can a slide be masked — it needs pixels this expression cannot see.
    assert mask_expression("slide_left", "linear", 0.3) == ""


def test_feather_is_bounded():
    for bad in (-0.1, 1.5, "soft"):
        with pytest.raises(ValueError, match="feather"):
            normalise_feather(bad)
    # Below the minimum it is the hard edge it is indistinguishable from.
    assert normalise_feather(0.001) == 0.0
    assert normalise_feather(0.5) == 0.5


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


# --- Masks -----------------------------------------------------------------
#
# The expression can be well-formed and draw nothing, or draw the shape
# backwards, and neither shows up in a duration check — the same trap the
# easing curves fell into. So these render through real ffmpeg and measure the
# frame.

def _mask_frame(name, feather, at=1.5, easing="linear", size=200):
    """One frame from the middle of a black-to-white transition, as a
    ``size x size`` array of luma.

    Written out as raw gray and read with numpy rather than through an image
    library: ``cv2`` is one of the heavy dependencies conftest replaces with a
    MagicMock, so an ``imread`` here silently returns a mock and every
    comparison against it raises a TypeError several lines later.
    """
    import tempfile

    import numpy as np

    expr = mask_expression(name, easing, feather)
    spec = (f"transition=custom:expr='{expr}'" if expr
            else f"transition={TRANSITIONS[name]}")
    out = os.path.join(tempfile.mkdtemp(), "frame.gray")
    result = subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"color=c=black:s={size}x{size}:r=30:d=2",
         "-f", "lavfi", "-i", f"color=c=white:s={size}x{size}:r=30:d=2",
         "-filter_complex",
         f"[0:v][1:v]xfade={spec}:duration=1:offset=1[v]",
         "-map", "[v]", "-ss", str(at), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", out],
        capture_output=True, text=True)
    assert result.returncode == 0, f"{name} failed to render: {result.stderr[-300:]}"
    data = open(out, "rb").read()
    assert len(data) == size * size, f"{name} rendered no frame"
    return np.frombuffer(data, dtype=np.uint8).reshape(size, size)


def _soft_pixels(frame) -> int:
    """Pixels that are neither of the two source clips — i.e. the soft edge."""
    return int(((frame > 25) & (frame < 230)).sum())


@pytest.mark.parametrize("name", sorted(MASKS))
def test_every_mask_is_partway_through_at_its_midpoint(name):
    """The failure this catches is a mask that renders cleanly and does
    nothing — all black or all white halfway through, which is a transition
    that is really a cut with extra steps."""
    share = _mask_frame(name, 0.0 if TRANSITIONS[name] == "" else 0.25).mean() / 255
    assert 0.1 < share < 0.9, f"{name} is {share:.0%} handed over at its midpoint"


@pytest.mark.parametrize("name", sorted(MASKS))
def test_every_mask_finishes(name):
    """A mask that never reaches 1 leaves a sliver of the outgoing clip on
    screen for ever. Scaling progress by (1+feather) is what prevents it, and
    it is exactly the kind of thing that is right in the algebra and wrong in
    the code."""
    frame = _mask_frame(name, 0.4, at=1.99)
    assert frame.mean() / 255 > 0.97, "the outgoing clip is still visible at the end"


@pytest.mark.parametrize("name", ["wipe_left", "wipe_down", "iris_open",
                                  "diamond_open", "clock", "blinds", "checker"])
def test_feather_widens_the_edge(name):
    """The point of the whole mask mechanism: a hard edge has no in-between
    pixels and a feathered one has a band of them."""
    hard = _soft_pixels(_mask_frame(name, 0.0))
    soft = _soft_pixels(_mask_frame(name, 0.35))
    assert soft > hard * 3 + 500, f"{name}: hard {hard}px, soft {soft}px"


def test_a_wipe_moves_the_way_its_name_says():
    """The bug that survives every other check: a mask written in P rather
    than in (1-P) plays backwards and looks entirely plausible. A left wipe
    must be showing the incoming clip on the *right* of the frame halfway
    through."""
    frame = _mask_frame("wipe_left", 0.0, at=1.5)
    left = frame[:, : frame.shape[1] // 4].mean()
    right = frame[:, -frame.shape[1] // 4:].mean()
    assert right > left + 100, "wipe_left is running the wrong way"

    frame = _mask_frame("wipe_right", 0.0, at=1.5)
    left = frame[:, : frame.shape[1] // 4].mean()
    right = frame[:, -frame.shape[1] // 4:].mean()
    assert left > right + 100, "wipe_right is running the wrong way"


def test_an_iris_opens_from_the_middle():
    frame = _mask_frame("iris_open", 0.0, at=1.5)
    h, w = frame.shape
    centre = frame[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10].mean()
    corner = frame[:20, :20].mean()
    assert centre > corner + 100, "the iris is not opening from the centre"
    # ...and the closing one does the opposite.
    frame = _mask_frame("iris_close", 0.0, at=1.5)
    centre = frame[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10].mean()
    corner = frame[:20, :20].mean()
    assert corner > centre + 100, "the closing iris is not closing inwards"


def test_grain_is_stable_rather_than_crawling():
    """The dissolve mask is a hash of position, not a random: an expression is
    evaluated per pixel with no memory, so anything actually random would
    boil. Two renders of the same frame must be identical."""
    import numpy as np

    first = _mask_frame("grain", 0.0, at=1.5)
    second = _mask_frame("grain", 0.0, at=1.5)
    assert np.array_equal(first, second)


def _runs_of_incoming(column) -> int:
    """How many separate stretches of the incoming clip a column passes
    through — which for a banded mask is the number of bands."""
    import numpy as np

    white = (column > 128).astype(int)
    # A run starts wherever white begins, counting the top of the frame.
    return int(white[0]) + int(((np.diff(white)) == 1).sum())


@pytest.mark.parametrize("name,bands", [("blinds", 6), ("blinds_fine", 14),
                                        ("blinds_v", 6), ("blinds_v_fine", 14)])
def test_blinds_have_the_bands_their_name_claims(name, bands):
    """Each band hands over on its own, so a line across the frame passes
    through one stretch of the incoming clip per band. This is what separates
    a blind from a plain wipe, and nothing about the length of the render
    would show the difference."""
    frame = _mask_frame(name, 0.0, at=1.5, size=240)
    line = frame[:, 120] if name.startswith("blinds") and "_v" not in name \
        else frame[120, :]
    assert _runs_of_incoming(line) == bands


def test_the_families_only_name_real_transitions():
    """The UI builds its menu from these, so a typo here is a menu entry that
    fails at render time rather than at import."""
    for _, items in FAMILIES:
        for name in items:
            assert name in TRANSITIONS, f"{name} is not a transition"


def test_a_reel_actually_builds_with_a_feathered_mask(three_clips, tmp_path):
    """End to end: the expression has to survive being embedded in a
    filtergraph, quoted, alongside everything else build_reel does."""
    out = str(tmp_path / "masked.mp4")
    build_reel(three_clips, out, kind="iris_open", duration=0.6, feather=0.3,
               log_fn=lambda *_: None)
    assert os.path.exists(out) and os.path.getsize(out) > 0
