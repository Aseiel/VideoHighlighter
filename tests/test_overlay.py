"""
Tests for modules.overlay — the graphics drawn over a finished reel.

Pillow is a real dependency of this suite rather than one of the mocked ones,
so these draw actual frames and look at the pixels. That matters more than
usual here: an element that silently draws nothing, or draws itself off the
edge of the frame, produces a render that completes successfully and looks
exactly like one with the feature turned off.
"""

from __future__ import annotations

import pytest

from modules.overlay import (
    ELEMENTS,
    MAX_POINTS,
    Box,
    ElevationProfile,
    Readout,
    RouteMap,
    Scene,
    Ticker,
    _thin,
    build_scene,
    burn_overlay,
    frames,
    make_elements,
)


def _scene(**kw) -> Scene:
    base = dict(duration=20.0, width=360, height=640,
                marks=[(0.0, 0.4), (5.0, 0.1), (10.0, 0.6), (15.0, 0.9)],
                elevations=[100, 180, 140, 260, 200, 300, 220, 150],
                length=33000.0, climb=1000.0)
    base.update(kw)
    return Scene(**base)


def _draw(element, scene, t=10.0):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (scene.width, scene.height), (0, 0, 0, 0))
    element.draw(ImageDraw.Draw(image, "RGBA"), scene, t)
    return image


def _lit(image) -> int:
    """Pixels the element actually put down."""
    return sum(1 for pixel in image.getdata() if pixel[3] > 0)


# ---------------------------------------------------------------------------
# The two kinds of progress, which is the thing that was wrong
# ---------------------------------------------------------------------------

def test_the_bar_only_ever_moves_forward():
    """A reel is ordered as a story, so the shot on screen jumps about the
    route: the hook is whichever shot is most striking and the payoff is
    whichever ends it. Tying the counters to that made them read 12.6 km,
    9.3 km, 20.0 km, 3.2 km, and a distance that goes backwards is worse than
    no distance at all."""
    scene = _scene()

    seen = [scene.progress_at(t) for t in range(0, 21)]

    assert seen == sorted(seen)
    assert seen[0] == pytest.approx(0.0)
    assert seen[-1] == pytest.approx(1.0)


def test_the_marker_follows_the_shot_on_screen():
    """The dot is a pointer, not a bar, so it is allowed — required — to jump
    back to wherever the current shot was filmed."""
    scene = _scene()

    assert scene.marker_at(4.0) == pytest.approx(0.4)
    assert scene.marker_at(9.0) == pytest.approx(0.1)
    assert scene.marker_at(19.0) == pytest.approx(0.9)


def test_the_marker_eases_between_shots_rather_than_teleporting():
    scene = _scene()
    part_way = scene.marker_at(5.15)

    assert 0.1 < part_way < 0.4, "the dot jumped instead of travelling"


def test_progress_is_bounded_outside_the_reel():
    scene = _scene()

    assert scene.progress_at(-5) == 0.0
    assert scene.progress_at(999) == 1.0


def test_a_scene_with_no_marks_still_reads():
    """No GPS at all: the bar runs and the marker follows it."""
    scene = _scene(marks=[])

    assert scene.progress_at(10.0) == pytest.approx(0.5)
    assert scene.marker_at(10.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Placing the marks
# ---------------------------------------------------------------------------

def test_marks_account_for_the_overlap_between_shots():
    """A cut starts earlier than the sum of the ones before it, because every
    transition overlaps two shots. If this disagrees with Edl.duration the
    graphics drift further out of step with the picture on every cut."""
    from modules.edl import Cut, Edl

    edl = Edl(width=360, height=640, cuts=[
        Cut(source="a.mp4", start=0, end=3.0, transition="crossfade",
            transition_duration=0.6),
        Cut(source="b.mp4", start=0, end=3.0, transition="crossfade",
            transition_duration=0.6),
        Cut(source="c.mp4", start=0, end=3.0, transition="cut"),
    ])

    scene = build_scene(edl, None, {})

    assert [round(m[0], 2) for m in scene.marks] == [0.0, 2.4, 4.8]
    assert scene.duration == pytest.approx(edl.duration, abs=0.01)


def test_a_scene_without_a_track_still_has_a_mark_per_cut():
    from modules.edl import Cut, Edl

    edl = Edl(cuts=[Cut(source=f"{i}.mp4", start=0, end=2.0, transition="cut")
                    for i in range(4)])

    scene = build_scene(edl, None, None)

    assert len(scene.marks) == 4
    assert all(progress == 0.0 for _, progress in scene.marks)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ELEMENTS))
def test_every_element_draws_something(name):
    """The failure this catches is an element that renders cleanly and puts
    down no pixels, which is indistinguishable from the feature being off."""
    scene = _scene()
    element = make_elements([name], scene, _FakeTrack())[0]

    assert _lit(_draw(element, scene)) > 50, f"{name} drew nothing"


@pytest.mark.parametrize("name", sorted(ELEMENTS))
def test_no_element_draws_outside_the_frame(name):
    """Everything is positioned in fractions of the frame, so an element that
    escapes it is a layout bug that only shows on one delivery size."""
    scene = _scene()
    element = make_elements([name], scene, _FakeTrack())[0]
    image = _draw(element, scene)

    assert image.size == (scene.width, scene.height)
    assert _lit(image) < scene.width * scene.height, "the element filled the frame"


def test_the_profile_fills_as_the_reel_plays():
    scene = _scene()
    profile = ElevationProfile(box=Box(0.05, 0.7, 0.9, 0.2))

    early = _lit(_draw(profile, scene, 1.0))
    late = _lit(_draw(profile, scene, 19.0))

    assert late > early * 1.5, "the profile is not filling in"


def test_the_readout_counts_up():
    scene = _scene()
    readout = Readout(box=Box(0.05, 0.5, 0.9, 0.05))

    early = _lit(_draw(readout, scene, 0.5))
    late = _lit(_draw(readout, scene, 19.5))

    assert early > 0 and late > 0
    assert early != late, "the numbers never changed"


def test_a_readout_with_nothing_to_say_draws_nothing():
    scene = _scene(length=0.0, climb=0.0)

    assert _lit(_draw(Readout(), scene)) == 0


def test_an_element_that_throws_does_not_stop_the_frame():
    """One element failing must cost that element, not the render."""
    class Broken(ElevationProfile):
        def draw(self, *_args):
            raise RuntimeError("nope")

    scene = _scene()
    produced = list(frames(scene, [Broken(), Ticker(box=Box(0.05, 0.5, 0.9, 0.02))],
                           fps=2))

    assert len(produced) == 40
    assert any(byte for byte in produced[10]), "nothing was drawn at all"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def test_a_long_track_is_thinned_before_it_is_drawn():
    """A four-hour track is fifteen thousand points, and drawing all of them
    took 3.9 seconds per frame — half an hour for a twenty-second reel."""
    thinned = _thin(list(range(15000)))

    assert len(thinned) <= MAX_POINTS + 1
    assert thinned[0] == 0
    assert thinned[-1] == 14999, "the end of the route was dropped"


def test_thinning_leaves_a_short_track_alone():
    assert _thin([1, 2, 3]) == [1, 2, 3]


def test_static_geometry_is_built_once():
    """It is the same drawing on every one of several hundred frames."""
    scene = _scene()
    profile = ElevationProfile(box=Box(0.05, 0.7, 0.9, 0.2))

    first = profile.shape(scene)

    assert profile.shape(scene) is first


def test_the_right_number_of_frames_is_produced():
    scene = _scene(duration=3.0)

    assert len(list(frames(scene, [Ticker()], fps=10))) == 30


# ---------------------------------------------------------------------------
# Failing safely
# ---------------------------------------------------------------------------

def test_no_elements_means_the_reel_passes_through(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"not really a video")
    dst = tmp_path / "out.mp4"

    burn_overlay(str(src), str(dst), _scene(), [], log_fn=lambda *_: None)

    assert dst.read_bytes() == b"not really a video"


def test_unknown_element_names_are_skipped():
    """A preset naming an element a later build removed should cost that
    element, not the render."""
    scene = _scene()

    assert make_elements(["elevation", "nonsense"], scene, None)
    assert make_elements(["nonsense"], scene, None) == []
    assert make_elements(None, scene, None) == []


class _FakeTrack:
    """Only what make_elements reads off a track."""
    points = [(None, 53.5 + i * 0.001, -1.9 + i * 0.002, 100 + i)
              for i in range(50)]
