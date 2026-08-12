"""
Tests for modules.shot_window — which seconds of a clip are worth using.

The measurement itself needs OpenCV, which conftest replaces with a MagicMock,
so what is pinned here is everything downstream of it: that a window is chosen
by score rather than by position, that the search never hands back a stretch
that was already spoken for, and that a clip nobody could measure behaves
exactly as it did before this module existed. Those are the properties the
planner depends on; the pixel maths is validated against real footage, where
the answer can actually be looked at.
"""

from __future__ import annotations

import pytest

from modules.shot_window import (
    BLOWN,
    DARK,
    SAMPLES_PER_SECOND,
    ClipWindows,
    Sample,
    _score,
    profile,
)


def _clip(qualities, *, path="clip.mp4", rate=SAMPLES_PER_SECOND) -> ClipWindows:
    """A measured clip whose samples have the usable scores given.

    Bypasses the scorer so a test can state the shape it wants to select from
    rather than reverse-engineering four signals that produce it.
    """
    samples = [Sample(t=i / rate, usable=q) for i, q in enumerate(qualities)]
    return ClipWindows(path=path, duration=len(qualities) / rate,
                       samples=samples, measured=True)


# ---------------------------------------------------------------------------
# Choosing a window
# ---------------------------------------------------------------------------

def test_the_best_window_is_the_good_part_not_the_first_part():
    """The whole reason the module exists: a clip that opens on the camera
    being placed and settles later must offer the later part."""
    clip = _clip([0.1] * 16 + [0.9] * 16)          # 2s bad, 2s good

    window = clip.best(1.5)

    assert window.start == pytest.approx(2.0, abs=0.2)
    assert window.duration == pytest.approx(1.5)


def test_a_clip_that_starts_well_is_not_moved():
    """Preferring later windows on principle would be its own bug — a clip
    that is fine at the top must still start at the top."""
    clip = _clip([0.9] * 32)

    assert clip.best(1.5).start == pytest.approx(0.0)


def test_ties_go_to_the_earliest_window():
    """A clip of uniform quality has no reason to wander to its end, and the
    reel keeps reading in shooting order if it does not."""
    clip = _clip([0.7] * 40)

    assert clip.best(1.0).start == pytest.approx(0.0)


def test_a_window_never_starts_before_what_was_already_taken():
    """Two slices of one source overlapping is the reel repeating itself."""
    clip = _clip([0.9] * 8 + [0.2] * 24)   # the good part is at the very top

    second = clip.best(1.0, after=2.0)

    assert second.start >= 2.0 - 1e-6


def test_a_window_never_runs_past_the_end_of_the_clip():
    clip = _clip([0.5] * 32)               # 4 seconds

    window = clip.best(1.5, after=3.0)

    assert window.end <= clip.duration + 1e-6


def test_asking_for_more_than_is_left_gives_what_is_left():
    """A short shot is a far smaller problem than a missing one."""
    clip = _clip([0.5] * 24)               # 3 seconds

    window = clip.best(10.0)

    assert window.duration == pytest.approx(3.0)


def test_one_terrible_moment_sinks_a_window_the_average_would_keep():
    """A stretch that is excellent either side of a lurch is not a usable
    stretch, and scoring on the mean alone says it is."""
    steady = _clip([0.6] * 32)
    spiked = _clip([0.95] * 8 + [0.0] * 2 + [0.95] * 22)

    assert spiked.score_at(0.0, 1.5) < steady.score_at(0.0, 1.5)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_settle_is_when_the_clip_first_becomes_good():
    clip = _clip([0.1] * 16 + [0.9] * 16)

    assert clip.settle == pytest.approx(2.0, abs=0.3)


def test_settle_survives_a_single_bad_sample_in_a_good_stretch():
    """Requiring every sample to clear the bar means one blink disqualifies
    the stretch, and the clip then reports that it settles at zero — the one
    answer that is certainly wrong for a clip that opens badly."""
    clip = _clip([0.05] * 8 + [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9] + [0.9] * 16)

    assert clip.settle > 0.5


def test_head_penalty_measures_how_much_worse_the_opening_is():
    assert _clip([0.2] * 8 + [0.9] * 24).head_penalty == pytest.approx(0.7, abs=0.05)
    assert _clip([0.9] * 32).head_penalty == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _sample(**kw) -> Sample:
    base = dict(t=0.0, coherence=0.9, shift=0.05, jerk=0.05,
                sharpness=1000.0, brightness=0.5, drift=0.0)
    base.update(kw)
    return Sample(**base)


def test_a_steady_sharp_frame_scores_well():
    samples = [_sample(t=i / 8) for i in range(8)]
    _score(samples)

    assert samples[0].usable > 0.8


def test_incoherent_movement_scores_worse_than_coherent():
    """A camera being carried and a camera panning move by similar amounts.
    Coherence is what tells them apart, so it has to dominate."""
    panning = [_sample(t=i / 8, coherence=0.95, shift=0.4) for i in range(8)]
    carried = [_sample(t=i / 8, coherence=0.10, shift=0.4) for i in range(8)]
    _score(panning)
    _score(carried)

    assert panning[0].usable > carried[0].usable + 0.25


def test_changing_direction_scores_worse_than_holding_it():
    """A deliberate pan holds its speed however fast it is; a hand hunting
    for framing does not."""
    smooth = [_sample(t=i / 8, shift=0.5, jerk=0.02) for i in range(8)]
    hunting = [_sample(t=i / 8, shift=0.5, jerk=2.0) for i in range(8)]
    _score(smooth)
    _score(hunting)

    assert smooth[0].usable > hunting[0].usable


def test_sharpness_is_judged_against_the_clip_rather_than_a_constant():
    """The variance of a Laplacian on a hedge and on a wall differ by an order
    of magnitude at identical focus, so an absolute threshold measures the
    subject and not the picture. The same soft frame must score badly in a
    sharp clip and fine in a uniformly soft one."""
    mixed = [_sample(t=i / 8, sharpness=2000.0) for i in range(7)]
    mixed.append(_sample(t=7 / 8, sharpness=200.0))
    _score(mixed)

    flat = [_sample(t=i / 8, sharpness=200.0) for i in range(8)]
    _score(flat)

    assert mixed[-1].usable < flat[-1].usable - 0.1


def test_a_picture_nobody_can_see_is_vetoed():
    """A lens cap, a pocket or a blown sky, all of which are otherwise
    perfectly steady and would score well on every other term."""
    dark = [_sample(t=i / 8, brightness=DARK / 2) for i in range(8)]
    blown = [_sample(t=i / 8, brightness=(BLOWN + 1) / 2) for i in range(8)]
    fine = [_sample(t=i / 8, brightness=0.5) for i in range(8)]
    for group in (dark, blown, fine):
        _score(group)

    assert dark[0].usable < fine[0].usable - 0.2
    assert blown[0].usable < fine[0].usable - 0.2


def test_an_exposure_ramp_scores_worse_than_a_settled_one():
    """A camera coming out of a bag into daylight is sharp and steady and
    still visibly wrong for a second and a half."""
    ramping = [_sample(t=i / 8, drift=0.8) for i in range(8)]
    settled = [_sample(t=i / 8, drift=0.0) for i in range(8)]
    _score(ramping)
    _score(settled)

    assert ramping[0].usable < settled[0].usable


# ---------------------------------------------------------------------------
# Failing safely
# ---------------------------------------------------------------------------

def test_an_unmeasured_clip_hands_back_exactly_what_was_asked_for():
    """Measurement failing should cost the improvement, never the reel: with
    no samples the answer is the old behaviour, which is to start where the
    caller pointed."""
    clip = ClipWindows(path="x.mp4", duration=6.0)

    window = clip.best(2.0, after=1.5)

    assert not clip.measured
    assert window.start == pytest.approx(1.5)
    assert window.duration == pytest.approx(2.0)
    assert clip.settle == 0.0
    assert clip.head_penalty == 0.0


def test_a_clip_that_cannot_be_opened_does_not_raise(tmp_path):
    """Anything the decoder does — a missing file, a codec it will not touch,
    an OpenCV that hands back something nobody expected — comes back as an
    unmeasured clip rather than as an exception. Only ``measured`` is asserted
    because with cv2 mocked the other fields are whatever the mock invented."""
    clip = profile(str(tmp_path / "does_not_exist.mp4"),
                   use_cache=False, log_fn=lambda *_: None)

    assert not clip.measured
    assert clip.samples == []


def test_an_empty_clip_has_no_opinion():
    clip = ClipWindows(path="x.mp4", duration=0.0)

    assert clip.best(2.0).duration == 0.0
    assert clip.score_at(0.0, 1.0) == 0.0
