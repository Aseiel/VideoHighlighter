"""
Tests for modules.shot_look — do two clips look like the same view.

The module makes one narrow claim: that the *smaller* of a layout agreement
and a colour agreement is high only when both are, and that either on its own
is not usable. That claim is what these tests pin, because it is the whole
reason the module is shaped the way it is — and because a later change to
"just average the two" would look tidier, pass any test of the plumbing, and
quietly bring back the false pairs the conjunction exists to reject.

Measuring a real clip needs OpenCV, which conftest replaces with a MagicMock,
so the descriptors here are written by hand.
"""

from __future__ import annotations

import pytest

from modules.shot_look import (
    GRID,
    SAME_VIEW,
    Look,
    look,
    same_view,
    similarity,
)


def _look(structure, colour, path="clip.mp4") -> Look:
    """A measured clip with the two descriptors given, each normalised the way
    the real measurement leaves them."""
    import math

    length = math.sqrt(sum(v * v for v in colour)) or 1.0
    return Look(path=path, structure=list(structure),
                colour=[v / length for v in colour], measured=True)


def _flat(value, n=GRID * GRID):
    return [value] * n


# ---------------------------------------------------------------------------
# The conjunction, which is the whole idea
# ---------------------------------------------------------------------------

def test_both_descriptors_must_agree():
    """Matching layout with a different palette is a different view — two
    footpaths, one in heather and one in grass — and matching palette with a
    different layout is too."""
    layout_a = [1.0] + _flat(0.0, GRID * GRID - 1)
    layout_b = _flat(0.0, GRID * GRID - 1) + [1.0]

    same_layout_other_colour = similarity(
        _look(layout_a, [1, 0, 0, 0]), _look(layout_a, [0, 0, 0, 1]))
    same_colour_other_layout = similarity(
        _look(layout_a, [1, 0, 0, 0]), _look(layout_b, [1, 0, 0, 0]))

    assert same_layout_other_colour < SAME_VIEW
    assert same_colour_other_layout < SAME_VIEW


def test_a_clip_is_the_same_view_as_itself():
    item = _look([1.0] + _flat(0.0, GRID * GRID - 1), [1, 2, 3, 4])

    assert similarity(item, item) == pytest.approx(1.0, abs=1e-6)
    assert same_view(item, item)


def test_agreeing_on_both_reads_as_a_repeat():
    structure = [0.9, 0.4] + _flat(0.0, GRID * GRID - 2)
    nearly = [0.89, 0.42] + _flat(0.0, GRID * GRID - 2)

    assert same_view(_look(structure, [5, 3, 1, 0]),
                     _look(nearly, [5, 3, 1, 0]))


def test_similarity_is_the_weaker_of_the_two_not_the_average():
    """Averaging is the tidier-looking version and it is wrong: a pair that
    matches perfectly on colour and not at all on layout would average to
    something respectable, which is exactly the false positive this module
    was built to reject."""
    a = _look([1.0] + _flat(0.0, GRID * GRID - 1), [1, 0, 0, 0])
    b = _look(_flat(0.0, GRID * GRID - 1) + [1.0], [1, 0, 0, 0])

    score = similarity(a, b)

    assert score == pytest.approx(0.0, abs=1e-6)
    assert score < 0.5   # the average would be about this


# ---------------------------------------------------------------------------
# Failing safely
# ---------------------------------------------------------------------------

def test_an_unmeasured_clip_reports_nothing_rather_than_a_difference():
    """0 means "no idea", and the caller reads it as "not a repeat" — which
    leaves the edit exactly as it would have been without this module."""
    measured = _look([1.0] + _flat(0.0, GRID * GRID - 1), [1, 0, 0, 0])

    assert similarity(measured, Look(path="x.mp4")) == 0.0
    assert similarity(Look(path="x.mp4"), measured) == 0.0
    assert not same_view(measured, Look(path="x.mp4"))
    assert similarity(measured, None) == 0.0


def test_descriptors_of_different_sizes_do_not_raise():
    """A cache written by an older build with a different grid."""
    small = Look(path="a.mp4", structure=[1.0, 0.0], colour=[1.0], measured=True)
    big = _look([1.0] + _flat(0.0, GRID * GRID - 1), [1, 0, 0, 0])

    assert similarity(small, big) == 0.0


def test_a_clip_that_cannot_be_read_comes_back_unmeasured(tmp_path):
    item = look(str(tmp_path / "absent.mp4"), use_cache=False,
                log_fn=lambda *_: None)

    assert not item.measured


def test_the_threshold_is_set_for_precision():
    """A regression guard on the constant. Measured over 210 pairs of a real
    shoot, 0.88 reported no false pairs and every step below it admitted them
    faster than true ones; the module is used to reorder shots, where a false
    pair silently drops good footage for a reason nobody can see."""
    assert SAME_VIEW >= 0.86
