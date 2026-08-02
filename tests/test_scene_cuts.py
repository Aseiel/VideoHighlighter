"""Tests for the shot-cut policy — when to trust 70, and when to stop trusting it.

The distributions here are built to match measurement rather than intuition. The
"film" fixture uses the median (14.2) and MAD (4.8) taken from a real 5-minute
stretch whose maximum difference anywhere was 47 — a video on which the fixed
threshold of 70 cannot fire even once. The "broadcast" fixture is the opposite
case, material that clears 70 regularly, and its job in this file is to prove
the fallback leaves it alone.

That asymmetry is the whole design: the fixed threshold works on the footage it
was tuned for, so a fix that changed those results would be a regression wearing
a fix's clothes.
"""
from __future__ import annotations

import numpy as np
import pytest

from modules.scene_cuts import (
    DEFAULT_Z,
    MIN_PLAUSIBLE_CUTS_PER_MINUTE,
    MIN_SAMPLES,
    adaptive_threshold,
    cuts_from_diffs,
    looks_undetected,
    resolve,
    suppress_adjacent,
    scenes_from_cuts,
)


def _film(n=3600, cuts=70, seed=0):
    """Graded feature: median ~14, MAD ~4.8, nothing anywhere near 70."""
    rng = np.random.default_rng(seed)
    d = np.abs(rng.normal(14.2, 4.8 / 1.4826, n))
    d[rng.choice(n, size=cuts, replace=False)] = rng.uniform(30, 47, cuts)
    return d


def _broadcast(n=3600, cuts=120, seed=1):
    """High-contrast material that clears the fixed threshold regularly."""
    rng = np.random.default_rng(seed)
    d = np.abs(rng.normal(25.0, 8.0, n))
    d[rng.choice(n, size=cuts, replace=False)] = rng.uniform(75, 140, cuts)
    return d


class TestAdaptiveThreshold:
    def test_outliers_do_not_inflate_their_own_bar(self):
        """Cuts are the high outliers; a mean-based bar would hide them."""
        flat = np.full(1000, 14.0)
        spiked = flat.copy()
        spiked[:50] = 200.0
        assert adaptive_threshold(spiked) == pytest.approx(
            adaptive_threshold(flat), abs=0.5)

    def test_scales_with_the_videos_own_spread(self):
        rng = np.random.default_rng(3)
        tight = np.abs(rng.normal(14.0, 1.0, 2000))
        loose = np.abs(rng.normal(14.0, 6.0, 2000))
        assert adaptive_threshold(loose) > adaptive_threshold(tight)

    def test_empty_is_zero(self):
        assert adaptive_threshold([]) == 0.0

    def test_a_flat_distribution_does_not_divide_by_zero(self):
        assert adaptive_threshold(np.full(500, 10.0)) == pytest.approx(10.0, abs=1e-3)


class TestPlausibility:
    def test_one_cut_every_two_minutes_is_the_floor(self):
        assert looks_undetected(0, 60.0) is True
        assert looks_undetected(5, 60.0) is True
        assert looks_undetected(600, 60.0) is False

    def test_zero_length_video_is_not_a_failure(self):
        assert looks_undetected(0, 0.0) is False


class TestResolve:
    def test_the_reported_bug_recalibrates(self):
        """60 minutes, max difference 47, threshold 70 -> one scene."""
        d = _film()
        assert cuts_from_diffs(d, 70.0).size == 0

        cuts, used, recalibrated = resolve(d, minutes=5.0, threshold=70.0)
        assert recalibrated is True
        assert used < 70.0
        assert cuts.size > 0

    def test_footage_that_already_works_is_untouched(self):
        """The guarantee that makes this safe to ship."""
        d = _broadcast()
        cuts, used, recalibrated = resolve(d, minutes=5.0, threshold=70.0)
        assert recalibrated is False
        assert used == 70.0
        assert np.array_equal(cuts, cuts_from_diffs(d, 70.0))

    def test_a_slow_but_real_edit_is_not_second_guessed(self):
        d = _film(cuts=0)
        d[::100] = 95.0                      # 36 clear cuts in 5 minutes
        _cuts, used, recalibrated = resolve(d, minutes=5.0, threshold=70.0)
        assert recalibrated is False
        assert used == 70.0

    def test_a_short_clip_keeps_the_configured_threshold(self):
        d = _film(n=MIN_SAMPLES - 1, cuts=0)
        _cuts, used, recalibrated = resolve(d, minutes=0.2, threshold=70.0)
        assert recalibrated is False
        assert used == 70.0

    def test_static_footage_is_not_given_invented_cuts(self):
        """A locked-off camera has no shot structure; do not manufacture one."""
        rng = np.random.default_rng(5)
        d = np.abs(rng.normal(2.0, 0.05, 3600))    # sensor noise only
        cuts, _used, recalibrated = resolve(d, minutes=5.0, threshold=70.0)
        assert recalibrated is False
        assert cuts.size == 0

    def test_recalibration_never_returns_fewer_cuts(self):
        d = _film()
        before = cuts_from_diffs(d, 70.0).size
        cuts, _used, _r = resolve(d, minutes=5.0, threshold=70.0)
        assert cuts.size >= before

    def test_the_derived_rate_is_plausible_for_a_film(self):
        """Guards the z constant: the fix must not shatter the video."""
        d = _film()
        cuts, _used, _r = resolve(d, minutes=5.0, threshold=70.0)
        per_minute = cuts.size / 5.0
        assert 1.0 < per_minute < 40.0, f"{per_minute:.1f} cuts/min is not an edit"

    def test_a_lower_configured_threshold_is_honoured(self):
        """Someone who tuned their own number must not be overridden."""
        d = _film()
        cuts, used, recalibrated = resolve(d, minutes=5.0, threshold=30.0)
        assert recalibrated is False
        assert used == 30.0
        assert cuts.size > 0


class TestSuppressAdjacent:
    def test_a_cut_spanning_several_samples_becomes_one(self):
        """Observed on real footage as 0.1-second scenes."""
        diffs = np.full(100, 5.0)
        diffs[40:43] = [80.0, 95.0, 85.0]
        cuts = cuts_from_diffs(diffs, 70.0)
        assert cuts.size == 3
        assert list(suppress_adjacent(cuts, diffs, min_gap=6)) == [41]

    def test_it_keeps_the_strongest_not_the_earliest(self):
        """The largest difference is the frame the cut happened on."""
        diffs = np.full(100, 5.0)
        diffs[40:43] = [72.0, 74.0, 99.0]
        assert list(suppress_adjacent(cuts_from_diffs(diffs, 70.0),
                                      diffs, min_gap=6)) == [42]

    def test_genuinely_separate_cuts_both_survive(self):
        diffs = np.full(100, 5.0)
        diffs[[20, 60]] = 90.0
        assert list(suppress_adjacent(cuts_from_diffs(diffs, 70.0),
                                      diffs, min_gap=6)) == [20, 60]

    def test_a_gap_of_one_is_a_no_op(self):
        diffs = np.full(20, 5.0)
        diffs[[3, 4]] = 90.0
        assert list(suppress_adjacent([3, 4], diffs, min_gap=1)) == [3, 4]

    def test_empty_stays_empty(self):
        assert suppress_adjacent([], np.zeros(10), min_gap=6).size == 0

    def test_resolve_applies_it(self):
        d = _broadcast()
        d[490:513] = 25.0                 # clear the fixture's own cuts nearby
        d[500:503] = [90.0, 120.0, 95.0]  # then plant one three-sample cut
        cuts, _used, _r = resolve(d, minutes=5.0, threshold=70.0, min_gap=6)
        assert [c for c in cuts if 490 <= c <= 512] == [501]


class TestScenesFromCuts:
    def test_covers_the_whole_video_with_no_gaps(self):
        scenes = scenes_from_cuts([10.0, 25.0, 40.0], duration=60.0)
        assert scenes[0][0] == 0.0
        assert scenes[-1][1] == 60.0
        for a, b in zip(scenes[:-1], scenes[1:]):
            assert a[1] == b[0]

    def test_no_cuts_is_one_scene(self):
        assert scenes_from_cuts([], duration=60.0) == [(0.0, 60.0)]

    def test_duplicate_and_out_of_range_cuts_make_no_empty_scenes(self):
        scenes = scenes_from_cuts([0.0, 10.0, 10.0, 60.0, 99.0], duration=60.0)
        assert all(e > s for s, e in scenes)
        assert scenes == [(0.0, 10.0), (10.0, 60.0)]

    def test_zero_duration_is_empty(self):
        assert scenes_from_cuts([1.0], duration=0.0) == []

    def test_cuts_arrive_sorted_regardless_of_input_order(self):
        scenes = scenes_from_cuts([40.0, 10.0, 25.0], duration=60.0)
        assert [s for s, _ in scenes] == [0.0, 10.0, 25.0, 40.0]
