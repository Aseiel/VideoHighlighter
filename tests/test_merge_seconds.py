"""
Characterization tests for `compute_forbidden._merge_seconds`.

This function turns a set of "this whole second is forbidden" integer markers
into merged (start, end) float ranges, bridging gaps up to `merge_gap` seconds.
It drives the AVOID(skip) flow: any second that lands inside one of these
ranges has its highlight score zeroed.

We freeze current behaviour as a regression baseline before any pipeline
refactor.
"""

from __future__ import annotations

import pytest

# Import under shim cover (see conftest.py — cv2 is replaced by MagicMock so
# the top-level `import cv2` in compute_forbidden.py does not need a real
# OpenCV install).
from modules.compute_forbidden import _merge_seconds


class TestEmptyAndTrivial:
    def test_empty_input_returns_empty_list(self):
        assert _merge_seconds(set()) == []

    def test_empty_list_input_returns_empty_list(self):
        assert _merge_seconds([]) == []

    def test_single_second_becomes_one_one_long_range(self):
        # Convention: a single forbidden second N produces the half-open
        # range [N, N+1). We pin this so callers (FFmpeg cuts, scoring
        # zeroing) keep getting the same right-edge.
        assert _merge_seconds({5}) == [(5.0, 6.0)]


class TestMerging:
    def test_contiguous_seconds_merge_into_one_range(self):
        # 10, 11, 12 are all adjacent; gap = 1 between each. Default
        # merge_gap=2.0, so they collapse to one range ending at the last
        # second + 1.
        assert _merge_seconds({10, 11, 12}) == [(10.0, 13.0)]

    def test_seconds_within_default_merge_gap_collapse(self):
        # Gap of 2 between 10 and 12 is <= default merge_gap=2.0, so they
        # merge. End of the merged range is 12 + 1 = 13.
        assert _merge_seconds({10, 12}) == [(10.0, 13.0)]

    def test_seconds_beyond_merge_gap_stay_separate(self):
        # Gap of 3 between 10 and 13 is > default merge_gap=2.0, so they
        # stay as two ranges.
        assert _merge_seconds({10, 13}) == [(10.0, 11.0), (13.0, 14.0)]

    def test_custom_merge_gap_narrower(self):
        # merge_gap=0 means only strictly-adjacent (gap of 1) seconds merge.
        # 10 and 12 have gap=2 > 0, so they should stay separate.
        # Behaviour pin: `cur - prev <= merge_gap`, so gap=2 with
        # merge_gap=0 does NOT merge.
        assert _merge_seconds({10, 12}, merge_gap=0) == [
            (10.0, 11.0),
            (12.0, 13.0),
        ]

    def test_custom_merge_gap_wider(self):
        # merge_gap=5 bridges anything within 5 seconds.
        assert _merge_seconds({10, 14, 18}, merge_gap=5) == [(10.0, 19.0)]


class TestOrderingAndDuplicates:
    def test_input_order_does_not_matter(self):
        # Input is a set / unordered iterable. Result must be sorted.
        assert _merge_seconds({30, 10, 20}) == [(10.0, 11.0), (20.0, 21.0), (30.0, 31.0)]

    def test_list_with_duplicates_collapses(self):
        # Set-like dedupe before processing. List input with dupes must not
        # produce duplicate ranges or crash.
        assert _merge_seconds([5, 5, 5, 6]) == [(5.0, 7.0)]


class TestReturnTypes:
    def test_returns_list_of_float_tuples(self):
        result = _merge_seconds({1, 2, 3})
        assert isinstance(result, list)
        assert len(result) == 1
        start, end = result[0]
        # Tuples of (float, float) — downstream consumers (FFmpeg arg
        # builders) rely on this.
        assert isinstance(start, float)
        assert isinstance(end, float)


class TestRealisticScenarios:
    def test_two_separate_appearances_in_video(self):
        # Avoided person appears at seconds 30-32 and again at 120-121.
        # Expected: two distinct forbidden ranges.
        seconds = {30, 31, 32, 120, 121}
        assert _merge_seconds(seconds) == [(30.0, 33.0), (120.0, 122.0)]

    def test_burst_with_short_gap_collapses(self):
        # Tracking flickers: appears at 30, lost at 31, back at 32.
        # Default merge_gap=2.0 bridges the gap. Expected single range
        # 30 -> 33.
        seconds = {30, 32}
        assert _merge_seconds(seconds) == [(30.0, 33.0)]
