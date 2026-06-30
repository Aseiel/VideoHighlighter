"""
Characterization tests for `auto_segments.cluster_points`.

When `CLIP_TIME=0`, the pipeline builds variable-length highlight regions from
point signals (motion events, motion peaks, audio peaks, etc). `cluster_points`
is the first transform: turn a sorted list of timestamps into padded clusters
that respect a max gap.

We freeze the adaptive-padding behaviour so future refactor cannot accidentally
shift highlight boundaries.
"""

from __future__ import annotations

import math

import pytest

from modules.auto_segments import cluster_points


class TestEmptyAndSingleton:
    def test_empty_returns_empty(self):
        assert cluster_points([]) == []

    def test_single_timestamp_creates_one_cluster(self):
        # One timestamp at t=10. Default min_pad=0.5, max_pad=2.0.
        # Adaptive pad rule (raw_dur < 1.0 -> pad=1.0), so cluster spans
        # [10 - 1.0, 10 + 1.0] = (9.0, 11.0).
        clusters = cluster_points([10.0])
        assert clusters == [(9.0, 11.0)]


class TestClustering:
    def test_close_timestamps_merge_into_one_cluster(self):
        # Gaps of 1.0 each, max_gap default 2.0 -> all merge.
        # Raw cluster span = 10 -> 12 = 2.0s. raw_dur >= 1.0 -> pad=0.5.
        # Final: (10 - 0.5, 12 + 0.5) = (9.5, 12.5).
        clusters = cluster_points([10.0, 11.0, 12.0])
        assert clusters == [(9.5, 12.5)]

    def test_far_timestamps_stay_separate(self):
        # Gap of 5.0 between 10 and 15 > default max_gap=2.0 -> separate.
        clusters = cluster_points([10.0, 15.0])
        # Each is a singleton (raw_dur = 0 < 1.0 -> pad=1.0).
        assert clusters == [(9.0, 11.0), (14.0, 16.0)]

    def test_unsorted_input_is_sorted_first(self):
        clusters = cluster_points([15.0, 10.0])
        assert clusters == [(9.0, 11.0), (14.0, 16.0)]


class TestAdaptivePadding:
    def test_short_cluster_gets_more_padding(self):
        # Two timestamps 0.5s apart -> raw_dur = 0.5 < 1.0 -> pad=1.0 each side.
        clusters = cluster_points([10.0, 10.5])
        # Span = (10 - 1.0, 10.5 + 1.0) = (9.0, 11.5).
        assert clusters == [(9.0, 11.5)]

    def test_long_cluster_gets_less_padding(self):
        # Span >= 1.0s -> pad=0.5 each side.
        clusters = cluster_points([10.0, 11.5])
        assert clusters == [(9.5, 12.0)]


class TestCustomParameters:
    def test_max_gap_controls_merging(self):
        # max_gap=0.5 means timestamps 1.0s apart do NOT merge.
        clusters = cluster_points([10.0, 11.0], max_gap=0.5)
        assert clusters == [(9.0, 11.0), (10.0, 12.0)]

    def test_min_pad_floor(self):
        # min_pad=2.0 means every cluster gets at least 2.0s padding per side,
        # but capped by max_pad=2.0 in the default ceiling rule.
        clusters = cluster_points([10.0], min_pad=2.0, max_pad=2.0)
        assert clusters == [(8.0, 12.0)]

    def test_clamps_to_zero_on_left_edge(self):
        # Padding cannot make the start negative. The function clamps at 0.0.
        clusters = cluster_points([0.5])
        # Singleton -> pad=1.0 -> would be (-0.5, 1.5), clamps to (0.0, 1.5).
        assert clusters == [(0.0, 1.5)]


class TestReturnTypes:
    def test_returns_list_of_float_tuples(self):
        clusters = cluster_points([10.0])
        assert isinstance(clusters, list)
        assert len(clusters) == 1
        start, end = clusters[0]
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert start <= end
