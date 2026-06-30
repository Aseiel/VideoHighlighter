"""
Tests for `modules.manual_avoid`.

Pin the parser's permissive input contract and the overlap / combine logic
that the future "right-click → Avoid this range" timeline action will
depend on.
"""

from __future__ import annotations

import pytest

from modules.manual_avoid import (
    combine,
    merge_overlapping,
    parse_ranges,
)


# ---------------------------------------------------------------------------
# parse_ranges
# ---------------------------------------------------------------------------
class TestParseRangesTuples:
    def test_empty_input(self):
        assert parse_ranges([]) == []
        assert parse_ranges(None) == []

    def test_single_tuple(self):
        assert parse_ranges([(1.0, 2.0)]) == [(1.0, 2.0)]

    def test_tuple_with_int_and_float(self):
        assert parse_ranges([(1, 2.5)]) == [(1.0, 2.5)]

    def test_list_instead_of_tuple(self):
        assert parse_ranges([[1.0, 2.0]]) == [(1.0, 2.0)]

    def test_multiple_unsorted_get_sorted(self):
        assert parse_ranges([(5.0, 6.0), (1.0, 2.0)]) == [(1.0, 2.0), (5.0, 6.0)]


class TestParseRangesDict:
    def test_dict_with_start_and_end(self):
        assert parse_ranges([{"start": 1.0, "end": 2.0}]) == [(1.0, 2.0)]

    def test_dict_with_time_strings(self):
        assert parse_ranges([{"start": "01:30", "end": "01:45"}]) == [(90.0, 105.0)]

    def test_dict_missing_field_raises(self):
        with pytest.raises(ValueError, match="missing"):
            parse_ranges([{"start": 1.0}])


class TestParseRangesString:
    def test_dash_separated(self):
        assert parse_ranges(["1.0-2.0"]) == [(1.0, 2.0)]

    def test_mmss_dash_separated(self):
        assert parse_ranges(["01:30-01:45"]) == [(90.0, 105.0)]

    def test_hhmmss_dash_separated(self):
        assert parse_ranges(["01:00:00-01:00:15"]) == [(3600.0, 3615.0)]

    def test_to_separator(self):
        assert parse_ranges(["1.0 to 2.0"]) == [(1.0, 2.0)]

    def test_unicode_arrow_separator(self):
        # Movie-makers paste from Notion / Google Docs where → replaces -.
        assert parse_ranges(["00:30 → 00:45"]) == [(30.0, 45.0)]

    def test_em_dash_separator(self):
        # Same source, em-dash variant.
        assert parse_ranges(["00:30—00:45"]) == [(30.0, 45.0)]

    def test_string_without_separator_raises(self):
        with pytest.raises(ValueError, match="separator"):
            parse_ranges(["just a number"])


class TestParseRangesCleaning:
    def test_negative_start_clamps_to_zero(self):
        assert parse_ranges([(-1.0, 5.0)]) == [(0.0, 5.0)]

    def test_zero_length_range_dropped(self):
        # start == end is degenerate; drop it silently so the GUI can
        # send a "just clicked, no drag" event without breaking anything.
        assert parse_ranges([(5.0, 5.0)]) == []

    def test_inverted_range_dropped(self):
        # User drags the timeline right-to-left -> end < start. Drop
        # silently rather than swap; the GUI handles direction.
        assert parse_ranges([(10.0, 5.0)]) == []

    def test_unrecognised_entry_raises(self):
        with pytest.raises(ValueError, match="unrecognised"):
            parse_ranges([42])  # int is neither a known entry type


# ---------------------------------------------------------------------------
# merge_overlapping
# ---------------------------------------------------------------------------
class TestMergeOverlapping:
    def test_empty(self):
        assert merge_overlapping([]) == []

    def test_single_range(self):
        assert merge_overlapping([(1.0, 2.0)]) == [(1.0, 2.0)]

    def test_disjoint_stays_separate(self):
        assert merge_overlapping([(1.0, 2.0), (5.0, 6.0)]) == [(1.0, 2.0), (5.0, 6.0)]

    def test_overlapping_merges(self):
        assert merge_overlapping([(1.0, 5.0), (3.0, 7.0)]) == [(1.0, 7.0)]

    def test_touching_at_zero_tolerance_merges(self):
        # The end of A == start of B counts as overlapping under default
        # tolerance 0.
        assert merge_overlapping([(1.0, 5.0), (5.0, 7.0)]) == [(1.0, 7.0)]

    def test_gap_within_tolerance_bridges(self):
        # 0.5s gap with tolerance 1.0 -> merge.
        assert merge_overlapping([(1.0, 5.0), (5.5, 7.0)], gap_tolerance=1.0) == [(1.0, 7.0)]

    def test_gap_beyond_tolerance_stays_separate(self):
        assert merge_overlapping([(1.0, 5.0), (6.0, 7.0)], gap_tolerance=0.5) == [
            (1.0, 5.0),
            (6.0, 7.0),
        ]

    def test_chain_collapses_to_single_range(self):
        ranges = [(1.0, 3.0), (2.0, 5.0), (4.0, 8.0)]
        assert merge_overlapping(ranges) == [(1.0, 8.0)]

    def test_unsorted_input_handled_defensively(self):
        # Most callers go through parse_ranges (which sorts), but the
        # signature accepts any sequence — pin defensiveness.
        ranges = [(5.0, 7.0), (1.0, 5.0)]
        assert merge_overlapping(ranges) == [(1.0, 7.0)]


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------
class TestCombine:
    def test_both_empty(self):
        assert combine([], []) == []

    def test_only_auto(self):
        auto = [(1.0, 2.0), (5.0, 6.0)]
        assert combine(auto, []) == auto

    def test_only_manual(self):
        manual = [(1.0, 2.0)]
        assert combine([], manual) == manual

    def test_disjoint_auto_and_manual(self):
        auto = [(1.0, 2.0)]
        manual = [(5.0, 6.0)]
        assert combine(auto, manual) == [(1.0, 2.0), (5.0, 6.0)]

    def test_overlapping_auto_and_manual_merge(self):
        # Realistic: auto-detected "person X appears 10-15s", user manually
        # adds 12-20s "skip this whole scene".
        auto = [(10.0, 15.0)]
        manual = [(12.0, 20.0)]
        assert combine(auto, manual) == [(10.0, 20.0)]

    def test_combine_is_idempotent(self):
        ranges = [(1.0, 5.0), (3.0, 7.0)]
        once = combine(ranges, [])
        twice = combine(once, [])
        assert once == twice == [(1.0, 7.0)]

    def test_compatible_with_pipeline_merge_gap_tolerance(self):
        # compute_forbidden._merge_seconds uses merge_gap=2.0 by default.
        # When the user wants the same bridging behaviour for manual ranges,
        # they pass gap_tolerance=2.0. Pin that.
        auto = [(10.0, 12.0)]
        manual = [(13.5, 15.0)]
        assert combine(auto, manual, gap_tolerance=2.0) == [(10.0, 15.0)]
