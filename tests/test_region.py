"""
Characterization tests for `auto_segments.Region`.

`Region` is the candidate-segment data class used during auto-segmentation. The
overlap math drives merging: two adjacent or overlapping regions should
collapse into one. The score addition policy is what makes "many signals fired
here" rank above "one signal fired here" during selection.
"""

from __future__ import annotations

import pytest

from modules.auto_segments import Region


class TestConstruction:
    def test_basic_construction(self):
        r = Region(10.0, 20.0, score=5.0, sources=["motion"])
        assert r.start == 10.0
        assert r.end == 20.0
        assert r.score == 5.0
        assert r.sources == ["motion"]

    def test_default_sources_is_empty_list_not_shared(self):
        # Common Python gotcha: mutable default argument. If sources defaults
        # to a shared list, modifying one Region's sources mutates all of
        # them. Pin that the implementation gives each Region its own list.
        r1 = Region(0.0, 1.0)
        r2 = Region(0.0, 1.0)
        r1.sources.append("scene")
        assert r2.sources == []

    def test_inputs_are_coerced_to_float(self):
        r = Region(10, 20, score=5)
        assert isinstance(r.start, float)
        assert isinstance(r.end, float)
        assert isinstance(r.score, float)


class TestDuration:
    def test_normal_duration(self):
        assert Region(10.0, 20.0).duration == 10.0

    def test_zero_duration(self):
        assert Region(10.0, 10.0).duration == 0.0

    def test_inverted_clamps_to_zero(self):
        # Defensive: if start > end, duration should not go negative.
        assert Region(20.0, 10.0).duration == 0.0


class TestOverlaps:
    def test_disjoint_regions_do_not_overlap_with_zero_tolerance(self):
        a = Region(0.0, 10.0)
        b = Region(20.0, 30.0)
        assert a.overlaps(b, gap_tolerance=0.0) is False

    def test_touching_regions_overlap(self):
        # End of A == start of B. With gap_tolerance=0, this should count as
        # touching (a.start <= b.end and b.start <= a.end both hold).
        a = Region(0.0, 10.0)
        b = Region(10.0, 20.0)
        assert a.overlaps(b, gap_tolerance=0.0) is True

    def test_default_gap_tolerance_bridges_short_gaps(self):
        # Default gap_tolerance=1.5. Regions 1.0s apart should be considered
        # overlapping for merge purposes.
        a = Region(0.0, 10.0)
        b = Region(11.0, 20.0)
        assert a.overlaps(b) is True

    def test_gap_beyond_tolerance_does_not_overlap(self):
        a = Region(0.0, 10.0)
        b = Region(12.0, 20.0)
        assert a.overlaps(b, gap_tolerance=1.0) is False

    def test_overlap_is_symmetric(self):
        a = Region(0.0, 10.0)
        b = Region(5.0, 15.0)
        assert a.overlaps(b) == b.overlaps(a)


class TestMerge:
    def test_merge_takes_union_of_spans(self):
        a = Region(0.0, 10.0, score=3.0, sources=["motion"])
        b = Region(5.0, 15.0, score=4.0, sources=["audio"])
        merged = a.merge(b)
        assert merged.start == 0.0
        assert merged.end == 15.0

    def test_merge_sums_scores(self):
        # Pin: merge adds scores rather than maxing them. The "many signals
        # here" intuition is built on this.
        a = Region(0.0, 10.0, score=3.0)
        b = Region(5.0, 15.0, score=4.0)
        merged = a.merge(b)
        assert merged.score == 7.0

    def test_merge_concatenates_sources(self):
        a = Region(0.0, 10.0, sources=["motion"])
        b = Region(5.0, 15.0, sources=["audio", "scene"])
        merged = a.merge(b)
        assert merged.sources == ["motion", "audio", "scene"]

    def test_merge_with_disjoint_still_takes_union(self):
        # The merge method itself is span-blind; it just unions. The
        # `overlaps` check is the caller's responsibility before merging.
        a = Region(0.0, 5.0, score=1.0)
        b = Region(20.0, 25.0, score=2.0)
        merged = a.merge(b)
        assert (merged.start, merged.end) == (0.0, 25.0)
        assert merged.score == 3.0

    def test_merge_returns_new_region_not_mutating(self):
        a = Region(0.0, 10.0, score=3.0, sources=["motion"])
        b = Region(5.0, 15.0, score=4.0, sources=["audio"])
        _ = a.merge(b)
        # Originals untouched.
        assert (a.start, a.end, a.score, a.sources) == (0.0, 10.0, 3.0, ["motion"])
        assert (b.start, b.end, b.score, b.sources) == (5.0, 15.0, 4.0, ["audio"])
