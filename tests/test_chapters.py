"""Tests for chapterization — the partition layer over the shot list.

No CLIP and no video here. Every operator in `modules/chapters.py` is numpy over
a shot-signature array, so a synthetic film — a known number of locations, a
known number of shots each, dialogue alternation inside them — pins down exactly
what the partition does. The embeddings only decide how far apart two locations
sit; these tests fix how that distance is read.

The case that matters most is `test_dialogue_does_not_split_a_scene`: naive
per-cut novelty fails precisely there, and the windowed comparison exists to
survive it.
"""
from __future__ import annotations

import numpy as np
import pytest

from modules.chapters import (
    DEFAULT_SHOT_WINDOW,
    boundary_novelty,
    chapterize,
    describe_pace,
    format_timestamp,
    l2_normalize,
    pick_boundaries,
    robust_threshold,
    shot_signatures,
    to_ffmetadata,
    to_youtube,
)

DIM = 16


def _unit(index: int, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    """A basis vector standing for one location, optionally roughened."""
    v = np.zeros(DIM, dtype=np.float32)
    v[index % DIM] = 1.0
    if jitter:
        rng = np.random.default_rng(seed)
        v = v + jitter * rng.standard_normal(DIM).astype(np.float32)
    return l2_normalize(v.reshape(1, -1))[0]


def _film(locations: int = 4, shots_per_location: int = 20,
          shot_seconds: float = 5.0, dialogue: bool = True):
    """Build a synthetic film: contiguous runs of shots per location.

    With `dialogue`, shots inside a location alternate between two nearby setups
    — shot-reverse-shot. That alternation is a real, large, per-cut change that
    carries no chapter information, which is the trap being tested.
    """
    scenes, sigs = [], []
    t = 0.0
    for loc in range(locations):
        base = _unit(loc)
        other = l2_normalize((base + 0.6 * _unit(loc + 8)).reshape(1, -1))[0]
        for shot in range(shots_per_location):
            scenes.append((t, t + shot_seconds))
            sigs.append(other if (dialogue and shot % 2) else base)
            t += shot_seconds
    return scenes, np.stack(sigs).astype(np.float32), t


def _per_second(scenes, sigs, duration):
    """Expand shot signatures to the per-second arrays a CLIP index hands over."""
    ts = np.arange(0.0, duration, 1.0)
    emb = np.zeros((len(ts), DIM), dtype=np.float32)
    for (start, end), sig in zip(scenes, sigs):
        sel = (ts >= start) & (ts < end)
        emb[sel] = sig
    return ts, emb


# ---------------------------------------------------------------------------
# boundary_novelty
# ---------------------------------------------------------------------------
class TestBoundaryNovelty:
    def test_peaks_at_the_location_changes(self):
        scenes, sigs, _ = _film(locations=4, shots_per_location=20)
        novelty = boundary_novelty(sigs, window=DEFAULT_SHOT_WINDOW)

        # Real boundaries sit at shot 20, 40, 60.
        for edge in (20, 40, 60):
            assert novelty[edge] > 0.5, f"missed the change at shot {edge}"

    def test_dialogue_does_not_split_a_scene(self):
        """The whole reason for a shot window rather than an adjacent-cut diff."""
        scenes, sigs, _ = _film(locations=4, shots_per_location=20, dialogue=True)
        novelty = boundary_novelty(sigs, window=DEFAULT_SHOT_WINDOW)

        real = {20, 40, 60}
        interior = [novelty[i] for i in range(1, len(novelty)) if i not in real]
        assert max(interior) < min(novelty[i] for i in real), (
            "a cut inside a dialogue scene scored as high as a location change"
        )

    def test_first_shot_has_no_cut_before_it(self):
        _, sigs, _ = _film()
        assert boundary_novelty(sigs)[0] == 0.0

    def test_ends_are_truncated_not_padded(self):
        """Padding would manufacture a boundary a few shots into the film."""
        sigs = np.repeat(_unit(0).reshape(1, -1), 30, axis=0)
        novelty = boundary_novelty(sigs, window=DEFAULT_SHOT_WINDOW)
        assert np.allclose(novelty, 0.0, atol=1e-5)

    def test_handles_degenerate_input(self):
        assert boundary_novelty(np.zeros((0, DIM), dtype=np.float32)).size == 0
        assert boundary_novelty(np.zeros((1, DIM), dtype=np.float32)).size == 1


# ---------------------------------------------------------------------------
# thresholds and picking
# ---------------------------------------------------------------------------
class TestRobustThreshold:
    def test_outliers_do_not_raise_their_own_bar(self):
        flat = np.full(100, 0.05, dtype=np.float32)
        spiked = flat.copy()
        spiked[[10, 40, 70]] = 5.0
        assert robust_threshold(spiked) == pytest.approx(robust_threshold(flat), abs=1e-3)

    def test_empty_is_zero(self):
        assert robust_threshold(np.zeros(0, dtype=np.float32)) == 0.0


class TestPickBoundaries:
    def test_respects_the_minimum_chapter_length(self):
        # 30 s shots, so the cluster sits well past the head guard but its
        # members are only 30 s from each other.
        novelty = np.zeros(50, dtype=np.float32)
        novelty[[10, 11, 12]] = [0.9, 0.95, 0.92]
        starts = np.arange(50, dtype=np.float64) * 30.0

        picked = pick_boundaries(novelty, starts, min_gap=90.0, max_run=0.0)
        assert picked == [11], "took more than the strongest cut of the cluster"

    def test_prefers_the_strongest_cut_not_the_first(self):
        novelty = np.zeros(50, dtype=np.float32)
        novelty[[10, 12]] = [0.6, 0.95]
        starts = np.arange(50, dtype=np.float64) * 30.0

        assert pick_boundaries(novelty, starts, min_gap=90.0, max_run=0.0) == [12]

    def test_no_stub_chapter_at_the_head(self):
        """The truncated window makes the first cut score high on any video."""
        novelty = np.zeros(40, dtype=np.float32)
        novelty[1] = 0.99
        starts = np.arange(40, dtype=np.float64) * 10.0

        assert pick_boundaries(novelty, starts, min_gap=90.0, max_run=0.0,
                               duration=400.0) == []

    def test_target_count_overrides_the_threshold(self):
        novelty = np.linspace(0.0, 1.0, 60).astype(np.float32)
        starts = np.arange(60, dtype=np.float64) * 30.0

        picked = pick_boundaries(novelty, starts, min_gap=60.0, max_run=0.0,
                                 target=5, duration=1800.0)
        assert len(picked) == 4, "5 chapters needs 4 interior boundaries"

    def test_target_beyond_capacity_returns_fewer(self):
        novelty = np.linspace(0.0, 1.0, 10).astype(np.float32)
        starts = np.arange(10, dtype=np.float64) * 10.0

        picked = pick_boundaries(novelty, starts, min_gap=90.0, max_run=0.0,
                                 target=8, duration=100.0)
        assert len(picked) <= 1

    def test_no_stub_chapter_at_the_tail(self):
        novelty = np.zeros(40, dtype=np.float32)
        novelty[38] = 0.99                       # a strong cut 20 s from the end
        starts = np.arange(40, dtype=np.float64) * 10.0

        picked = pick_boundaries(novelty, starts, min_gap=90.0, max_run=0.0,
                                 duration=400.0)
        assert picked == []

    def test_a_long_flat_run_is_split_anyway(self):
        novelty = np.full(200, 0.01, dtype=np.float32)
        novelty[100] = 0.02                      # barely anything, but the best
        starts = np.arange(200, dtype=np.float64) * 10.0

        picked = pick_boundaries(novelty, starts, min_gap=90.0, max_run=600.0,
                                 z=99.0, duration=2000.0)
        assert picked, "a 2000 s video with max_run=600 got no boundary"
        assert all((b - a) <= 600.0 + 1e-6 for a, b in zip(
            [0.0] + [starts[i] for i in picked],
            [starts[i] for i in picked] + [2000.0],
        ))


# ---------------------------------------------------------------------------
# shot_signatures
# ---------------------------------------------------------------------------
class TestShotSignatures:
    def test_averages_the_seconds_inside_each_shot(self):
        scenes, sigs, duration = _film(locations=2, shots_per_location=4)
        ts, emb = _per_second(scenes, sigs, duration)

        got, keep = shot_signatures(ts, emb, scenes)
        assert len(got) == len(scenes)
        assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)
        assert got[0] @ sigs[0] == pytest.approx(1.0, abs=1e-4)

    def test_drops_shots_that_fell_between_samples(self):
        # The middle shot opens and closes strictly between two 1 s samples, so
        # nothing was ever measured inside it.
        scenes = [(0.0, 10.2), (10.2, 10.8), (10.8, 20.0)]
        ts = np.arange(0.0, 20.0, 1.0)
        emb = np.repeat(_unit(0).reshape(1, -1), len(ts), axis=0)

        got, keep = shot_signatures(ts, emb, scenes)
        assert list(keep) == [0, 2], "the sub-sample shot invented a signature"
        assert len(got) == 2


# ---------------------------------------------------------------------------
# chapterize
# ---------------------------------------------------------------------------
class TestChapterize:
    def test_finds_the_locations_of_a_synthetic_film(self):
        scenes, sigs, duration = _film(locations=4, shots_per_location=24,
                                       shot_seconds=5.0)
        ts, emb = _per_second(scenes, sigs, duration)

        chapters = chapterize(duration, scenes, ts, emb, log_fn=lambda *_: None)

        # 4 locations x 24 shots x 5 s = 120 s each, 480 s total.
        assert len(chapters) == 4
        for ch, expected in zip(chapters[1:], (120.0, 240.0, 360.0)):
            assert ch["start"] == pytest.approx(expected, abs=5.0)

    def test_the_result_is_a_partition(self):
        scenes, sigs, duration = _film(locations=5, shots_per_location=20)
        ts, emb = _per_second(scenes, sigs, duration)

        chapters = chapterize(duration, scenes, ts, emb, log_fn=lambda *_: None)

        assert chapters[0]["start"] == 0.0
        assert chapters[-1]["end"] == pytest.approx(duration)
        for a, b in zip(chapters[:-1], chapters[1:]):
            assert a["end"] == b["start"], "a gap or an overlap between chapters"
        assert [c["number"] for c in chapters] == list(range(1, len(chapters) + 1))

    def test_every_boundary_lands_on_a_real_cut(self):
        scenes, sigs, duration = _film(locations=4, shots_per_location=22)
        ts, emb = _per_second(scenes, sigs, duration)
        cuts = {round(s, 2) for s, _ in scenes}

        chapters = chapterize(duration, scenes, ts, emb, log_fn=lambda *_: None)
        for ch in chapters[1:]:
            assert ch["start"] in cuts, "a chapter began mid-shot"

    def test_target_count_is_honoured(self):
        scenes, sigs, duration = _film(locations=8, shots_per_location=20)
        ts, emb = _per_second(scenes, sigs, duration)

        chapters = chapterize(duration, scenes, ts, emb, target=6,
                              log_fn=lambda *_: None)
        assert len(chapters) == 6

    def test_falls_back_to_shot_length_without_embeddings(self):
        scenes, _, duration = _film(locations=4, shots_per_location=20)

        chapters = chapterize(duration, scenes, log_fn=lambda *_: None)

        assert chapters, "the fallback produced no partition"
        assert all(c["method"] == "shot-length" for c in chapters)
        cuts = {round(s, 2) for s, _ in scenes}
        for ch in chapters[1:]:
            assert ch["start"] in cuts

    def test_a_video_with_no_structure_is_one_chapter(self):
        chapters = chapterize(120.0, [(0.0, 120.0)], log_fn=lambda *_: None)
        assert len(chapters) == 1
        assert chapters[0]["method"] == "single"
        assert chapters[0]["end"] == pytest.approx(120.0)

    def test_zero_duration_is_empty(self):
        assert chapterize(0.0, [(0.0, 10.0)], log_fn=lambda *_: None) == []

    def test_result_is_json_serialisable(self):
        import json

        scenes, sigs, duration = _film(locations=3, shots_per_location=20)
        ts, emb = _per_second(scenes, sigs, duration)
        chapters = chapterize(duration, scenes, ts, emb, log_fn=lambda *_: None)

        json.dumps(chapters)   # no numpy scalars may escape


# ---------------------------------------------------------------------------
# description and export
# ---------------------------------------------------------------------------
class TestDescription:
    @pytest.mark.parametrize("shots,seconds,expected", [
        (60, 120.0, "fast-cut"),
        (12, 120.0, "steady"),
        (4, 120.0, "held"),
        (0, 120.0, "unmeasured"),
    ])
    def test_pace_bands(self, shots, seconds, expected):
        assert describe_pace(shots, seconds) == expected

    @pytest.mark.parametrize("seconds,expected", [
        (0.0, "0:00:00"), (61.0, "0:01:01"), (3600.0, "1:00:00"), (3725.4, "1:02:05"),
    ])
    def test_timestamp_format(self, seconds, expected):
        assert format_timestamp(seconds) == expected


class TestCachedSignatures:
    """`cached_index_arrays` must never build an index — only reuse one."""

    def test_a_missing_cache_falls_back_rather_than_encoding(self, tmp_path):
        from modules.chapters import cached_index_arrays

        ts, emb = cached_index_arrays(str(tmp_path / "nope.mp4"),
                                      cache_dir=str(tmp_path))
        assert ts is None and emb is None

    def test_chapters_for_video_still_partitions_without_a_cache(self, tmp_path):
        from modules.chapters import chapters_for_video

        scenes, _, duration = _film(locations=4, shots_per_location=20)
        chapters = chapters_for_video(str(tmp_path / "nope.mp4"), scenes,
                                      duration, cache_dir=str(tmp_path),
                                      log_fn=lambda *_: None)
        assert chapters, "no partition without a visual index"
        assert all(c["method"] == "shot-length" for c in chapters)
        assert chapters[0]["start"] == 0.0
        assert chapters[-1]["end"] == pytest.approx(duration)


class TestExport:
    def _chapters(self):
        return [
            {"start": 0.0, "end": 600.0, "timestamp": "0:00:00", "title": "Chapter 1"},
            {"start": 600.0, "end": 1200.5, "timestamp": "0:10:00", "title": "Chapter 2"},
        ]

    def test_ffmetadata_shape(self):
        out = to_ffmetadata(self._chapters())
        assert out.startswith(";FFMETADATA1")
        assert out.count("[CHAPTER]") == 2
        assert "START=600000" in out and "END=1200500" in out
        assert "TIMEBASE=1/1000" in out

    def test_ffmetadata_escapes_special_characters(self):
        out = to_ffmetadata([{"start": 0.0, "end": 10.0, "title": "a=b;c#d"}])
        assert "title=a\\=b\\;c\\#d" in out

    def test_ffmetadata_skips_empty_spans(self):
        out = to_ffmetadata([{"start": 5.0, "end": 5.0, "title": "x"}])
        assert "[CHAPTER]" not in out

    def test_youtube_format(self):
        assert to_youtube(self._chapters()) == "0:00:00 Chapter 1\n0:10:00 Chapter 2"
