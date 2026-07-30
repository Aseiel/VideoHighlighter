"""Tests for `modules.highlight_select` — score curve to cut segments.

Pure numpy, no video: that is the point of the module existing separately. The
behaviour locked in here is the greedy selection as it ran inline in
`pipeline.py`, so a later change to how moments are spread across the video has
something to be compared against.
"""

from __future__ import annotations

import numpy as np

from modules.highlight_select import (
    bucket_ceiling,
    peak_confidence_by_sec,
    rank_seconds,
    select_fixed_window_segments,
    swap_segment,
)


def _score(n=120, **peaks):
    """Zeroed per-second score curve with named seconds given points."""
    score = np.zeros(n)
    for sec, points in peaks.items():
        score[int(sec.lstrip("s"))] = points
    return score


def _total(segments):
    return sum(end - start for start, end in segments)


# --- peak_confidence_by_sec -------------------------------------------------

def test_peak_confidence_keeps_the_highest_per_second():
    detections = {10: [("a", 0.4), ("b", 0.9), ("c", 0.7)]}
    assert peak_confidence_by_sec(detections) == {10: 0.9}


def test_peak_confidence_skips_seconds_with_no_detections():
    assert peak_confidence_by_sec({10: [], 11: [("a", 0.5)]}) == {11: 0.5}


# --- rank_seconds -----------------------------------------------------------

def test_ranking_is_by_descending_score():
    score = _score(n=10, s3=5.0, s7=9.0, s1=2.0)
    assert list(rank_seconds(score, duration_mode="MAX")) == [7, 3, 1]


def test_max_mode_ignores_zero_scoring_seconds():
    score = _score(n=10, s4=3.0)
    assert list(rank_seconds(score, duration_mode="MAX")) == [4]


def test_exact_mode_ranks_every_second_so_the_budget_can_be_filled():
    score = _score(n=10, s4=3.0)
    ranked = list(rank_seconds(score, duration_mode="EXACT"))
    assert ranked[0] == 4
    assert sorted(ranked) == list(range(10))


def test_confidence_breaks_ties_between_equal_scores():
    score = _score(n=10, s2=5.0, s8=5.0)
    ranked = rank_seconds(
        score, duration_mode="MAX", confidence_by_sec={2: 0.1, 8: 0.9}
    )
    assert list(ranked) == [8, 2]


def test_confidence_does_not_outrank_score():
    score = _score(n=10, s2=9.0, s8=1.0)
    ranked = rank_seconds(
        score, duration_mode="MAX", confidence_by_sec={2: 0.1, 8: 0.9}
    )
    assert list(ranked) == [2, 8]


# --- select_fixed_window_segments -------------------------------------------

def test_window_is_centred_on_the_peak_second():
    score = _score(n=120, s60=10.0)
    segments = select_fixed_window_segments(
        score, video_duration=120, clip_time=10,
        target_duration=10, duration_mode="MAX",
    )
    assert segments == [(55, 65)]


def test_segments_never_overlap():
    score = _score(n=120, s60=10.0, s62=9.0, s64=8.0, s100=7.0)
    segments = select_fixed_window_segments(
        score, video_duration=120, clip_time=10,
        target_duration=60, duration_mode="MAX",
    )
    taken = set()
    for start, end in segments:
        seconds = set(range(int(start), int(end)))
        assert not (seconds & taken)
        taken |= seconds


def test_segments_come_back_in_chronological_order():
    score = _score(n=200, s20=3.0, s100=9.0, s180=6.0)
    segments = select_fixed_window_segments(
        score, video_duration=200, clip_time=10,
        target_duration=30, duration_mode="MAX",
    )
    assert segments == sorted(segments)


def test_max_mode_stops_at_the_budget():
    score = np.linspace(10, 1, 120)
    segments = select_fixed_window_segments(
        score, video_duration=120, clip_time=10,
        target_duration=40, duration_mode="MAX",
    )
    assert _total(segments) <= 40


def test_max_mode_returns_less_than_the_budget_when_little_scored():
    score = _score(n=120, s60=10.0)
    segments = select_fixed_window_segments(
        score, video_duration=120, clip_time=10,
        target_duration=60, duration_mode="MAX",
    )
    assert _total(segments) == 10


def test_exact_mode_lands_on_the_target_even_with_a_flat_score():
    score = np.zeros(300)
    segments = select_fixed_window_segments(
        score, video_duration=300, clip_time=10,
        target_duration=45, duration_mode="EXACT",
    )
    assert _total(segments) == 45


def test_the_last_segment_is_trimmed_to_fit_the_remaining_budget():
    score = _score(n=120, s10=9.0, s40=8.0, s70=7.0)
    segments = select_fixed_window_segments(
        score, video_duration=120, clip_time=10,
        target_duration=25, duration_mode="MAX",
    )
    assert _total(segments) == 25
    assert min(end - start for start, end in segments) == 5


def test_a_window_at_the_very_end_is_pulled_back_inside_the_video():
    score = _score(n=100, s99=10.0)
    segments = select_fixed_window_segments(
        score, video_duration=100, clip_time=10,
        target_duration=10, duration_mode="MAX",
    )
    assert segments == [(90, 100)]
    assert segments[0][1] <= 100


def test_nothing_scored_gives_no_segments_in_max_mode():
    segments = select_fixed_window_segments(
        np.zeros(120), video_duration=120, clip_time=10,
        target_duration=60, duration_mode="MAX",
    )
    assert segments == []


def test_greedy_selection_concentrates_where_the_score_does():
    """The behaviour the coverage control is meant to change.

    Every high score sits in the last quarter, so the whole cut comes from
    there and the first three quarters of the video are never represented.
    """
    score = np.zeros(400)
    score[300:400] = np.linspace(10, 5, 100)
    segments = select_fixed_window_segments(
        score, video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX",
    )
    assert _total(segments) == 60
    assert all(start >= 295 for start, _ in segments)


# --- coverage: best moments <-> full story ----------------------------------

def _lopsided(n=400, hot_from=300):
    """A score curve whose every good moment sits in one stretch."""
    score = np.zeros(n)
    score[hot_from:] = np.linspace(10, 5, n - hot_from)
    # Weak but non-zero elsewhere, so the earlier stretches have candidates.
    score[:hot_from] = np.linspace(0.5, 1.0, hot_from)
    return score


def test_ceiling_at_zero_coverage_is_the_whole_budget():
    assert bucket_ceiling(coverage=0.0, clip_time=10, target_duration=60) == 60


def test_ceiling_at_full_coverage_is_one_buckets_fair_share():
    assert bucket_ceiling(coverage=1.0, clip_time=10, target_duration=60) == 10


def test_ceiling_moves_monotonically_with_coverage():
    ceilings = [
        bucket_ceiling(coverage=c, clip_time=10, target_duration=60)
        for c in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert ceilings == sorted(ceilings, reverse=True)


def test_zero_coverage_is_exactly_the_old_behaviour():
    score = _lopsided()
    assert select_fixed_window_segments(
        score, video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX", coverage=0.0,
    ) == select_fixed_window_segments(
        score, video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX",
    )


def test_full_coverage_reaches_the_start_of_a_lopsided_video():
    segments = select_fixed_window_segments(
        _lopsided(), video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX", coverage=1.0,
    )
    assert min(start for start, _ in segments) < 100


def test_full_coverage_puts_a_clip_in_every_bucket():
    segments = select_fixed_window_segments(
        _lopsided(), video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX", coverage=1.0,
    )
    # By midpoint, not by start: a clip is centred on its peak second, and that
    # is the second the ceiling is charged against.
    buckets = {int((start + end) / 2 / 400 * 6) for start, end in segments}
    assert len(buckets) == 6


def test_coverage_still_spends_the_whole_budget():
    for coverage in (0.0, 0.5, 1.0):
        segments = select_fixed_window_segments(
            _lopsided(), video_duration=400, clip_time=10,
            target_duration=60, duration_mode="MAX", coverage=coverage,
        )
        assert _total(segments) == 60, coverage


def test_coverage_never_starves_exact_mode_of_its_target():
    """Only one stretch has any candidates at all, yet EXACT must still fill."""
    score = np.zeros(400)
    score[300:320] = 9.0
    segments = select_fixed_window_segments(
        score, video_duration=400, clip_time=10,
        target_duration=90, duration_mode="EXACT", coverage=1.0,
    )
    assert _total(segments) == 90


def test_partial_coverage_keeps_the_top_moment():
    """Mid-slider still takes the single best moment first."""
    segments = select_fixed_window_segments(
        _lopsided(), video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX", coverage=0.5,
    )
    assert (295, 305) in segments


def test_coverage_is_clamped_to_a_sane_range():
    over = select_fixed_window_segments(
        _lopsided(), video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX", coverage=5.0,
    )
    under = select_fixed_window_segments(
        _lopsided(), video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX", coverage=-3.0,
    )
    assert over == select_fixed_window_segments(
        _lopsided(), video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX", coverage=1.0,
    )
    assert under == select_fixed_window_segments(
        _lopsided(), video_duration=400, clip_time=10,
        target_duration=60, duration_mode="MAX", coverage=0.0,
    )


def test_segments_never_overlap_at_any_coverage():
    for coverage in (0.0, 0.3, 0.7, 1.0):
        taken = set()
        for start, end in select_fixed_window_segments(
            _lopsided(), video_duration=400, clip_time=10,
            target_duration=60, duration_mode="MAX", coverage=coverage,
        ):
            seconds = set(range(int(start), int(end)))
            assert not (seconds & taken), coverage
            taken |= seconds


# --- exclude: time the selection may not touch -----------------------------

def test_excluded_time_is_never_selected():
    score = _score(n=120, s60=10.0, s20=9.0)
    segments = select_fixed_window_segments(
        score, video_duration=120, clip_time=10,
        target_duration=20, duration_mode="MAX", exclude=[(50, 70)],
    )
    assert all(not (start < 70 and end > 50) for start, end in segments)


def test_excluding_a_moment_yields_another_rather_than_a_shorter_cut():
    score = _score(n=200, s60=10.0, s120=9.0, s170=8.0)
    full = select_fixed_window_segments(
        score, video_duration=200, clip_time=10,
        target_duration=20, duration_mode="MAX",
    )
    without = select_fixed_window_segments(
        score, video_duration=200, clip_time=10,
        target_duration=20, duration_mode="MAX", exclude=[(55, 65)],
    )
    assert _total(full) == _total(without) == 20
    assert (55, 65) in full and (55, 65) not in without


def test_excluding_everything_leaves_nothing():
    score = _score(n=120, s60=10.0)
    assert select_fixed_window_segments(
        score, video_duration=120, clip_time=10,
        target_duration=20, duration_mode="MAX", exclude=[(0, 120)],
    ) == []


# --- swap: give me a different one -----------------------------------------

def _swappable():
    return _score(n=400, s60=10.0, s150=9.0, s250=8.0, s340=7.0)


def test_swap_replaces_only_the_chosen_clip():
    score = _swappable()
    segments = select_fixed_window_segments(
        score, video_duration=400, clip_time=10,
        target_duration=30, duration_mode="MAX",
    )
    swapped = swap_segment(score, segments=segments, index=0,
                           video_duration=400, clip_time=10)
    assert swapped is not None
    assert segments[0] not in swapped
    assert segments[1] in swapped and segments[2] in swapped


def test_swap_keeps_the_highlight_the_same_length():
    score = _swappable()
    segments = select_fixed_window_segments(
        score, video_duration=400, clip_time=10,
        target_duration=30, duration_mode="MAX",
    )
    swapped = swap_segment(score, segments=segments, index=1,
                           video_duration=400, clip_time=10)
    assert _total(swapped) == _total(segments)


def test_swap_returns_segments_in_chronological_order():
    score = _swappable()
    segments = select_fixed_window_segments(
        score, video_duration=400, clip_time=10,
        target_duration=30, duration_mode="MAX",
    )
    swapped = swap_segment(score, segments=segments, index=0,
                           video_duration=400, clip_time=10)
    assert swapped == sorted(swapped)


def test_swap_offers_the_next_best_moment():
    score = _swappable()
    segments = select_fixed_window_segments(
        score, video_duration=400, clip_time=10,
        target_duration=30, duration_mode="MAX",
    )          # takes 60, 150, 250
    swapped = swap_segment(score, segments=segments, index=0,
                           video_duration=400, clip_time=10)
    assert (335, 345) in swapped, "the best unused moment should be offered"


def test_swapping_repeatedly_keeps_offering_new_moments():
    """What `rejected` is for: the caller feeds the result back in and
    accumulates what was turned down, exactly as a "swap again" button would."""
    score = _score(n=400, s60=10.0, s150=9.0, s250=8.0, s340=7.0, s30=6.0)
    segments = [(145, 155), (245, 255)]
    rejected = []
    seen = set()

    for _ in range(3):
        turned_down = segments[0]
        swapped = swap_segment(score, segments=segments, index=0,
                               video_duration=400, clip_time=10,
                               rejected=rejected)
        assert swapped is not None
        rejected.append(turned_down)
        offered = [s for s in swapped if s not in segments][0]
        assert offered not in seen, "the same moment was offered twice"
        seen.add(offered)
        segments = swapped

    assert len(seen) == 3


def test_swap_says_no_when_the_video_has_nothing_else():
    score = _score(n=120, s60=10.0)
    segments = [(55, 65)]
    assert swap_segment(score, segments=segments, index=0,
                        video_duration=120, clip_time=10) is None


def test_swap_never_offers_back_the_clip_being_replaced():
    score = _score(n=400, s60=10.0, s150=9.0)
    segments = [(55, 65)]
    swapped = swap_segment(score, segments=segments, index=0,
                           video_duration=400, clip_time=10)
    assert swapped is not None and (55, 65) not in swapped


def test_swap_rejects_an_index_that_is_not_there():
    score = _swappable()
    for bad in (5, -1):
        try:
            swap_segment(score, segments=[(0, 10)], index=bad,
                         video_duration=400, clip_time=10)
        except IndexError:
            continue
        raise AssertionError(f"index {bad} should have raised")


def test_exact_mode_can_always_offer_something():
    """EXACT ranks every second, so a swap has an answer even on a flat score."""
    score = np.zeros(400)
    segments = [(100, 110)]
    swapped = swap_segment(score, segments=segments, index=0,
                           video_duration=400, clip_time=10,
                           duration_mode="EXACT")
    assert swapped is not None and _total(swapped) == 10
