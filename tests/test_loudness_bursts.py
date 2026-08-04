"""Tests for `modules.loudness_bursts` — turning measurements into events.

Pure stdlib and numpy, no ffmpeg and no video file: `group` and `event_seconds`
take the candidate dicts `reaction_bursts.find_candidates` produces, so the
policy layer is testable without decoding anything.

Deliberately generic labels and synthetic curves throughout — the module holds
no opinion about what makes a sound, and neither do its tests.
"""

from __future__ import annotations

import numpy as np

from modules.loudness_bursts import (
    DEFAULT_EDGE_GUARD,
    DEFAULT_MERGE_GAP,
    event_seconds,
    group,
)
from modules.reaction_bursts import find_candidates


def _candidate(start, end, z, modulation=0.05, mod_hz=1.0):
    """One entry shaped like `find_candidates` output."""
    return {
        "start": float(start),
        "end": float(end),
        "duration": float(end - start),
        "peak_second": int(start),
        "timestamp": "0:00",
        "z": float(z),
        "burst_db": 10.0,
        "modulation": float(modulation),
        "peak_modulation": float(modulation),
        "mod_hz": float(mod_hz),
    }


def test_edge_guard_drops_opening_and_closing_material():
    cands = [
        _candidate(10, 12, 3.0),      # inside the opening guard
        _candidate(600, 602, 2.5),    # content
        _candidate(3590, 3592, 4.0),  # inside the closing guard
    ]
    events = group(cands, total_seconds=3600)
    assert [e["peak_second"] for e in events] == [600]


def test_adjacent_candidates_merge_into_one_event():
    cands = [
        _candidate(600, 602, 2.2),
        _candidate(615, 617, 3.1),   # 13s later -> same event
        _candidate(640, 642, 2.0),   # 23s after that -> still same event
    ]
    events = group(cands, total_seconds=3600)
    assert len(events) == 1
    assert events[0]["runs"] == 3
    # The event carries the strongest run's peak, not the first run's.
    assert events[0]["z"] == 3.1
    assert events[0]["peak_second"] == 615
    assert events[0]["start"] == 600.0
    assert events[0]["end"] == 642.0


def test_distant_candidates_stay_separate():
    cands = [_candidate(600, 602, 2.2), _candidate(700, 702, 2.4)]
    events = group(cands, total_seconds=3600)
    assert len(events) == 2


def test_merge_gap_is_a_policy_knob_not_a_constant():
    """60s was measured to swallow separately-identified moments; 30s does not."""
    cands = [_candidate(600, 602, 2.2), _candidate(650, 652, 2.4)]
    assert len(group(cands, total_seconds=3600, merge_gap=30)) == 2
    assert len(group(cands, total_seconds=3600, merge_gap=60)) == 1


def test_events_are_ranked_by_excursion():
    cands = [
        _candidate(600, 602, 2.1),
        _candidate(1200, 1202, 4.4),
        _candidate(1800, 1802, 3.2),
    ]
    events = group(cands, total_seconds=3600)
    assert [e["z"] for e in events] == [4.4, 3.2, 2.1]


def test_event_seconds_covers_the_whole_span_not_just_the_peak():
    events = group([_candidate(600, 604, 2.5)], total_seconds=3600)
    secs = event_seconds(events)
    assert secs == {600, 601, 602, 603, 604}


def test_empty_input_is_not_an_error():
    assert group([], total_seconds=3600) == []
    assert event_seconds([]) == set()


def test_guards_can_exceed_the_video_without_crashing():
    cands = [_candidate(50, 52, 3.0)]
    assert group(cands, total_seconds=100, edge_guard=DEFAULT_EDGE_GUARD) == []


# --- the ranking option this module depends on -----------------------------

def _curves(loud_seconds, *, n=200, modulation_at=None):
    """Synthetic per-second curves: quiet everywhere, loud where asked."""
    z = np.zeros(n)
    for sec, value in loud_seconds.items():
        z[sec] = value
    burst = z * 4.0
    mod = np.full(n, 0.02)
    for sec, value in (modulation_at or {}).items():
        mod[sec] = value
    return z, burst, mod, np.full(n, 1.0)


def test_rank_by_z_puts_the_largest_excursion_first():
    """The shipped default ranks rhythm first, which buries the loudest moment.

    This is the whole reason `rank_by` exists: on material whose rhythm is
    speech, ordering by modulation promotes dialogue over the excursions.
    """
    z, burst, mod, hz = _curves(
        {40: 3.0, 41: 3.0, 100: 5.0, 101: 5.0},
        modulation_at={40: 0.30, 41: 0.30},   # rhythmic but not loud-est
    )
    by_mod = find_candidates(z, burst, mod, hz, z_threshold=2.0, min_duration=1.0)
    by_z = find_candidates(z, burst, mod, hz, z_threshold=2.0, min_duration=1.0,
                           rank_by="z")

    assert by_mod[0]["peak_second"] == 40      # rhythm wins by default
    assert by_z[0]["peak_second"] == 100       # excursion wins when asked
    # Ordering only — the same candidates come back either way.
    assert {c["peak_second"] for c in by_mod} == {c["peak_second"] for c in by_z}


def test_min_duration_decides_membership_and_is_the_consequential_knob():
    """1.5s (which suits sustained applause) reports nothing on brief events."""
    z, burst, mod, hz = _curves({100: 5.0, 101: 5.0})
    assert find_candidates(z, burst, mod, hz,
                           z_threshold=2.0, min_duration=1.0)
    assert not find_candidates(z, burst, mod, hz,
                               z_threshold=2.0, min_duration=3.0)
