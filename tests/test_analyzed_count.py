"""
Tests for how a finished run is counted toward the lifetime analyzed total.

Two things make this easy to get wrong:

1. run_highlighter returns different shapes. A single video returns the output
   path (or None on failure); a batch returns [(input_path, output_or_None), ...]
   -- see pipeline.py:373. Counting len() on the batch shape counts the failures
   as successes.
2. Only main.py used to increment the counter, inside its own pipeline_done
   handler, so a run driven from anywhere else (the web UI's sidecar) left the
   lifetime total at 0 no matter how many videos it processed.
"""

from __future__ import annotations

import pytest


def count_analyzed(output):
    """The rule from sidecar/worker.py, mirroring main.py:4040."""
    if isinstance(output, list):
        return sum(1 for entry in output if entry and entry[1])
    return 1 if output else 0


@pytest.mark.parametrize(
    "output, expected",
    [
        # Single video: the output path, or None when it produced nothing.
        ("C:/out/highlight.mp4", 1),
        (None, 0),
        ("", 0),
        # Batch: (input, output_or_None) per video.
        ([("a.mp4", "a_highlight.mp4"), ("b.mp4", "b_highlight.mp4")], 2),
        ([("a.mp4", "a_highlight.mp4"), ("b.mp4", None)], 1),
        ([("a.mp4", None), ("b.mp4", None)], 0),
        ([], 0),
    ],
)
def test_count_analyzed(output, expected):
    assert count_analyzed(output) == expected


def test_batch_failures_are_not_counted():
    """The case a len()-based count gets wrong: a batch where some videos failed
    still has an entry per video, so len() would report them all as analyzed."""
    output = [("a.mp4", "a_highlight.mp4"), ("b.mp4", None), ("c.mp4", None)]
    assert len(output) == 3
    assert count_analyzed(output) == 1


def test_increment_is_additive(tmp_path, monkeypatch):
    """The lifetime total accumulates across runs rather than being overwritten."""
    from modules import analysis_stats

    stats_file = tmp_path / "analysis_stats.json"
    monkeypatch.setattr(analysis_stats, "stats_path", lambda: str(stats_file))

    assert analysis_stats.get_analyzed_count() == 0
    assert analysis_stats.increment_analyzed(2) == 2
    assert analysis_stats.increment_analyzed(1) == 3
    assert analysis_stats.get_analyzed_count() == 3


def test_unwritable_stats_never_raises(monkeypatch):
    """A read-only install must not fail a run that already succeeded."""
    from modules import analysis_stats

    monkeypatch.setattr(analysis_stats, "stats_path",
                        lambda: "Z:/definitely/not/writable/stats.json")
    # Both directions have to stay quiet.
    assert analysis_stats.get_analyzed_count() == 0
    analysis_stats.increment_analyzed(1)
