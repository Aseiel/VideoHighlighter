"""
Tests for the manual-avoid persistence layer.

Ranges marked in the timeline viewer used to live only in that window's scene,
so main.py could read them (same process) but nothing else could. Persisting
them per video makes the ranges the source of truth, which is what lets the web
UI's sidecar feed them into a run.
"""

from __future__ import annotations

import json

import pytest

from modules import manual_avoid


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    """Point the store at a temp file so tests never touch the real cache."""
    path = tmp_path / "cache" / "manual_avoid.json"
    monkeypatch.setattr(manual_avoid, "store_path", lambda: str(path))
    return path


def test_round_trip():
    manual_avoid.save_ranges("a.mp4", [(5.0, 10.0)])
    assert manual_avoid.load_ranges("a.mp4") == [(5.0, 10.0)]


def test_saved_ranges_are_merged_and_sorted():
    manual_avoid.save_ranges("a.mp4", [(30.0, 40.0), (5.0, 10.0), (8.0, 12.0)])
    assert manual_avoid.load_ranges("a.mp4") == [(5.0, 12.0), (30.0, 40.0)]


def test_ranges_are_keyed_per_video():
    manual_avoid.save_ranges("a.mp4", [(1.0, 2.0)])
    manual_avoid.save_ranges("b.mp4", [(9.0, 9.5)])
    assert manual_avoid.load_ranges("a.mp4") == [(1.0, 2.0)]
    assert manual_avoid.load_ranges("b.mp4") == [(9.0, 9.5)]


def test_unknown_video_is_empty():
    assert manual_avoid.load_ranges("never-seen.mp4") == []


def test_saving_empty_removes_the_key(_tmp_store):
    manual_avoid.save_ranges("a.mp4", [(1.0, 2.0)])
    manual_avoid.save_ranges("a.mp4", [])
    assert manual_avoid.load_ranges("a.mp4") == []
    # The key is dropped rather than left as [], so the file doesn't accumulate
    # dead entries as users clear ranges.
    assert json.loads(_tmp_store.read_text()) == {}


def test_corrupt_store_is_treated_as_empty(_tmp_store):
    _tmp_store.parent.mkdir(parents=True, exist_ok=True)
    _tmp_store.write_text("{not json")
    # A broken file must never block a run.
    assert manual_avoid.load_ranges("a.mp4") == []


def test_save_creates_parent_directory(_tmp_store):
    assert not _tmp_store.parent.exists()
    manual_avoid.save_ranges("a.mp4", [(1.0, 2.0)])
    assert _tmp_store.exists()
