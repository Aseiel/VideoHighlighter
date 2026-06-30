"""
Characterization tests for `auto_segments.snap_to_scene`.

`snap_to_scene` is the scene-boundary-aware alignment helper. Given a
timestamp and a list of (start, end) scene spans, it returns the scene the
timestamp lies in (within `max_snap` tolerance), or `None` if no scene is
close enough.

The behaviour pinned here is the **first-match-wins** loop and the
inclusive tolerance window.
"""

from __future__ import annotations

import pytest

from modules.auto_segments import snap_to_scene


class TestInsideScene:
    def test_inside_scene_returns_its_span(self):
        scenes = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        assert snap_to_scene(5.0, scenes) == (0.0, 10.0)

    def test_inside_later_scene_returns_its_span(self):
        # NOTE: scenes must be spaced wider than 2*max_snap apart for a
        # timestamp inside the *second* scene to actually snap to it. With
        # default max_snap=5.0 and back-to-back scenes (0-10, 10-20), t=15
        # is within 5s of the FIRST scene's end and the first-match-wins
        # loop returns (0.0, 10.0). Pin that here by using spaced scenes.
        scenes = [(0.0, 10.0), (50.0, 60.0), (100.0, 110.0)]
        assert snap_to_scene(55.0, scenes) == (50.0, 60.0)


class TestTolerance:
    def test_within_default_snap_before_scene_start_returns_scene(self):
        # Default max_snap=5.0; timestamp 8.0 is within 5.0 of scene start
        # 10.0, so it should snap into (10.0, 20.0).
        scenes = [(10.0, 20.0)]
        assert snap_to_scene(8.0, scenes) == (10.0, 20.0)

    def test_within_default_snap_after_scene_end_returns_scene(self):
        # Default max_snap=5.0; timestamp 24.0 is within 5.0 after scene end
        # 20.0, so it should snap.
        scenes = [(10.0, 20.0)]
        assert snap_to_scene(24.0, scenes) == (10.0, 20.0)

    def test_beyond_tolerance_returns_none(self):
        # 30.0 is 10s past scene end 20.0 -> outside max_snap=5.0.
        scenes = [(10.0, 20.0)]
        assert snap_to_scene(30.0, scenes) is None

    def test_custom_max_snap_widens_window(self):
        scenes = [(10.0, 20.0)]
        assert snap_to_scene(30.0, scenes, max_snap=15.0) == (10.0, 20.0)


class TestEdgeCases:
    def test_no_scenes_returns_none(self):
        assert snap_to_scene(5.0, []) is None

    def test_first_match_wins(self):
        # If overlapping scenes are passed (degenerate but possible from
        # detector flicker), the first matching one is returned.
        scenes = [(0.0, 100.0), (10.0, 20.0)]
        # Both contain t=15.0 but the loop returns the first.
        assert snap_to_scene(15.0, scenes) == (0.0, 100.0)

    def test_timestamp_at_exact_scene_start(self):
        scenes = [(10.0, 20.0)]
        assert snap_to_scene(10.0, scenes) == (10.0, 20.0)

    def test_timestamp_at_exact_scene_end(self):
        scenes = [(10.0, 20.0)]
        assert snap_to_scene(20.0, scenes) == (10.0, 20.0)
