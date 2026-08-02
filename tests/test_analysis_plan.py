"""Tests for second-run behaviour — the bug class that keeps coming back.

Every case here is a *pair* of runs: one that writes a cache, one that reuses
it with different scoring settings. That framing is the point. The pipeline's
first run has been correct throughout; what broke, repeatedly and silently, was
the run after it, and no test that considered a single run in isolation could
have caught any of the three instances found so far.

`TestTheInvariant` is the durable half. The others pin down today's behaviour;
that one fails when someone adds a *new* detector gated by a scoring weight
without deciding what a cached run should do about it — which is exactly how
this bug arrived each time.
"""
from __future__ import annotations

import pytest

from modules.analysis_plan import (
    GATED_ARTIFACTS,
    SIGNATURE_GATED,
    describe,
    gate_is_open,
    is_empty,
    needs_backfill,
    plan,
)

MOTION_ON = {"scene_points": 5, "motion_event_points": 0, "motion_peak_points": 0}
MOTION_OFF = {"scene_points": 0, "motion_event_points": 0, "motion_peak_points": 0}
AUDIO_ON = {"audio_peak_points": 3}
AUDIO_OFF = {"audio_peak_points": 0}


class TestGate:
    def test_any_one_point_opens_the_shared_detector(self):
        """Scenes, events and peaks are one pass — wanting any means running it."""
        for key in ("scene_points", "motion_event_points", "motion_peak_points"):
            assert gate_is_open({key: 1}, "motion"), key

    def test_all_zero_is_closed(self):
        assert not gate_is_open(MOTION_OFF, "motion")

    def test_a_missing_setting_reads_as_zero(self):
        assert not gate_is_open({}, "motion")

    def test_an_unknown_artifact_is_closed(self):
        assert not gate_is_open({"anything": 5}, "not_an_artifact")


class TestIsEmpty:
    def test_all_empty_is_empty(self):
        assert is_empty([], [], [])

    def test_any_populated_proves_the_pass_ran(self):
        """A silent video is not the same as a detector that never ran."""
        assert not is_empty([], [1.0], [])

    def test_no_values_at_all_is_empty(self):
        assert is_empty()


class TestTheRerunBug:
    """The exact sequence that broke motion, then audio."""

    def test_points_raised_after_a_cache_written_with_them_off(self):
        # Run 1 had motion off, so the cache holds empty lists.
        cached = ([], [], [])
        # Run 2 raises scene points. Same inputs -> same cache key -> cache hit.
        assert needs_backfill("motion", MOTION_ON, using_cache=True,
                              values=cached) is True

    def test_the_same_for_audio(self):
        assert needs_backfill("audio_peaks", AUDIO_ON, using_cache=True,
                              values=([],)) is True

    def test_a_populated_cache_is_left_alone(self):
        """The cache doing its job must not be mistaken for the bug."""
        assert needs_backfill("motion", MOTION_ON, using_cache=True,
                              values=([(0.0, 5.0)], [], [])) is False

    def test_a_run_that_does_not_want_it_does_not_compute_it(self):
        assert needs_backfill("motion", MOTION_OFF, using_cache=True,
                              values=([], [], [])) is False

    def test_a_fresh_run_never_backfills(self):
        """A non-cached run computes what it wants through the normal path."""
        assert needs_backfill("motion", MOTION_ON, using_cache=False,
                              values=([], [], [])) is False

    def test_turning_points_back_off_does_not_recompute(self):
        """Lowering a weight is not a reason to run a detector."""
        assert needs_backfill("audio_peaks", AUDIO_OFF, using_cache=True,
                              values=([],)) is False


class TestPlan:
    def test_covers_every_gated_artifact(self):
        result = plan({}, using_cache=True, values={})
        assert set(result) == set(GATED_ARTIFACTS)

    def test_reports_each_artifact_independently(self):
        config = {**MOTION_ON, **AUDIO_OFF}
        result = plan(config, using_cache=True,
                      values={"motion": ([], [], []), "audio_peaks": ([],)})
        assert result == {"motion": True, "audio_peaks": False}

    def test_a_missing_value_is_treated_as_empty_not_an_error(self):
        result = plan(MOTION_ON, using_cache=True, values={})
        assert result["motion"] is True

    def test_a_fresh_run_plans_nothing(self):
        config = {**MOTION_ON, **AUDIO_ON}
        assert not any(plan(config, using_cache=False, values={}).values())


class TestTheInvariant:
    """The part that stops this arriving a fourth time."""

    def test_every_gating_setting_is_either_in_the_signature_or_declared(self):
        """A setting that gates an artifact must be handled somewhere.

        Either it is part of the cache signature — so changing it forces a full
        re-analysis and the artifact is rebuilt — or it is declared in
        GATED_ARTIFACTS so a cached run can backfill. A gating setting in
        neither place is the bug, and this is what catches it.
        """
        from modules.video_cache import build_analysis_cache_params

        signature = set(build_analysis_cache_params(
            gui_config={}, config={}, sample_rate=1, video_duration=60.0))

        for artifact, settings in SIGNATURE_GATED.items():
            for name in settings:
                assert name in signature, (
                    f"{artifact} is gated by {name!r}, which is listed as "
                    "signature-gated but is not in the cache signature — a "
                    "cached run will silently reuse an empty artifact")

        for artifact, settings in GATED_ARTIFACTS.items():
            for name in settings:
                assert name not in signature, (
                    f"{artifact} is gated by {name!r}, which IS in the cache "
                    "signature — the backfill is dead code, and the declaration "
                    "is misleading. Move it to SIGNATURE_GATED")

    def test_the_two_registries_do_not_overlap(self):
        gated = {n for names in GATED_ARTIFACTS.values() for n in names}
        signed = {n for names in SIGNATURE_GATED.values() for n in names}
        assert not (gated & signed)

    def test_the_scoring_points_that_gate_detectors_are_all_declared(self):
        """The known offenders, named. Regression cover for the found bugs."""
        declared = {n for names in GATED_ARTIFACTS.values() for n in names}
        for name in ("scene_points", "motion_event_points",
                     "motion_peak_points", "audio_peak_points"):
            assert name in declared


class TestDescribe:
    def test_reads_as_a_sentence(self):
        assert "audio peaks" in describe("audio_peaks")
        assert describe("motion").startswith("🔁")
