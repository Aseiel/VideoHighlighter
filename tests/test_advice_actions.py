"""Advice as actions: which findings carry a settings change, and how several
chosen at once merge into one re-run (`modules.report.advice_actions`).

Reports here are plain dicts with only the fields the rules read, so the tests
need no video and no report builder. Shared by both editions, byte-identical.
"""
from __future__ import annotations

from modules.report.advice_actions import (
    ANALYSE, RESCORE, actionable, merge, plan_rerun, recorded_settings)
from modules.report.highlight_advice import (
    SUGGESTED_COVERAGE, SUGGESTED_WEIGHT, attach_advice, diagnose)

WEIGHTS = {"scene_points": 0, "motion_event_points": 0, "motion_peak_points": 0,
           "audio_peak_points": 0, "keyword_points": 0, "object_points": 0,
           "action_points": 0, "face_expression_points": 0}


def _report(settings=None, segments=None, totals=None, duration=600, **extra):
    s = {"duration_mode": "MAX", "target_duration": 90, "clip_time": 8,
         "coverage": 0.0, "multi_signal_boost": 1.2, "min_signals_for_boost": 2,
         "face_expression_labels": [], **WEIGHTS}
    s.update(settings or {})
    report = {
        "settings": s,
        "segments": segments if segments is not None else [
            {"start": 100, "end": 108, "score": 5, "breakdown": {"motion_peak": 5}},
            {"start": 120, "end": 128, "score": 4, "breakdown": {"motion_peak": 4}},
        ],
        "video": {"duration": duration},
        "totals": {"duration": 16},
    }
    if totals is not None:
        report["signal_totals"] = totals
    report.update(extra)
    return report


def _by_id(report, **kw):
    return {f.id: f for f in diagnose(report, **kw)}


# ── Which findings carry an action ───────────────────────────────────────


class TestActions:
    def test_unused_detector_offers_its_weight(self):
        rep = _report({"motion_peak_points": 5,
                       "detector_activity": {"motion_event": 80}})
        f = _by_id(rep)["unused_motion_event"]
        assert f.action["settings"] == {"motion_event_points": SUGGESTED_WEIGHT}
        assert "motion events" in f.action["label"]

    def test_unused_expressions_need_a_class_first(self):
        rep = _report({"motion_peak_points": 5,
                       "detector_activity": {"face": 80}})
        assert _by_id(rep)["unused_face"].action == {}

    def test_single_signal_suggests_the_busiest_switched_off_detector(self):
        rep = _report({"motion_peak_points": 5,
                       "detector_activity": {"scene": 3, "audio": 12}})
        f = _by_id(rep)["single_signal"]
        assert f.action["settings"] == {"audio_peak_points": SUGGESTED_WEIGHT}

    def test_single_signal_without_a_record_of_activity_offers_nothing(self):
        rep = _report({"motion_peak_points": 5})
        assert _by_id(rep)["single_signal"].action == {}

    def test_concentrated_cut_moves_the_coverage_slider(self):
        rep = _report({"motion_peak_points": 5}, duration=3600)
        f = _by_id(rep)["concentrated"]
        assert f.action["settings"] == {"coverage": SUGGESTED_COVERAGE}

    def test_boost_bar_lowers_only_while_it_still_needs_agreement(self):
        def seg(count):
            return {"start": 100, "end": 108, "score": 5,
                    "breakdown": {"motion_peak": 5},
                    "boost": {"applied": False, "signal_count": count}}
        rep = _report({"motion_peak_points": 5, "min_signals_for_boost": 3},
                      segments=[seg(2), seg(1)])
        assert _by_id(rep)["boost_never_fired"].action["settings"] == {
            "min_signals_for_boost": 2}
        rep = _report({"motion_peak_points": 5, "min_signals_for_boost": 2},
                      segments=[seg(1), seg(1)])
        assert _by_id(rep)["boost_never_fired"].action == {}

    def test_near_miss_signals_get_their_weights_raised(self):
        rep = _report({"motion_peak_points": 5, "audio_peak_points": 4},
                      near_misses=[{"signals_present": ["audio"]}])
        f = _by_id(rep)["near_miss_gap"]
        assert f.action["settings"] == {"audio_peak_points": 6.0}

    def test_judgement_calls_carry_no_action(self):
        rep = _report({"motion_peak_points": 5, "face_expression_points": 3})
        assert _by_id(rep)["expressions_unselected"].action == {}

    def test_actions_reach_the_report(self):
        rep = attach_advice(_report({"motion_peak_points": 5,
                                     "detector_activity": {"motion_event": 80}}))
        stored = {f["id"]: f for f in rep["advice"]}
        assert stored["unused_motion_event"]["action"]["settings"]
        assert actionable(rep["advice"])
        assert all(f["action"]["settings"] for f in actionable(rep["advice"]))


# ── Merging and planning a re-run ────────────────────────────────────────


class TestMerge:
    def test_settings_from_several_actions_combine(self):
        change, conflicts = merge([
            {"settings": {"scene_points": 4}},
            {"settings": {"coverage": 0.6}},
        ])
        assert change == {"scene_points": 4, "coverage": 0.6}
        assert conflicts == []

    def test_the_stronger_change_wins_and_the_conflict_is_reported(self):
        change, conflicts = merge([
            {"settings": {"coverage": 0.6, "min_signals_for_boost": 3}},
            {"settings": {"coverage": 0.9, "min_signals_for_boost": 2}},
        ])
        assert change == {"coverage": 0.9, "min_signals_for_boost": 2}
        assert {c["setting"] for c in conflicts} == {
            "coverage", "min_signals_for_boost"}

    def test_the_same_change_twice_is_not_a_conflict(self):
        change, conflicts = merge([{"settings": {"coverage": 0.6}},
                                   {"settings": {"coverage": 0.6}}])
        assert change == {"coverage": 0.6} and conflicts == []


class TestPlanRerun:
    def test_starts_from_what_the_report_recorded(self):
        rep = _report({"motion_peak_points": 5, "target_duration": 120})
        base = recorded_settings(rep)
        assert base["motion_peak_points"] == 5
        assert base["max_duration"] == 120 and "exact_duration" not in base
        rep["settings"]["duration_mode"] = "EXACT"
        assert recorded_settings(rep)["exact_duration"] == 120

    def test_a_weight_on_an_open_detector_only_rescores(self):
        rep = _report({"motion_peak_points": 5})
        plan = plan_rerun(rep, [{"id": "a", "action": {
            "settings": {"motion_event_points": 4}}}])
        assert plan["cost"] == RESCORE and plan["analyses"] == []
        assert plan["settings"]["motion_event_points"] == 4
        assert plan["settings"]["motion_peak_points"] == 5
        assert plan["lines"] == ["motion_event_points: 0 → 4"]
        assert plan["chosen"] == ["a"]

    def test_switching_on_a_skipped_detector_means_analysing(self):
        rep = _report({"motion_peak_points": 5})
        plan = plan_rerun(rep, [{"id": "a", "action": {
            "settings": {"audio_peak_points": 4}}}])
        assert plan["cost"] == ANALYSE and plan["analyses"] == ["audio_peaks"]

    def test_end_to_end_from_diagnosis(self):
        rep = attach_advice(_report({"motion_peak_points": 5,
                                     "detector_activity": {"motion_event": 80}},
                                    duration=3600))
        chosen = actionable(rep["advice"])
        plan = plan_rerun(rep, chosen)
        assert plan["change"]["coverage"] == SUGGESTED_COVERAGE
        assert plan["change"]["motion_event_points"] == SUGGESTED_WEIGHT
        assert plan["cost"] == RESCORE
