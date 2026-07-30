"""Tests for `modules.highlight_advice` — why a highlight disappointed.

Findings are computed from the report record and nothing else, so these run
without a video, a model, or the app. The point of each test is that a finding
fires when it should and, just as importantly, stays quiet when it should not:
an advisor that warns about everything is ignored as fast as one that is wrong.
"""

from __future__ import annotations

import numpy as np

from modules.highlight_advice import attach_advice, diagnose
from modules.highlight_report import build_report


def _report(n=600, segments=None, settings=None, objects=None, **kw):
    keys = ("scene", "motion_event", "motion_peak", "audio",
            "keyword", "object", "action", "beginning", "ending")
    sig = {k: np.zeros(n) for k in keys}
    for key, points in (kw.pop("signals", {}) or {}).items():
        for sec, value in points.items():
            sig[key][sec] = value
    base = {"clip_time": 10, "duration_mode": "MAX", "object_points": 5}
    base.update(settings or {})
    return build_report(
        video_path="a.mp4", video_duration=n, score=sum(sig.values()),
        signals=sig, segments=segments or [(95, 105)],
        object_detections=objects, settings=base, **kw)


def _ids(findings):
    return {f.id for f in findings}


class TestSingleSignal:
    def test_fires_when_only_one_signal_scored(self):
        rep = _report(signals={"object": {100: 10.0}})
        findings = diagnose(rep)
        assert "single_signal" in _ids(findings)

    def test_names_the_signals_that_are_switched_off(self):
        rep = _report(signals={"object": {100: 10.0}},
                      settings={"audio_peak_points": 0, "scene_points": 0})
        finding = next(f for f in diagnose(rep) if f.id == "single_signal")
        assert "audio peaks" in finding.detail
        assert finding.severity == "high"

    def test_quiet_when_several_signals_contributed(self):
        rep = _report(signals={"object": {100: 10.0}, "audio": {100: 4.0}})
        assert "single_signal" not in _ids(diagnose(rep))

    def test_position_bonuses_do_not_count_as_a_second_signal(self):
        rep = _report(signals={"object": {100: 10.0}, "beginning": {100: 2.0}})
        assert "single_signal" in _ids(diagnose(rep))


class TestSilentDetector:
    def test_fires_when_a_weighted_signal_never_scored(self):
        rep = _report(signals={"object": {100: 10.0}},
                      settings={"audio_peak_points": 4})
        assert "silent_audio" in _ids(diagnose(rep))

    def test_quiet_when_the_signal_is_weighted_zero(self):
        rep = _report(signals={"object": {100: 10.0}},
                      settings={"audio_peak_points": 0})
        assert "silent_audio" not in _ids(diagnose(rep))

    def test_quiet_when_the_signal_did_score(self):
        rep = _report(signals={"object": {100: 10.0}, "audio": {100: 4.0}},
                      settings={"audio_peak_points": 4})
        assert "silent_audio" not in _ids(diagnose(rep))


class TestFlatScore:
    def test_fires_when_every_clip_scored_the_same(self):
        rep = _report(signals={"object": {100: 5.0, 300: 5.0, 500: 5.0}},
                      segments=[(95, 105), (295, 305), (495, 505)])
        assert "flat_score" in _ids(diagnose(rep))

    def test_quiet_when_scores_differ(self):
        rep = _report(signals={"object": {100: 9.0, 300: 5.0, 500: 2.0}},
                      segments=[(95, 105), (295, 305), (495, 505)])
        assert "flat_score" not in _ids(diagnose(rep))

    def test_quiet_for_a_cut_too_short_to_judge(self):
        rep = _report(signals={"object": {100: 5.0, 300: 5.0}},
                      segments=[(95, 105), (295, 305)])
        assert "flat_score" not in _ids(diagnose(rep))


class TestConcentrated:
    def test_fires_when_every_clip_sits_in_one_stretch(self):
        rep = _report(signals={"object": {100: 9.0, 130: 8.0, 160: 7.0}},
                      segments=[(95, 105), (125, 135), (155, 165)])
        assert "concentrated" in _ids(diagnose(rep))

    def test_quiet_when_the_cut_already_spans_the_video(self):
        rep = _report(signals={"object": {50: 9.0, 300: 8.0, 550: 7.0}},
                      segments=[(45, 55), (295, 305), (545, 555)])
        assert "concentrated" not in _ids(diagnose(rep))

    def test_quiet_when_coverage_is_already_raised(self):
        """Do not tell someone to use a control they are already using."""
        rep = _report(signals={"object": {100: 9.0, 130: 8.0}},
                      segments=[(95, 105), (125, 135)],
                      settings={"coverage": 0.8})
        assert "concentrated" not in _ids(diagnose(rep))


class TestBoost:
    def test_fires_when_the_boost_is_configured_but_never_applied(self):
        rep = _report(signals={"object": {100: 10.0}},
                      settings={"multi_signal_boost": 1.2,
                                "min_signals_for_boost": 2})
        assert "boost_never_fired" in _ids(diagnose(rep))

    def test_quiet_when_the_boost_is_switched_off(self):
        rep = _report(signals={"object": {100: 10.0}},
                      settings={"multi_signal_boost": 1.0,
                                "min_signals_for_boost": 2})
        assert "boost_never_fired" not in _ids(diagnose(rep))


class TestNearMissGap:
    def test_fires_when_a_signal_only_appears_in_moments_not_taken(self):
        rep = _report(signals={"object": {100: 10.0}, "audio": {400: 6.0}},
                      segments=[(95, 105)])
        finding = next(f for f in diagnose(rep) if f.id == "near_miss_gap")
        assert "audio peaks" in finding.detail

    def test_quiet_when_the_same_signals_appear_in_both(self):
        rep = _report(signals={"object": {100: 10.0, 400: 6.0}},
                      segments=[(95, 105)])
        assert "near_miss_gap" not in _ids(diagnose(rep))


class TestDominantTag:
    def test_fires_when_one_tag_is_in_nearly_every_clip(self):
        rep = _report(
            signals={"object": {100: 9.0, 300: 8.0, 500: 7.0}},
            segments=[(95, 105), (295, 305), (495, 505)],
            objects={100: ["thing"], 300: ["thing"], 500: ["thing"]})
        finding = next(f for f in diagnose(rep) if f.id == "dominant_tag")
        assert "thing" in finding.detail

    def test_quiet_when_clips_differ(self):
        rep = _report(
            signals={"object": {100: 9.0, 300: 8.0, 500: 7.0}},
            segments=[(95, 105), (295, 305), (495, 505)],
            objects={100: ["a"], 300: ["b"], 500: ["c"]})
        assert "dominant_tag" not in _ids(diagnose(rep))


class TestRejected:
    def _rep(self):
        return _report(
            signals={"object": {100: 9.0, 300: 8.0, 500: 7.0}},
            segments=[(95, 105), (295, 305), (495, 505)],
            objects={100: ["unwanted", "a"], 300: ["unwanted", "b"],
                     500: ["c"]})

    def test_fires_on_what_the_rejected_clips_share(self):
        findings = diagnose(self._rep(), rejected=[(95, 105), (295, 305)])
        finding = next(f for f in findings if f.id == "rejected_share_a_tag")
        assert "unwanted" in finding.detail
        assert finding.severity == "high"

    def test_quiet_without_any_feedback(self):
        assert "rejected_share_a_tag" not in _ids(diagnose(self._rep()))

    def test_quiet_when_only_one_clip_was_rejected(self):
        findings = diagnose(self._rep(), rejected=[(95, 105)])
        assert "rejected_share_a_tag" not in _ids(findings)


class TestShortOfTarget:
    def test_fires_when_the_cut_is_well_under_the_budget(self):
        rep = _report(signals={"object": {100: 10.0}},
                      settings={"target_duration": 300})
        assert "short_of_target" in _ids(diagnose(rep))

    def test_quiet_when_the_budget_was_essentially_met(self):
        rep = _report(signals={"object": {100: 10.0}},
                      settings={"target_duration": 10})
        assert "short_of_target" not in _ids(diagnose(rep))

    def test_quiet_when_no_target_was_recorded(self):
        """Older reports have no target_duration; that is not a finding."""
        rep = _report(signals={"object": {100: 10.0}})
        assert "short_of_target" not in _ids(diagnose(rep))


class TestOrderingAndShape:
    def test_the_most_serious_finding_comes_first(self):
        rep = _report(signals={"object": {100: 10.0}},
                      settings={"multi_signal_boost": 1.2,
                                "min_signals_for_boost": 2})
        findings = diagnose(rep)
        assert findings[0].severity == "high"

    def test_every_finding_carries_its_evidence(self):
        rep = _report(signals={"object": {100: 10.0}})
        for finding in diagnose(rep):
            assert finding.evidence, f"{finding.id} has no evidence"
            assert finding.remedy and finding.topic

    def test_a_healthy_run_produces_no_high_severity_findings(self):
        rep = _report(
            signals={"object": {50: 9.0, 300: 6.0, 550: 3.0},
                     "audio": {50: 4.0, 300: 2.0}},
            segments=[(45, 55), (295, 305), (545, 555)],
            objects={50: ["a"], 300: ["b"], 550: ["c"]},
            settings={"audio_peak_points": 4, "target_duration": 30})
        assert not [f for f in diagnose(rep) if f.severity == "high"]

    def test_attach_advice_puts_findings_in_the_record(self):
        rep = attach_advice(_report(signals={"object": {100: 10.0}}))
        assert rep["advice"] and rep["advice"][0]["id"]
        assert isinstance(rep["advice"][0], dict), "must be JSON-serialisable"


class TestOlderReports:
    """The runs someone is unhappy with are often the ones already on disk."""

    def _schema1(self):
        """A record as written before signal totals were stored with it."""
        rep = _report(signals={"object": {100: 10.0}})
        rep.pop("signal_totals")
        return rep

    def test_totals_are_recovered_from_the_clips(self):
        from modules.highlight_advice import signal_totals
        assert signal_totals(self._schema1())["object"] == 10.0

    def test_findings_still_come_out(self):
        assert "single_signal" in _ids(diagnose(self._schema1()))

    def test_recorded_totals_are_preferred_when_present(self):
        from modules.highlight_advice import signal_totals
        rep = _report(signals={"object": {100: 10.0}})
        rep["signal_totals"] = {"object": 999.0}
        assert signal_totals(rep)["object"] == 999.0
