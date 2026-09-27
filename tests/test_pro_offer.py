"""Tests for when the free edition may mention Pro.

The failure that matters is nagging: an offer that shows up where the free
build did not actually stop, or keeps showing up after the user has seen it.
So what's pinned down is the decision -- which edges speak, the cooldowns, the
off switch, and that a Pro build never speaks at all.

Fixture classes are workshop objects, as elsewhere in the suite.
"""
from __future__ import annotations

import datetime as dt

import pytest

from modules.ui import pro_offer

NOW = dt.datetime(2026, 9, 27, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture(autouse=True)
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pro_offer, "state_path",
                        lambda: str(tmp_path / "pro_offer_state.json"))
    monkeypatch.setattr(pro_offer, "__edition__", "Free")
    return tmp_path


def _report(claims=1, faces=0):
    return {
        "vocabulary": {"classes": ["bench_vice", "lathe"], "events": []},
        "chapters": [{"number": 1, "method": "visual"}],
        "settings": {"detector_activity": {"face": faces, "action": 0}},
        "speech": {"words": 400},
        "unmeasured": ({"claims": [{"text": f"line {i}"} for i in range(claims)]}
                       if claims else {}),
    }


def _lacks(*missing):
    return lambda module: module not in missing


class TestMayOffer:
    def test_a_fresh_install_may(self):
        assert pro_offer.may_offer("rule_unbuildable", NOW)

    def test_a_pro_build_never_does(self, monkeypatch):
        monkeypatch.setattr(pro_offer, "__edition__", "Pro")
        assert not pro_offer.may_offer("rule_unbuildable", NOW)
        assert pro_offer.for_unbuildable_rule(NOW) is None

    def test_an_unknown_moment_never_does(self):
        assert not pro_offer.may_offer("startup", NOW)

    def test_switched_off_stays_off(self):
        pro_offer.set_enabled(False)
        assert not pro_offer.is_enabled()
        assert not pro_offer.may_offer("rule_unbuildable", NOW)
        assert not pro_offer.may_offer("report_unmeasured", NOW)

    def test_the_same_edge_waits_its_cooldown(self):
        pro_offer.mark_shown("rule_unbuildable", NOW)
        later = NOW + dt.timedelta(days=pro_offer.MOMENT_COOLDOWN_DAYS - 1)
        assert not pro_offer.may_offer("rule_unbuildable", later)
        later = NOW + dt.timedelta(days=pro_offer.MOMENT_COOLDOWN_DAYS)
        assert pro_offer.may_offer("rule_unbuildable", later)

    def test_any_offer_holds_back_the_others_for_a_while(self):
        pro_offer.mark_shown("rule_unbuildable", NOW)
        soon = NOW + dt.timedelta(days=pro_offer.GLOBAL_COOLDOWN_DAYS - 1)
        assert not pro_offer.may_offer("report_unmeasured", soon)
        later = NOW + dt.timedelta(days=pro_offer.GLOBAL_COOLDOWN_DAYS)
        assert pro_offer.may_offer("report_unmeasured", later)

    def test_a_corrupt_state_file_is_a_fresh_start(self, state_dir):
        (state_dir / "pro_offer_state.json").write_text("{not json")
        assert pro_offer.may_offer("rule_unbuildable", NOW)


class TestUnbuildableRule:
    def test_it_names_the_trial(self):
        offer = pro_offer.for_unbuildable_rule(NOW)
        assert offer.moment == "rule_unbuildable"
        assert f"{pro_offer.TRIAL_DAYS}-day trial" in offer.text


class TestReport:
    def test_speaks_when_only_a_missing_engine_could_measure_it(self):
        offer = pro_offer.for_report(_report(),
                                     installed=_lacks("llm.owl_detect"),
                                     now=NOW)
        assert offer is not None
        assert "open-vocabulary" in offer.text
        assert "1 thing that was said" in offer.text

    def test_silent_when_this_build_has_every_engine(self):
        assert pro_offer.for_report(_report(), installed=lambda _m: True,
                                    now=NOW) is None

    def test_silent_when_nothing_went_unmeasured(self):
        assert pro_offer.for_report(_report(claims=0),
                                    installed=_lacks("llm.owl_detect"),
                                    now=NOW) is None

    def test_a_route_this_run_could_not_use_anyway_is_not_named(self):
        # Face categories need a face scan; without one, lacking that engine
        # is not something Pro would fix for this report.
        missing = pro_offer.pro_only_routes(
            _report(faces=0), installed=_lacks("modules.vision.face_examples"))
        assert missing == []
        missing = pro_offer.pro_only_routes(
            _report(faces=900), installed=_lacks("modules.vision.face_examples"))
        assert [r.id for r in missing] == ["face_category"]

    def test_it_does_not_touch_the_report(self):
        report = _report(claims=0)
        report.pop("unmeasured")
        pro_offer.for_report(report, installed=_lacks("llm.owl_detect"), now=NOW)
        assert "unmeasured" not in report
