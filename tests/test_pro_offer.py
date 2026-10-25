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


class TestAsked:
    """The Pro entry in the expression picker answers a click, so it is not
    held back the way an interruption is."""

    def test_a_click_is_answered_inside_the_cooldowns(self):
        pro_offer.mark_shown("face_category", NOW)
        assert pro_offer.for_face_category(NOW) is not None

    def test_a_click_is_answered_when_switched_off(self):
        pro_offer.set_enabled(False)
        assert pro_offer.for_face_category(NOW) is not None

    def test_but_not_in_a_pro_build(self, monkeypatch):
        monkeypatch.setattr(pro_offer, "__edition__", "Pro")
        assert pro_offer.for_face_category(NOW) is None

    def test_offered_moments_still_wait(self):
        pro_offer.mark_shown("vr_region", NOW)
        assert pro_offer.for_vr(NOW) is None


class TestUnknownObjects:
    KNOWN = ["bench_vice", "lathe", "person"]

    def test_names_only_what_the_detector_lacks(self):
        assert pro_offer.unknown_objects(
            ["lathe", " drill_press ", "", "drill_press"], self.KNOWN) == ["drill_press"]

    def test_the_match_is_exact_like_the_run(self):
        assert pro_offer.unknown_objects(["Lathe"], self.KNOWN) == ["Lathe"]

    def test_the_offer_names_them_and_escapes_them(self):
        offer = pro_offer.for_unknown_objects(["drill<b>"], NOW)
        assert offer.moment == "object_unknown"
        assert "drill&lt;b&gt;" in offer.text
        assert "Teach a model" in offer.text

    def test_nothing_unknown_says_nothing(self):
        assert pro_offer.for_unknown_objects([], NOW) is None


class TestLrv:
    def _touch(self, folder, *names):
        for name in names:
            (folder / name).write_bytes(b"")

    def test_gopro_preview_beside_the_master(self, tmp_path):
        self._touch(tmp_path, "GX010527.MP4", "GL010527.LRV", "GL010528.LRV")
        assert pro_offer.lrv_companion(str(tmp_path / "GX010527.MP4")) == "GL010527.LRV"

    def test_insta360_preview_beside_the_master(self, tmp_path):
        self._touch(tmp_path, "VID_20260901_101500_00_042.mp4",
                    "LRV_20260901_101500_11_042.lrv")
        assert (pro_offer.lrv_companion(str(tmp_path / "VID_20260901_101500_00_042.mp4"))
                == "LRV_20260901_101500_11_042.lrv")

    def test_no_preview_no_offer(self, tmp_path):
        self._touch(tmp_path, "GX010527.MP4", "holiday.mp4")
        assert pro_offer.lrv_companion(str(tmp_path / "GX010527.MP4")) is None
        assert pro_offer.lrv_companion(str(tmp_path / "holiday.mp4")) is None
        assert pro_offer.for_lrv(str(tmp_path / "GX010527.MP4"), None, NOW) is None

    def test_an_lrv_added_on_its_own_is_its_own_preview(self, tmp_path):
        assert pro_offer.lrv_companion(str(tmp_path / "GL010527.LRV")) == "GL010527.LRV"

    def test_the_offer_names_both_files(self, tmp_path):
        offer = pro_offer.for_lrv(str(tmp_path / "GX010527.MP4"), "GL010527.LRV", NOW)
        assert "GX010527.MP4" in offer.text and "GL010527.LRV" in offer.text


class TestFooter:
    def test_links_to_the_pro_page(self):
        assert f'href="{pro_offer.PRO_URL}"' in pro_offer.footer_text()


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
