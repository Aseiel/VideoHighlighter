"""Tests for re-running composition rules over detections that may be cached.

The bug this module was written for is invisible by construction: a user edits a
rule, re-runs, and the report is unchanged — because the detections came from
cache and the engine, which lived inside the detection branch, never executed.
Nothing failed and nothing was logged. `test_runs_on_a_cached_pass` is the guard,
and it is the only test here that would have caught the original.

The rest are about idempotence, which is what running on every pass makes
necessary. Cached detections already carry the previous pass's event names, and
the cached boxes already carry the previous pass's event boxes. Three distinct
ways that goes wrong if the strip is missed:

* `test_running_twice_changes_nothing` — events double up in the detections;
* `test_a_deleted_rule_does_not_outlive_its_file` — an event nobody has a rule
  for any more stays in the report for ever;
* `test_event_boxes_are_not_fed_back_in_as_detections` — the worst, because the
  result is still plausible: a rule matching against its own previous output.

Class names here are workshop objects. Neither the engine nor this module holds
a vocabulary, which is why any will do.
"""
from __future__ import annotations

import yaml

from modules.compose_events import apply_rules, rules_fingerprint, strip_events

RULES = {"events": [{
    "name": "clamped_board", "label": "Clamped Board",
    "rules": [{"source": "clamp", "region": "board",
               "min_count": 2, "max_count": 999}],
    "window_secs": 2.0, "persist_secs": 0.5}]}


def _rules_file(tmp_path, spec=None, name="composition_rules.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec or RULES), encoding="utf-8")
    return str(path)


def _boxes():
    """Frames with two clamps inside one board — enough to fire the rule."""
    board = [0.2, 0.2, 0.6, 0.6]
    left, right = [0.3, 0.3, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]
    return [{"timestamp": float(t),
             "objects": ["board", "clamp", "clamp"],
             "bboxes": [board, left, right],
             "confidences": [0.9, 0.8, 0.7]}
            for t in (10.0, 10.5, 11.0, 11.5, 12.0)]


def _detections():
    return {10: ["board", "clamp"], 11: ["board", "clamp"], 12: ["board", "clamp"]}


class TestRunsEverywhere:
    def test_the_rule_fires_and_lands_in_the_detections(self, tmp_path):
        dets, boxes, names, hits = apply_rules(
            _detections(), _boxes(), rules_path=_rules_file(tmp_path),
            log_fn=lambda _m: None)
        assert names == ["clamped_board"] and hits > 0
        assert any("clamped_board" in v for v in dets.values())
        assert any("clamped_board" in (f.get("objects") or []) for f in boxes)

    def test_runs_on_a_cached_pass(self, tmp_path):
        # The original bug. A cached pass arrives with detections and boxes
        # already in hand and no detector about to run; the rules must still be
        # applied, or editing one changes nothing and says nothing.
        dets, _boxes_out, names, hits = apply_rules(
            _detections(), _boxes(), rules_path=_rules_file(tmp_path),
            previous_names=[], log_fn=lambda _m: None)
        assert hits > 0 and names == ["clamped_board"]
        assert any("clamped_board" in v for v in dets.values())

    def test_a_new_rule_fires_over_boxes_from_an_older_pass(self, tmp_path):
        first = apply_rules(_detections(), _boxes(),
                            rules_path=_rules_file(tmp_path, {"events": []}),
                            log_fn=lambda _m: None)
        # The user adds a rule and re-runs without re-detecting anything.
        dets, _b, names, hits = apply_rules(
            first[0], first[1], rules_path=_rules_file(tmp_path),
            previous_names=first[2], log_fn=lambda _m: None)
        assert names == ["clamped_board"] and hits > 0
        assert any("clamped_board" in v for v in dets.values())


class TestIdempotence:
    def test_running_twice_changes_nothing(self, tmp_path):
        path = _rules_file(tmp_path)
        first = apply_rules(_detections(), _boxes(), rules_path=path,
                            log_fn=lambda _m: None)
        second = apply_rules(first[0], first[1], rules_path=path,
                             previous_names=first[2], log_fn=lambda _m: None)
        assert second[0] == first[0]
        assert len(second[1]) == len(first[1])

    def test_a_deleted_rule_does_not_outlive_its_file(self, tmp_path):
        first = apply_rules(_detections(), _boxes(),
                            rules_path=_rules_file(tmp_path),
                            log_fn=lambda _m: None)
        assert any("clamped_board" in v for v in first[0].values())
        # Same video, same boxes, rule gone.
        dets, boxes, names, _h = apply_rules(
            first[0], first[1],
            rules_path=_rules_file(tmp_path, {"events": []}),
            previous_names=first[2], log_fn=lambda _m: None)
        assert names == []
        assert not any("clamped_board" in v for v in dets.values())
        assert not any("clamped_board" in (f.get("objects") or []) for f in boxes)

    def test_event_boxes_are_not_fed_back_in_as_detections(self, tmp_path):
        path = _rules_file(tmp_path)
        first = apply_rules(_detections(), _boxes(), rules_path=path,
                            log_fn=lambda _m: None)
        composed = [f for f in first[1]
                    if "clamped_board" in (f.get("objects") or [])]
        assert composed, "the first pass must have produced event boxes"
        second = apply_rules(first[0], first[1], rules_path=path,
                             previous_names=first[2], log_fn=lambda _m: None)
        # Exactly as many event frames as the first pass — not more, which is
        # what a rule matching against its own output would produce.
        assert len([f for f in second[1]
                    if "clamped_board" in (f.get("objects") or [])]) \
            == len(composed)

    def test_detector_frames_are_never_dropped(self, tmp_path):
        boxes = _boxes()
        out = apply_rules(_detections(), boxes, rules_path=_rules_file(tmp_path),
                          log_fn=lambda _m: None)
        for frame in boxes:
            assert frame in out[1]


class TestDegradedCases:
    def test_no_rules_file_strips_and_adds_nothing(self, tmp_path):
        first = apply_rules(_detections(), _boxes(),
                            rules_path=_rules_file(tmp_path),
                            log_fn=lambda _m: None)
        dets, _b, names, hits = apply_rules(
            first[0], first[1], rules_path=str(tmp_path / "gone.yaml"),
            previous_names=first[2], log_fn=lambda _m: None)
        assert names == [] and hits == 0
        assert not any("clamped_board" in v for v in dets.values())

    def test_a_broken_rules_file_costs_the_events_not_the_detections(self, tmp_path):
        path = tmp_path / "broken.yaml"
        path.write_text("events: [ this is not: valid: yaml", encoding="utf-8")
        dets, boxes, names, _h = apply_rules(
            _detections(), _boxes(), rules_path=str(path),
            log_fn=lambda _m: None)
        assert names == []
        assert dets and boxes           # the run keeps everything it detected

    def test_no_boxes_is_not_an_error(self, tmp_path):
        dets, boxes, _n, hits = apply_rules(
            {}, [], rules_path=_rules_file(tmp_path), log_fn=lambda _m: None)
        assert dets == {} and boxes == [] and hits == 0

    def test_the_inputs_are_not_mutated(self, tmp_path):
        dets, boxes = _detections(), _boxes()
        before_dets, before_len = dict(dets), len(boxes)
        apply_rules(dets, boxes, rules_path=_rules_file(tmp_path),
                    log_fn=lambda _m: None)
        assert dets == before_dets and len(boxes) == before_len


class TestStripAndFingerprint:
    def test_strip_keeps_a_detector_frame_at_the_same_second(self):
        boxes = [{"timestamp": 10.0, "objects": ["clamp"], "bboxes": [[0, 0, 1, 1]]},
                 {"timestamp": 10.0, "objects": ["clamped_board"], "bboxes": []}]
        _dets, kept = strip_events({}, boxes, ["clamped_board"])
        assert [f["objects"] for f in kept] == [["clamp"]]

    def test_strip_leaves_a_second_with_nothing_left_out_entirely(self):
        dets, _b = strip_events({5: ["clamped_board"], 6: ["clamp"]}, [],
                                ["clamped_board"])
        assert dets == {6: ["clamp"]}

    def test_the_fingerprint_changes_with_the_file(self, tmp_path):
        a = rules_fingerprint(_rules_file(tmp_path, RULES, "a.yaml"))
        b = rules_fingerprint(_rules_file(tmp_path, {"events": []}, "b.yaml"))
        assert a and b and a != b

    def test_a_missing_file_has_no_fingerprint(self, tmp_path):
        assert rules_fingerprint(str(tmp_path / "nope.yaml")) == ""
