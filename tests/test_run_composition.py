"""
Tests for `modules.analysis_ondemand.run_composition` — applying composition
rules to boxes already in the cache, without re-running detection.

The bug this addresses: in `pipeline.py` the engine only runs *inside* a fresh
object-detection pass, so editing a single rule meant re-detecting an entire
video to see the effect. The engine itself is pure, so it never needed the
detector at all.
"""

from __future__ import annotations

import json

import pytest

from modules import analysis_ondemand as ao
from video_ai_editor.composition_engine import CompositionEngine


RULES = """
events:
  - name: held_object
    label: Held object
    window_secs: 0.75
    persist_secs: 0.5
    rules:
      - {source: handle, region: tool, min_count: 1}
"""


def _frames(n=8, together=True):
    """Cache-shaped per-frame boxes; `handle` inside `tool` when together."""
    out = []
    for i in range(n):
        ts = i * 0.5
        handle = [0.30, 0.30, 0.05, 0.05] if together else [0.90, 0.90, 0.05, 0.05]
        out.append({
            "timestamp": ts,
            "objects": ["tool", "handle"],
            "bboxes": [[0.20, 0.20, 0.40, 0.40], handle],
            "confidences": [0.9, 0.8],
        })
    return out


@pytest.fixture
def rules_file(tmp_path, monkeypatch):
    path = tmp_path / "composition_rules.yaml"
    path.write_text(RULES, encoding="utf-8")
    monkeypatch.setattr("modules.app_paths.composition_rules_path",
                        lambda: str(path))
    return path


@pytest.fixture
def cached(monkeypatch):
    """Install a fake on-disk cache and capture what would be written."""
    state = {"cache": {}, "written": []}
    monkeypatch.setattr(ao, "read_cache", lambda video_path: state["cache"])
    return state


class TestEventNames:
    def test_engine_reports_the_names_it_can_emit(self, rules_file):
        assert CompositionEngine(str(rules_file)).event_names == ["held_object"]

    def test_empty_rules_file_reports_nothing(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("events: []\n", encoding="utf-8")
        assert CompositionEngine(str(p)).event_names == []


class TestRunComposition:
    def test_events_are_added_to_the_cached_seconds(self, rules_file, cached):
        cached["cache"] = {
            "object_bboxes": _frames(),
            "objects": [{"timestamp": 0, "objects": ["tool", "handle"], "count": 2}],
        }
        patch = ao.run_composition("v.mp4", log=lambda *a: None)

        secs = {e["timestamp"]: e["objects"] for e in patch["objects"]}
        assert "held_object" in secs[0]
        # real detections are preserved alongside the derived event
        assert "tool" in secs[0] and "handle" in secs[0]
        assert secs[0] == sorted(secs[0]), "objects should be sorted"

    def test_counts_are_rebuilt(self, rules_file, cached):
        cached["cache"] = {
            "object_bboxes": _frames(),
            "objects": [{"timestamp": 0, "objects": ["tool"], "count": 1}],
        }
        patch = ao.run_composition("v.mp4", log=lambda *a: None)
        entry = patch["objects"][0]
        assert entry["count"] == len(entry["objects"])

    def test_running_twice_is_idempotent(self, rules_file, cached):
        """The whole point: re-running after a rule edit must converge, not
        accumulate a second copy of every event."""
        cached["cache"] = {
            "object_bboxes": _frames(),
            "objects": [{"timestamp": 0, "objects": ["tool"], "count": 1}],
        }
        first = ao.run_composition("v.mp4", log=lambda *a: None)

        cached["cache"] = {"object_bboxes": _frames(), "objects": first["objects"]}
        second = ao.run_composition("v.mp4", log=lambda *a: None)

        assert first["objects"] == second["objects"]
        for entry in second["objects"]:
            assert entry["objects"].count("held_object") <= 1

    def test_a_rule_that_no_longer_matches_removes_its_events(self, rules_file, cached):
        """Deleting or breaking a rule must clear its previous output, since the
        strip list comes from the rules file rather than from the old result."""
        cached["cache"] = {
            "object_bboxes": _frames(),
            "objects": [{"timestamp": 0, "objects": ["tool"], "count": 1}],
        }
        first = ao.run_composition("v.mp4", log=lambda *a: None)
        assert any("held_object" in e["objects"] for e in first["objects"])

        # Same rules, but the handle is now nowhere near the tool.
        cached["cache"] = {"object_bboxes": _frames(together=False),
                           "objects": first["objects"]}
        second = ao.run_composition("v.mp4", log=lambda *a: None)
        assert not any("held_object" in e["objects"] for e in second["objects"])

    def test_composed_boxes_are_not_written_back(self, rules_file, cached):
        """object_bboxes is the detector's record. Folding derived entries into
        it would make them indistinguishable from detections on the next run."""
        cached["cache"] = {"object_bboxes": _frames(), "objects": []}
        patch = ao.run_composition("v.mp4", log=lambda *a: None)
        assert "object_bboxes" not in patch


class TestFailureMessages:
    def test_no_rules_file(self, monkeypatch, cached):
        monkeypatch.setattr("modules.app_paths.composition_rules_path", lambda: None)
        with pytest.raises(RuntimeError, match="composition_rules.yaml"):
            ao.run_composition("v.mp4", log=lambda *a: None)

    def test_no_cache_at_all(self, rules_file, cached):
        cached["cache"] = {}
        with pytest.raises(RuntimeError, match="analysis cache"):
            ao.run_composition("v.mp4", log=lambda *a: None)

    def test_cache_without_boxes_points_at_object_detection(self, rules_file, cached):
        cached["cache"] = {"objects": [], "motion": []}
        with pytest.raises(RuntimeError, match="object detection first"):
            ao.run_composition("v.mp4", log=lambda *a: None)

    def test_rules_file_with_no_events(self, tmp_path, monkeypatch, cached):
        p = tmp_path / "empty.yaml"
        p.write_text("events: []\n", encoding="utf-8")
        monkeypatch.setattr("modules.app_paths.composition_rules_path", lambda: str(p))
        cached["cache"] = {"object_bboxes": _frames()}
        with pytest.raises(RuntimeError, match="No events defined"):
            ao.run_composition("v.mp4", log=lambda *a: None)


class TestReadCache:
    def test_missing_video_returns_empty(self, monkeypatch):
        monkeypatch.setattr(ao, "_cache_files", lambda vp: ("", []))
        assert ao.read_cache("nope.mp4") == {}

    def test_newest_file_wins(self, tmp_path, monkeypatch):
        import os
        import time

        old = tmp_path / "a.cache.json"
        new = tmp_path / "b.cache.json"
        old.write_text(json.dumps({"which": "old"}), encoding="utf-8")
        time.sleep(0.01)
        new.write_text(json.dumps({"which": "new"}), encoding="utf-8")
        os.utime(old, (1, 1))            # force the mtime ordering

        monkeypatch.setattr(ao, "_cache_files", lambda vp: ("h", [old, new]))
        assert ao.read_cache("v.mp4")["which"] == "new"

    def test_unreadable_file_is_skipped(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.cache.json"
        good = tmp_path / "good.cache.json"
        bad.write_text("{not json", encoding="utf-8")
        good.write_text(json.dumps({"ok": True}), encoding="utf-8")

        monkeypatch.setattr(ao, "_cache_files", lambda vp: ("h", [bad, good]))
        assert ao.read_cache("v.mp4") == {"ok": True}


class TestEventConfidence:
    """An event is only as sure as the weakest detection it needed.

    This used to emit a flat 1.0 into the same `confidences` array every
    detector writes its certainty into, so anything reading that field took a
    rule match to mean the detector was certain — and a rule firing on two
    barely-there detections was indistinguishable from one firing on two solid
    ones.
    """

    def _engine(self, tmp_path):
        rules = tmp_path / "rules.yaml"
        rules.write_text(RULES, encoding="utf-8")
        return CompositionEngine(str(rules))

    def _frames(self, handle_conf, tool_conf, n=6):
        return [{"timestamp": float(t),
                 "objects": ["handle", "tool"],
                 "bboxes": [[0.4, 0.4, 0.1, 0.1], [0.3, 0.3, 0.4, 0.4]],
                 "confidences": [handle_conf, tool_conf]}
                for t in range(n)]

    def _confidences(self, engine, frames):
        _events, overlay = engine.run(frames)
        return [c for entry in overlay for c in entry["confidences"]]

    def test_the_weakest_detection_sets_the_event_confidence(self, tmp_path):
        confs = self._confidences(self._engine(tmp_path),
                                  self._frames(0.31, 0.90))
        assert confs and all(c == pytest.approx(0.31) for c in confs)

    def test_a_solidly_supported_event_reads_as_solid(self, tmp_path):
        confs = self._confidences(self._engine(tmp_path),
                                  self._frames(0.88, 0.92))
        assert confs and all(c == pytest.approx(0.88) for c in confs)

    def test_weak_and_strong_events_are_told_apart(self, tmp_path):
        weak = self._confidences(self._engine(tmp_path), self._frames(0.31, 0.9))
        strong = self._confidences(self._engine(tmp_path), self._frames(0.9, 0.92))
        assert max(weak) < min(strong)

    def test_confidence_is_never_a_flat_one(self, tmp_path):
        """The specific regression: 1.0 regardless of what supported it."""
        confs = self._confidences(self._engine(tmp_path),
                                  self._frames(0.42, 0.55))
        assert confs and 1.0 not in confs


# ------------------------------------------- signal rules need no cache

SIGNAL_RULES = """
events:
  - name: loud_stretch
    label: Loud stretch
    signals:
      - {signal: level, min: 5.0}
"""


def test_signal_rules_run_without_any_analysis_cache(tmp_path, monkeypatch):
    """A rule measured from the file itself must not require a previous pass.

    Signal conditions read nothing out of the cache, and `merge_into_cache`
    seeds an entry when there is none — so demanding one put the fastest
    operation in the app behind the slowest, a full detection pass whose output
    those rules then ignore.
    """
    rules = tmp_path / "rules.yaml"
    rules.write_text(SIGNAL_RULES, encoding="utf-8")
    monkeypatch.setattr("modules.app_paths.composition_rules_path",
                        lambda: str(rules))
    monkeypatch.setattr(ao, "read_cache", lambda _p: {})
    monkeypatch.setattr("modules.composition_signals.gather",
                        lambda *a, **k: {"level": [0.0, 9.0, 9.0, 0.0]})

    patch = ao.run_composition("v.mp4", log=lambda *a: None)
    seconds = {row["timestamp"] for row in patch["objects"]}
    assert seconds == {1, 2}
    assert patch["composed_event_names"] == ["loud_stretch"]


def test_spatial_rules_still_require_a_cache(tmp_path, monkeypatch):
    """The guard is only relaxed where it was unnecessary."""
    rules = tmp_path / "rules.yaml"
    rules.write_text(RULES, encoding="utf-8")
    monkeypatch.setattr("modules.app_paths.composition_rules_path",
                        lambda: str(rules))
    monkeypatch.setattr(ao, "read_cache", lambda _p: {})
    with pytest.raises(RuntimeError, match="analysis"):
        ao.run_composition("v.mp4", log=lambda *a: None)


# ------------------------------------------- the run fetches what rules read

MIXED_RULES = """
events:
  - name: held_object
    label: Held object
    rules:
      - {source: handle, region: tool, min_count: 1}
  - name: off_rule
    label: Off rule
    enabled: false
    rules:
      - {source: ghost, region: shelf, min_count: 1}
    signals:
      - {signal: vocal_density_pct, min: 90}
"""


@pytest.fixture
def mixed_rules(tmp_path, monkeypatch):
    path = tmp_path / "composition_rules.yaml"
    path.write_text(MIXED_RULES, encoding="utf-8")
    monkeypatch.setattr("modules.app_paths.composition_rules_path",
                        lambda: str(path))
    return path


class TestFetchesItsOwnInputs:
    """A rule set states what it reads, so the run can go and get it.

    Without this, a spatial rule needed the user to work out which classes it
    named, type them into the objects field, run *that*, and only then press
    Run — for information already written in the rules file.
    """

    def test_missing_classes_start_a_detection_pass(self, rules_file, cached):
        cached["cache"] = {"objects": [], "object_bboxes": []}
        asked = {}

        def detect(classes, progress=None, cancel=None):
            asked["classes"] = classes
            return {"objects": [{"timestamp": 0, "objects": ["tool", "handle"],
                                 "count": 2}],
                    "object_bboxes": _frames()}

        patch = ao.run_composition("v.mp4", detect_fn=detect, log=lambda *a: None)

        assert asked["classes"] == ["handle", "tool"]
        secs = {e["timestamp"]: e["objects"] for e in patch["objects"]}
        assert "held_object" in secs[0]
        # what the pass found goes back too, or the classes exist as boxes and
        # are missing from the list the timeline reads
        assert "tool" in secs[0]
        assert patch["object_bboxes"], "the detector's boxes are kept"

    def test_cached_classes_are_not_re_detected(self, rules_file, cached):
        cached["cache"] = {"object_bboxes": _frames(), "objects": []}

        def detect(classes, progress=None, cancel=None):
            raise AssertionError("should not re-detect what the cache holds")

        patch = ao.run_composition("v.mp4", detect_fn=detect, log=lambda *a: None)
        assert any("held_object" in e["objects"] for e in patch["objects"])
        assert "object_bboxes" not in patch, "nothing was detected here"

    def test_a_partially_cached_rule_re_detects_every_class_it_names(
            self, rules_file, cached):
        """Both classes, not just the absent one: a spatial rule asks whether
        one sits inside another *in the same frame*, and two passes at
        different times cannot be joined across."""
        half = [dict(f, objects=["tool"], bboxes=[f["bboxes"][0]],
                     confidences=[0.9]) for f in _frames()]
        cached["cache"] = {"object_bboxes": half, "objects": []}
        asked = {}

        def detect(classes, progress=None, cancel=None):
            asked["classes"] = classes
            return {"objects": [], "object_bboxes": _frames()}

        ao.run_composition("v.mp4", detect_fn=detect, log=lambda *a: None)
        assert asked["classes"] == ["handle", "tool"]

    def test_the_stale_boxes_for_those_classes_are_replaced_not_doubled(
            self, rules_file, cached):
        """The re-detected classes appear once, from the new pass. Two entries
        per timestamp would leave the engine reading half a frame at a time."""
        old = [dict(f, objects=["tool"], bboxes=[[0.8, 0.8, 0.1, 0.1]],
                    confidences=[0.4]) for f in _frames()]
        cached["cache"] = {"object_bboxes": old, "objects": []}

        def detect(classes, progress=None, cancel=None):
            return {"objects": [], "object_bboxes": _frames()}

        patch = ao.run_composition("v.mp4", detect_fn=detect, log=lambda *a: None)
        assert len(patch["object_bboxes"]) == len(_frames())
        assert all(e["objects"] == ["tool", "handle"]
                   for e in patch["object_bboxes"])
        assert any("held_object" in e["objects"] for e in patch["objects"])

    def test_boxes_of_other_classes_survive_the_pass(self, rules_file, cached):
        """The cache may be an hours-long pipeline run; re-detecting one class
        must not cost the rest."""
        other = [{"timestamp": 0.0, "objects": ["tool", "unrelated"],
                  "bboxes": [[0.2, 0.2, 0.4, 0.4], [0.7, 0.7, 0.1, 0.1]],
                  "confidences": [0.9, 0.5]}]
        cached["cache"] = {"object_bboxes": other, "objects": []}

        def detect(classes, progress=None, cancel=None):
            return {"objects": [], "object_bboxes": _frames()}

        patch = ao.run_composition("v.mp4", detect_fn=detect, log=lambda *a: None)
        kept = [n for e in patch["object_bboxes"] for n in e["objects"]]
        assert "unrelated" in kept
        assert kept.count("unrelated") == 1

    def test_without_a_detect_fn_it_still_says_what_to_run(self, rules_file,
                                                           cached):
        """Starting a long pass unasked is the caller's decision, not this
        function's."""
        cached["cache"] = {"objects": [], "object_bboxes": []}
        with pytest.raises(RuntimeError, match="object detection first"):
            ao.run_composition("v.mp4", log=lambda *a: None)


class TestDisabledRulesCostNothing:
    def test_an_unticked_rule_does_not_trigger_detection(self, mixed_rules,
                                                          cached):
        cached["cache"] = {"object_bboxes": _frames(), "objects": []}

        def detect(classes, progress=None, cancel=None):
            raise AssertionError(f"unticked rule sent us detecting {classes}")

        ao.run_composition("v.mp4", detect_fn=detect, log=lambda *a: None)

    def test_an_unticked_rule_does_not_trigger_a_measurement(self, mixed_rules):
        """The vocal measurement decodes the whole file — paying for it because
        of a rule that is switched off is the wait this avoids."""
        from modules import composition_signals

        assert composition_signals.signal_names(str(mixed_rules)) == set()

    def test_its_name_is_still_stripped(self, mixed_rules, cached):
        """`event_names` keeps disabled events on purpose: their last output
        has to go, or switching a rule off leaves its events on the timeline
        for ever."""
        cached["cache"] = {
            "object_bboxes": _frames(),
            "objects": [{"timestamp": 0, "objects": ["off_rule"], "count": 1}],
        }
        patch = ao.run_composition("v.mp4", log=lambda *a: None)
        assert not any("off_rule" in e["objects"] for e in patch["objects"])


class TestObjectClasses:
    def test_the_engine_reports_the_classes_its_rules_read(self, rules_file):
        assert CompositionEngine(str(rules_file)).object_classes == ["handle",
                                                                    "tool"]

    def test_disabled_events_are_left_out(self, mixed_rules):
        assert CompositionEngine(str(mixed_rules)).object_classes == ["handle",
                                                                     "tool"]

    def test_a_signal_only_rule_set_needs_no_classes(self, tmp_path):
        p = tmp_path / "signals.yaml"
        p.write_text(SIGNAL_RULES, encoding="utf-8")
        assert CompositionEngine(str(p)).object_classes == []
