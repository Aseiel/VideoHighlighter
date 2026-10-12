"""Objects found per sample, and rules that name a class from them (``evidence``).

A fake detector reads how many "tokens" a frame holds from the frame itself,
so the composition engine's real rules are exercised on known answers: one
token on the plate is one class, two is the other.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from modules.teach import cli, evidence, health, review, sort, status
from modules.teach.project import (
    ACCEPTED, ACTIONS, AUTO, NONE, PENDING, Project,
)
from modules.vision.detection_backend import Detection
from tests.test_teach import FakeEmbedder, _add_samples, _reader

RULES = """\
events:
  - name: alpha move
    rules:
      - {source: token, region: plate, relation: inside, min_count: 1, max_count: 1}
  - name: beta move
    rules:
      - {source: token, region: plate, relation: inside, min_count: 2}
"""


class TokenDetector:
    """A plate, and as many tokens on it as the frame's first pixel says."""

    def detect(self, frame):
        h, w = frame.shape[:2]
        tokens = int(frame[0, 0, 0])
        out = [Detection(1, "plate", 0.9, 0.2 * w, 0.2 * h, 0.8 * w, 0.8 * h)]
        for i in range(tokens):
            x = (0.3 + 0.2 * i) * w
            out.append(Detection(0, "token", 0.8, x, 0.4 * h, x + 0.05 * w, 0.5 * h))
        return out


def _frames(tokens: dict):
    def read(path, count, **_):
        n = tokens[os.path.basename(path)]
        return [np.full((90, 160, 3), n, np.uint8)] * count
    return read


@pytest.fixture
def setup(tmp_path):
    p = Project.create(str(tmp_path / "proj"), ACTIONS)
    p.add_class("alpha move")
    p.add_class("beta move")
    truth = _add_samples(p, [0] * 6 + [1] * 6)
    tokens = {name: k + 1 for name, k in truth.items()}       # alpha: 1, beta: 2
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    p.settings.detector_model = str(model)
    p.settings.detector_classes = "token,plate"
    p.save()
    return p, truth, tokens


def test_status_asks_to_detect_and_detections_are_cached(setup):
    p, truth, tokens = setup
    assert status.next_step(p)["args"] == ["detect"]
    out = evidence.detect_samples(p, TokenDetector(), frame_reader=_frames(tokens))
    assert out["detected"] == 12 and out["frames_with"]["token"] == 12 * 12
    assert evidence.todo(p) == []
    entry = evidence.load(p)["samples"][p.samples[0].id][0]
    assert entry["objects"] == ["plate", "token"] and len(entry["bboxes"][0]) == 4
    assert status.next_step(p)["args"] == ["sort"]


def test_rules_tell_one_from_two(setup):
    p, truth, tokens = setup
    evidence.detect_samples(p, TokenDetector(), frame_reader=_frames(tokens))
    with open(evidence.rules_path(p), "w", encoding="utf-8") as handle:
        handle.write(RULES)
    guesses = evidence.rule_guesses(p)
    assert len(guesses) == 12
    for s in p.samples:
        want = "alpha move" if truth[os.path.basename(s.path)] == 0 else "beta move"
        assert guesses[s.id][0] == want and guesses[s.id][1] >= 0.4


def test_sort_takes_the_rule_guess_review_measures_it_and_a_proven_rule_decides(setup):
    p, truth, tokens = setup
    p.settings.auto_accept = False
    p.settings.rule_auto_min_checked = 3
    evidence.detect_samples(p, TokenDetector(), frame_reader=_frames(tokens))
    with open(evidence.rules_path(p), "w", encoding="utf-8") as handle:
        handle.write(RULES)
    result = sort.sort_project(p, FakeEmbedder(), frame_reader=_reader(truth))
    assert result["rules"]["proposed"] == {"alpha move": 6, "beta move": 6}
    assert all(s.proposed_by == "rule" and s.margin == 0.0 for s in p.samples)

    sheet = review.next_sheet(p, size=6, frame_reader=lambda *a, **k: [],
                              renderer=lambda *a, **k: None)
    assert "(rule)" in sheet["items"][0]["caption"]
    review.apply_verdicts(p, sheet["sheet"], accept_rest=True)
    checked = [s for s in p.samples if s.is_human]
    assert checked and all(s.guess_from == "rule" for s in checked)
    found = health.sorting(p)["by_source"]["rule"]
    assert found["accuracy"] == 1.0 and found["right"] == len(checked)

    # Proven on enough checks, the rule now decides the rest alone, under the
    # same spot checks as any auto-accept.
    sort.sort_project(p, FakeEmbedder(), frame_reader=_reader(truth))
    record = evidence.rule_record(p)
    proven = [c for c in record if evidence.trusted(p, c, record)]
    assert proven
    auto = [s for s in p.samples if s.decided_by == AUTO]
    assert auto and all(s.label in proven and s.auto_label == s.label for s in auto)
    assert all(s.verdict == PENDING for s in p.samples
               if s.proposed_by == "rule" and s.proposed not in proven
               and not s.is_human)


def test_two_rules_firing_is_no_guess(setup):
    p, truth, tokens = setup
    evidence.detect_samples(p, TokenDetector(), frame_reader=_frames(tokens))
    with open(evidence.rules_path(p), "w", encoding="utf-8") as handle:
        handle.write(RULES.replace("min_count: 1, max_count: 1", "min_count: 1"))
    guesses = evidence.rule_guesses(p)
    beta = [s.id for s in p.samples if truth[os.path.basename(s.path)] == 1]
    assert not any(sid in guesses for sid in beta)      # both rules hold on two


def test_the_more_specific_rule_wins_and_rules_written_for_the_timeline_map(tmp_path):
    p = Project.create(str(tmp_path / "proj"), ACTIONS)
    truth = _add_samples(p, [0] * 4 + [1] * 4)
    tokens = {name: k + 1 for name, k in truth.items()}
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    p.settings.detector_model, p.settings.detector_classes = str(model), "token,plate"
    p.save()
    evidence.detect_samples(p, TokenDetector(), frame_reader=_frames(tokens))
    # A timeline rules file: names are not classes, `class:` says which. One
    # token fires both rules on a two-token sample; the one with more
    # conditions wins there.
    with open(evidence.rules_path(p), "w", encoding="utf-8") as handle:
        handle.write("""\
events:
  - name: one_on_plate
    label: One
    class: alpha move
    rules:
      - {source: token, region: plate, min_count: 1}
  - name: two_on_plate
    class: beta move
    rules:
      - {source: token, region: plate, min_count: 2}
      - {source: plate, region: plate, min_count: 1}
  - name: loud
    signals:
      - {signal: vocal_density_pct, min: 90}
""")
    code, out = cli.run(["--project", p.root, "rules", "--add-classes"])
    assert out["classes_added"] == ["alpha move", "beta move"]
    assert out["events"] == {"one_on_plate": "alpha move", "two_on_plate": "beta move"}
    p = Project.load(p.root)
    guesses = evidence.rule_guesses(p)
    for s in p.samples:
        want = "alpha move" if truth[os.path.basename(s.path)] == 0 else "beta move"
        assert guesses[s.id][0] == want


def test_cli_rules_init_and_report(setup, monkeypatch):
    p, truth, tokens = setup
    code, out = cli.run(["--project", p.root, "rules"])
    assert code == 0 and out["exists"] is False
    code, out = cli.run(["--project", p.root, "rules", "--init"])
    assert code == 0 and out["exists"] and os.path.exists(out["rules"])
    assert out["fired"] == {}                             # the template has no events
    assert "<class>" in open(out["rules"], encoding="utf-8").read()


def test_class_names_from_a_labels_file(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text('["a", "b"]', encoding="utf-8")
    assert evidence.class_names(str(f)) == ["a", "b"]
    f.write_text('{"names": {"1": "y", "0": "x"}}', encoding="utf-8")
    assert evidence.class_names(str(f)) == ["x", "y"]
    assert evidence.class_names("p, q") == ["p", "q"]
    assert NONE == evidence._class_of("nothing", ["p"]) and evidence._class_of("z", []) == ""
