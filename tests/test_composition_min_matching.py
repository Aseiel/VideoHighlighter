"""Composition rules made of several cues: "any k of these", "this one must be
among them", alternatives in one value, and seconds that carry several values.
Shared: the engine and the rules file are the same in both editions."""
from __future__ import annotations

from video_ai_editor.composition_engine import CompositionEngine


def _engine(tmp_path, text):
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return CompositionEngine(str(path))


RULE = """
events:
- name: cue
  label: Cue
  min_matching: {k}
  signals:
  - signal: said
    equals: "bolí|au|to boli"
    within_secs: 2
  - signal: "face:a described face"
    min: 95
  - signal: question
    equals: male
    within_secs: 3
"""


def _signals(said_at=None, face_at=None, question_at=None):
    s = {"said": {}, "face:a described face": {}, "question": {}}
    for sec in range(60):
        s["face:a described face"][sec] = 50.0
    if said_at is not None:
        s["said"][said_at] = ["mě", "to", "bolí", "mě to", "to bolí"]
    if face_at is not None:
        s["face:a described face"][face_at] = 99.0
    if question_at is not None:
        s["question"][question_at] = ["male"]
    return s


def _fired(events, name="cue"):
    return sorted(sec for sec, names in events.items() if name in names)


def test_two_of_three_cues_are_enough(tmp_path):
    eng = _engine(tmp_path, RULE.format(k=2))
    events, overlay = eng.run([], signals=_signals(said_at=30, face_at=31))
    # "said" reaches 2 s either side; the face holds only at 31.
    assert _fired(events) == [31]
    ev = next(e for e in overlay if e["timestamp"] == 31.0)
    assert sorted(ev["evidence"]["cue"]) == sorted(["said", "face:a described face"])


def test_one_cue_alone_is_not(tmp_path):
    eng = _engine(tmp_path, RULE.format(k=2))
    events, _ = eng.run([], signals=_signals(said_at=30))
    assert _fired(events) == []


def test_without_min_matching_every_cue_must_hold(tmp_path):
    eng = _engine(tmp_path, RULE.format(k=0))
    events, _ = eng.run([], signals=_signals(said_at=30, face_at=31))
    assert _fired(events) == []
    events, _ = eng.run([], signals=_signals(said_at=30, face_at=31, question_at=31))
    assert 31 in _fired(events)


def test_alternatives_in_one_equals_and_list_values(tmp_path):
    eng = _engine(tmp_path, RULE.format(k=1).replace('"bolí|au|to boli"', '"au|to boli"'))
    events, _ = eng.run([], signals=_signals(said_at=10))
    assert _fired(events) == []                       # "to bolí" is not "to boli"
    s = _signals()
    s["said"][10] = ["to", "boli", "to boli"]
    events, _ = eng.run([], signals=s)
    assert 10 in _fired(events)



def test_a_required_cue_must_be_among_them(tmp_path):
    text = RULE.format(k=2).replace('    within_secs: 2\n', '    within_secs: 2\n    required: true\n', 1)
    eng = _engine(tmp_path, text)
    events, _ = eng.run([], signals=_signals(face_at=31, question_at=31))
    assert _fired(events) == []                       # two cues, but not the words
    events, _ = eng.run([], signals=_signals(said_at=31, question_at=31))
    assert 31 in _fired(events)


def test_the_advanced_tab_keeps_required_and_lists():
    from modules.rules.rules_file import carry_signal_fields
    original = [{"signal": "said", "equals": "bolí|au", "required": True, "within_secs": 5},
                {"signal": "question", "any_of": ["male", "unknown"]}]
    rebuilt = [{"signal": "said", "equals": "bolí|au|aua", "within_secs": 5},
               {"signal": "question"}]
    out = carry_signal_fields(original, rebuilt)
    assert out[0] == {"signal": "said", "equals": "bolí|au|aua", "within_secs": 5, "required": True}
    assert out[1]["any_of"] == ["male", "unknown"]
    out = carry_signal_fields(original, [{"signal": "question", "equals": "male"}])
    assert "any_of" not in out[0]                     # the table's value wins
