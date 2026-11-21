"""What a composed event's conditions measured, per bar (modules/rules/rule_evidence.py)."""
from __future__ import annotations

from modules.rules import rule_evidence as ev

EVENT = {"name": "slow_pass", "signals": [
    {"signal": "past_line:pin:ring", "max": 0.5, "sustained_secs": 3},
    {"signal": "vocal_effort", "min": 1.0, "within_secs": 1},
    {"signal": "expression", "any_of": ["happy"]},
]}


def _signals():
    past = [None] * 20
    effort = [0.0] * 20
    for s, v in zip(range(10, 14), (0.1, 0.3, 0.2, 0.25)):
        past[s] = v
    effort[9], effort[12] = 1.7, 1.2            # 9 is inside within_secs of the run
    return {"past_line:pin:ring": past, "vocal_effort": effort,
            "expression": {10: "happy", 11: "happy", 12: "sad"}}


def test_runs_join_consecutive_seconds():
    assert ev.runs([3, 4, 5, 9, 10, 15]) == [(3, 5), (9, 10), (15, 15)]


def test_each_condition_reads_its_own_numbers_over_the_run():
    composed = {s: ["slow_pass"] for s in range(10, 14)}
    out = ev.evidence([EVENT], composed, _signals())
    (run,) = out["slow_pass"]
    assert (run["start"], run["end"]) == (10, 14)
    past, effort, expr = run["signals"]
    assert past["max"] == 0.3 and past["min"] == 0.1 and past["need"] == "<= 0.5"
    assert effort["max"] == 1.7                   # widened by within_secs
    assert expr["labels"] == {"happy": 2, "sad": 1}


def test_events_without_signals_or_hits_are_left_out():
    spatial = {"name": "touching", "rules": [{"source": "pin", "region": "ring"}]}
    out = ev.evidence([spatial, EVENT], {5: ["touching"]}, _signals())
    assert out == {}


def test_the_bar_finds_its_run_and_says_it_in_words():
    composed = {s: ["slow_pass"] for s in range(10, 14)}
    stored = ev.evidence([EVENT], composed, _signals())
    run = ev.for_bar(stored, "Slow_Pass", 10.0, 14.0)
    text = "\n".join(ev.describe(run))
    assert "Measured here (0:10-0:14)" in text
    assert "pin past the line on ring: deepest 30% of its length" in text
    assert "(needs <= 50%)" in text
    assert "vocal effort: 0 to 1.7, average 0.48" in text and "(needs >= 1)" in text
    assert "expression: happy 2s, sad 1s" in text
    assert ev.for_bar(stored, "slow_pass", 40.0, 42.0) is None
    assert ev.for_bar({}, "slow_pass", 10.0, 14.0) is None
