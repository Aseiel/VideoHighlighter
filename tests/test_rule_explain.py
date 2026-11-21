"""A composed event's tooltip says what its rule checked (modules/rules/rule_explain.py)."""
from __future__ import annotations

import textwrap

from modules.rules import rule_explain as rx

EVENT = {
    "name": "slow_pass", "label": "Slow pass", "min_duration_secs": 2,
    "ignore_start_secs": 120, "min_matching": 2,
    "rules": [{"source": "pin", "region": "ring", "relation": "touches",
               "min_count": 1, "max_count": 999}],
    "signals": [
        {"signal": "past_line:pin:ring", "max": 0.5, "sustained_secs": 3, "required": True},
        {"signal": "vocal_effort", "min": 1.0, "within_secs": 2},
        {"signal": "face_text:a tired face", "min": 70, "within_secs": 2},
    ],
}


def test_every_condition_and_threshold_is_in_the_explanation():
    text = "\n".join(rx.explain(EVENT))
    assert "pin touches ring: at least 1" in text
    assert "any 2 of these 3" in text
    assert "how far pin is past the line on ring" in text and "<= 0.5" in text
    assert "for 3 s in a row" in text and "(required)" in text
    assert "vocal effort >= 1 within 2 s" in text
    assert 'face looks like "a tired face"' in text and ">= 70" in text
    assert "lasting at least 2 s" in text and "not in the first 120 s" in text


def test_all_signals_needed_reads_as_all():
    text = "\n".join(rx.explain({**EVENT, "min_matching": 3}))
    assert "and all of these" in text and "any 3" not in text


def test_counts_and_relations_are_spelled_out():
    assert rx._spatial_line({"source": "a", "region": "b", "max_count": 0}) == "a inside b: none"
    assert rx._spatial_line({"source": "a", "region": "b", "min_count": 2, "max_count": 3}) == "a inside b: 2-3"
    line = rx._spatial_line({"source": "a", "region": "b", "relation": "overlaps",
                             "min_overlap": 0.6, "outline": True})
    assert line == "a overlaps b: at least 1, 60% of it inside (by outline)"


def test_explain_event_reads_the_rules_file_and_follows_edits(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent("""
        events:
          - name: slow_pass
            label: Slow pass
            signals:
              - {signal: vocal_effort, min: 1.0}
        """), encoding="utf-8")
    lines = rx.explain_event("slow_pass", str(path))
    assert lines[0] == 'Why "Slow pass":' and "vocal effort >= 1" in lines[-1]
    assert rx.explain_event("missing", str(path)) == []

    import os
    import time
    path.write_text(path.read_text(encoding="utf-8").replace("1.0", "2.5"), encoding="utf-8")
    later = time.time() + 5
    os.utime(path, (later, later))
    assert "vocal effort >= 2.5" in rx.explain_event("slow_pass", str(path))[-1]


def test_no_file_means_no_explanation(tmp_path):
    assert rx.explain_event("slow_pass", str(tmp_path / "nope.yaml")) == []
