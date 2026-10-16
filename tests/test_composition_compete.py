"""``compete_with``: a source counts only for the region class it is closest to.

Two region classes side by side in the frame: one source meets both, and a
rule on each fires on the same thing. Naming the neighbour in ``compete_with``
gives the source to one of them, across events as well as within one. Without
it, nothing changes.
"""
from __future__ import annotations

import asyncio
import textwrap

import pytest
import yaml

from modules.rules import shapes
from modules.rules.rules_file import TABLE_RULE_KEYS, carry_rule_fields
from video_ai_editor.composition_engine import CompositionEngine

# A holder straddling the border between two regions: its centre is inside
# both boxes, 70% of its area is in A and 60% in B.
HOLDER = [0.40, 0.40, 0.20, 0.20]
A = [0.20, 0.30, 0.34, 0.40]
B = [0.48, 0.30, 0.32, 0.40]


def _engine(tmp_path, events: str):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(events), encoding="utf-8")
    return CompositionEngine(str(path))


def _two_events(tmp_path, compete: bool, relation: str = "inside"):
    a_vs = ", compete_with: [b]" if compete else ""
    b_vs = ", compete_with: [a]" if compete else ""
    return _engine(tmp_path, f"""
        events:
          - name: on_a
            window_secs: 0
            persist_secs: 0
            rules:
              - {{source: holder, region: a, relation: {relation}{a_vs}}}
          - name: on_b
            window_secs: 0
            persist_secs: 0
            rules:
              - {{source: holder, region: b, relation: {relation}{b_vs}}}
        """)


def _frames(dets, n=3):
    """dets: list of (class, box)."""
    return [{
        "timestamp": float(i),
        "objects": [d[0] for d in dets],
        "bboxes": [d[1] for d in dets],
        "confidences": [0.9] * len(dets),
    } for i in range(n)]


def _fired(engine, frames) -> set:
    events, _ = engine.run(frames)
    return {name for names in events.values() for name in names}


def test_without_it_one_source_fires_both_neighbours(tmp_path):
    engine = _two_events(tmp_path, compete=False)
    frames = _frames([("holder", HOLDER), ("a", A), ("b", B)])
    assert _fired(engine, frames) == {"on_a", "on_b"}       # the old answer


@pytest.mark.parametrize("relation", ["inside", "overlaps", "touches"])
def test_with_it_the_source_goes_to_the_closer_one_across_events(tmp_path, relation):
    engine = _two_events(tmp_path, compete=True, relation=relation)
    frames = _frames([("holder", HOLDER), ("a", A), ("b", B)])
    assert _fired(engine, frames) == {"on_a"}

    # The regions swapped: the same source now sits mostly in b.
    frames = _frames([("holder", HOLDER), ("a", B), ("b", A)])
    assert _fired(engine, frames) == {"on_b"}


def test_a_rival_that_is_not_in_the_frame_changes_nothing(tmp_path):
    engine = _two_events(tmp_path, compete=True)
    assert _fired(engine, _frames([("holder", HOLDER), ("b", B)])) == {"on_b"}


def test_the_rule_still_needs_its_own_relation(tmp_path):
    # Closer to A than to B, but its centre is in neither: nothing fires.
    engine = _two_events(tmp_path, compete=True)
    far_a, far_b = [0.0, 0.0, 0.42, 0.42], [0.58, 0.0, 0.42, 0.42]
    assert _fired(engine, _frames([("holder", [0.3, 0.5, 0.2, 0.2]),
                                   ("a", far_a), ("b", far_b)])) == set()


def test_inside_both_wholly_goes_to_the_one_it_sits_in_the_middle_of(tmp_path):
    engine = _two_events(tmp_path, compete=True)
    big = [0.10, 0.10, 0.80, 0.80]          # centre 0.5, 0.5
    offset = [0.30, 0.30, 0.50, 0.50]       # centre 0.55, 0.55
    holder = [0.45, 0.45, 0.10, 0.10]       # centre 0.5, 0.5: wholly in both
    assert _fired(engine, _frames([("holder", holder), ("a", big), ("b", offset)])) == {"on_a"}


def test_an_exact_tie_counts_for_both(tmp_path):
    engine = _two_events(tmp_path, compete=True)
    same = [0.3, 0.3, 0.4, 0.4]
    assert _fired(engine, _frames([("holder", HOLDER), ("a", same), ("b", same)])) == {
        "on_a", "on_b"}


def test_within_one_event_each_source_goes_to_its_own_region(tmp_path):
    engine = _engine(tmp_path, """
        events:
          - name: both
            window_secs: 0
            persist_secs: 0
            rules:
              - {source: holder, region: a, compete_with: [b]}
              - {source: holder, region: b, compete_with: [a]}
        """)
    left = [0.25, 0.40, 0.10, 0.10]         # well inside A only
    # One source in A and one straddling but mostly in A: B gets nobody.
    assert _fired(engine, _frames([("holder", left), ("holder", HOLDER),
                                   ("a", A), ("b", B)])) == set()
    right = [0.65, 0.40, 0.10, 0.10]        # well inside B only
    assert _fired(engine, _frames([("holder", left), ("holder", right),
                                   ("a", A), ("b", B)])) == {"both"}


def test_an_empty_list_is_the_same_as_none(tmp_path):
    engine = _engine(tmp_path, """
        events:
          - name: on_a
            window_secs: 0
            persist_secs: 0
            rules:
              - {source: holder, region: a, compete_with: []}
        """)
    assert _fired(engine, _frames([("holder", HOLDER), ("a", A), ("b", B)])) == {"on_a"}


def test_load_accepts_one_class_and_drops_its_own_region(tmp_path):
    engine = _engine(tmp_path, """
        events:
          - name: e
            rules:
              - {source: holder, region: a, compete_with: b}
              - {source: holder, region: b, compete_with: [a, b, a, " c "]}
        """)
    rules = engine._specs[0].rules
    assert rules[0].compete_with == ("b",)
    assert rules[1].compete_with == ("a", "c")


def test_load_refuses_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="compete_with"):
        _engine(tmp_path, """
            events:
              - name: e
                rules:
                  - {source: holder, region: a, compete_with: {b: 1}}
            """)


def test_rivals_are_asked_for(tmp_path):
    engine = _engine(tmp_path, """
        events:
          - name: e
            rules:
              - {source: holder, region: a, compete_with: [b], outline: true, max_gap: 0.01}
        """)
    assert engine.object_classes == ["a", "b", "holder"]
    assert engine.outline_pairs == [("holder", "a", 0.01), ("holder", "b", 0.01)]


def test_closeness_ranks_overlap_then_gap_then_centres():
    src = {"box": HOLDER}
    assert shapes.closeness(src, {"box": A}) > shapes.closeness(src, {"box": B})
    # No overlap with either: the nearer edge wins.
    near, far = {"box": [0.61, 0.4, 0.1, 0.1]}, {"box": [0.70, 0.4, 0.1, 0.1]}
    assert shapes.closeness(src, near) > shapes.closeness(src, far)


def test_a_table_save_keeps_it():
    original = [{"source": "holder", "region": "a", "compete_with": ["b"]}]
    rebuilt = [{"source": "holder", "region": "a", "min_count": 1, "max_count": 999}]
    assert carry_rule_fields(original, rebuilt, owned=TABLE_RULE_KEYS | {"relation", "outline"}) == [
        {"source": "holder", "region": "a", "min_count": 1, "max_count": 999,
         "compete_with": ["b"]}]


def test_a_web_ui_save_keeps_it(tmp_path, monkeypatch):
    from sidecar.server import CompRulesRequest, get_composition_rules, save_composition_rules

    rules = tmp_path / "composition_rules.yaml"
    rules.write_text(textwrap.dedent("""
        events:
          - name: on_a
            rules:
              - {source: holder, region: a, compete_with: [b]}
        """), encoding="utf-8")
    monkeypatch.setattr("modules.system.app_paths.composition_rules_path", lambda: str(rules))
    monkeypatch.setattr("modules.system.app_paths.user_data_dir", lambda: str(tmp_path))

    rows = asyncio.run(get_composition_rules())["rules"]
    rows[0]["min_count"] = 2
    assert asyncio.run(save_composition_rules(CompRulesRequest(rules=rows)))["ok"]
    saved = yaml.safe_load(rules.read_text(encoding="utf-8"))["events"][0]["rules"][0]
    assert saved["compete_with"] == ["b"] and saved["min_count"] == 2
    CompositionEngine(str(rules))               # and it still loads
