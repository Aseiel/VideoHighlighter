"""Composition rules over MCP: read, change one event (keeping the file's
comments), refuse what the engine cannot read, and list events."""
from __future__ import annotations

import pytest

from mcp_server import composition_tools as ct

RULES = """# measurements behind a threshold
events:
- name: kept
  label: Kept
  # why 8 s
  min_duration_secs: 8
  signals:
  - signal: vocal_density_pct
    min: 90
"""


@pytest.fixture
def rules(tmp_path, monkeypatch):
    path = tmp_path / "composition_rules.yaml"
    path.write_text(RULES, encoding="utf-8")
    monkeypatch.setattr(ct, "_rules_path", lambda: str(path))
    return path


def test_rules_are_read_as_written(rules):
    out = ct.get_rules()
    assert out["events"][0]["name"] == "kept" and out["events"][0]["min_duration_secs"] == 8


def test_an_event_is_added_and_replaced_and_comments_survive(rules):
    pytest.importorskip("ruamel.yaml")
    ev = {"name": "new_one", "label": "New", "min_matching": 2,
          "signals": [{"signal": "said", "equals": "a|b", "required": True},
                      {"signal": "question", "equals": "male"}]}
    assert ct.set_rule(ev)["action"] == "added"
    ev["label"] = "Renamed"
    assert ct.set_rule(ev)["action"] == "replaced"
    text = rules.read_text(encoding="utf-8")
    assert "# measurements behind a threshold" in text and "# why 8 s" in text
    names = [e["name"] for e in ct.get_rules()["events"]]
    assert names == ["kept", "new_one"]
    assert ct.get_rules()["events"][1]["label"] == "Renamed"


def test_a_rule_the_engine_cannot_read_is_refused_and_nothing_written(rules):
    before = rules.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        ct.set_rule({"name": "broken", "rules": [{"source": "x"}]})   # no region
    with pytest.raises(ValueError):
        ct.set_rule({"name": "empty"})
    assert rules.read_text(encoding="utf-8") == before


def test_an_event_is_removed(rules):
    ct.set_rule({"name": "temp", "signals": [{"signal": "said", "equals": "a"}]})
    assert ct.remove_rule("temp")["events"] == ["kept"]
    with pytest.raises(ValueError):
        ct.remove_rule("temp")


def test_events_are_read_as_stretches():
    cache = {"composed_event_names": ["a", "b"],
             "objects": [{"timestamp": t, "objects": ["a"]} for t in (10, 11, 12, 20)]
             + [{"timestamp": 30, "objects": ["person"]}]}
    assert ct.events_in(cache) == {"a": [{"start": 10.0, "end": 13.0}, {"start": 20.0, "end": 21.0}],
                                   "b": []}


def test_the_server_lists_the_tools():
    import asyncio
    pytest.importorskip("mcp")
    from mcp_server.server import build_server
    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert {"get_rules", "set_rule", "remove_rule", "start_apply_rules",
            "get_events", "get_transcript"} <= names
