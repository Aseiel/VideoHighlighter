"""The web UI's rules save must not delete what its rows cannot show.

Its rows are spatial rules only; the file also holds signal conditions,
signal-only events, event and rule fields, and top-level settings. Called
directly (asyncio.run), as test_combine_endpoint does.
"""
from __future__ import annotations

import asyncio
import textwrap

import yaml

import sidecar.server as server
from sidecar.server import CompRulesRequest, save_composition_rules


def test_saving_rows_keeps_everything_else(tmp_path, monkeypatch):
    rules = tmp_path / "composition_rules.yaml"
    rules.write_text(textwrap.dedent("""
        outliner: sam
        events:
          - name: both
            label: Both
            min_duration_secs: 3
            signals:
              - {signal: loudness, min: 0.5}
            rules:
              - {source: a, region: b, min_count: 1, relation: touches, outline: true}
          - name: audio_only
            signals:
              - {signal: loudness, min: 0.9}
          - name: removed
            rules:
              - {source: x, region: y}
        """), encoding="utf-8")
    monkeypatch.setattr("modules.system.app_paths.composition_rules_path", lambda: str(rules))
    monkeypatch.setattr("modules.system.app_paths.user_data_dir", lambda: str(tmp_path))

    rows = [{"name": "both", "label": "Both", "source": "a", "region": "b",
             "min_count": 2, "max_count": 999, "window_secs": 0.75, "persist_secs": 0.5}]
    result = asyncio.run(save_composition_rules(CompRulesRequest(rules=rows)))
    assert result["ok"], result

    saved = yaml.safe_load(rules.read_text(encoding="utf-8"))
    assert saved["outliner"] == "sam"
    events = {e["name"]: e for e in saved["events"]}
    assert set(events) == {"both", "audio_only"}           # "removed" was deleted in the UI
    both = events["both"]
    assert both["min_duration_secs"] == 3
    assert both["signals"] == [{"signal": "loudness", "min": 0.5}]
    assert both["rules"] == [{"source": "a", "region": "b", "min_count": 2,
                              "max_count": 999, "relation": "touches", "outline": True}]
    assert server is not None


def test_relation_and_outline_round_trip_through_the_web_ui(tmp_path, monkeypatch):
    from sidecar.server import get_composition_rules

    rules = tmp_path / "composition_rules.yaml"
    rules.write_text(textwrap.dedent("""
        events:
          - name: e
            rules:
              - {source: a, region: b, relation: Touches, outline: true, max_gap: 0.02}
        """), encoding="utf-8")
    monkeypatch.setattr("modules.system.app_paths.composition_rules_path", lambda: str(rules))
    monkeypatch.setattr("modules.system.app_paths.user_data_dir", lambda: str(tmp_path))

    rows = asyncio.run(get_composition_rules())["rules"]
    assert rows[0]["relation"] == "touches" and rows[0]["outline"] is True

    # Switched back to the defaults in the UI: they must stick.
    rows[0]["relation"], rows[0]["outline"] = "inside", False
    assert asyncio.run(save_composition_rules(CompRulesRequest(rules=rows)))["ok"]
    saved = yaml.safe_load(rules.read_text(encoding="utf-8"))["events"][0]["rules"][0]
    assert "relation" not in saved and "outline" not in saved
    assert saved["max_gap"] == 0.02               # no column for it: kept
