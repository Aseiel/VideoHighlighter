"""Keeping a rules file whole when a table rewrites it.

The Advanced tab edits composition rules as table rows and saves by
rebuilding the file from them. Anything without a column (a rule's
``relation`` or ``outline``, a top-level ``outliner``) would be dropped on
the first Save, silently, with nothing on screen to show it going. It has
happened before with event fields (``min_duration_secs``). These helpers
carry such fields across, so a field added to the engine survives the table
without needing a column first.
"""
from __future__ import annotations

# The spatial-rule fields the table has columns for; everything else on a
# rule is the file's to keep.
TABLE_RULE_KEYS = frozenset({"source", "region", "min_count", "max_count"})


def carry_rule_fields(original_rules, rebuilt_rules) -> list:
    """``rebuilt_rules`` with each one's extra fields restored from its original.

    A rebuilt rule is matched to the first unused original rule with the same
    source and region, so two rules on the same pair keep their own settings
    in order, and a rule that was edited to a new pair starts clean.
    """
    pool = [dict(r) for r in (original_rules or []) if isinstance(r, dict)]
    used = set()
    out = []
    for rule in rebuilt_rules:
        merged = dict(rule)
        for i, old in enumerate(pool):
            if i in used:
                continue
            if old.get("source") == rule.get("source") and old.get("region") == rule.get("region"):
                used.add(i)
                for key, value in old.items():
                    if key not in TABLE_RULE_KEYS:
                        merged.setdefault(key, value)
                break
        out.append(merged)
    return out


def top_level_fields(raw) -> dict:
    """Everything in a rules file except its events (e.g. ``outliner``)."""
    return {k: v for k, v in (raw or {}).items() if k != "events"} if isinstance(raw, dict) else {}
