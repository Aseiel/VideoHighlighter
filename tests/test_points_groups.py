"""Each box on the scoring page appears once.

The scoring page is built from one `groups` tuple of (title, rows). A Qt widget
has one parent, so a group listed twice does not show two working copies: the
second copy takes the spin boxes and pickers away from the first, which is left
with only the buttons `_points_row_with_button` builds afresh on each call.
Speech, Objects & actions, Face expression and Where in the video were listed
twice from 0.10.0 to 0.13.0, and the page showed four boxes with no inputs.

Source-level for the same reason as test_preview_wiring: CI has no Qt.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _group_titles():
    """Titles of every `groups = ((title, rows), ...)` tuple in main.py."""
    tree = ast.parse((REPO / "main.py").read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Tuple):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "groups" for t in node.targets):
            continue
        titles = [el.elts[0].value for el in node.value.elts
                  if isinstance(el, ast.Tuple) and el.elts
                  and isinstance(el.elts[0], ast.Constant)]
        found.append((node.lineno, titles))
    return found


def test_the_scoring_page_is_found():
    assert any("Speech" in titles for _, titles in _group_titles())


def test_no_group_is_listed_twice():
    for line, titles in _group_titles():
        repeated = sorted({t for t in titles if titles.count(t) > 1})
        assert not repeated, f"main.py:{line} lists {repeated} more than once"
