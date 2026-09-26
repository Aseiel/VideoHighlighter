"""The click-to-review window, driven offscreen.

Skipped where PySide6 cannot start (CI installs only the test requirements).
What matters is that a click means the same verdict the command line would
record, so the window is driven with real clicks and the project is read back.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

from modules.teach import review_window  # noqa: E402
from modules.teach.project import (  # noqa: E402
    ACCEPTED, ACTIONS, NEGATIVE, NONE, PENDING, REJECTED, UNSURE, Project, Sample,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _project(tmp_path):
    p = Project.create(str(tmp_path / "w"), ACTIONS)
    p.add_class("alpha move")
    p.add_class("beta move")
    guesses = ["alpha move", "alpha move", "beta move", UNSURE, NONE]
    for i, guess in enumerate(guesses):
        p.samples.append(Sample(id=f"v001__{i:08d}", source="v001",
                                path=str(tmp_path / f"{i}.mp4"), start=i * 5.0,
                                duration=5.0, proposed=guess, margin=0.5,
                                scores={"alpha move": 0.6, "beta move": 0.3}))
    p.save()
    return p


def test_verdicts_from_tile_states():
    states = {1: ("accept", "a"), 2: ("accept", "b"), 3: ("reject", ""),
              4: ("none", ""), 5: ("undecided", "")}
    guesses = {1: "a", 2: "a", 3: "a", 4: "b", 5: "_unsure"}
    assert review_window.verdict_args(states, guesses) == {
        "accept": "1", "reject": "3", "negative": "4", "relabel": ["2=b"]}


def test_clicking_through_a_batch(app, tmp_path):
    p = _project(tmp_path)
    window = review_window.ReviewWindow(p.root, size=10,
                                        frame_reader=lambda *a, **k: [])
    tiles = {t.item["sample"]: t for t in window.tiles}
    assert len(tiles) == 5
    states = {sid: t.state for sid, t in tiles.items()}
    assert states["v001__00000000"] == "accept"          # guessed a class
    assert states["v001__00000003"] == "undecided"       # unsure
    assert states["v001__00000004"] == "none"            # none of these

    QTest.mouseClick(tiles["v001__00000001"], Qt.LeftButton)       # -> reject
    tiles["v001__00000002"].set_state("accept", "alpha move")      # right-click relabel
    QTest.keyClick(tiles["v001__00000003"], Qt.Key_R)              # unsure -> reject
    result = window.save(and_next=True)
    assert result["errors"] == []

    got = {s.id: (s.verdict, s.label) for s in Project.load(p.root).samples}
    assert got == {
        "v001__00000000": (ACCEPTED, "alpha move"),
        "v001__00000001": (REJECTED, ""),
        "v001__00000002": (ACCEPTED, "alpha move"),
        "v001__00000003": (REJECTED, ""),
        "v001__00000004": (NEGATIVE, ""),
    }
    # Everything was decided, so the next batch is empty and says so.
    assert window.tiles == [] and "Nothing left" in window.header.text()
    window.close()


def test_undecided_tiles_stay_in_the_queue(app, tmp_path):
    p = _project(tmp_path)
    window = review_window.ReviewWindow(p.root, size=10, frame_reader=lambda *a, **k: [])
    window.save(and_next=False)
    left = [s.id for s in Project.load(p.root).samples if s.verdict == PENDING]
    assert left == ["v001__00000003"]
