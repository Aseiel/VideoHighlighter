"""The Datasets tab: list, new dataset, and the four steps, driven by the real CLI.

Offscreen; skipped where PySide6 cannot start. Samples are the same stand-ins
as ``test_teach.py``, and the embedder is the fake one.
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from modules.teach import cli, dataset_panel, embed  # noqa: E402
from modules.teach.project import ACTIONS, NAMED, Project  # noqa: E402
from tests.test_teach import FakeEmbedder, _add_samples, _reader  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Projects live under tmp_path; one exists, its footage already cut."""
    parent = tmp_path / "teach"
    monkeypatch.setattr("modules.teach.project.projects_root", lambda: str(parent))
    p = Project.create(str(parent / "first"), ACTIONS, name="first")
    truth = _add_samples(p, [0] * 10 + [1] * 10)
    monkeypatch.setattr(cli, "make_embedder", lambda name="clip": FakeEmbedder())
    monkeypatch.setattr(embed, "read_frames", _reader(truth))
    return parent, p


def _wait(panel, app):
    deadline = time.time() + 10
    while panel.runner.busy and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_lists_datasets_and_opens_one_on_the_step_it_needs(app, home):
    parent, p = home
    panel = dataset_panel.DatasetsPanel()
    _wait(panel, app)
    assert [x["name"] for x in panel.list.projects] == ["first"]
    assert panel.list.table.item(0, 6).text() == "Not ready"
    panel.list.open_selected()
    assert panel.stack.currentWidget() is panel.wizard
    # Footage is cut and nothing is named: grouping is next, on step 2.
    assert panel.wizard.steps.currentRow() == dataset_panel.GROUPS


def test_group_name_and_health_from_the_buttons(app, home):
    parent, p = home
    panel = dataset_panel.DatasetsPanel()
    _wait(panel, app)
    panel.open(p.root)
    page = panel.wizard.groups
    page.k.setValue(2)
    page.regroup.click()
    _wait(panel, app)
    assert len(page.cards) == 2 and "0 of 2 groups named" in page.progress.text()

    card = page.cards[0]
    card.thumbs[1].mousePressEvent(_left_click())
    assert card.wrong() == "2"
    card.name.setCurrentText("alpha move")
    card.name_it()
    _wait(panel, app)
    project = Project.load(p.root)
    named = [g for g in project.groups if g.state == NAMED]
    assert len(named) == 1 and named[0].name == "alpha move" and len(named[0].wrong) == 1
    assert "1 of 2 groups named" in page.progress.text()
    assert "left to check in step 3" in panel.wizard.message.text()

    page.cards[0].say(["--nothing"])
    _wait(panel, app)
    health = panel.wizard.health
    # One class and "nothing", both big enough to train, from one video.
    assert health.verdict == "needs_videos" and health.train_btn.isEnabled()
    assert health.sources.rowCount() == 1          # only group guesses so far
    assert "2 groups named" in page.progress.text()


def test_new_dataset_runs_init_and_opens_it(app, home, monkeypatch):
    parent, _ = home
    panel = dataset_panel.DatasetsPanel()
    _wait(panel, app)

    class Dialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return QtWidgets.QDialog.Accepted

        def args(self):
            return ["--project", str(parent / "second"), "init", "--task", "actions",
                    "--name", "second"]

    monkeypatch.setattr(dataset_panel, "NewDatasetDialog", Dialog)
    panel.list.new()
    _wait(panel, app)
    assert os.path.exists(parent / "second" / "project.json")
    assert panel.stack.currentWidget() is panel.wizard
    assert panel.wizard.steps.currentRow() == dataset_panel.VIDEOS


def _left_click():
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    return QMouseEvent(QMouseEvent.MouseButtonPress, QPointF(5, 5), QPointF(5, 5),
                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
