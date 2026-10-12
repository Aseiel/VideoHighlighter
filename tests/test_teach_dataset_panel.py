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
    assert "left to check in step 4" in panel.wizard.message.text()

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


def test_the_crop_switch_sets_focus_and_the_button_says_crop(app, home):
    parent, p = home
    panel = dataset_panel.DatasetsPanel()
    _wait(panel, app)
    panel.open(p.root)
    page = panel.wizard.videos
    assert not page.crop.isChecked()
    page.crop.click()
    _wait(panel, app)
    assert Project.load(p.root).settings.focus is True
    assert page.crop.isChecked() and page.prepare.isEnabled()
    assert page.prepare.text() == "Crop to people and group them"


def test_browse_selects_a_range_and_moves_it_to_a_new_class(app, home, monkeypatch):
    parent, p = home
    panel = dataset_panel.DatasetsPanel()
    _wait(panel, app)
    panel.open(p.root)
    page = panel.wizard.browser
    panel.wizard.browse(("pending",))
    assert panel.wizard.steps.currentRow() == dataset_panel.BROWSE
    assert len(page.tiles.thumbs) == 20
    from PySide6.QtCore import Qt
    page.tiles._clicked(page.tiles.thumbs[2], Qt.NoModifier)
    page.tiles._clicked(page.tiles.thumbs[5], Qt.ShiftModifier)
    assert [t.n for t in page.tiles.selected()] == [3, 4, 5, 6]
    monkeypatch.setattr(dataset_panel.QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("alpha move", True)))
    menu = dataset_panel.move_menu(panel.wizard, [t.sample.id for t in
                                                  page.tiles.selected()], page)
    next(a for a in menu.actions() if a.text() == "New class…").trigger()
    _wait(panel, app)
    project = Project.load(p.root)
    assert len(project.accepted("alpha move")) == 4
    assert "Moved 4 samples to 'alpha move'" in panel.wizard.message.text()
    # Browse stays on the same folder, now 4 shorter, and lists the new class.
    assert len(page.tiles.thumbs) == 16
    labels = [page.folders.item(i).text() for i in range(page.folders.count())]
    assert any(t.startswith("alpha move") and "(4)" in t for t in labels)


def test_the_folder_and_each_sample_path_can_be_seen_and_copied(app, home):
    parent, p = home
    panel = dataset_panel.DatasetsPanel()
    _wait(panel, app)
    panel.open(p.root)
    assert panel.wizard.where.text() == p.root
    page = panel.wizard.browser
    panel.wizard.browse(("pending",))
    first = page.tiles.thumbs[0]
    assert os.path.abspath(first.sample.path) in first.toolTip()
    page.tiles.select_all(True)
    page._copy_paths()
    copied = QtWidgets.QApplication.clipboard().text().splitlines()
    assert len(copied) == 20 and copied[0] == os.path.abspath(first.sample.path)


def test_what_a_command_prints_shows_as_progress_while_it_runs(app, home):
    import sys
    import threading

    parent, p = home
    go_on = threading.Event()

    def run_cli(argv):
        if argv[-1] != "auto":
            return cli.run(argv)
        print("auto: cut", file=sys.stderr)
        print("cut S001: 20/40", file=sys.stderr)
        print("\r 12%|bar", end="", file=sys.stderr)   # a redraw, not yet a line
        go_on.wait(10)
        return 0, {"ran": []}

    panel = dataset_panel.DatasetsPanel(run_cli=run_cli)
    _wait(panel, app)
    panel.open(p.root)
    wizard = panel.wizard
    seen = []
    panel.runner.progress.connect(seen.append)
    wizard.command(["auto"])
    deadline = time.time() + 10
    while len(seen) < 2 and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert seen == ["auto: cut", "cut S001: 20/40"]
    assert not wizard.bar.isHidden()
    assert (wizard.bar.value(), wizard.bar.maximum()) == (20, 40)
    assert "Cutting into samples" in wizard.message.text()
    assert "cut S001: 20/40" in wizard.message.text()
    print("from the UI thread", file=sys.stderr)        # not the command's
    app.processEvents()
    assert len(seen) == 2
    go_on.set()
    _wait(panel, app)
    assert wizard.bar.isHidden()
    assert not isinstance(sys.stderr, dataset_panel._Lines)


@pytest.fixture
def recycle_bin(monkeypatch):
    """moveToTrash, without filling the real Recycle Bin."""
    import shutil

    from PySide6.QtCore import QFile
    binned = []

    def move(path):
        binned.append(path)
        shutil.rmtree(path)
        return True
    monkeypatch.setattr(QFile, "moveToTrash", staticmethod(move))
    return binned


@pytest.mark.parametrize("answer", ["Yes", "Cancel"])
def test_delete_asks_first_then_bins_the_dataset(app, home, recycle_bin, monkeypatch,
                                                  answer):
    parent, p = home
    panel = dataset_panel.DatasetsPanel()
    _wait(panel, app)
    asked = []

    def exec_(box):
        asked.append(box.text())
        return getattr(QtWidgets.QMessageBox, answer)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", exec_)
    panel.list.table.selectRow(0)
    panel.list.delete_selected()
    _wait(panel, app)
    assert asked and "first" in asked[0]
    if answer == "Yes":
        assert recycle_bin == [p.root] and not os.path.exists(p.root)
        assert panel.list.projects == []
    else:
        assert recycle_bin == [] and os.path.exists(p.root)
        assert len(panel.list.projects) == 1


def test_the_rules_box_sets_the_detector_and_reports_what_waits(app, home, tmp_path):
    parent, p = home
    model = tmp_path / "model.onnx"
    model.write_bytes(b"x")
    panel = dataset_panel.DatasetsPanel()
    _wait(panel, app)
    panel.open(p.root)
    page = panel.wizard.check
    assert "No detector set" in page.rules_note.text()
    assert not page.detect_btn.isEnabled()
    page.detector.setText(str(model))
    page.detector.editingFinished.emit()
    _wait(panel, app)
    page.det_classes.setText("token, plate")
    page.det_classes.editingFinished.emit()
    _wait(panel, app)
    project = Project.load(p.root)
    assert project.settings.detector_model == str(model)
    assert project.settings.detector_classes == "token, plate"
    assert "20 not looked at yet" in page.rules_note.text()
    assert page.detect_btn.isEnabled()
