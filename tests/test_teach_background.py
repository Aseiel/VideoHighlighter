"""Teaching in the background: only when the app is free, and it says what waits.

The engine runs without Qt; the Qt part is driven offscreen (skipped where
PySide6 cannot start).
"""
from __future__ import annotations

import os
import time

import pytest

from modules.teach import background
from modules.teach.project import OBJECTS, Project


def _project(tmp_path):
    p = Project.create(str(tmp_path / "bg"), OBJECTS)
    p.add_class("widget")
    p.save()
    return p


def test_one_pass_runs_auto_and_counts_the_questions(tmp_path):
    from modules.teach.boxes import store
    from modules.vision.label_store import LabelledBox

    p = _project(tmp_path)
    labels = store(p)
    for t in (1.0, 2.0):
        labels.add(LabelledBox(video="v.mp4", time=t, class_name="widget",
                               box=(0.1, 0.1, 0.2, 0.2), source="found"))
    labels.save()
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        return 0, {"ran": [{"args": ["find"]}], "stopped_at": {}}

    report = background.run_once(p.root, run_cli=fake_cli)
    assert calls == [["--project", p.root, "auto", "--train"]]
    assert report["ran"] == ["find"] and report["questions"] == 2
    assert report["next"]["args"]
    line = background.summary(report)
    assert "find" in line and "2 questions waiting" in line
    assert background.summary({"error": "boom"}) == "Background: stopped (boom)"


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _settle(app, teacher):
    deadline = time.time() + 5
    while teacher.running and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_it_waits_until_the_app_is_free(app, tmp_path):
    free = {"now": False}
    passes, reports = [], []
    teacher = background.BackgroundTeacher(
        lambda: str(tmp_path), lambda: free["now"],
        run=lambda root, train: passes.append(root) or {"questions": 3})
    teacher.report.connect(reports.append)

    assert teacher.tick() is False and passes == []
    free["now"] = True
    assert teacher.tick() is True
    assert teacher.tick() is False                   # one pass at a time
    _settle(app, teacher)
    assert passes == [str(tmp_path)] and reports == [{"questions": 3}]


def test_a_failing_pass_is_reported_not_raised(app, tmp_path):
    reports = []

    def broken(root, train):
        raise SystemExit("no")

    teacher = background.BackgroundTeacher(lambda: str(tmp_path), lambda: True, run=broken)
    teacher.report.connect(reports.append)
    teacher.tick()
    _settle(app, teacher)
    assert reports and "SystemExit" in reports[0]["error"]
    assert not teacher.running


def _waiting_project(root, n):
    from modules.teach.boxes import store
    from modules.vision.label_store import LabelledBox

    p = Project.create(root, OBJECTS)
    p.add_class("widget")
    p.settings.background = True
    p.save()
    labels = store(p)
    for t in range(n):
        labels.add(LabelledBox(video="v.mp4", time=float(t), class_name="widget",
                               box=(0.1, 0.1, 0.2, 0.2), source="found"))
    labels.save()
    return p


def test_only_opted_in_projects_are_touched_and_they_take_turns(tmp_path, monkeypatch):
    from modules.teach import project as project_mod

    monkeypatch.setattr(project_mod, "projects_root", lambda: str(tmp_path))
    a = Project.create(str(tmp_path / "a"), OBJECTS)
    a.add_class("widget")
    a.add_source(str(tmp_path / "v.mp4"))           # not cut yet: auto work to do
    a.settings.background = True
    a.save()
    b = Project.create(str(tmp_path / "b"), OBJECTS)
    b.add_class("widget")
    b.add_source(str(tmp_path / "v.mp4"))
    b.settings.background = True
    b.save()
    c = Project.create(str(tmp_path / "c"), OBJECTS)   # not opted in
    c.add_class("widget")
    c.add_source(str(tmp_path / "v.mp4"))
    c.save()

    roots = background.opted_in()
    assert roots == [a.root, b.root]
    assert background.pick(roots) == a.root
    assert background.pick(roots, last=a.root) == b.root
    assert background.pick(roots, last=b.root) == a.root
    # Nothing unattended left (a judge step next: it needs footage): not picked.
    for p in (a, b):
        p.sources = []
        p.save()
    assert background.pick(roots) == ""


def test_teaching_from_the_player_seeds_a_background_project(tmp_path, monkeypatch):
    from modules.teach import from_player
    from modules.teach import project as project_mod

    monkeypatch.setattr(project_mod, "projects_root", lambda: str(tmp_path))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    result = from_player.teach(str(video), 3.5, (0.2, 0.3, 0.1, 0.1), "  red   kite ")
    assert result["class"] == "red kite" and result["seeds"] == 1
    assert result["root"] == str(tmp_path / "red-kite")
    assert background.opted_in() == [result["root"]]
    assert "Learning to find “red kite”" in from_player.message(result)
    again = from_player.teach(str(video), 9.0, (0.5, 0.3, 0.1, 0.1), "red kite")
    assert again["seeds"] == 2 and "2 so far" in from_player.message(again)
    with pytest.raises(ValueError):
        from_player.teach(str(video), 1.0, (0.2, 0.3, 0.1, 0.1), "   ")


def test_the_panel_counts_waiting_questions_and_stays_out_of_the_way(app, tmp_path,
                                                                    monkeypatch):
    from modules.teach import project as project_mod
    from modules.teach import teach_panel

    monkeypatch.setattr(project_mod, "projects_root", lambda: str(tmp_path))
    _waiting_project(str(tmp_path / "taught"), 4)
    panel = teach_panel.TeachPanel(run_cli=lambda argv: (0, {}))
    panel.project.setText(str(tmp_path / "mine"))
    _project(tmp_path)
    # Freshly touched: not idle, so nothing runs.
    panel.idle.last = time.monotonic()
    assert panel._free() is False
    panel.idle.last = time.monotonic() - background.IDLE_SECONDS - 1
    assert panel._free() is True
    panel._review = object()                          # a review window is open
    assert panel._free() is False
    panel._review = None

    panel._background_report({"root": str(tmp_path / "taught"), "ran": ["find"],
                              "questions": 4, "next": {"why": "Check 4 proposed boxes."}})
    assert panel.review_btn.text() == "Check guesses… (4)"
    assert "taught: find; 4 questions waiting" in panel.background_status.text()

    # Check guesses goes to the project with questions, not the empty one typed in.
    opened = []
    panel._open_review = opened.append
    panel.review()
    assert opened == [str(tmp_path / "taught")]


def test_the_switch_is_the_project_setting(app, tmp_path, monkeypatch):
    from modules.teach import project as project_mod
    from modules.teach import teach_panel

    monkeypatch.setattr(project_mod, "projects_root", lambda: str(tmp_path))
    p = _project(tmp_path)
    panel = teach_panel.TeachPanel(run_cli=lambda argv: (0, {}))
    panel.project.setText(p.root)
    panel._show_background_setting()
    assert not panel.background_box.isChecked()
    panel.background_box.setChecked(True)
    assert Project.load(p.root).settings.background is True
    assert background.opted_in() == [p.root]


def test_a_box_drawn_in_the_player_becomes_a_seed(app, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtCore import QRectF

    from modules.teach import project as project_mod
    from video_ai_editor import realtime_overlay as ro

    monkeypatch.setattr(project_mod, "projects_root", lambda: str(tmp_path))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    shown = []
    monkeypatch.setattr(ro.QMessageBox, "information", lambda *a: shown.append(a[2]))
    monkeypatch.setattr(ro.QMessageBox, "warning", lambda *a: shown.append("warn: " + a[2]))
    player = SimpleNamespace(
        _scene=SimpleNamespace(sceneRect=lambda: QRectF(0, 0, 200, 100)),
        _player=SimpleNamespace(position=lambda: 12500), video_path=str(video))
    player._normalized_region = lambda rect: ro.RealtimeOverlayPreview._normalized_region(
        player, rect)

    # Dragged past the right edge: kept inside the frame.
    roi, ts = player._normalized_region(QRectF(150, 20, 80, 30))
    assert ts == 12.5 and roi == pytest.approx((0.75, 0.2, 0.25, 0.3))

    ro.RealtimeOverlayPreview._teach_model(player, "kite", QRectF(20, 10, 40, 30))
    assert shown and "Learning to find “kite”" in shown[0]
    root = str(tmp_path / "kite")
    assert background.opted_in() == [root]
    from modules.teach.boxes import store
    (seed_box,) = store(Project.load(root)).accepted()
    assert seed_box.time == 12.5 and seed_box.box == pytest.approx((0.1, 0.1, 0.2, 0.3))
