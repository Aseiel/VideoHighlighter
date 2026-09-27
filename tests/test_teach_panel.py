"""The in-app "From videos" tab: buttons that run the teach CLI.

Offscreen; skipped where PySide6 cannot start.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from modules.teach import teach_panel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(panel, app):
    import time
    deadline = time.time() + 5
    while panel._thread is not None and panel._thread.isRunning() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_start_runs_quick_with_the_fields_and_shows_the_next_step(app, tmp_path):
    calls = []

    def fake_cli(argv):
        calls.append(argv)
        return 0, {"ran": [{"args": ["cut"]}, {"args": ["sort"]}],
                   "stopped_at": {"why": "Review guesses: 'x' has 1 of 100."}}

    panel = teach_panel.TeachPanel(run_cli=fake_cli)
    panel.project.setText("p1")
    panel.examples.setText(str(tmp_path / "ex"))
    panel.videos.setText(str(tmp_path / "vids"))
    panel.focus.setChecked(True)
    panel.start()
    _wait(panel, app)
    assert calls == [["--project", "p1", "quick", "--task", "actions", "--examples",
                      str(tmp_path / "ex"), "--videos", str(tmp_path / "vids"), "--focus"]]
    text = panel.output.toPlainText()
    assert "done: cut" in text and "Next: Review guesses" in text
    assert panel.start_btn.isEnabled()


def test_doctor_results_read_as_a_checklist():
    text = teach_panel.describe({"ready": False, "checks": [
        {"name": "clip", "ok": False, "level": "required", "detail": "missing",
         "fix": "install the pack"},
        {"name": "ffmpeg", "ok": True, "level": "required", "detail": "found", "fix": ""}]})
    assert "MISSING clip: missing  -> install the pack" in text
    assert "OK ffmpeg" in text and "Not ready" in text


def test_an_error_is_shown_not_raised(app):
    panel = teach_panel.TeachPanel(run_cli=lambda argv: (_ for _ in ()).throw(OSError("disk")))
    panel._run(["status"])
    _wait(panel, app)
    assert "OSError: disk" in panel.output.toPlainText()


def test_check_guesses_needs_a_project(app, tmp_path, monkeypatch):
    opened = []
    panel = teach_panel.TeachPanel(run_cli=lambda argv: (0, {}), open_review=opened.append)
    panel.project.setText(str(tmp_path / "nothing-here"))
    panel.review()
    assert opened == [] and "Start a project" in panel.output.toPlainText()
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "project.json").write_text("{}")
    panel.project.setText(str(tmp_path / "proj"))
    panel.review()
    assert opened == [str(tmp_path / "proj")]
