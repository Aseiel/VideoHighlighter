"""The startup splash: safe to call from anywhere, and always comes down again.

`stage()` is deliberately callable before Qt exists, before any splash is up,
and after one is torn down — `main.py` calls it from between its imports and
`signal_timeline_viewer` calls it whether or not anyone opened a splash first.
That freedom is only safe if none of those paths can raise, which is what most
of this file pins down.

The other half is the teardown. The splash is always-on-top and frameless, so
one that survives its own failure sits over the app with nothing behind it and
no way to dismiss it.
"""

from __future__ import annotations

import sys

import pytest

from modules import startup_splash


@pytest.fixture(autouse=True)
def _no_splash_left_behind():
    """Every test starts and ends with nothing on screen — a leaked splash
    would otherwise be inherited by whatever test ran next."""
    startup_splash.finish()
    yield
    startup_splash.finish()


class TestStageIsAlwaysSafe:
    """No Qt needed: these are the calls that happen before it is imported."""

    def test_stage_without_a_splash_is_a_no_op(self, capsys):
        startup_splash.stage("Loading the video engine…")

        assert not startup_splash.active()
        # Still logged: a slow launch is diagnosed from debug.log afterwards,
        # which is the only record when the build shows no text at all.
        assert "Loading the video engine" in capsys.readouterr().out

    def test_finish_without_a_splash_is_a_no_op(self):
        startup_splash.finish()          # must not raise

        assert not startup_splash.active()

    def test_stage_after_finish_is_a_no_op(self):
        startup_splash.finish()
        startup_splash.stage("late arrival")

        assert not startup_splash.active()


class _FakeNativeSplash:
    """Stands in for PyInstaller's pyi_splash, which only exists inside a
    frozen build."""

    def __init__(self):
        self.closed = False
        self.texts = []

    def is_alive(self):
        return not self.closed

    def update_text(self, text):
        self.texts.append(text)

    def close(self):
        self.closed = True


@pytest.fixture
def frozen_with_native_splash(monkeypatch):
    """A frozen build whose bootloader splash is up and already imported — and
    whose _PYI_SPLASH_IPC is therefore gone, because pyi_splash deletes the
    variable as it reads it."""
    native = _FakeNativeSplash()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(sys.modules, "pyi_splash", native)
    monkeypatch.delenv("_PYI_SPLASH_IPC", raising=False)
    return native


class TestNativeSplash:
    """The bootloader's splash is always-on-top and outlives the import phase,
    so the only thing that takes it down is us asking it to."""

    def test_it_closes_once_the_ipc_variable_is_gone(self, frozen_with_native_splash):
        startup_splash.close_native_splash()

        assert frozen_with_native_splash.closed

    def test_stage_still_reaches_it_after_the_first_call(self, frozen_with_native_splash):
        """main.py stages several times during the import phase. The first of
        those consumes the variable; the rest must still find the splash."""
        startup_splash.stage("Loading the video engine…")
        startup_splash.stage("Loading the detection runtime…")

        assert frozen_with_native_splash.texts == [
            "Loading the video engine…", "Loading the detection runtime…",
        ]

    def test_a_closed_splash_is_left_alone(self, frozen_with_native_splash):
        frozen_with_native_splash.close()

        startup_splash.close_native_splash()
        startup_splash.stage("late arrival")

        assert frozen_with_native_splash.texts == []

    def test_no_import_is_attempted_without_the_variable(self, monkeypatch):
        """A build made without --splash has neither the variable nor the
        module, and probing for it must not try to import one."""
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.delitem(sys.modules, "pyi_splash", raising=False)
        monkeypatch.delenv("_PYI_SPLASH_IPC", raising=False)

        startup_splash.close_native_splash()      # must not raise

        assert "pyi_splash" not in sys.modules


# The panel itself needs a QApplication; the rest of the module does not.
pytestmark_qt = pytest.importorskip("PySide6", reason="Qt not available")

from PySide6.QtWidgets import QApplication, QWidget    # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


class TestSplashLifecycle:
    def test_begin_shows_and_finish_clears(self, qt_app):
        splash = startup_splash.begin("VideoHighlighter", "Version 9.9", steps=3)

        assert splash is not None
        assert startup_splash.active()

        startup_splash.finish()

        assert not startup_splash.active()

    def test_begin_replaces_a_splash_already_up(self, qt_app):
        """Opening the viewer while the launch splash is somehow still up must
        leave exactly one on screen, not two stacked always-on-top windows."""
        first = startup_splash.begin("First", steps=3)
        second = startup_splash.begin("Second", steps=3)

        assert second is not first
        assert startup_splash.active()

    def test_stage_advances_the_bar_but_never_completes_it(self, qt_app):
        """Claiming 100% while work is still running is the thing progress bars
        get wrong; the last step belongs to finish()."""
        splash = startup_splash.begin("Opening timeline viewer", steps=3)
        bar = splash._bar

        startup_splash.stage("one")
        after_one = bar.value()
        for text in ("two", "three", "four", "five"):
            startup_splash.stage(text)

        assert after_one == 1
        assert bar.value() < bar.maximum(), "bar hit 100% before finish()"

    def test_finish_completes_the_bar(self, qt_app):
        splash = startup_splash.begin("Opening timeline viewer", steps=3)
        bar = splash._bar
        startup_splash.stage("one")

        splash.complete()

        assert bar.value() == bar.maximum()

    def test_stage_text_reaches_the_panel(self, qt_app):
        splash = startup_splash.begin("Opening timeline viewer", steps=3)

        startup_splash.stage("Drawing the signal timeline…")

        assert splash._stage.text() == "Drawing the signal timeline…"


class TestContextManager:
    def test_splash_closes_on_the_way_out(self, qt_app):
        with startup_splash.splash("Opening timeline viewer", steps=3):
            assert startup_splash.active()

        assert not startup_splash.active()

    def test_splash_closes_even_when_the_block_raises(self, qt_app):
        """A viewer that dies half-way through construction must not strand an
        always-on-top window over the app."""
        with pytest.raises(RuntimeError):
            with startup_splash.splash("Opening timeline viewer", steps=3):
                raise RuntimeError("viewer blew up")

        assert not startup_splash.active()


class TestFinishRaisesTheWindow:
    def test_window_is_raised_into_the_gap(self, qt_app):
        startup_splash.begin("Opening timeline viewer", steps=3)
        window = QWidget()
        calls = []
        window.raise_ = lambda: calls.append("raise")
        window.activateWindow = lambda: calls.append("activate")

        startup_splash.finish(window)

        assert calls == ["raise", "activate"]

    def test_a_none_window_is_fine(self, qt_app):
        """The viewer passes None when construction failed — there is no window
        to raise, and that path must not raise either."""
        startup_splash.begin("Opening timeline viewer", steps=3)

        startup_splash.finish(None)

        assert not startup_splash.active()
