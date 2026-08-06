"""The logo while you wait, and an honest line about what is being loaded.

Two things are slow enough to need it: launching the app (dominated by the
import phase in a frozen build) and opening the timeline viewer (~6s of widget
construction, most of it the signal timeline and the assistant panel). Both
block the GUI thread, so without this the window simply is not there yet and
the app looks hung.

There are two surfaces, because no single one covers the whole launch:

``pyi_splash``
    PyInstaller's native splash, drawn by the bootloader *before* Python
    starts. It is the only thing that can cover the import phase — by the time
    any of our code runs, the slowest part of an exe launch is already over.
    Needs ``--splash`` at build time; absent from a source run. **Image only:**
    the ``--splash`` CLI flag hardcodes ``text_pos=None``, which disables the
    text overlay entirely, so ``update_text`` is accepted and silently drawn
    nowhere. The calls below are kept because they cost nothing and start
    working the day the build moves to a .spec that sets ``text_pos`` — until
    then, stage() during the import phase is a debug.log timing and no more.

``begin()`` / ``stage()`` / ``finish()``
    A Qt panel with the logo, a stage line and a step bar, for everything
    after Qt exists. Also used on its own for the timeline viewer.

``stage()`` drives whichever of the two is live, so callers never branch. It is
a no-op when neither is — that is what lets `signal_timeline_viewer` report its
own progress without knowing whether anyone is watching, and without importing
this module's Qt half at all.

**Qt is imported lazily, on the first `begin()`.** Importing this module has to
stay free of it: `main.py` reports progress from before it has imported Qt
itself, and the order it imports things in on Windows is deliberate.
"""

from __future__ import annotations

import os
import sys
import time

from modules.app_paths import resource_path

LOGO_FILE = os.path.join("assets", "icon.png")

_current = None          # the live Qt splash, if any
_started_at = 0.0        # for the debug-log timings
_splash_class = None     # built on first use, see _stage_splash_class()


# ──────────────────────────────────────────────────────────────────
# PyInstaller's native splash (frozen builds only)
# ──────────────────────────────────────────────────────────────────

def _pyi():
    """The bootloader's splash, or None. Missing from source runs and from any
    build made without --splash, so every call site has to tolerate None."""
    if not getattr(sys, "frozen", False):
        return None
    try:
        import pyi_splash
        return pyi_splash if pyi_splash.is_alive() else None
    except Exception:
        return None


def close_native_splash():
    """Dismiss the bootloader splash. Called once the Qt panel is up — two
    splashes on screen at once looks broken."""
    native = _pyi()
    if native is not None:
        try:
            native.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────
# The Qt panel
# ──────────────────────────────────────────────────────────────────

def _stage_splash_class():
    """Define the widget class on first use. See the module docstring on why
    the Qt import cannot happen at module scope."""
    global _splash_class
    if _splash_class is not None:
        return _splash_class

    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtGui import QPixmap, QPainter, QPainterPath
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QLabel, QFrame, QProgressBar, QApplication,
    )
    from modules.ui.theme import DARK as THEME

    def _rounded(pix: QPixmap, radius: int) -> QPixmap:
        """Clip the artwork to a rounded tile.

        The logo is a dark image on a dark card, so its square edge reads as an
        accidental seam. Rounding it makes the boundary look like the app-icon
        tile it is."""
        out = QPixmap(pix.size())
        out.fill(Qt.GlobalColor.transparent)
        painter = QPainter(out)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(pix.rect()), radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pix)
        painter.end()
        return out

    class _StageSplash(QWidget):
        """Frameless card: logo, title, step bar, and the current stage."""

        def __init__(self, title: str, subtitle: str, steps: int, parent=None):
            super().__init__(parent, Qt.WindowType.SplashScreen
                             | Qt.WindowType.FramelessWindowHint
                             | Qt.WindowType.WindowStaysOnTopHint)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)

            card = QFrame()
            card.setObjectName("splashCard")
            card.setStyleSheet(f"""
                QFrame#splashCard {{
                    background-color: {THEME.surface};
                    border: 1px solid {THEME.border_strong};
                    border-radius: {THEME.radius_card}px;
                }}
            """)
            outer.addWidget(card)

            lay = QVBoxLayout(card)
            lay.setContentsMargins(28, 26, 28, 22)
            lay.setSpacing(10)

            logo = QLabel()
            logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Every label here is explicitly transparent: the app palette gives
            # them a window-coloured background, which on top of the card reads
            # as a stack of grey blocks rather than text on a panel.
            logo.setStyleSheet("background: transparent;")
            pix = QPixmap(resource_path(LOGO_FILE))
            if not pix.isNull():
                logo.setPixmap(_rounded(pix.scaled(
                    112, 112,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation), 18))
            lay.addWidget(logo)

            head = QLabel(title)
            head.setAlignment(Qt.AlignmentFlag.AlignCenter)
            head.setStyleSheet(f"color: {THEME.text}; font-size: 16px; "
                              f"font-weight: 600; background: transparent;")
            lay.addWidget(head)

            if subtitle:
                sub = QLabel(subtitle)
                sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
                sub.setStyleSheet(f"color: {THEME.text_mute}; font-size: 11px; "
                                  f"background: transparent;")
                lay.addWidget(sub)

            lay.addSpacing(4)

            # Determinate on purpose: an indeterminate bar cannot animate while
            # the GUI thread is blocked building widgets, so it would just sit
            # there looking stuck. Stepping it per stage is both honest and
            # visibly alive.
            self._bar = QProgressBar()
            self._bar.setRange(0, max(1, steps))
            self._bar.setValue(0)
            self._bar.setTextVisible(False)
            self._bar.setFixedHeight(4)
            self._bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {THEME.surface_alt};
                    border: none; border-radius: 2px;
                }}
                QProgressBar::chunk {{
                    background-color: {THEME.accent}; border-radius: 2px;
                }}
            """)
            lay.addWidget(self._bar)

            self._stage = QLabel("Starting…")
            self._stage.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._stage.setStyleSheet(f"color: {THEME.text_dim}; "
                                      f"font-size: 12px; background: transparent;")
            lay.addWidget(self._stage)

            self.setFixedSize(380, 300)
            self._centre_on_screen(parent)

        def _centre_on_screen(self, parent):
            """Centre on the screen the app is actually on, not always the
            primary one — on a two-monitor desktop the splash showing up on the
            other screen from the window it belongs to is worse than none."""
            screen = None
            if parent is not None and parent.window().screen() is not None:
                screen = parent.window().screen()
            if screen is None:
                screen = QApplication.primaryScreen()
            if screen is None:
                return
            area = screen.availableGeometry()
            self.move(area.x() + (area.width() - self.width()) // 2,
                      area.y() + (area.height() - self.height()) // 2)

        def set_stage(self, text: str):
            self._stage.setText(text)
            # Hold at one short of full until finish(): claiming 100% while
            # there is still work to do is the thing progress bars get wrong.
            self._bar.setValue(min(self._bar.value() + 1, self._bar.maximum() - 1))

        def complete(self):
            self._bar.setValue(self._bar.maximum())

    _splash_class = _StageSplash
    return _splash_class


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def begin(title: str, subtitle: str = "", steps: int = 6, parent=None):
    """Show the splash. ``steps`` is how many stage() calls to expect — it only
    sizes the bar, so being off by one just makes it move in slightly wrong
    increments. Replaces any splash already up."""
    global _current, _started_at
    finish()
    try:
        _current = _stage_splash_class()(title, subtitle, steps, parent)
        _current.show()
        _process_events()
        # Only now: the native splash stays up until its replacement is
        # actually painted, so the logo never blinks out mid-launch.
        close_native_splash()
    except Exception as e:
        print(f"⚠️ Startup splash unavailable: {e}")
        _current = None
    _started_at = time.perf_counter()
    return _current


def stage(text: str):
    """Report what is being loaded now. Safe to call from anywhere, at any
    time: it drives whichever surface is live and does nothing when none is.
    Always logged with a timing, so a slow launch can be read back from
    debug.log after the fact."""
    elapsed = (time.perf_counter() - _started_at) if _started_at else 0.0
    print(f"⏳ [{elapsed:5.1f}s] {text}")

    native = _pyi()
    if native is not None:
        try:
            native.update_text(text)
        except Exception:
            pass

    if _current is not None:
        try:
            _current.set_stage(text)
            _process_events()
        except Exception:
            pass


def finish(window=None):
    """Take the splash down. Pass the window it was covering so it is raised
    into the gap the splash leaves."""
    global _current, _started_at
    close_native_splash()
    if _current is not None:
        try:
            _current.complete()
            _process_events()
            _current.close()
            _current.deleteLater()
        except Exception:
            pass
        _current = None
    if window is not None:
        try:
            window.raise_()
            window.activateWindow()
        except Exception:
            pass
    _started_at = 0.0


def active() -> bool:
    return _current is not None


def _process_events():
    """Repaint the splash from inside a blocking constructor.

    User input is deliberately excluded: the window being built is not wired up
    yet, and a click delivered into a half-constructed viewer is a crash with a
    confusing traceback."""
    try:
        from PySide6.QtCore import QEventLoop
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    except Exception:
        pass


class splash:
    """Context-manager form, for a block that should always take the splash
    down again::

        with splash("Opening timeline viewer", steps=6, parent=self):
            window = SignalTimelineWindow(...)
        window.show()
    """

    def __init__(self, title: str, subtitle: str = "", steps: int = 6, parent=None):
        self._args = (title, subtitle, steps, parent)

    def __enter__(self):
        return begin(*self._args)

    def __exit__(self, *exc):
        finish()
        return False
