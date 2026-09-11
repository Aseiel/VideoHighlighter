"""Write down what the app is being asked to draw on.

A window that dies when it is resized to fill a large, heavily scaled display
leaves nothing behind: the failure is in Qt or the driver, below the Python
frame, so `debug.log` ends mid-sentence and the traceback that would name the
cause never exists. What can be recorded is the *shape of the problem* — how
many screens, how big, at what device pixel ratio, with which scale factor, and
what size the window had reached when the log stopped.

That is what this module is for. It proves or refutes the obvious theory (a
4K panel at 200% asking for a surface twice the size anyone tested) without
anybody having to reproduce the crash on hardware they do not own.
"""

from __future__ import annotations

import os
import sys


def _screen_line(screen) -> str:
    geometry = screen.geometry()
    available = screen.availableGeometry()
    return (f"   {screen.name() or 'screen'}: "
            f"{geometry.width()}x{geometry.height()} at "
            f"({geometry.x()},{geometry.y()}), "
            f"usable {available.width()}x{available.height()}, "
            f"ratio {screen.devicePixelRatio():g}, "
            f"{screen.logicalDotsPerInch():.0f} logical DPI, "
            f"{screen.physicalDotsPerInch():.0f} physical")


def describe(app) -> list:
    """Lines describing every screen, plus the scale factors in play."""
    lines = []
    try:
        screens = list(app.screens())
        primary = app.primaryScreen()
    except Exception as e:  # noqa: BLE001 - diagnostics must not raise
        return [f"   (could not read the screens: {type(e).__name__}: {e})"]

    for screen in screens:
        try:
            mark = " (primary)" if screen is primary else ""
            lines.append(_screen_line(screen) + mark)
        except Exception as e:  # noqa: BLE001
            lines.append(f"   (a screen would not describe itself: {e})")

    factor = os.environ.get("QT_SCALE_FACTOR")
    rounding = os.environ.get("QT_SCALE_FACTOR_ROUNDING_POLICY")
    lines.append(f"   QT_SCALE_FACTOR={factor or '(unset)'}, "
                 f"rounding={rounding or '(default)'}, "
                 f"platform={sys.platform}")
    return lines


def log(app, log_fn=print) -> None:
    """Report the display setup once, at startup."""
    log_fn("🖥️ Displays:")
    for line in describe(app):
        log_fn(line)


def log_window_size(window, label: str, log_fn=print) -> None:
    """Report a window's size, in the units that matter for a crash.

    Both numbers are here on purpose: Qt lays out in logical pixels, the driver
    allocates in physical ones, and a surface that is fine at 1920 wide may not
    be at 3840. A log that carries only one of them cannot tell those apart.
    """
    try:
        size = window.size()
        ratio = window.devicePixelRatioF()
        state = "maximised" if window.isMaximized() else (
            "fullscreen" if window.isFullScreen() else "windowed")
        log_fn(f"🖥️ {label}: {size.width()}x{size.height()} logical, "
               f"{int(size.width() * ratio)}x{int(size.height() * ratio)} "
               f"physical, {state}")
    except Exception as e:  # noqa: BLE001 - diagnostics must not raise
        log_fn(f"🖥️ {label}: size unavailable ({type(e).__name__}: {e})")
