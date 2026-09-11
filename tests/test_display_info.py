"""The display report: the only evidence a resize crash leaves.

A crash below the Python frame writes no traceback, so what ends up in
debug.log is whatever was logged before it. That makes this diagnostic code the
witness, and a witness that raises, or that reports only logical pixels, is
worse than none.
"""

from __future__ import annotations

import types

from modules import display_info


def _screen(name, w, h, ratio=1.0, dpi=96.0):
    rect = types.SimpleNamespace(width=lambda: w, height=lambda: h,
                                 x=lambda: 0, y=lambda: 0)
    return types.SimpleNamespace(
        name=lambda: name,
        geometry=lambda: rect,
        availableGeometry=lambda: rect,
        devicePixelRatio=lambda: ratio,
        logicalDotsPerInch=lambda: dpi,
        physicalDotsPerInch=lambda: dpi * ratio,
    )


class TestDescribingScreens:
    def test_it_reports_size_and_scaling_together(self, monkeypatch):
        """Either number alone is useless: Qt lays out in logical pixels and
        the driver allocates in physical ones."""
        screen = _screen("DISPLAY1", 1920, 1080, ratio=2.0, dpi=192.0)
        app = types.SimpleNamespace(screens=lambda: [screen],
                                    primaryScreen=lambda: screen)

        lines = "\n".join(display_info.describe(app))

        assert "1920x1080" in lines
        assert "ratio 2" in lines
        assert "192 logical DPI" in lines
        assert "(primary)" in lines

    def test_every_screen_is_listed(self, monkeypatch):
        first, second = _screen("A", 3840, 2160), _screen("B", 1920, 1080)
        app = types.SimpleNamespace(screens=lambda: [first, second],
                                    primaryScreen=lambda: second)

        lines = display_info.describe(app)

        assert any("3840x2160" in l for l in lines)
        assert any("1920x1080" in l for l in lines)

    def test_the_scale_factor_is_recorded(self, monkeypatch):
        monkeypatch.setenv("QT_SCALE_FACTOR", "0.75")
        app = types.SimpleNamespace(screens=lambda: [], primaryScreen=lambda: None)

        assert any("QT_SCALE_FACTOR=0.75" in l for l in display_info.describe(app))

    def test_an_app_that_will_not_answer_is_not_an_error(self):
        def explode():
            raise RuntimeError("no screens yet")

        app = types.SimpleNamespace(screens=explode, primaryScreen=explode)

        assert "could not read the screens" in display_info.describe(app)[0]


class TestWindowSize:
    def test_it_logs_logical_and_physical(self):
        window = types.SimpleNamespace(
            size=lambda: types.SimpleNamespace(width=lambda: 1920,
                                               height=lambda: 1080),
            devicePixelRatioF=lambda: 2.0,
            isMaximized=lambda: True,
            isFullScreen=lambda: False)
        said = []

        display_info.log_window_size(window, "Main window", said.append)

        assert "1920x1080 logical" in said[0]
        assert "3840x2160 physical" in said[0]
        assert "maximised" in said[0]

    def test_a_window_that_will_not_answer_is_not_an_error(self):
        """This runs from a resize handler. Raising here would turn a
        diagnostic into the crash it exists to explain."""
        def explode():
            raise RuntimeError("gone")

        window = types.SimpleNamespace(size=explode, devicePixelRatioF=explode,
                                       isMaximized=explode, isFullScreen=explode)
        said = []

        display_info.log_window_size(window, "Main window", said.append)

        assert "size unavailable" in said[0]
