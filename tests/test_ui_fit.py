"""Sizing a compact button to the text it has to show.

The bug this replaces was quiet: small buttons carried hard-coded pixel widths
that left no room for the padding the theme adds, so the label was drawn
centred inside a box too narrow for it and lost a few pixels from *each* end.
"Refresh" in a 60px button reads as "efres" — not obviously truncated, just
wrong, which is why it shipped.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QPushButton

from modules.ui import theme
from modules.ui.fit import CHROME_PX, fit_icon_button, fit_width


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    theme.apply(application)
    yield application


class TestChromeMatchesTheStylesheet:
    def test_it_accounts_for_the_padding_the_theme_sets(self):
        # The number here is not independent of the stylesheet — that is the
        # point. A button styled `padding: 6px 14px` with a 1px border spends
        # 30px before a glyph is drawn, and sizing that ignores it clips.
        assert CHROME_PX >= theme.BUTTON_CHROME_H
        assert theme.BUTTON_CHROME_H == 2 * (theme.BUTTON_PADDING_H
                                             + theme.BUTTON_BORDER)

    def test_the_stylesheet_really_uses_that_padding(self):
        # Guards the other direction: someone editing the QSS by hand would
        # otherwise silently desync it from what fit_width assumes.
        qss = theme.build_qss(theme.DARK)
        assert f"padding: 6px {theme.BUTTON_PADDING_H}px;" in qss


class TestFitWidth:
    @pytest.mark.parametrize("label", ["Open", "Refresh", "Labels…", "Save"])
    def test_the_label_fits_with_its_padding(self, app, label):
        button = QPushButton(label)
        width = fit_width(button)
        assert width >= button.fontMetrics().horizontalAdvance(label) + CHROME_PX

    def test_the_labels_that_used_to_clip_now_get_more_room(self, app):
        # The three that were visibly wrong on screen, with the widths they
        # were pinned to.
        for label, old in (("Open", 56), ("Refresh", 60), ("Labels…", 62)):
            assert fit_width(QPushButton(label)) > old, label

    def test_it_sets_a_minimum_not_a_fixed_size(self, app):
        # A layout must stay free to make the button wider; pinning the size is
        # what caused this in the first place.
        button = QPushButton("Open")
        fit_width(button)
        assert button.minimumWidth() > 0
        assert button.maximumWidth() > button.minimumWidth()

    def test_an_explicit_minimum_wins_when_it_is_larger(self, app):
        button = QPushButton("Go")
        assert fit_width(button, minimum=200) == 200

    def test_it_can_measure_text_the_button_does_not_carry_yet(self, app):
        # For buttons whose label changes at runtime: size for the longest.
        button = QPushButton("Stop")
        wide = fit_width(button, "Render Highlight Video")
        assert wide > fit_width(QPushButton("Stop"))

    def test_an_empty_label_does_not_explode(self, app):
        assert fit_width(QPushButton("")) >= CHROME_PX


class TestFitIconButton:
    def test_it_is_square_and_reaches_the_touch_target(self, app):
        button = QPushButton()
        side = fit_icon_button(button)
        assert button.minimumWidth() == side
        assert button.minimumHeight() == side
