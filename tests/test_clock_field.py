"""The player clock that doubles as "go to time"."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from video_ai_editor.clock_field import ClockField


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def field(app):
    f = ClockField()
    f.show()
    f.show_time("47:50 / 48:31")
    yield f
    f.close()


def test_reads_like_the_clock_until_clicked(field):
    assert field.text() == "47:50 / 48:31" and field.isReadOnly()
    QTest.mouseClick(field, Qt.LeftButton)
    assert field.editing() and not field.isReadOnly()
    assert field.text() == "47:50" and field.selectedText() == "47:50"


def test_enter_jumps_to_the_typed_time(field):
    jumps = []
    field.jumpRequested.connect(jumps.append)
    field.start_editing()
    field.setText("1:09:05")
    QTest.keyClick(field, Qt.Key_Return)
    assert jumps == [4145.0]
    assert not field.editing() and field.text() == "47:50 / 48:31"


def test_a_bad_time_stays_open_and_jumps_nowhere(field):
    jumps = []
    field.jumpRequested.connect(jumps.append)
    field.start_editing()
    field.setText("soon")
    QTest.keyClick(field, Qt.Key_Return)
    assert jumps == [] and field.editing()


def test_player_ticks_do_not_overwrite_typing_and_escape_restores(field):
    field.start_editing()
    field.setText("21:3")
    field.show_time("47:51 / 48:31")
    assert field.text() == "21:3"
    QTest.keyClick(field, Qt.Key_Escape)
    assert not field.editing() and field.text() == "47:51 / 48:31"
