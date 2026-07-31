"""The Search panel's results list survives being refreshed repeatedly.

The empty-state label is long-lived and reused, but it lives inside the layout
that each refresh clears. Deleting it there only *schedules* its destruction,
so it survives the call that removed it and dies on the next turn of the event
loop — and the refresh after that touches a freed C++ object.

That is a crash you cannot reach without an event loop and two refreshes, which
is exactly why it shipped. These drive both.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="Qt not available in this environment")

from PySide6.QtWidgets import QApplication                        # noqa: E402
from PySide6.QtCore import QCoreApplication, QEventLoop           # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qt_app):
    from video_ai_editor.search_panel import SearchPanel

    return SearchPanel(cache_data={}, video_duration=600.0,
                       on_jump=lambda t: None,
                       on_add_clip=lambda s, e: None,
                       video_path=None)


def _settle():
    """Let deleteLater() actually run, which is when the damage would show."""
    QCoreApplication.processEvents(QEventLoop.AllEvents)
    QCoreApplication.sendPostedEvents(None, 52)   # QEvent.DeferredDelete
    QCoreApplication.processEvents(QEventLoop.AllEvents)


class TestRefreshLifecycle:
    def test_empty_then_populated_then_empty(self, panel):
        for segments in ([], [(1.0, 2.0)], [], [(3.0, 4.0), (5.0, 6.0)], []):
            panel._refresh_results(segments)
            _settle()

    def test_the_empty_state_label_is_never_destroyed(self, panel):
        panel._refresh_results([])
        _settle()
        panel._refresh_results([(1.0, 2.0)])
        _settle()
        # Touching it is the operation that used to raise.
        panel._no_results_lbl.hide()
        panel._no_results_lbl.show()

    def test_switching_expressions_repeatedly_does_not_crash(self, panel):
        """The path that surfaced it: pick a label, then another, then another."""
        panel._expr_seconds = {
            10: {"label": "happy", "confidence": 0.9, "faces": 1},
            11: {"label": "happy", "confidence": 0.9, "faces": 1},
            40: {"label": "sad", "confidence": 0.8, "faces": 1},
        }
        labels = [panel._expr_combo.itemData(i)
                  for i in range(panel._expr_combo.count())]
        for label in ("happy", "anger", "sad", "happy", "surprise"):
            panel._expr_combo.setCurrentIndex(labels.index(label))
            _settle()

    def test_rows_are_replaced_not_accumulated(self, panel):
        panel._refresh_results([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
        _settle()
        panel._refresh_results([(7.0, 8.0)])
        _settle()
        # One row plus the trailing stretch.
        assert panel._seg_list_layout.count() == 2
