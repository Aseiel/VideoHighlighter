"""Guard the signal types the Search panel's scan depends on.

A face scan is keyed by second — integers. Qt's `Signal(dict)` marshals through
QVariantMap, which is keyed by string, so an int-keyed dict crosses the
connection as an empty one. The scan then succeeds, logs its findings, and the
panel reports that nothing was found: no exception, no failed signal, just a
silent hole between a worker and its receiver.

That shipped once. These tests exist so it cannot ship twice.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="Qt not available in this environment")

from PySide6.QtCore import QObject, Signal          # noqa: E402
from PySide6.QtWidgets import QApplication          # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _scan():
    return {10: {"label": "happy", "confidence": 0.9, "faces": 1},
            3641: {"label": "sad", "confidence": 0.7, "faces": 2}}


class TestSignalMarshalling:
    def test_a_dict_signal_silently_drops_integer_keys(self, qt_app):
        """The behaviour being guarded against, asserted so the reason is visible."""
        class Sender(QObject):
            sig = Signal(dict)

        received = []
        sender = Sender()
        sender.sig.connect(received.append)
        sender.sig.emit(_scan())
        assert received == [{}], "if this ever passes through, Qt changed"

    def test_an_object_signal_carries_the_scan_intact(self, qt_app):
        class Sender(QObject):
            sig = Signal(object)

        received = []
        sender = Sender()
        sender.sig.connect(received.append)
        sender.sig.emit(_scan())
        assert received == [_scan()]


class TestWorkerSignals:
    def test_the_scan_worker_emits_its_result_as_an_object(self, qt_app):
        """A scan keyed by second must survive the thread boundary."""
        from video_ai_editor.search_panel import _ExpressionScanWorker

        received = []
        worker = _ExpressionScanWorker("nonexistent.mp4")
        worker.done.connect(received.append)
        worker.done.emit(_scan())
        assert received == [_scan()]

    def test_a_scan_that_found_nothing_still_arrives(self, qt_app):
        from video_ai_editor.search_panel import _ExpressionScanWorker

        received = []
        worker = _ExpressionScanWorker("nonexistent.mp4")
        worker.done.connect(received.append)
        worker.done.emit({})
        assert received == [{}]
