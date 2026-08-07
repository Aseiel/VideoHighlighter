"""Tests for the per-class object rows in the timeline's Layers panel.

The panel had a nested checkbox list under EVENTS and nothing under OBJECTS, so
hiding one detected class meant opening a modal to do what its neighbour did
inline. That asymmetry reads as the objects group being broken rather than as a
control living elsewhere, which is how it was reported.

Two things here are worth pinning and neither is cosmetic.

`test_composed_events_are_not_listed_as_objects` is the subtle one. Composed
events are stored inside `objects` in the cache, because that is what object
scoring reads. If the object list does not subtract them, every event appears in
both groups, and unticking it in one leaves it drawn by the other — a control
that visibly does nothing.

`test_toggling_drives_the_same_scene_state_as_the_dialog` is the other: the
Advanced dialog and this list are two views of one piece of state, and the
moment either keeps its own copy they disagree about what is hidden.

The window itself is never constructed — it wants a video, a cache and a
waveform. The methods under test only touch a container widget and the scene, so
they are called against a stub holding exactly those.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _Scene:
    """Just the surface the object rows read and write."""

    def __init__(self, classes, cache, visible=None):
        self.object_classes = list(classes)
        self.cache_data = cache
        self.visible_objects = {c: True for c in classes}
        if visible:
            self.visible_objects.update(visible)
        self.filtered = []

    def set_object_filter(self, name, visible):
        self.visible_objects[name] = visible
        self.filtered.append((name, visible))

    def set_all_objects_visible(self, visible):
        for name in self.visible_objects:
            self.visible_objects[name] = visible
        self.filtered.append(("*", visible))


class _Bar:
    def __init__(self):
        self.messages = []

    def showMessage(self, text, _ms=0):
        self.messages.append(text)


class _Stub:
    """A window-shaped object with only what these methods touch."""

    def __init__(self, app, scene):
        from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

        self.signal_scene = scene
        self._object_box = QWidget()
        QVBoxLayout(self._object_box)
        self._object_fold = QPushButton("▾")
        self._object_rows_expanded = True
        self._bar = _Bar()
        self._buttons = QPushButton

    def statusBar(self):
        return self._bar


@pytest.fixture
def bound(app):
    """The real methods, bound to the stub — no window is ever built."""
    from signal_timeline_viewer import SignalTimelineWindow

    def make(classes, cache, visible=None):
        stub = _Stub(app, _Scene(classes, cache, visible))
        for name in ("refresh_object_checkboxes", "_build_object_header",
                     "_apply_object_fold", "_toggle_object_fold",
                     "_toggle_object", "_set_all_object_rows", "_mini_button"):
            setattr(stub, name,
                    getattr(SignalTimelineWindow, name).__get__(stub))
        return stub
    return make


def _cache(rows):
    return {"objects": [{"timestamp": t, "objects": names} for t, names in rows]}


class TestRows:
    def test_a_checkbox_per_detected_class(self, bound):
        stub = bound(["Penis", "Pussy"],
                     _cache([(1, ["Penis"]), (2, ["Penis", "Pussy"])]))
        stub.refresh_object_checkboxes()
        assert set(stub.object_checkboxes) == {"Penis", "Pussy"}

    def test_each_row_carries_how_many_seconds_it_covers(self, bound):
        stub = bound(["Penis"], _cache([(1, ["Penis"]), (2, ["Penis"])]))
        stub.refresh_object_checkboxes()
        assert "(2s)" in stub.object_checkboxes["Penis"].text()

    def test_composed_events_are_not_listed_as_objects(self, bound):
        # They live inside `objects` in the cache because object scoring reads
        # that. Listing them here too would give every event two controls, one
        # of which appears to do nothing.
        stub = bound(["Penis"], _cache([(1, ["Penis", "Double_Vaginal"])]))
        stub.refresh_object_checkboxes()
        assert "Double_Vaginal" not in stub.object_checkboxes

    def test_a_hidden_class_comes_back_unticked(self, bound):
        stub = bound(["Penis", "Pussy"], _cache([(1, ["Penis"])]),
                     visible={"Pussy": False})
        stub.refresh_object_checkboxes()
        assert stub.object_checkboxes["Penis"].isChecked()
        assert not stub.object_checkboxes["Pussy"].isChecked()

    def test_nothing_detected_leaves_no_rows_and_no_caret(self, bound):
        stub = bound([], _cache([]))
        stub.refresh_object_checkboxes()
        assert stub.object_checkboxes == {}
        assert not stub._object_fold.isVisible()

    def test_the_placeholder_class_is_not_offered_as_a_filter(self, bound):
        # `_extract_object_classes` returns ['Unknown'] when it found nothing;
        # that is a placeholder, and a checkbox for it filters nothing.
        stub = bound(["Unknown"], _cache([]))
        stub.refresh_object_checkboxes()
        assert stub.object_checkboxes == {}


class TestToggling:
    def test_toggling_drives_the_same_scene_state_as_the_dialog(self, bound):
        from PySide6.QtCore import Qt

        stub = bound(["Penis"], _cache([(1, ["Penis"])]))
        stub.refresh_object_checkboxes()
        stub.object_checkboxes["Penis"].setChecked(False)
        assert ("Penis", False) in stub.signal_scene.filtered
        assert stub.signal_scene.visible_objects["Penis"] is False

    def test_show_none_hides_every_class_in_one_rebuild(self, bound):
        stub = bound(["Penis", "Pussy"], _cache([(1, ["Penis"])]))
        stub.refresh_object_checkboxes()
        stub._set_all_object_rows(False)
        assert ("*", False) in stub.signal_scene.filtered
        assert not any(cb.isChecked() for cb in stub.object_checkboxes.values())


class TestFold:
    def test_folded_with_everything_shown_says_nothing_extra(self, bound):
        stub = bound(["Penis"], _cache([(1, ["Penis"])]))
        stub.refresh_object_checkboxes()
        stub._toggle_object_fold()
        assert stub._object_fold.text() == "▸"

    def test_folded_with_something_hidden_carries_the_count(self, bound):
        # A folded group that hides a class must not look identical to a
        # complete one, or a filter is invisible from the panel that set it.
        stub = bound(["Penis", "Pussy"], _cache([(1, ["Penis"])]),
                     visible={"Pussy": False})
        stub.refresh_object_checkboxes()
        stub._toggle_object_fold()
        assert stub._object_fold.text() == "▸ 1/2"
