"""Tests for the per-category action rows in the timeline's Layers panel.

OBJECTS and EVENTS each had a nested checkbox per row; ACTIONS had only the
group checkbox, so one action category could not be hidden without the
Advanced dialog. Same stub approach as test_layer_object_rows: the window is
never built, the real methods are bound to an object holding a container
widget and the scene.
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
    """Just the surface the action rows read and write. The bar counting is
    the real scene's, bound below, so the counts are what the row draws."""

    def __init__(self, names, detections, visible=None, gap=0.5):
        self.merge_threshold = gap
        self.min_action_confidence = 0.0
        self.max_action_confidence = 1.0
        self.action_types = list(names)
        self._detections = detections
        self.visible_actions = {n: True for n in names}
        if visible:
            self.visible_actions.update(visible)
        self.filtered = []

    def _actions_list(self):
        return self._detections

    def set_action_filter(self, name, visible):
        self.visible_actions[name] = visible
        self.filtered.append((name, visible))

    def set_all_actions_visible(self, visible):
        for name in self.visible_actions:
            self.visible_actions[name] = visible
        self.filtered.append(("*", visible))


class _Bar:
    def showMessage(self, _text, _ms=0):
        pass


class _Stub:
    def __init__(self, scene):
        from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

        self.signal_scene = scene
        self._action_box = QWidget()
        QVBoxLayout(self._action_box)
        self._action_fold = QPushButton("▾")
        self._action_rows_expanded = True
        self._bar = _Bar()

    def statusBar(self):
        return self._bar


@pytest.fixture
def bound(app):
    from signal_timeline_viewer import SignalTimelineWindow

    from video_ai_editor.signal_timeline import SignalTimelineScene

    def make(names, detections, visible=None, gap=0.5):
        scene = _Scene(names, detections, visible, gap)
        for name in ("action_bar_counts", "_merge_intervals", "_action_confidence_ok"):
            setattr(scene, name, getattr(SignalTimelineScene, name).__get__(scene))
        scene._action_intervals = SignalTimelineScene._action_intervals   # static
        stub = _Stub(scene)
        for name in ("refresh_action_checkboxes", "_build_action_header",
                     "_apply_action_fold", "_toggle_action_fold",
                     "_toggle_action", "_set_all_action_rows", "_mini_button"):
            setattr(stub, name,
                    getattr(SignalTimelineWindow, name).__get__(stub))
        return stub
    return make


def _dets(rows):
    return [{"timestamp": t, "action_name": n, "confidence": 0.5} for t, n in rows]


class TestRows:
    def test_a_checkbox_per_category(self, bound):
        stub = bound(["Category A", "Category B"],
                     _dets([(1.0, "category a"), (2.0, "category b")]))
        stub.refresh_action_checkboxes()
        assert set(stub.action_checkboxes) == {"Category A", "Category B"}

    def test_counts_bars_at_the_current_gap(self, bound):
        # 1.0 and 1.5 touch, 4.2 is 2.2s after the first bar ends: two bars.
        stub = bound(["Category A"],
                     _dets([(1.0, "Category A"), (1.5, "Category A"), (4.2, "Category A")]))
        stub.refresh_action_checkboxes()
        assert stub.action_checkboxes["Category A"].text() == "Category A (2)"

    def test_a_wider_gap_merges_into_fewer_bars(self, bound):
        dets = _dets([(1.0, "Category A"), (3.5, "Category A"), (6.0, "Category A")])
        narrow = bound(["Category A"], dets, gap=0.5)
        narrow.refresh_action_checkboxes()
        wide = bound(["Category A"], dets, gap=2.5)
        wide.refresh_action_checkboxes()
        assert narrow.action_checkboxes["Category A"].text() == "Category A (3)"
        assert wide.action_checkboxes["Category A"].text() == "Category A (1)"

    def test_merge_off_counts_every_detection(self, bound):
        stub = bound(["Category A"],
                     _dets([(1.0, "Category A"), (1.2, "Category A")]), gap=0.0)
        stub.refresh_action_checkboxes()
        assert stub.action_checkboxes["Category A"].text() == "Category A (2)"

    def test_a_hidden_category_still_says_what_it_would_show(self, bound):
        stub = bound(["Category A"], _dets([(1.0, "Category A")]),
                     visible={"Category A": False})
        stub.refresh_action_checkboxes()
        assert stub.action_checkboxes["Category A"].text() == "Category A (1)"

    def test_a_hidden_category_comes_back_unticked(self, bound):
        stub = bound(["Category A", "Category B"], _dets([(1.0, "Category A")]),
                     visible={"Category B": False})
        stub.refresh_action_checkboxes()
        assert stub.action_checkboxes["Category A"].isChecked()
        assert not stub.action_checkboxes["Category B"].isChecked()

    def test_the_placeholder_is_not_offered_as_a_filter(self, bound):
        stub = bound(["Unknown"], [])
        stub.refresh_action_checkboxes()
        assert stub.action_checkboxes == {}
        assert not stub._action_fold.isVisible()


class TestToggling:
    def test_toggling_drives_the_scene_state(self, bound):
        stub = bound(["Category A"], _dets([(1.0, "Category A")]))
        stub.refresh_action_checkboxes()
        stub.action_checkboxes["Category A"].setChecked(False)
        assert stub.signal_scene.visible_actions["Category A"] is False

    def test_show_none_hides_every_category_in_one_rebuild(self, bound):
        stub = bound(["Category A", "Category B"], _dets([(1.0, "Category A")]))
        stub.refresh_action_checkboxes()
        stub._set_all_action_rows(False)
        assert ("*", False) in stub.signal_scene.filtered
        assert not any(cb.isChecked() for cb in stub.action_checkboxes.values())


class TestFold:
    def test_folded_with_something_hidden_carries_the_count(self, bound):
        stub = bound(["Category A", "Category B"], _dets([(1.0, "Category A")]),
                     visible={"Category B": False})
        stub.refresh_action_checkboxes()
        stub._toggle_action_fold()
        assert stub._action_fold.text() == "▸ 1/2"
