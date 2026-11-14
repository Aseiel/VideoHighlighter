"""What a long film's shot lane costs the timeline while it plays.

A 97-minute film has over a thousand shots, and the timeline draws each as a
bar with a text label. Zoomed out to the whole film, two things made every
playback tick expensive enough to freeze the window on a slow laptop:

- each label was drawn from glyph outlines, squeezed a hundredfold sideways
  into an unreadable smear (BarLabel now skips that), and
- moving the playhead listed every item in the scene just to check the
  playhead was still one of them (SignalTimelineScene._holds asks the item).
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QImage, QPainter, QTransform, QColor
from PySide6.QtWidgets import QApplication, QGraphicsScene, QStyleOptionGraphicsItem

from video_ai_editor.timeline_bars import BarLabel
from video_ai_editor.signal_timeline import SignalTimelineScene


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _cache(shots):
    duration = shots * 5.0
    return {
        "video_metadata": {"duration": duration, "fps": 24},
        "scenes": [{"start": i * 5.0, "end": (i + 1) * 5.0} for i in range(shots)],
    }


def _ink(label, transform) -> int:
    """How many pixels painting `label` under `transform` actually changes."""
    img = QImage(400, 60, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0))
    p = QPainter(img)
    p.setWorldTransform(transform)
    option = QStyleOptionGraphicsItem()
    option.exposedRect = label.boundingRect()
    label.paint(p, option, None)
    p.end()
    black = QColor(0, 0, 0).rgba()
    return sum(1 for y in range(img.height()) for x in range(img.width())
               if img.pixel(x, y) != black)


class TestBarLabel:
    def test_unscaled_text_is_legible(self):
        assert BarLabel.legible(QTransform())

    def test_squeezed_past_half_is_not(self):
        assert not BarLabel.legible(QTransform.fromScale(0.4, 1.0))
        assert not BarLabel.legible(QTransform.fromScale(0.0067, 1.0))

    def test_half_squeeze_and_stretch_still_draw(self):
        assert BarLabel.legible(QTransform.fromScale(0.5, 1.0))
        assert BarLabel.legible(QTransform.fromScale(8.0, 1.0))

    def test_a_squeezed_label_paints_nothing(self, app):
        label = BarLabel("Shot 601")
        label.setDefaultTextColor(QColor(255, 255, 255))
        assert _ink(label, QTransform()) > 0
        assert _ink(label, QTransform.fromScale(0.05, 1.0)) == 0

    def test_shot_bars_carry_bar_labels(self, app):
        scene = SignalTimelineScene(_cache(40), 200.0)
        labels = [i for i in scene.items() if isinstance(i, BarLabel)]
        assert labels, "the shot lane drew no labels at all"


class TestHolds:
    def test_an_item_in_the_scene(self, app):
        scene = SignalTimelineScene(_cache(3), 15.0)
        item = scene.addRect(0, 0, 1, 1)
        assert scene._holds(item)

    def test_none_and_another_scenes_item(self, app):
        scene = SignalTimelineScene(_cache(3), 15.0)
        other = QGraphicsScene()
        assert not scene._holds(None)
        assert not scene._holds(other.addRect(0, 0, 1, 1))

    def test_an_item_a_rebuild_deleted(self, app):
        scene = SignalTimelineScene(_cache(3), 15.0)
        item = scene.addRect(0, 0, 1, 1)
        scene.clear()
        assert not scene._holds(item)

    def test_the_playhead_survives_a_rebuild(self, app):
        scene = SignalTimelineScene(_cache(30), 150.0)
        scene.set_current_time(10.0)
        first = scene.current_time_line
        assert scene._holds(first)
        scene.build_timeline()
        scene.set_current_time(20.0)
        assert scene._holds(scene.current_time_line)
        assert scene.current_time_line.line().x1() == pytest.approx(
            20.0 * scene.pixels_per_second)
