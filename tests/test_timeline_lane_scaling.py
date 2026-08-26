"""How tall the signal timeline is, and how far it may be squashed.

Two separate things used to go wrong once enough signals were switched on, and
together they made the timeline unreadable exactly when it had the most to say.

The scene reserved its height from a budget computed ahead of the drawing: one
lane per *track name*, meaning one per object class, per action, per query. The
drawing collapses each of those groups into a single lane, so a run
with a dozen detected classes claimed several hundred scene pixels that nothing
was ever drawn in. The view then scaled that mostly-empty scene to fit, and the
lanes — the part anyone is looking at — were squashed into a strip at the top
with the ruler stranded far below them.

And the squashing had no floor. The row names are painted in the gutter at a
fixed font size, so they do not shrink with the lanes they label; past a certain
pitch they simply overlap into an unreadable stack.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from video_ai_editor.signal_timeline import SignalTimelineScene, SignalTimelineView


DURATION = 600.0


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def cache_with(classes=1, actions=1):
    """A cache whose OBJECTS group holds `classes` distinct classes."""
    names = [f"class_{i}" for i in range(classes)]
    acts = [f"act_{i}" for i in range(actions)]
    return {
        "objects": [{"timestamp": float(t), "objects": [names[t % len(names)]]}
                    for t in range(300)],
        "actions": [{"timestamp": float(t) * 2, "action_name": acts[t % len(acts)],
                     "confidence": 0.8} for t in range(150)],
        "scenes": [{"start": t * 20.0, "end": t * 20.0 + 15} for t in range(20)],
        "motion_events": [t * 3.0 for t in range(100)],
        "motion_peaks": [t * 5.0 for t in range(80)],
        "audio_peaks": [t * 4.0 for t in range(90)],
        "transcript": {"segments": [{"start": t * 6.0, "end": t * 6.0 + 4, "text": "a"}
                                    for t in range(80)]},
    }


def build(app, **kwargs):
    scene = SignalTimelineScene(cache_data=cache_with(**kwargs),
                                video_duration=DURATION)
    return scene


class TestTheSceneIsAsTallAsWhatWasDrawn:
    def test_more_classes_in_a_group_do_not_make_the_scene_taller(self, app):
        """The regression. OBJECTS is one lane whether it holds 1 class or 20."""
        one = build(app, classes=1)
        many = build(app, classes=20)

        assert [n for n, _ in many.row_labels] == [n for n, _ in one.row_labels]
        assert many.sceneRect().height() == one.sceneRect().height()

    def test_more_action_types_do_not_either(self, app):
        one = build(app, actions=1)
        many = build(app, actions=12)
        assert many.sceneRect().height() == one.sceneRect().height()

    def test_the_last_lane_ends_near_the_bottom(self, app):
        """No empty band under the content — only the ruler's own strip."""
        scene = build(app, classes=20)
        last_top = max(y for _, y in scene.row_labels)
        slack = scene.sceneRect().height() - last_top
        # The filmstrip is the bottom lane and the tallest; anything much beyond
        # it plus the ruler band is reserved-but-never-drawn space.
        assert slack < 120

    def test_the_ruler_gets_its_strip(self, app):
        """The bottom band is reserved, so labels are not painted over a lane."""
        scene = build(app)
        assert scene.sceneRect().height() > max(y for _, y in scene.row_labels)

    def test_hiding_a_layer_gives_its_height_back(self, app):
        scene = build(app)
        tall = scene.sceneRect().height()
        scene.visible_layers['motion_peaks'] = False
        scene.build_timeline()
        assert scene.sceneRect().height() < tall


def fitted(app, scene, view_height):
    """A view of `scene`, that tall, with the vertical fit applied."""
    view = SignalTimelineView(scene)
    view.setFixedSize(1200, view_height)   # resize() alone does not stick offscreen
    app.processEvents()
    view._fit_vertical()
    return view


class TestSquashingHasAFloor:
    def test_a_short_viewport_does_not_squash_lanes_into_a_smear(self, app):
        scene = build(app, classes=20)
        view = fitted(app, scene, 80)      # far shorter than the lanes need

        unclamped = view.viewport().height() / scene.sceneRect().height()
        pitch_scale = SignalTimelineView.MIN_LANE_PITCH_PX / (
            scene.layer_height + scene.layer_spacing)
        assert unclamped < pitch_scale, "viewport was not short enough to test this"

        pitch = (scene.layer_height + scene.layer_spacing) * view.transform().m22()
        assert pitch >= SignalTimelineView.MIN_LANE_PITCH_PX - 0.01

    def test_the_scene_then_scrolls_instead(self, app):
        scene = build(app, classes=20)
        view = fitted(app, scene, 80)

        on_screen = scene.sceneRect().height() * view.transform().m22()
        assert on_screen > view.viewport().height()
        assert view.verticalScrollBar().maximum() > 0

    def test_a_tall_viewport_still_stretches_to_fill(self, app):
        """The floor is a floor, not a fixed scale — fitting still happens."""
        scene = build(app)
        short = fitted(app, scene, 400).transform().m22()
        tall = fitted(app, scene, 1400).transform().m22()
        assert tall > short
        assert tall > 1.0
