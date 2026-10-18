"""Motion strength on the timeline: the curve the motion pass keeps, and the
rows that draw it behind their bars.

Every motion peak used to be the same orange box, so the timeline could say
*where* the peaks were but never which was the big one. The pass now keeps its
per-frame levels, binned, and the rows rank and shade their bars from them —
while staying the same draggable boxes they were.
"""
from __future__ import annotations

import pytest

from modules.segments import motion_curve as mc

FPS = 30.0


def samples(levels_by_second, per_second=6):
    """Detector-shaped rows: (frame index, level, scene id)."""
    rows = []
    for sec, level in enumerate(levels_by_second):
        for k in range(per_second):
            frame = int((sec + k / per_second) * FPS)
            rows.append((frame, level if k == 0 else level * 0.1, 0))
    return rows


class TestBuildCurve:
    def test_bins_keep_the_spike_not_the_average(self):
        # One sampled frame at 100, the rest of its second near zero. A mean
        # would report ~17; the spike is the point.
        curve = mc.build_curve([(0, 100.0, 0), (5, 1.0, 0), (10, 1.0, 0)],
                               FPS, duration=1.0)
        assert curve["values"][0] == 1.0

    def test_normalised_to_the_videos_own_maximum(self):
        curve = mc.build_curve(samples([10, 40, 20]), FPS, duration=3.0)
        assert max(curve["values"]) == 1.0
        assert curve["step"] == mc.STEP
        assert len(curve["values"]) == 6

    def test_covers_the_whole_duration_even_when_the_end_is_still(self):
        curve = mc.build_curve(samples([5]), FPS, duration=10.0)
        assert len(curve["values"]) == 20

    def test_scene_cuts_are_left_out(self):
        # A cut changes the whole picture at once; it is not the video's
        # biggest movement.
        rows = [(0, 1.0), (15, 50.0), (30, 1.0)]
        curve = mc.build_curve(rows, FPS, duration=1.5, exclude_frames={15})
        assert curve["values"] == [1.0, 0.0, 1.0]

    def test_nothing_to_bin_is_no_curve_rather_than_a_flat_one(self):
        assert mc.build_curve([], FPS, duration=10.0) is None
        assert mc.build_curve(samples([5]), 0, duration=10.0) is None


class TestStrength:
    curve = {"step": 0.5, "values": [0.1, 0.1, 1.0, 0.1, 0.5, 0.1, 0.1, 0.1]}

    def test_is_measured_against_the_videos_average(self):
        avg = sum(self.curve["values"]) / 8
        assert mc.strength(self.curve, 1.0, 1.5) == pytest.approx(1.0 / avg)

    def test_a_merged_bar_takes_its_strongest_moment(self):
        assert (mc.strength(self.curve, 0.0, 2.5)
                == pytest.approx(mc.strength(self.curve, 1.0, 1.5)))

    def test_bar_ending_on_a_bin_edge_does_not_borrow_the_next_bin(self):
        # [0.5, 1.0) is bin 1 only; bin 2 holds the spike.
        assert mc.strength(self.curve, 0.5, 1.0) < 1.0

    def test_no_curve_or_no_motion_has_no_strength(self):
        assert mc.strength(None, 1.0, 2.0) is None
        assert mc.strength({"step": 0.5, "values": [0.0, 0.0]}, 0.0, 1.0) is None


class TestPeakTime:
    curve = {"step": 0.5, "values": [0.1, 0.4, 0.9, 0.2]}

    def test_is_the_middle_of_the_strongest_bin_in_the_window(self):
        assert mc.peak_time(self.curve, 0.0, 2.0) == 1.25
        assert mc.level_at(self.curve, 1.25) == 0.9

    def test_without_a_curve_falls_back_to_the_start(self):
        assert mc.peak_time(None, 3.0, 4.0) == 3.0
        assert mc.level_at(None, 3.0) == 0.0


class TestRanks:
    def test_strongest_first_and_capped(self):
        assert mc.ranks([1.0, 5.0, 3.0, 4.0], 2) == {1: 1, 3: 2}

    def test_unknown_strengths_are_never_ranked(self):
        assert mc.ranks([None, 2.0, None], 5) == {1: 1}

    def test_ties_keep_timeline_order(self):
        assert mc.ranks([2.0, 2.0], 2) == {0: 1, 1: 2}


# --------------------------------------------------------------------------- #
# The timeline rows
# --------------------------------------------------------------------------- #
DURATION = 60.0
PEAKS = [5.0, 20.0, 35.0, 50.0]


def curve_with_spikes(spikes):
    values = [0.05] * int(DURATION / mc.STEP)
    for t, v in spikes.items():
        values[int(t / mc.STEP)] = v
    return {"step": mc.STEP, "values": values}


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def peak_scene(app, curve):
    from video_ai_editor.signal_timeline import SignalTimelineScene
    cache = {"motion_peaks": PEAKS, "motion_events": []}
    if curve is not None:
        cache["motion_curve"] = curve
    scene = SignalTimelineScene(cache_data=cache, video_duration=DURATION)
    scene.merge_threshold = 0
    scene.build_timeline()
    return scene


def peak_bars(scene):
    seen, out = set(), []
    for bar in scene.bars:
        if 'Peak' in str(bar.label) and id(bar) not in seen:
            seen.add(id(bar))
            out.append(bar)
    return sorted(out, key=lambda b: b.start_time)


def text_items(scene):
    from PySide6.QtWidgets import QGraphicsTextItem
    return [i for i in scene.items() if isinstance(i, QGraphicsTextItem)]


class TestTimelineRows:
    def test_strongest_peak_ranks_first_and_is_drawn_most_opaque(self, app):
        scene = peak_scene(app, curve_with_spikes({5.0: 0.3, 20.0: 1.0,
                                                   35.0: 0.6, 50.0: 0.2}))
        bars = peak_bars(scene)
        assert [b.start_time for b in bars] == PEAKS
        assert bars[1].metadata["rank"] == "1 of 4"
        assert bars[2].metadata["rank"] == "2 of 4"
        alphas = [b.get_alpha() for b in bars]
        assert alphas[1] == max(alphas) and alphas[3] == min(alphas)
        assert "video average" in bars[1].metadata["strength"]
        # The rank is drawn on the curve, not carried in the label, which goes
        # with the bar into the edit timeline.
        assert not bars[1].label.startswith("★")
        assert any(i.toPlainText().startswith("★1 ")
                   for i in text_items(scene))

    def test_the_burst_just_before_a_peak_counts_towards_it(self, app):
        # The detector stamps the peak at the end of its burst.
        scene = peak_scene(app, curve_with_spikes({18.5: 1.0, 35.0: 0.6}))
        bars = peak_bars(scene)
        assert bars[1].metadata["rank"] == "1 of 4"

    def test_bars_keep_their_labels(self, app):
        # A label parented through the constructor was deleted with its
        # Python wrapper on PySide 6.10, so no bar on the timeline had one.
        scene = peak_scene(app, curve_with_spikes({20.0: 1.0}))
        bar = peak_bars(scene)[1]
        kids = [k.toPlainText() for k in bar.graphics_rect.childItems()]
        assert kids and kids[0]

    def test_envelope_takes_no_clicks_and_bars_stay_grabbable(self, app):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QGraphicsPathItem
        from video_ai_editor.signal_timeline import SignalTimelineView

        scene = peak_scene(app, curve_with_spikes({20.0: 1.0}))
        paths = [i for i in scene.items() if isinstance(i, QGraphicsPathItem)
                 and i.zValue() in (-4, 5)]
        assert {p.zValue() for p in paths} == {-4, 5}, "fill behind, outline above"
        assert all(p.acceptedMouseButtons() == Qt.MouseButton.NoButton
                   for p in paths)

        # Under the outline and the peak marker, the view still finds the bar
        # -- which is what lets a press there start the bar's drag.
        view = SignalTimelineView(scene)
        view.resize(1200, 600)
        bar = peak_bars(scene)[1]
        item = bar.graphics_rect
        centre = item.mapToScene(item.rect().center())
        assert any(i.zValue() > 0 and i.acceptedMouseButtons() == Qt.MouseButton.NoButton
                   for i in scene.items(centre)), "nothing drawn over the bar"
        assert view._bar_at(view.mapFromScene(centre)) is item

    def test_a_cache_without_a_curve_draws_as_it_always_did(self, app):
        from PySide6.QtWidgets import QGraphicsPathItem

        scene = peak_scene(app, None)
        bars = peak_bars(scene)
        assert [b.label for b in bars] == ["Peak"] * len(PEAKS)
        assert all(b.alpha is None for b in bars)
        assert not [i for i in scene.items() if isinstance(i, QGraphicsPathItem)
                    and i.zValue() in (-4, 5)]
