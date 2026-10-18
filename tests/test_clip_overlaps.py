"""Edit-timeline clips that cover the same source footage.

When two clips share source time, that footage plays twice in the cut. The
edit timeline marks such clips and, when a clip that was just added overlaps,
asks once whether to merge them, trim the new one, or keep both.
"""
from __future__ import annotations

import pytest

from video_ai_editor import clip_overlaps as co

# The pair from the report: clip 2 lies wholly inside clip 3.
CLIP_2 = (978.4, 991.6)
CLIP_3 = (963.2, 994.9)


class TestFinding:
    def test_shared_time(self):
        assert co.shared(CLIP_2, CLIP_3) == pytest.approx(13.2)
        assert co.shared((0, 5), (5, 9)) == 0.0

    def test_halves_of_a_cut_are_not_an_overlap(self):
        assert co.pairs([(10.0, 20.0), (20.0, 30.0)]) == []

    def test_pairs_and_map(self):
        clips = [(0, 10), CLIP_2, (100, 110), CLIP_3]
        assert [(i, j) for i, j, _s in co.pairs(clips)] == [(1, 3)]
        assert set(co.by_clip(clips)) == {1, 3}

    def test_groups_are_connected_runs(self):
        clips = [(0, 10), (8, 20), (18, 30), (100, 110), (105, 115)]
        assert co.groups(clips, co.pairs(clips)) == [[0, 1, 2], [3, 4]]


class TestResolving:
    def test_merge_spans_the_group_at_its_first_position(self):
        clips = [(0, 10), CLIP_2, (100, 110), CLIP_3]
        out = co.merge(clips, [[1, 3]])
        assert out == [(0, 10), (963.2, 994.9), (100, 110)]

    def test_trim_drops_a_clip_wholly_inside_another(self):
        assert co.trim([CLIP_3, CLIP_2], [1]) == [CLIP_3]

    def test_trim_keeps_the_footage_around_another(self):
        out = co.trim([CLIP_2, CLIP_3], [1])
        assert out == [CLIP_2, (963.2, 978.4), (991.6, 994.9)]

    def test_trim_never_leaves_a_sliver(self):
        assert co.trim([(0, 10), (0.2, 10)], [1]) == [(0, 10)]

    def test_two_new_clips_overlapping_each_other_keep_it_once(self):
        out = co.trim([(0, 10), (5, 15)], [0, 1])
        assert sum(e - s for s, e in out) == pytest.approx(15.0)


# --------------------------------------------------------------------------- #
# The edit scene
# --------------------------------------------------------------------------- #
@pytest.fixture()
def scene(monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import video_ai_editor.edit_timeline as et

    def no_cache(*_a, **_k):
        raise RuntimeError("thumbnails not needed here")

    monkeypatch.setattr(et, "ThumbnailCache", no_cache)
    s = et.EditTimelineScene("missing.mp4", 1200.0)
    found = []
    s.overlaps_found.connect(lambda pairs: found.append(list(pairs)))
    s.found = found
    yield s
    s.cleanup()


def settle():
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()


class TestScene:
    def test_adding_an_overlapping_clip_asks_once(self, scene):
        scene.add_clip(*CLIP_2)
        settle()
        assert scene.found == []
        scene.add_clip(*CLIP_3)
        settle()
        assert len(scene.found) == 1
        (i, j, s), = scene.found[0]
        assert (i, j) == (0, 1) and s == pytest.approx(13.2)
        assert set(scene.overlap_map) == {0, 1}

    def test_a_batch_of_adds_asks_once(self, scene):
        for clip in [(0, 10), (5, 15), (12, 20), (100, 110)]:
            scene.add_clip(*clip)
        settle()
        assert len(scene.found) == 1
        assert len(scene.found[0]) == 2

    def test_merge(self, scene):
        scene.add_clip(*CLIP_2)
        scene.add_clip(*CLIP_3)
        settle()
        scene.merge_new_overlaps()
        assert scene.clips == [(963.2, 994.9)]
        assert scene.found[-1] == []
        assert scene.overlap_map == {}

    def test_trim_new_clip(self, scene):
        scene.add_clip(*CLIP_2)
        settle()
        scene.add_clip(*CLIP_3)
        settle()
        scene.trim_new_overlaps()
        assert scene.clips == [CLIP_2, (963.2, 978.4), (991.6, 994.9)]
        assert scene.found[-1] == []

    def test_keep_both_is_remembered(self, scene):
        scene.add_clip(*CLIP_2)
        scene.add_clip(*CLIP_3)
        settle()
        scene.keep_new_overlaps()
        assert scene.found[-1] == []
        # Still marked on the clips, and not asked about again on the next add.
        assert set(scene.overlap_map) == {0, 1}
        scene.add_clip(500, 510)
        settle()
        assert scene.found[-1] == []

    def test_deleting_the_overlap_closes_the_notice(self, scene):
        scene.add_clip(*CLIP_2)
        scene.add_clip(*CLIP_3)
        settle()
        scene.clips.pop(1)
        scene.build_timeline()
        assert scene.found[-1] == []

    def test_right_click_merge(self, scene):
        scene.add_clip(0, 10)
        scene.add_clip(*CLIP_2)
        scene.add_clip(*CLIP_3)
        settle()
        scene.merge_overlaps_of(2)
        assert scene.clips == [(0, 10), (963.2, 994.9)]

    def test_a_loaded_edit_is_marked_but_not_asked_about(self, monkeypatch):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import video_ai_editor.edit_timeline as et

        monkeypatch.setattr(et, "ThumbnailCache",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
        monkeypatch.setattr(et.EditTimelineScene, "load_initial_clips",
                            lambda self: self.clips.extend([CLIP_2, CLIP_3]))
        s = et.EditTimelineScene("missing.mp4", 1200.0)
        found = []
        s.overlaps_found.connect(found.append)
        settle()
        assert found == []
        assert set(s.overlap_map) == {0, 1}
        s.cleanup()


class TestUndo:
    """Ctrl+Z on the edit timeline. Recorded where every edit ends
    (build_timeline), so each kind of edit is covered without its own code."""

    def test_undo_a_merge(self, scene):
        scene.add_clip(*CLIP_2)
        scene.add_clip(*CLIP_3)
        settle()
        scene.merge_new_overlaps()
        settle()
        assert scene.clips == [CLIP_3]
        assert scene.undo()
        assert scene.clips == [CLIP_2, CLIP_3]
        assert set(scene.overlap_map) == {0, 1}

    def test_a_burst_of_adds_is_one_step(self, scene):
        scene.add_clip(0, 10)
        settle()
        for clip in [(20, 30), (40, 50), (60, 70)]:
            scene.add_clip(*clip)
        settle()
        scene.undo()
        assert scene.clips == [(0, 10)]

    def test_redo_and_a_new_edit_clears_it(self, scene):
        scene.add_clip(0, 10)
        settle()
        scene.add_clip(20, 30)
        settle()
        scene.undo()
        assert scene.redo()
        assert scene.clips == [(0, 10), (20, 30)]
        scene.undo()
        scene.add_clip(40, 50)
        settle()
        assert not scene.can_redo()

    def test_each_kind_of_edit_is_undoable(self, scene):
        scene.add_clip(0, 10)
        scene.add_clip(20, 30)
        settle()
        for change in (lambda: scene.cut_clip_at(5.0),
                       lambda: scene.trim_clip_end(0, 4.0),
                       lambda: scene.reorder_clip(0, 10_000)):
            before = list(scene.clips)
            change()
            settle()
            assert scene.clips != before
            scene.undo()
            assert scene.clips == before

    def test_nothing_to_undo_after_load(self, scene):
        assert not scene.can_undo()
        assert scene.undo() is False

    def test_window_leaves_edit_playback_before_stepping(self):
        import signal_timeline_viewer as stv

        calls = []

        class Win:
            _edit_playback_active = True

            def _leave_edit_playback(self):
                calls.append("leave")

            def statusBar(self):
                class Bar:
                    def showMessage(self, *a):
                        calls.append("msg")
                return Bar()

        stv.SignalTimelineWindow._step_edit_history(Win(), lambda: True, "x")
        assert calls == ["leave"]
        Win._edit_playback_active = False
        stv.SignalTimelineWindow._step_edit_history(Win(), lambda: False, "x")
        assert calls == ["leave", "msg"]


class TestReorder:
    """Dragging a clip: its left edge past a neighbour's centre moves it past
    that neighbour, measured on the layout as it is while dragging."""

    def lefts(self, scene):
        return [item.pos().x() for item in scene.clip_items]

    def test_one_place_right(self, scene):
        for clip in [(0, 4), (10, 14), (20, 24)]:      # 4 s = 200 px wide each
            scene.add_clip(*clip)
        settle()
        b_left = self.lefts(scene)[1]
        scene.reorder_clip(0, b_left + 120)            # past B's centre (+100)
        assert scene.clips == [(10, 14), (0, 4), (20, 24)]

    def test_to_the_end(self, scene):
        for clip in [(0, 4), (10, 14), (20, 24)]:
            scene.add_clip(*clip)
        settle()
        scene.reorder_clip(0, 10_000)
        assert scene.clips == [(10, 14), (20, 24), (0, 4)]

    def test_short_of_the_centre_stays(self, scene):
        for clip in [(0, 4), (10, 14), (20, 24)]:
            scene.add_clip(*clip)
        settle()
        b_left = self.lefts(scene)[1]
        scene.reorder_clip(0, b_left + 60)             # not yet past B's centre
        assert scene.clips == [(0, 4), (10, 14), (20, 24)]

    def test_one_place_left(self, scene):
        for clip in [(0, 4), (10, 14), (20, 24)]:
            scene.add_clip(*clip)
        settle()
        b_left = self.lefts(scene)[1]
        scene.reorder_clip(2, b_left + 80)             # left of B's centre
        assert scene.clips == [(0, 4), (20, 24), (10, 14)]
