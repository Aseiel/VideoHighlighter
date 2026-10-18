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
