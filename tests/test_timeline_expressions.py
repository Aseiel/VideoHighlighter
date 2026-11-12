"""The EXPRESSIONS row: the face scan on the signal timeline.

The scan has existed for a while — it scores a signal, feeds the rules and fills
a section of the report — and none of that could be *looked at*. A per-second
reading you cannot lay beside the waveform and the footage is one you have to
take on trust, and this classifier is exactly the kind that should not be.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from modules.vision import face_scan
from video_ai_editor.signal_timeline import SignalTimelineScene

DURATION = 120.0


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def reading(label, confidence=0.9):
    return {"label": label, "confidence": confidence, "faces": 1}


SCAN = {
    **{s: reading("neutral", 0.6) for s in range(0, 10)},
    10: reading("happy", 0.7), 11: reading("happy", 0.95), 12: reading("happy"),
    # One unread second inside a held expression is not a cut.
    14: reading("happy", 0.8),
    60: reading("happy"),
    90: reading("surprise", 0.99),
}


def scene_with(app, scan):
    scene = SignalTimelineScene(cache_data={}, video_duration=DURATION)
    scene.set_expressions(scan)
    return scene


def bars(scene, label=None):
    return [b for b in scene.bars
            if (b.metadata or {}).get("type") == "expression"
            and (label is None or b.metadata["query"] == label)]


def test_no_scan_draws_no_row(app):
    scene = SignalTimelineScene(cache_data={}, video_duration=DURATION)
    assert "EXPRESSIONS" not in [name for name, _y in scene.row_labels]
    assert not scene.layer_has_data("expressions")


def test_a_held_expression_is_one_bar(app):
    scene = scene_with(app, SCAN)
    assert "EXPRESSIONS" in [name for name, _y in scene.row_labels]
    assert scene.layer_has_data("expressions")

    happy = sorted(bars(scene, "happy"), key=lambda b: b.start_time)
    assert [(b.start_time, b.end_time) for b in happy] == [(10.0, 15.0),
                                                           (60.0, 61.0)]
    # The bar carries the strongest reading inside it, not the first.
    assert happy[0].metadata["max_confidence"] == pytest.approx(0.95)
    assert happy[0].metadata["merged_count"] == 4


def test_each_class_found_gets_its_own_lane(app):
    scene = scene_with(app, SCAN)
    lanes = {label: {b.y_position for b in bars(scene, label)}
             for label in ("neutral", "happy", "surprise")}
    assert all(len(ys) == 1 for ys in lanes.values())
    assert len({next(iter(ys)) for ys in lanes.values()}) == 3


def test_arrows_step_through_what_a_face_shows_not_through_neutral(app):
    scene = scene_with(app, SCAN)
    assert scene._nav_timestamps_expressions() == [10.0, 60.0, 90.0]
    # With nothing but neutral there is still something to step through.
    only_neutral = scene_with(app, {s: reading("neutral") for s in (5, 50)})
    assert only_neutral._nav_timestamps_expressions() == [5.0, 50.0]


def test_the_scan_survives_a_cache_reload(app):
    """It is not in cache_data, so re-ingesting the cache must not drop it."""
    scene = scene_with(app, SCAN)
    scene.reload_cache_data({"audio_peaks": [1.0]})
    assert bars(scene, "surprise")


def test_the_row_reads_the_scan_saved_beside_the_cache(app, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = str(tmp_path / "clip.mp4")
    face_scan.save(SCAN, face_scan.cache_path_for(video), video_path=video)
    scene = SignalTimelineScene(cache_data={}, video_duration=DURATION,
                                video_path=video)
    assert len(bars(scene, "happy")) == 2


def test_a_stopped_scan_is_not_saved_as_the_whole_answer(tmp_path):
    video = str(tmp_path / "clip.mp4")

    class Detector:
        def detect_faces(self, frame):
            return []

    class Classifier:
        def load(self):
            return True

        def classify(self, crops):
            return []

    def frames(*a, **k):
        return iter(())

    import modules.vision.face_scan as fs
    original = fs.iter_video_frames
    fs.iter_video_frames = frames
    try:
        fs.scan_video(video, cache_dir=str(tmp_path), detector=Detector(),
                      classifier=Classifier(), cancel_fn=lambda: True,
                      log_fn=lambda _m: None)
        assert not (tmp_path / "clip_faces.json").exists()
        fs.scan_video(video, cache_dir=str(tmp_path), detector=Detector(),
                      classifier=Classifier(), log_fn=lambda _m: None)
        assert (tmp_path / "clip_faces.json").exists()
    finally:
        fs.iter_video_frames = original
