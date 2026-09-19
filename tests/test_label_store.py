"""Tests for labelled examples and the COCO dataset built from them.

No video is decoded here. The split arithmetic and the COCO document are pure,
and they are where the mistakes that matter live — a bad split produces a
validation score that flatters every later comparison, and a bad document
produces a dataset that trains happily against the wrong classes.
"""

from __future__ import annotations

import json

import pytest

from modules.vision.label_store import (
    ACCEPTED,
    NEGATIVE,
    PENDING,
    REJECTED,
    LabelStore,
    LabelledBox,
    coco_document,
    from_labeler_export,
    segments,
    split_segments,
)


def _box(video="a.mp4", moment=0.0, name="alpha", verdict=ACCEPTED,
         box=(0.25, 0.25, 0.5, 0.5), source="hand"):
    return LabelledBox(video=video, time=moment, class_name=name,
                       box=box, source=source, verdict=verdict)


# ── the store ────────────────────────────────────────────────────────────

def test_only_accepted_labels_count(tmp_path):
    store = LabelStore(str(tmp_path / "labels.json"))
    store.extend([
        _box(name="alpha"), _box(name="alpha"),
        _box(name="beta", verdict=PENDING),
        _box(name="gamma", verdict=REJECTED),
    ])
    assert store.counts() == {"alpha": 2}
    assert store.class_names() == ["alpha"]
    assert len(store.pending()) == 1


def test_class_order_does_not_move_when_a_proposal_is_rejected(tmp_path):
    """Category ids come from this list, so it must be stable.

    If rejecting one pending proposal reordered the classes, every previously
    exported model's ``labels.json`` would start naming detections wrongly.
    """
    store = LabelStore(str(tmp_path / "labels.json"))
    store.extend([_box(name="zulu"), _box(name="alpha"), _box(name="mike")])
    before = store.class_names()
    store.add(_box(name="beta", verdict=PENDING))
    assert store.class_names() == before == ["alpha", "mike", "zulu"]


def test_a_store_survives_a_round_trip(tmp_path):
    path = tmp_path / "labels.json"
    store = LabelStore(str(path))
    store.add(_box(moment=1.5, source="category", box=(0.1, 0.2, 0.3, 0.4)))
    store.save()

    reloaded = LabelStore(str(path)).load()
    assert len(reloaded.boxes) == 1
    restored = reloaded.boxes[0]
    assert restored.time == 1.5
    assert restored.source == "category"
    assert restored.box == (0.1, 0.2, 0.3, 0.4)


def test_an_unreadable_store_starts_empty_rather_than_raising(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert LabelStore(str(path)).load().boxes == []


def test_boxes_convert_to_pixels_and_are_clamped_to_the_frame():
    inside = _box(box=(0.25, 0.5, 0.25, 0.25))
    assert inside.pixels(400, 200) == (100.0, 100.0, 100.0, 50.0)
    # A box drawn past the edge is trimmed, not left to produce a negative area.
    spilling = _box(box=(0.9, 0.9, 0.5, 0.5))
    x, y, w, h = spilling.pixels(100, 100)
    assert (x + w, y + h) == (100.0, 100.0)


# ── the split ────────────────────────────────────────────────────────────

def test_labels_close_in_time_form_one_segment():
    boxes = [_box(moment=t) for t in (0.0, 1.0, 2.0, 30.0, 31.0)]
    groups = segments(boxes)
    assert [len(g) for g in groups] == [3, 2]


def test_different_videos_never_share_a_segment():
    boxes = [_box(video="a.mp4", moment=0.0), _box(video="b.mp4", moment=0.1)]
    assert len(segments(boxes)) == 2


def test_a_segment_is_never_split_across_train_and_validation():
    """The trap this module exists to avoid.

    Frames a second apart in one shot are near-identical. Scatter them across
    the split and validation scores the model on pictures it trained on: the
    loss falls beautifully and means nothing, and "never promote a worse model"
    compares two fictions.
    """
    boxes = [_box(video="a.mp4", moment=t)
             for t in (0.0, 1.0, 2.0, 60.0, 61.0, 120.0, 121.0, 180.0)]
    train, val = split_segments(segments(boxes), val_fraction=0.5, seed=1)

    def moments(groups):
        return {b.time for g in groups for b in g}

    assert moments(train) & moments(val) == set()
    # And each original run of nearby moments landed wholly on one side.
    for group in segments(boxes):
        times = {b.time for b in group}
        assert times <= moments(train) or times <= moments(val)


def test_the_split_is_deterministic_for_a_seed():
    groups = segments([_box(moment=t * 60) for t in range(10)])
    first = split_segments(groups, seed=3)
    second = split_segments(groups, seed=3)
    assert [len(g) for g in first[1]] == [len(g) for g in second[1]]


def test_training_is_never_left_empty():
    groups = segments([_box(moment=0.0), _box(moment=60.0)])
    train, val = split_segments(groups, val_fraction=0.99)
    assert train and val


def test_too_few_segments_yields_no_validation_rather_than_a_fake_one():
    """An empty val set is honest; one borrowed from training is not.

    Stated against a single group directly, rather than against labels that
    happen to form one: how many groups a set of labels makes depends on
    :func:`auto_span`, and this is a claim about the split, not the grouping.
    """
    train, val = split_segments([[_box(moment=0.0), _box(moment=1.0)]])
    assert len(train) == 1 and val == []


# ── the COCO document ────────────────────────────────────────────────────

def _frames():
    return {("a.mp4", 0.0): ("a_000000000.jpg", 400, 200)}


def test_categories_are_one_based_and_follow_the_class_order():
    doc = coco_document([_box(name="beta")], ["alpha", "beta"], _frames())
    assert doc["categories"] == [{"id": 1, "name": "alpha"},
                                 {"id": 2, "name": "beta"}]
    assert doc["annotations"][0]["category_id"] == 2


def test_annotations_carry_pixel_boxes_and_areas():
    doc = coco_document([_box(box=(0.25, 0.5, 0.25, 0.25))], ["alpha"], _frames())
    annotation = doc["annotations"][0]
    assert annotation["bbox"] == [100.0, 100.0, 100.0, 50.0]
    assert annotation["area"] == 5000.0
    assert annotation["iscrowd"] == 0


def test_negatives_become_images_with_no_annotations():
    """A detector trained only on positives never learns what the class is not."""
    doc = coco_document([_box(verdict=NEGATIVE)], ["alpha"], _frames())
    assert len(doc["images"]) == 1
    assert doc["annotations"] == []


def test_a_degenerate_box_is_dropped_rather_than_trained_on():
    doc = coco_document([_box(box=(0.5, 0.5, 0.0, 0.0))], ["alpha"], _frames())
    assert doc["annotations"] == []


def test_a_class_outside_the_list_is_not_silently_renumbered():
    doc = coco_document([_box(name="unknown")], ["alpha"], _frames())
    assert doc["annotations"] == []


def test_a_box_whose_frame_could_not_be_read_is_skipped():
    doc = coco_document([_box(moment=99.0)], ["alpha"], _frames())
    assert doc["annotations"] == []
    assert len(doc["images"]) == 1


# ── importing the labeller's work ────────────────────────────────────────

def test_labeler_points_become_boxes_centred_on_the_click(tmp_path):
    """The labeller stores points; the box size is an assumption, not a label.

    The construction matches ``training/train_yolox_dataset.py`` exactly, so
    the two import paths cannot disagree about what a labelled point means.
    """
    export = tmp_path / "export.json"
    export.write_text(json.dumps({
        "video": "clip.mp4", "fps": 25.0,
        "frame_width": 1000, "frame_height": 500,
        "keyframes": [{"frame_number": 50, "points": {"alpha": [[500, 250]]}}],
    }), encoding="utf-8")

    boxes = from_labeler_export(str(export), box_fraction=0.10)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.source == "labeler"
    assert box.time == pytest.approx(2.0)          # frame 50 at 25 fps
    # Centred on the click: 0.5 - 0.10/2 = 0.45 on both axes.
    assert box.box == pytest.approx((0.45, 0.45, 0.10, 0.10))


def test_imported_points_arrive_pending_for_review():
    """A fixed box around a click is a proposal about position, not extent."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "e.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"video": "c.mp4", "fps": 30.0,
                       "frame_width": 100, "frame_height": 100,
                       "keyframes": [{"frame_number": 0,
                                      "points": {"alpha": [10, 10]}}]}, fh)
        assert from_labeler_export(path)[0].verdict == PENDING


def test_an_export_without_a_frame_size_is_refused(tmp_path):
    # Without it the points cannot be normalised, and a guessed size would
    # produce boxes that are wrong everywhere but look fine in the JSON.
    export = tmp_path / "e.json"
    export.write_text(json.dumps({"video": "c.mp4", "fps": 30.0,
                                  "keyframes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="frame size"):
        from_labeler_export(str(export))


def test_continuous_labelling_still_yields_more_than_one_segment():
    """The cap that makes a densely-labelled video splittable at all.

    Labels a third of a second apart never exceed the gap, so without a span
    cap they form one segment of hundreds — which `split_segments` refuses to
    break, leaving the run with no validation set and nothing saying so.
    """
    boxes = [_box(moment=i * 0.32) for i in range(200)]   # ~64 seconds, no gaps
    groups = segments(boxes)
    assert len(groups) > 1, "continuous labelling collapsed into one segment"
    # And each group still spans no more than the cap.
    for group in groups:
        assert group[-1].time - group[0].time < 30.0


def test_a_continuously_labelled_video_gets_a_validation_split():
    boxes = [_box(moment=i * 0.32) for i in range(200)]
    train, val = split_segments(segments(boxes), val_fraction=0.25)
    assert train and val
