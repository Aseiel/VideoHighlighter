"""Outlines in composition rules: where a box says the wrong thing.

Each case is one a bounding box gets wrong and an outline gets right, run
through the real engine, plus the guarantee everything else rests on: a rule
over plain boxes answers exactly as it always did.
"""
from __future__ import annotations

import textwrap

import pytest

from modules.rules import shapes
from video_ai_editor.composition_engine import CompositionEngine


def _engine(tmp_path, rule: str):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(f"""
        events:
          - name: ev
            label: Ev
            window_secs: 0
            persist_secs: 0
            rules:
              - {rule}
        """), encoding="utf-8")
    return CompositionEngine(str(path))


def _frames(dets, n=3):
    """dets: list of (class, box, contour-or-None)."""
    return [{
        "timestamp": float(i),
        "objects": [d[0] for d in dets],
        "bboxes": [d[1] for d in dets],
        "confidences": [0.9] * len(dets),
        "contours": [d[2] for d in dets],
    } for i in range(n)]


def _fires(engine, frames) -> bool:
    events, _ = engine.run(frames)
    return any("ev" in names for names in events.values())


# An L-shaped outline: an arm reaching right from a body on the left. Its box
# is the whole square; most of the square is empty.
REACH_BOX = [0.0, 0.0, 0.6, 0.6]
REACH = [[0.0, 0.0], [0.2, 0.0], [0.2, 0.4], [0.6, 0.4], [0.6, 0.6], [0.0, 0.6]]
# A small thing in the empty corner of that box: top right.
THING_BOX = [0.45, 0.05, 0.1, 0.1]


def test_boxes_alone_answer_as_they_always_did(tmp_path):
    engine = _engine(tmp_path, "{source: thing, region: holder}")
    frames = _frames([("holder", REACH_BOX, None), ("thing", THING_BOX, None)])
    assert _fires(engine, frames)          # centre in the box: the old answer


def test_an_outline_stops_the_empty_part_of_a_box_counting(tmp_path):
    engine = _engine(tmp_path, "{source: thing, region: holder, outline: true}")
    frames = _frames([("holder", REACH_BOX, REACH), ("thing", THING_BOX, None)])
    assert not _fires(engine, frames)      # the corner is outside the outline


def test_overlaps_measures_area_not_centres(tmp_path):
    # A thing two-thirds inside the arm's outline.
    thing = [0.3, 0.35, 0.15, 0.15]
    engine = _engine(tmp_path, "{source: thing, region: holder, relation: overlaps, "
                               "min_overlap: 0.6, outline: true}")
    assert _fires(engine, _frames([("holder", REACH_BOX, REACH), ("thing", thing, None)]))
    strict = _engine(tmp_path, "{source: thing, region: holder, relation: overlaps, "
                               "min_overlap: 0.9, outline: true}")
    assert not _fires(strict, _frames([("holder", REACH_BOX, REACH), ("thing", thing, None)]))


def test_diagonal_things_whose_boxes_overlap_do_not_touch(tmp_path):
    # Two thin diagonal bars, parallel, apart: their boxes overlap heavily.
    a = [[0.10, 0.50], [0.50, 0.10], [0.52, 0.12], [0.12, 0.52]]
    b = [[0.30, 0.70], [0.70, 0.30], [0.72, 0.32], [0.32, 0.72]]
    engine = _engine(tmp_path, "{source: a, region: b, relation: touches, outline: true}")
    boxes_only = _frames([("a", [0.1, 0.1, 0.42, 0.42], None),
                          ("b", [0.3, 0.3, 0.42, 0.42], None)])
    outlines = _frames([("a", [0.1, 0.1, 0.42, 0.42], a),
                        ("b", [0.3, 0.3, 0.42, 0.42], b)])
    assert _fires(engine, boxes_only)      # the boxes do meet...
    assert not _fires(engine, outlines)    # ...the things never do
    near = _engine(tmp_path, "{source: a, region: b, relation: touches, max_gap: 0.3, "
                             "outline: true}")
    assert _fires(near, outlines)


def test_a_misspelt_relation_fails_at_load(tmp_path):
    with pytest.raises(ValueError, match="relation"):
        _engine(tmp_path, "{source: a, region: b, relation: inisde}")


def test_matched_outlines_reach_the_overlay(tmp_path):
    engine = _engine(tmp_path, "{source: thing, region: holder, relation: overlaps, "
                               "min_overlap: 0.1, outline: true}")
    _, overlay = engine.run(_frames([("holder", REACH_BOX, REACH),
                                     ("thing", [0.3, 0.35, 0.15, 0.15], None)]))
    assert overlay and overlay[0]["event_contours"][0][0] == REACH


# --- the geometry itself ---------------------------------------------------------

def test_area_centroid_and_containment():
    square = shapes.box_polygon([0, 0, 1, 1])
    assert shapes.area(square) == pytest.approx(1.0)
    assert shapes.centroid(square) == pytest.approx((0.5, 0.5))
    assert shapes.contains(square, (1.0, 0.5))          # on the edge counts
    assert not shapes.contains(square, (1.01, 0.5))
    assert shapes.centroid([(0, 0), (1, 1), (2, 2)]) == pytest.approx((1, 1))


def test_overlap_fraction_is_accurate_at_any_scale():
    for scale in (1.0, 0.01):
        src = shapes.box_polygon([0, 0, 1 * scale, 1 * scale])
        region = shapes.box_polygon([0.5 * scale, 0, 1 * scale, 1 * scale])
        assert shapes.overlap_fraction(src, region) == pytest.approx(0.5, abs=0.02)


def test_gap_between_shapes():
    a = shapes.box_polygon([0, 0, 0.1, 0.1])
    b = shapes.box_polygon([0.3, 0, 0.1, 0.1])
    assert shapes.gap(a, b) == pytest.approx(0.2)
    assert shapes.gap(a, shapes.box_polygon([0.05, 0.05, 0.1, 0.1])) == 0.0
    assert shapes.gap(shapes.box_polygon([0, 0, 1, 1]), a) == 0.0   # one inside the other


def test_simplify_keeps_the_shape_and_drops_the_noise():
    import math
    circle = [(0.5 + 0.3 * math.cos(t / 100 * 2 * math.pi),
               0.5 + 0.3 * math.sin(t / 100 * 2 * math.pi)) for t in range(100)]
    small = shapes.simplify(circle, 0.005)
    assert 8 <= len(small) <= 40          # from 100
    assert shapes.area(small) == pytest.approx(shapes.area(circle), rel=0.05)


# --- the Advanced tab's save must not drop what it has no column for ------------

def test_a_table_save_keeps_rule_fields_it_has_no_column_for():
    from modules.rules.rules_file import carry_rule_fields, top_level_fields

    original = [
        {"source": "a", "region": "b", "min_count": 1, "relation": "touches",
         "max_gap": 0.02, "outline": True},
        {"source": "a", "region": "b", "min_count": 2, "relation": "overlaps"},
        {"source": "c", "region": "d", "outline": True},
    ]
    rebuilt = [
        {"source": "a", "region": "b", "min_count": 3, "max_count": 999},   # edited count
        {"source": "a", "region": "b", "min_count": 2, "max_count": 999},
        {"source": "c", "region": "e", "min_count": 1, "max_count": 999},   # new region
    ]
    out = carry_rule_fields(original, rebuilt)
    assert out[0] == {"source": "a", "region": "b", "min_count": 3, "max_count": 999,
                      "relation": "touches", "max_gap": 0.02, "outline": True}
    assert out[1]["relation"] == "overlaps"
    assert "outline" not in out[2]                  # a different rule now
    assert top_level_fields({"outliner": "sam", "events": []}) == {"outliner": "sam"}


def test_a_column_set_back_to_its_default_stays_default():
    from modules.rules.rules_file import TABLE_RULE_KEYS, carry_rule_fields

    original = [{"source": "a", "region": "b", "relation": "touches", "outline": True,
                 "max_gap": 0.02}]
    # The Advanced tab owns relation/outline and writes only non-defaults: a
    # rule switched back to inside / no outline arrives without the keys.
    rebuilt = [{"source": "a", "region": "b", "min_count": 1, "max_count": 999}]
    out = carry_rule_fields(original, rebuilt,
                            owned=TABLE_RULE_KEYS | {"relation", "outline"})
    assert out == [{"source": "a", "region": "b", "min_count": 1, "max_count": 999,
                    "max_gap": 0.02}]


def test_outlines_in_the_cache_only_change_rules_that_asked_for_them(tmp_path):
    frames = _frames([("holder", REACH_BOX, REACH), ("thing", THING_BOX, None)])
    boxes_rule = _engine(tmp_path, "{source: thing, region: holder}")
    outline_rule = _engine(tmp_path, "{source: thing, region: holder, outline: true}")
    assert _fires(boxes_rule, frames)          # the box answer, outlines or not
    assert not _fires(outline_rule, frames)
