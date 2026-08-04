"""Tests for `modules.level_by_class` — comparing level between labelled classes.

Synthetic levels and abstract class names throughout: the module is told what
the labels are and holds no opinion about them, so the tests use ``class_a`` /
``class_b`` and would read identically for any subject matter.

The behaviour that matters most here is the *refusal*: a comparison that cannot
be resolved must say so rather than rank on noise.
"""

from __future__ import annotations

import numpy as np

from modules.level_by_class import (
    annotate,
    classes_at,
    peak_in_range,
    summarise,
)


def _labels(spans):
    """{class_name: [(start, end), ...]} -> {second: [names]}"""
    out: dict = {}
    for name, ranges in spans.items():
        for start, end in ranges:
            for sec in range(start, end + 1):
                out.setdefault(sec, []).append(name)
    return out


def test_describes_each_class_that_has_enough_seconds():
    levels = np.full(600, -30.0)
    levels[100:160] = -20.0          # class_a is 10 dB louder
    labels = _labels({"class_a": [(100, 159)], "class_b": [(300, 359)]})
    out = summarise(levels, labels)
    names = {c["name"]: c for c in out["classes"]}
    assert names["class_a"]["median_dbfs"] == -20.0
    assert names["class_b"]["median_dbfs"] == -30.0
    assert names["class_a"]["seconds"] == 60


def test_a_class_with_too_few_seconds_is_not_described():
    levels = np.full(600, -30.0)
    labels = _labels({"class_a": [(100, 159)], "class_b": [(300, 302)]})
    out = summarise(levels, labels)
    assert [c["name"] for c in out["classes"]] == ["class_a"]
    # Nothing to compare a lone class against.
    assert out["comparison"] is None


def test_a_large_consistent_difference_is_reported_as_resolvable():
    rng = np.random.default_rng(0)
    levels = np.full(1200, -35.0) + rng.normal(0, 0.5, 1200)
    labels: dict = {}
    # Ten interleaved stretches so each has nearby control material.
    for i in range(10):
        a0 = 60 + i * 100
        b0 = a0 + 40
        levels[a0:a0 + 20] += 9.0          # class_a clearly louder
        for sec in range(a0, a0 + 20):
            labels.setdefault(sec, []).append("class_a")
        for sec in range(b0, b0 + 20):
            labels.setdefault(sec, []).append("class_b")
    out = summarise(levels, labels)
    comp = out["comparison"]
    assert comp is not None
    assert comp["resolvable"] is True
    assert comp["louder"] == "class_a"
    assert comp["median_difference_db"] > 7.0
    assert "Louder during class_a" in comp["headline"]


def test_a_difference_smaller_than_the_noise_is_refused():
    """The measured case: a ~1 dB effect against ~5 dB of scatter.

    The module must say it cannot resolve this, and must not rank the classes
    as though it had.
    """
    rng = np.random.default_rng(1)
    levels = np.full(1200, -35.0)
    labels: dict = {}
    for i in range(10):
        a0 = 60 + i * 100
        b0 = a0 + 40
        # Each stretch sits at its own level -- passage noise far exceeding any
        # class difference, which is what the paired design has to survive.
        passage = rng.normal(0, 5.0)
        levels[a0:a0 + 20] += passage + 1.3
        levels[b0:b0 + 20] += passage
        for sec in range(a0, a0 + 20):
            labels.setdefault(sec, []).append("class_a")
        for sec in range(b0, b0 + 20):
            labels.setdefault(sec, []).append("class_b")
    # Scatter the paired differences so the sign test cannot clear 5%.
    for i in range(0, 10, 2):
        a0 = 60 + i * 100
        levels[a0:a0 + 20] -= 6.0
    out = summarise(levels, labels)
    comp = out["comparison"]
    assert comp is not None
    assert comp["resolvable"] is False
    assert "No resolvable level difference" in comp["headline"]
    # The refusal must carry the number that justifies it.
    assert comp["min_detectable_db"] > abs(comp["median_difference_db"])
    assert "cannot measure it" in comp["detail"]


def test_pairing_ignores_stretches_with_no_nearby_control():
    levels = np.full(4000, -30.0)
    labels = _labels({"class_a": [(100, 160)], "class_b": [(3000, 3060)]})
    out = summarise(levels, labels)
    # Both are described, but they are 48 minutes apart so nothing pairs.
    assert len(out["classes"]) == 2
    assert out["comparison"] is None


def test_classes_at_and_annotate_report_facts_not_inferences():
    labels = _labels({"class_a": [(100, 120)], "class_b": [(110, 130)]})
    assert classes_at(labels, 105) == ["class_a"]
    assert classes_at(labels, 115) == ["class_a", "class_b"]
    assert classes_at(labels, 500) == []

    events = [{"start": 100, "end": 130, "peak_second": 105, "z": 3.0}]
    out = annotate(events, labels)
    assert out[0]["classes_at_peak"] == ["class_a"]
    assert out[0]["classes_spanning"] == ["class_a", "class_b"]
    # The original event data survives untouched.
    assert out[0]["z"] == 3.0


def test_empty_input_is_not_an_error():
    assert summarise([], {}) == {"classes": [], "comparison": None,
                                 "loudest": None}
    assert annotate([], {}) == []


def test_loudest_labelled_second_is_reported_without_statistics():
    """One measurement and one set of labels — no aggregate, no significance."""
    levels = np.full(600, -30.0)
    levels[142] = -8.0                      # the loudest labelled second
    levels[500] = -4.0                      # louder, but carries no label
    labels = _labels({"class_a": [(100, 160)], "class_b": [(300, 360)]})
    out = summarise(levels, labels)
    assert out["loudest"]["second"] == 142
    assert out["loudest"]["timestamp"] == "2:22"
    assert out["loudest"]["level_dbfs"] == -8.0
    assert out["loudest"]["classes"] == ["class_a"]


def test_peak_in_range_is_about_the_clip_not_the_video():
    levels = np.full(600, -30.0)
    levels[120] = -12.0        # loudest inside the clip
    levels[400] = -4.0         # louder, but outside it
    labels = _labels({"class_a": [(100, 160)]})
    out = peak_in_range(levels, labels, 100, 160, video_median=-30.0)
    assert out["second"] == 120
    assert out["timestamp"] == "2:00"
    assert out["level_dbfs"] == -12.0
    assert out["classes"] == ["class_a"]
    # Quoted against the video so the number survives a change of mastering.
    assert out["vs_video_db"] == 18.0


def test_peak_in_range_reports_an_unlabelled_peak_honestly():
    levels = np.full(600, -30.0)
    levels[500] = -10.0
    out = peak_in_range(levels, {}, 480, 520)
    assert out["second"] == 500
    assert out["classes"] == []       # nothing labelled -> say nothing, not guess


def test_peak_in_range_clamps_to_the_available_levels():
    levels = np.full(100, -30.0)
    assert peak_in_range(levels, {}, 90, 500)["second"] < 100
    assert peak_in_range([], {}, 0, 10) is None
    assert peak_in_range(levels, {}, 500, 600) is None


def test_restricting_to_named_classes_excludes_the_rest():
    levels = np.full(600, -30.0)
    labels = _labels({"class_a": [(100, 160)], "raw_part": [(100, 160)]})
    out = summarise(levels, labels, classes=["class_a"])
    assert [c["name"] for c in out["classes"]] == ["class_a"]
