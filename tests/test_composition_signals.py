"""
Tests for signal conditions in `video_ai_editor.composition_engine`.

The engine could only ever express spatial relations — "class A inside class B".
Several of the measurements the app already makes are one value per second with
no box to be inside anything (audio level, vocal brightness, an expression
reading), so a combination that spanned both kinds could not be written down at
all and had to be correlated by hand after the fact.

These tests pin the three things that make the addition safe to rely on: that a
signal condition genuinely gates a spatial rule rather than decorating it, that
a rule made only of signal conditions fires on a video with no detections, and
that a missing measurement never counts as a satisfied one.
"""

from __future__ import annotations

import textwrap

import pytest

from video_ai_editor.composition_engine import CompositionEngine


def _rules(tmp_path, body: str):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return CompositionEngine(str(path))


def _frames(n=8, together=True, start=0.0):
    """Cache-shaped frames; `handle` sits inside `tool` when together."""
    out = []
    for i in range(n):
        handle = [0.30, 0.30, 0.05, 0.05] if together else [0.90, 0.90, 0.05, 0.05]
        out.append({
            "timestamp": start + i * 0.5,
            "objects": ["tool", "handle"],
            "bboxes": [[0.25, 0.25, 0.30, 0.30], handle],
            "confidences": [0.9, 0.8],
        })
    return out


# --------------------------------------------------------------- gating

SPATIAL_PLUS_SIGNAL = """
    events:
      - name: gated
        label: Gated
        window_secs: 0.75
        persist_secs: 0.5
        signals:
          - {signal: effort, min: 1.0}
        rules:
          - {source: handle, region: tool, min_count: 1}
    """


def test_signal_condition_blocks_a_satisfied_spatial_rule(tmp_path):
    """The spatial half holds throughout; the signal is what decides."""
    engine = _rules(tmp_path, SPATIAL_PLUS_SIGNAL)

    quiet, _ = engine.run(_frames(), {"effort": [0.0] * 10})
    assert quiet == {}, "spatial rule fired although its signal was below min"

    loud, _ = engine.run(_frames(), {"effort": [2.0] * 10})
    assert loud, "spatial rule did not fire although its signal was satisfied"
    assert "gated" in loud[0]


def test_signal_gates_second_by_second(tmp_path):
    """A curve that only rises for part of the run fires only for that part.

    The trailing second is the majority-vote window doing its job, not the
    signal leaking: at ts=3.0 the 0.75 s window still holds the frame at 2.5,
    which fired, and half the window is enough. Pinned here because it is the
    behaviour a user will see — a signal condition inherits the same smoothing
    a spatial one has always had, so an event outlasts its cause by up to
    ``window_secs`` either way.
    """
    engine = _rules(tmp_path, SPATIAL_PLUS_SIGNAL)
    # 8 frames at 0.5 s covers seconds 0-3; make only second 2 loud.
    events, _ = engine.run(_frames(), {"effort": [0.0, 0.0, 5.0, 0.0]})
    assert set(events) == {2, 3}


def test_smoothing_can_be_turned_off(tmp_path):
    """With no window there is no spill, and gating is exact."""
    engine = _rules(tmp_path, """
        events:
          - name: gated
            label: Gated
            window_secs: 0.0
            signals:
              - {signal: effort, min: 1.0}
            rules:
              - {source: handle, region: tool, min_count: 1}
        """)
    events, _ = engine.run(_frames(), {"effort": [0.0, 0.0, 5.0, 0.0]})
    assert set(events) == {2}


def test_omitting_signals_entirely_is_the_old_behaviour(tmp_path):
    """A spatial-only rule set must not care that the feature now exists."""
    engine = _rules(tmp_path, """
        events:
          - name: held
            label: Held
            rules:
              - {source: handle, region: tool, min_count: 1}
        """)
    with_none, _ = engine.run(_frames())
    with_empty, _ = engine.run(_frames(), {})
    assert with_none == with_empty
    assert with_none, "spatial-only rule stopped firing"


# ------------------------------------------------------- signal-only rules

SIGNAL_ONLY = """
    events:
      - name: combined
        label: Combined
        signals:
          - {signal: effort, min: 1.5}
          - {signal: reading, equals: surprise}
    """


def test_signal_only_rule_fires_without_any_detections(tmp_path):
    """The whole point: no boxes exist, and the rule still has an answer."""
    engine = _rules(tmp_path, SIGNAL_ONLY)
    events, boxes = engine.run([], {
        "effort": [0.0, 2.0, 2.0, 0.2],
        "reading": {1: "surprise", 2: "neutral"},
    })
    assert set(events) == {1}, "expected only the second where both held"
    assert events[1] == ["combined"]
    assert [b["timestamp"] for b in boxes] == [1.0]

    # Marked across the whole frame. A signal-only event is a statement about
    # the moment, not about a thing in it: with no box nothing draws and the
    # event is invisible during playback, while a box somewhere in particular
    # would claim a location the rule never established.
    assert boxes[0]["bboxes"] == [[0.0, 0.0, 1.0, 1.0]]
    # And no confidence: the rule held or it did not. A number here would go
    # into the field detectors write their certainty into and be read as one.
    assert boxes[0]["confidences"] == []


def test_both_conditions_are_required(tmp_path):
    engine = _rules(tmp_path, SIGNAL_ONLY)
    only_audio, _ = engine.run([], {"effort": [2.0, 2.0], "reading": {}})
    assert only_audio == {}
    only_label, _ = engine.run([], {"effort": [0.0, 0.0],
                                    "reading": {0: "surprise", 1: "surprise"}})
    assert only_label == {}


def test_a_missing_measurement_is_not_a_satisfied_one(tmp_path):
    """A signal the caller never supplied must not fire the rule.

    The failure this guards against is silent and total: a rule referring to a
    signal that no longer exists (renamed, or its detector did not run) would
    otherwise fire on every second of every video.
    """
    engine = _rules(tmp_path, SIGNAL_ONLY)
    events, _ = engine.run([], {"reading": {0: "surprise", 1: "surprise"}})
    assert events == {}

    # Present but short: seconds past the end of the curve have no value.
    events, _ = engine.run([], {
        "effort": [2.0],
        "reading": {0: "surprise", 1: "surprise", 2: "surprise"},
    })
    assert set(events) == {0}


# ------------------------------------------------------------- conditions

@pytest.mark.parametrize("condition,value,expected", [
    ("{signal: s, min: 1.0}", 1.0, True),
    ("{signal: s, min: 1.0}", 0.99, False),
    ("{signal: s, max: 1.0}", 1.0, True),
    ("{signal: s, max: 1.0}", 1.01, False),
    ("{signal: s, min: 1.0, max: 2.0}", 1.5, True),
    ("{signal: s, min: 1.0, max: 2.0}", 2.5, False),
    ("{signal: s, equals: surprise}", "surprise", True),
    # Labels compare case- and whitespace-insensitively: they come from model
    # vocabularies and hand-typed rules, and the two spell things differently.
    ("{signal: s, equals: surprise}", "Surprise", True),
    ("{signal: s, equals: surprise}", " SURPRISE ", True),
    ("{signal: s, equals: surprise}", "neutral", False),
    ("{signal: s, any_of: [surprise, anger]}", "anger", True),
    ("{signal: s, any_of: [surprise, anger]}", "happy", False),
])
def test_condition_boundaries(tmp_path, condition, value, expected):
    engine = _rules(tmp_path, f"""
        events:
          - name: e
            label: E
            signals:
              - {condition}
        """)
    events, _ = engine.run([], {"s": [value]})
    assert (events != {}) is expected


def test_a_numeric_bound_never_matches_a_label(tmp_path):
    """A threshold applied to a word must fail, not raise."""
    engine = _rules(tmp_path, """
        events:
          - name: e
            label: E
            signals:
              - {signal: s, min: 1.0}
        """)
    events, _ = engine.run([], {"s": ["surprise"]})
    assert events == {}


def test_string_keyed_signal_mapping(tmp_path):
    """A mapping that has been through JSON has string keys, and still works."""
    engine = _rules(tmp_path, """
        events:
          - name: e
            label: E
            signals:
              - {signal: reading, equals: surprise}
        """)
    events, _ = engine.run([], {"reading": {"3": "surprise"}})
    assert set(events) == {3}


def test_event_names_include_signal_only_rules(tmp_path):
    """Stripping a previous pass works by name, so the name has to be listed."""
    engine = _rules(tmp_path, SIGNAL_ONLY)
    assert engine.event_names == ["combined"]


# ------------------------------------------------------- sustained_secs

SUSTAINED = """
    events:
      - name: held
        label: Held
        signals:
          - {signal: s, min: 1.0, sustained_secs: 4}
    """


def test_sustained_rejects_a_spike_and_accepts_a_run(tmp_path):
    """The distinction the modifier exists for: a shape, not an instant."""
    engine = _rules(tmp_path, SUSTAINED)
    # one spike, then a four-second run
    events, _ = engine.run([], {"s": [5.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0]})
    assert set(events) == {3, 4, 5, 6}


def test_sustained_marks_the_whole_run_not_only_its_tail(tmp_path):
    """An event that began once the requirement was met would start four
    seconds after the thing it reports, and be cut from the wrong place."""
    engine = _rules(tmp_path, SUSTAINED)
    events, _ = engine.run([], {"s": [2.0] * 6})
    assert min(events) == 0


def test_sustained_needs_consecutive_seconds(tmp_path):
    """Four qualifying seconds broken by a gap are not a four-second run."""
    engine = _rules(tmp_path, SUSTAINED)
    events, _ = engine.run([], {"s": [2.0, 2.0, 0.0, 2.0, 2.0, 0.0]})
    assert events == {}


def test_sustained_run_at_the_very_end_still_counts(tmp_path):
    """The run is closed by the end of the signal, not only by a falling edge."""
    engine = _rules(tmp_path, SUSTAINED)
    events, _ = engine.run([], {"s": [0.0, 2.0, 2.0, 2.0, 2.0]})
    assert set(events) == {1, 2, 3, 4}


# ---------------------------------------------------------- within_secs

def test_within_secs_tolerates_signals_that_do_not_coincide(tmp_path):
    """Two signals sampled by different means rarely land on one second.

    The audio curve is dense and the expression reading is sparse; requiring
    them to agree exactly discards real co-occurrences over a detail of
    sampling, which is what this modifier is for.
    """
    engine = _rules(tmp_path, """
        events:
          - name: near
            label: Near
            signals:
              - {signal: effort, min: 1.0}
              - {signal: reading, equals: surprise, within_secs: 3}
        """)
    signals = {"effort": [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
               "reading": {2: "surprise"}}
    strict, _ = engine.run([], {**signals, "reading": {4: "surprise"}})
    assert set(strict) == {4}, "sanity: exact coincidence should fire"

    loose, _ = engine.run([], signals)
    assert set(loose) == {4}, "a reading 2s away should still satisfy the rule"


def test_within_secs_has_a_limit(tmp_path):
    engine = _rules(tmp_path, """
        events:
          - name: near
            label: Near
            signals:
              - {signal: effort, min: 1.0}
              - {signal: reading, equals: surprise, within_secs: 2}
        """)
    events, _ = engine.run([], {"effort": [0.0] * 9 + [5.0],
                                "reading": {2: "surprise"}})
    assert events == {}, "a reading 7s away should not satisfy a 2s window"


def test_sustained_and_within_compose(tmp_path):
    """Sustained is applied first, then the tolerance is grown around the run."""
    engine = _rules(tmp_path, """
        events:
          - name: shape
            label: Shape
            signals:
              - {signal: s, min: 1.0, sustained_secs: 3, within_secs: 1}
        """)
    # A lone spike must not be rescued by the tolerance window.
    spike, _ = engine.run([], {"s": [0.0, 5.0, 0.0, 0.0, 0.0, 0.0]})
    assert spike == {}
    # A qualifying run is widened by one second either side.
    run, _ = engine.run([], {"s": [0.0, 2.0, 2.0, 2.0, 0.0, 0.0]})
    assert set(run) == {0, 1, 2, 3, 4}


# --------------------------------------------------------------- enabled

def test_disabled_rule_does_not_fire(tmp_path):
    engine = _rules(tmp_path, """
        events:
          - name: off_rule
            label: Off
            enabled: false
            signals:
              - {signal: s, min: 1.0}
          - name: on_rule
            label: On
            signals:
              - {signal: s, min: 1.0}
        """)
    events, _ = engine.run([], {"s": [5.0, 5.0]})
    assert events, "the enabled rule should still fire"
    assert all("off_rule" not in names for names in events.values())
    assert all("on_rule" in names for names in events.values())


def test_disabled_rule_still_reports_its_name(tmp_path):
    """Off means "do not evaluate", not "forget about".

    `event_names` drives the strip of a previous pass. If a disabled rule
    dropped off that list, its last output would stay in the cache for ever and
    look exactly like the rule still running.
    """
    engine = _rules(tmp_path, """
        events:
          - name: off_rule
            label: Off
            enabled: false
            signals:
              - {signal: s, min: 1.0}
        """)
    assert engine.event_names == ["off_rule"]


def test_enabled_defaults_to_true(tmp_path):
    """Rule files written before the flag existed must keep working."""
    engine = _rules(tmp_path, """
        events:
          - name: legacy
            label: Legacy
            signals:
              - {signal: s, min: 1.0}
        """)
    events, _ = engine.run([], {"s": [5.0]})
    assert set(events) == {0}


# ----------------------------------------------------- min_duration_secs

MIN_DURATION = """
    events:
      - name: long_enough
        label: Long enough
        min_duration_secs: 4
        signals:
          - {signal: s, min: 1.0}
    """


def test_short_events_are_discarded_whole(tmp_path):
    """A brief flicker is not the thing being looked for.

    Distinct from `sustained_secs`, which constrains one measurement: with
    several conditions ANDed, the overlap where all of them hold is routinely
    shorter than any of them individually.
    """
    engine = _rules(tmp_path, MIN_DURATION)
    # 2s burst, then a 5s one
    events, _ = engine.run([], {"s": [0, 5, 5, 0, 0, 5, 5, 5, 5, 5, 0]})
    assert set(events) == {5, 6, 7, 8, 9}, "the 2s burst should have gone"


def test_a_run_is_measured_the_way_the_timeline_draws_it(tmp_path):
    """First second to last plus one, so a lone second is 1s and not 0."""
    engine = _rules(tmp_path, """
        events:
          - name: e
            label: E
            min_duration_secs: 1
            signals:
              - {signal: s, min: 1.0}
        """)
    events, _ = engine.run([], {"s": [0, 5, 0]})
    assert set(events) == {1}, "a single qualifying second should survive min 1"


def test_a_gap_splits_one_long_event_into_two_short_ones(tmp_path):
    """Two 2s bursts either side of a gap must not add up to 4s."""
    engine = _rules(tmp_path, MIN_DURATION)
    events, _ = engine.run([], {"s": [5, 5, 0, 0, 0, 0, 5, 5, 0]})
    assert events == {}


def test_min_duration_defaults_to_off(tmp_path):
    """Rule files written before the field existed keep their behaviour."""
    engine = _rules(tmp_path, """
        events:
          - name: e
            label: E
            signals:
              - {signal: s, min: 1.0}
        """)
    events, _ = engine.run([], {"s": [0, 5, 0]})
    assert set(events) == {1}


def test_short_events_lose_their_overlay_box_too(tmp_path):
    """A discarded event must not leave a marker drawn on the frame."""
    engine = _rules(tmp_path, MIN_DURATION)
    events, boxes = engine.run([], {"s": [0, 5, 5, 0, 0, 0]})
    assert events == {}
    assert boxes == [], f"a dropped event still emitted {len(boxes)} box(es)"


# ---------------------------------------------------- ignore_edges_secs

EDGE_GUARD = """
    events:
      - name: guarded
        label: Guarded
        ignore_edges_secs: 5
        signals:
          - {signal: s, min: 1.0}
    """


def test_edges_are_dropped(tmp_path):
    """Opening and closing material is titles and music beds, not content."""
    engine = _rules(tmp_path, EDGE_GUARD)
    # fires at 0-1 (start), 10-11 (middle), 18-19 (end) of a 20s signal
    s = [5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 5, 5]
    events, _ = engine.run([], {"s": s})
    assert set(events) == {10, 11}, f"edges survived: {sorted(events)}"


def test_edge_guard_uses_the_duration_it_is_given(tmp_path):
    """A longer file moves the closing guard, so the same second survives."""
    engine = _rules(tmp_path, EDGE_GUARD)
    s = [0] * 18 + [5, 5]
    near_end, _ = engine.run([], {"s": s})
    assert near_end == {}, "should be inside the closing guard of a 20s file"
    mid, _ = engine.run([], {"s": s}, duration=200.0)
    assert set(mid) == {18, 19}, "same seconds are mid-file in a 200s file"


def test_edge_guard_defaults_to_off(tmp_path):
    engine = _rules(tmp_path, """
        events:
          - name: e
            label: E
            signals:
              - {signal: s, min: 1.0}
        """)
    events, _ = engine.run([], {"s": [5, 0, 0, 0, 5]})
    assert set(events) == {0, 4}


def test_the_two_ends_are_guarded_separately(tmp_path):
    """A closing guard discards what a recording builds towards.

    Learned from a real miss: a symmetric 120s guard threw away a hand-marked
    episode that ended 94 seconds before its file did. Opening material is
    titles; closing material is often the payload.
    """
    engine = _rules(tmp_path, """
        events:
          - name: e
            label: E
            ignore_start_secs: 5
            ignore_end_secs: 0
            signals:
              - {signal: s, min: 1.0}
        """)
    s = [5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5]
    events, _ = engine.run([], {"s": s})
    assert set(events) == {18, 19}, "the opening was kept or the ending dropped"


def test_ignore_edges_secs_still_sets_both(tmp_path):
    """Rules written before the two ends were told apart keep working."""
    engine = _rules(tmp_path, """
        events:
          - name: e
            label: E
            ignore_edges_secs: 5
            signals:
              - {signal: s, min: 1.0}
        """)
    s = [5, 5] + [0] * 16 + [5, 5]
    assert engine.run([], {"s": s})[0] == {}
