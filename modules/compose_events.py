"""Run the composition rules over whatever detections a run ended up with.

The engine itself lives in ``video_ai_editor.composition_engine``. What lives
here is the one thing every caller of it needs and none of them should be
writing twice: applying it *idempotently* to a set of detections that may
already contain the results of a previous pass.

Why this is its own module
--------------------------

It began as a block inside the pipeline's object-detection branch, which meant
the rules only ever ran when objects were re-detected. Editing a rule and
re-running therefore changed nothing at all: the detections came from cache, the
engine never executed, and the report showed the previous rule set's events with
no indication that the file had been touched. The user's only remedy was to
delete the cache by hand and pay for a full re-detection -- minutes to hours --
to apply a change that costs milliseconds.

That is the wrong trade, because rules are cheap and detections are expensive.
Rules are a *reading* of boxes that already exist; nothing about changing one
requires looking at the video again. So the engine now runs on every pass, over
whatever boxes are to hand, and the cache keeps its job of storing the
expensive half.

Doing that safely is the whole content of this file. A cached detection set
already has the previous pass's event names merged into it and the previous
pass's event boxes appended to it, so running the engine again without removing
them first would double-count every event, leave events from deleted rules in
place for ever, and -- worst -- feed the previous pass's event boxes back in as
though they were detections, letting a rule match against its own output.

Stripping is by name, and by the full name list rather than by what fired.
A rule that matched nothing this time still has to have last time's matches
removed, or a deleted rule outlives the file it was deleted from.

The pipeline's side of this is an import and one call, deliberately: that file
diverges between the two editions and every line added to it is a line to be
ported by hand and a line that can drift.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterable, Mapping, Optional, Sequence


def rules_fingerprint(rules_path: Optional[str]) -> str:
    """A short digest of the rules file, or ``""`` when there is none.

    Not used to decide whether to run -- the engine always runs -- but to say
    in the log whether the rules changed since the pass whose detections are
    being reused. "Nothing matched" and "nothing matched, and these are the same
    rules as last time" send a user to different places.
    """
    if not rules_path or not os.path.exists(rules_path):
        return ""
    try:
        with open(rules_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:12]
    except OSError:
        return ""


def strip_events(object_detections: Optional[Mapping],
                 bbox_cache: Optional[Iterable[Mapping]],
                 names: Iterable[str]) -> tuple[dict, list]:
    """Remove every trace of a previous composition pass.

    Returns new containers rather than mutating: the caller may be holding the
    cached objects, and a cache quietly edited in place is one that gets written
    back in a state nobody chose.
    """
    drop = {str(n) for n in (names or ())}
    detections = {}
    for sec, found in (object_detections or {}).items():
        kept = [n for n in (found or []) if str(n) not in drop]
        if kept:
            detections[int(sec)] = kept

    boxes = []
    for frame in (bbox_cache or []):
        objects = [str(n) for n in (frame.get("objects") or [])]
        # A composed frame is one whose every object is an event name -- the
        # engine emits one such frame per event per timestamp. A detector frame
        # that happens to sit at the same second is untouched.
        if objects and all(name in drop for name in objects):
            continue
        boxes.append(frame)
    return detections, boxes


def apply_rules(object_detections: Optional[Mapping],
                bbox_cache: Optional[Sequence[Mapping]],
                *,
                rules_path: Optional[str],
                previous_names: Iterable[str] = (),
                log_fn=print) -> tuple:
    """Re-derive composed events. Returns ``(detections, boxes, names, hits)``.

    Safe to call on a freshly detected pass and on a cached one, in that order
    or any other, any number of times: the result depends only on the detector
    boxes and the current rules file.

    With no rules file, the previous pass's events are still removed and nothing
    is added — a run with no rules must not report events, and leaving stale
    ones in would be the same silent staleness this module exists to end.
    """
    detections = {int(k): list(v or []) for k, v in (object_detections or {}).items()}
    boxes = list(bbox_cache or [])
    previous = {str(n) for n in (previous_names or ())}

    engine = None
    names: list = []
    if rules_path and os.path.exists(rules_path):
        try:
            from video_ai_editor.composition_engine import CompositionEngine
            engine = CompositionEngine(rules_path)
            names = list(engine.event_names)
        except Exception as exc:
            # A malformed rules file must not cost the run its detections. The
            # previous pass's events are still stripped, because they no longer
            # correspond to a rule set anyone can read.
            log_fn(f"⚠️ Composition rules could not be loaded: {exc}")
            engine, names = None, []

    detections, boxes = strip_events(detections, boxes, previous | set(names))

    if engine is None or not boxes:
        if previous:
            log_fn("ℹ️ Composition engine: no rules in force; "
                   f"{len(previous)} event type(s) from the previous pass "
                   "removed.")
        return detections, boxes, names, 0

    composed, composed_boxes = engine.run(boxes)
    hits = sum(len(v) for v in composed.values())
    for sec, found in composed.items():
        sec = int(sec)
        detections[sec] = sorted(set(detections.get(sec, [])) | set(found))
    boxes = boxes + list(composed_boxes)

    changed = previous and previous != set(names)
    note = ""
    if changed:
        added = sorted(set(names) - previous)
        gone = sorted(previous - set(names))
        parts = []
        if added:
            parts.append(f"added {', '.join(added)}")
        if gone:
            parts.append(f"removed {', '.join(gone)}")
        note = f" (rules changed since the cached pass: {'; '.join(parts)})"
    if hits:
        log_fn(f"✅ Composition engine: {hits} event-hit(s) over "
               f"{len(composed)} second(s) from {len(names)} rule(s){note}")
    else:
        log_fn(f"ℹ️ Composition engine: {len(names)} rule(s), nothing "
               f"matched{note}")
    return detections, boxes, names, hits
