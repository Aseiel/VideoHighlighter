"""Find a shown thing in every sample, as one box to confirm.

Someone draws a box around the thing once (``seed``). From then on, each
sample is searched region by region: the regions the stock detector found
plus an overlapping grid (for things no detector knows), each embedded with
CLIP once and cached. A sample's score for the class is its best region's
likeness to the class's crops, calibrated against the whole footage the way
sample scores are (``scoring.background``): 0 is ordinary footage, 1 looks
like the boxes accepted so far.

A sample that clears ``settings.gate`` gets its best region as a ``found``
box, pending. That box is the whole question: "is this box around it?".
Accepting it accepts the sample as the class (``settle``); there is no
separate sample review for a seeded class. Boxes above the class's learnt
cutoff are accepted without asking (``cutoff``).

Every accepted box sharpens the class's likeness, so ``find`` is re-run
whenever a class has more accepted crops than when it last ran; the region
vectors are cached, so that costs a matrix product, not another pass of CLIP.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np

from modules.teach import cutoff
from modules.teach import embed as embed_mod
from modules.teach import scoring
from modules.teach.boxes import (
    MIN_AREA, _normalise, _read_at, crop_vectors, frame_times, seeded, shrink, store,
)
from modules.teach.project import ACCEPTED, AUTO_BOX, OBJECTS, PENDING, Project
from modules.vision.label_store import LabelledBox, LabelStore

FOUND = "found"
REGIONS_FILE = "regions.json"
FIND_FILE = "find.json"
DETECTED, TILE = "det", "tile"


def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def _write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=0)
    os.replace(tmp, path)


def _regions_of(frame, detector) -> list:
    """``(kind, (x1, y1, x2, y2))`` pixel regions worth comparing in one frame."""
    from llm.category_scoring import tile_rects

    h, w = frame.shape[:2]
    out = []
    for det in detector.detect(frame) if detector is not None else []:
        box = _normalise(det, w, h)
        if box[2] * box[3] >= MIN_AREA:
            out.append((DETECTED, (int(max(0, det.x1)), int(max(0, det.y1)),
                                   int(min(w, det.x2)), int(min(h, det.y2)))))
    for rect in tile_rects(w, h, 3, 0.5) + tile_rects(w, h, 4, 0.35):
        out.append((TILE, tuple(int(v) for v in rect)))
    return [(k, r) for k, r in out if r[2] - r[0] >= 8 and r[3] - r[1] >= 8]


def embed_regions(project: Project, detector, embedder, read_at: Callable,
                  progress: Optional[Callable] = None) -> dict:
    """``{sample id: {"time", "boxes", "kinds", "vectors"}}`` for every sample.

    One frame per sample, from its middle. Computed once; the boxes live in
    ``regions.json`` and the vectors in the project's vector cache.
    """
    cache = embed_mod.VectorCache(project.root, getattr(embedder, "model_id", ""))
    stored = _read_json(project.path(REGIONS_FILE), {})
    out, changed = {}, False
    todo = [s for s in project.samples if not s.unreadable]
    for n, sample in enumerate(todo, 1):
        entry = stored.get(sample.id)
        vectors = None
        if entry is not None and not entry.get("boxes"):
            continue                 # searched before, nothing readable in it
        if entry:
            got = [cache.get(f"region:{sample.id}:{i}") for i in range(len(entry["boxes"]))]
            if all(v is not None for v in got):
                vectors = np.stack(got)
        if vectors is None:
            moment = frame_times(sample.duration, 1)[0]
            frame = read_at(sample.path, moment)
            regions = _regions_of(frame, detector) if frame is not None else []
            if not regions:
                # Recorded as searched, or ``needed`` would ask for it forever.
                stored[sample.id] = {"time": moment, "kinds": [], "boxes": []}
                changed = True
                continue
            h, w = frame.shape[:2]
            crops = [frame[y1:y2, x1:x2] for _, (x1, y1, x2, y2) in regions]
            vectors = embed_mod.unit(embedder.images(crops))
            entry = {"time": moment, "kinds": [k for k, _ in regions],
                     "boxes": [[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
                               for _, (x1, y1, x2, y2) in regions]}
            for i, vector in enumerate(vectors):
                cache.put(f"region:{sample.id}:{i}", vector)
            stored[sample.id] = entry
            changed = True
        out[sample.id] = {**entry, "vectors": np.asarray(vectors, dtype=np.float32)}
        if progress:
            progress(n, len(todo))
    if changed:
        cache.save()
        _write_json(project.path(REGIONS_FILE), stored)
    return out


def score_class(crops: list, regions: dict) -> dict:
    """``{sample id: (calibrated score, index of the best region)}`` for one class."""
    if not crops or not regions:
        return {}
    stack = embed_mod.unit(np.stack(crops))
    proto = embed_mod.unit(stack.mean(axis=0))
    ids = list(regions)
    best = [int(np.argmax(regions[i]["vectors"] @ proto)) for i in ids]
    raw = np.array([float(regions[i]["vectors"][j] @ proto) for i, j in zip(ids, best)])
    floor = scoring.background(raw)
    # One seed box is its own prototype (likeness 1.0), which says nothing
    # about how alike the thing looks from frame to frame; until there are
    # two, the footage's own top end stands in, as for a class named in words.
    anchor = (float((stack @ proto).mean()) if len(stack) >= 2
              else float(np.percentile(raw, scoring.TEXT_ANCHOR_PERCENTILE)))
    calibrated = (raw - floor) / max(anchor - floor, scoring.MIN_SPREAD)
    return {i: (float(c), j) for i, c, j in zip(ids, calibrated, best)}


def crop_counts(project: Project, labels: LabelStore) -> dict:
    """Accepted boxes per class that someone looked at (what crops use)."""
    unchecked = cutoff.auto_decided(project)
    out: dict = {}
    for box in labels.accepted():
        if any(box.box) and cutoff.box_key(box) not in unchecked:
            out[box.class_name] = out.get(box.class_name, 0) + 1
    return out


def needed(project: Project, labels: Optional[LabelStore] = None) -> bool:
    """Would ``find`` change anything: new samples, or better-known classes?"""
    if project.task != OBJECTS:
        return False
    labels = labels or store(project)
    drawn = seeded(labels)
    if not drawn:
        return False
    regions = _read_json(project.path(REGIONS_FILE), {})
    if any(s.id not in regions for s in project.samples if not s.unreadable):
        return True
    last = _read_json(project.path(FIND_FILE), {}).get("crops", {})
    now = crop_counts(project, labels)
    return any(now.get(name, 0) > last.get(name, 0) for name in drawn)


def settle(project: Project, labels: LabelStore) -> dict:
    """Sample verdicts from found-box verdicts: an accepted box accepts its
    sample; a box taken back takes back the sample it accepted."""
    auto = cutoff.auto_decided(project)
    by_path = {s.path: s for s in project.samples}
    accepted = taken_back = 0
    for box in labels.boxes:
        if box.source != FOUND:
            continue
        sample = by_path.get(box.video)
        if sample is None:
            continue
        if box.verdict == ACCEPTED and sample.verdict == PENDING:
            by = AUTO_BOX if cutoff.box_key(box) in auto else "boxes"
            project.decide(sample, ACCEPTED, box.class_name, by=by)
            # Its one box is enough: more frames of it would be more
            # questions for little more to learn from.
            sample.boxes_tried = True
            accepted += 1
        elif (box.verdict == PENDING and sample.decided_by == AUTO_BOX
              and sample.label == box.class_name):
            sample.verdict, sample.label, sample.decided_by = PENDING, "", ""
            taken_back += 1
        elif (box.verdict == ACCEPTED and sample.decided_by == AUTO_BOX
              and cutoff.box_key(box) not in auto):
            sample.decided_by = "boxes"          # a person has now seen it
    project.save()
    return {"accepted": accepted, "taken_back": taken_back}


def find(project: Project, detector, embedder, *, read_at: Optional[Callable] = None,
         progress: Optional[Callable] = None) -> dict:
    """Search every sample for every seeded class; propose, auto-accept, settle."""
    if project.task != OBJECTS:
        raise ValueError("find is for object projects")
    read_at = read_at or _read_at
    labels = store(project)
    drawn = seeded(labels)
    if not drawn:
        raise ValueError("nothing to look for yet: draw a box around it first (seed)")
    crops = crop_vectors(project, labels, embedder, read_at)
    regions = embed_regions(project, detector, embedder, read_at, progress)
    known = {(b.video, round(b.time, 3), b.class_name) for b in labels.boxes
             if b.source == FOUND}
    by_id = {s.id: s for s in project.samples}
    gate = project.settings.gate
    proposed: dict = {}
    new_boxes = []
    for name in sorted(drawn):
        mine = crops.get(name) or []
        proto = embed_mod.unit(np.mean(mine, axis=0)) if mine else None
        for sid, (score, j) in score_class(mine, regions).items():
            sample = by_id.get(sid)
            entry = regions[sid]
            key = (sample.path, round(entry["time"], 3), name) if sample else None
            if sample is None or sample.verdict != PENDING or key in known or score < gate:
                continue
            box = tuple(entry["boxes"][j])
            if entry["kinds"][j] == TILE:
                # A grid region is coarse: pull it in around the thing.
                frame = read_at(sample.path, entry["time"])
                if frame is not None:
                    h, w = frame.shape[:2]
                    rect = (box[0] * w, box[1] * h, (box[0] + box[2]) * w,
                            (box[1] + box[3]) * h)
                    x1, y1, x2, y2 = shrink(embedder, frame, tuple(int(v) for v in rect),
                                            proto, float(entry["vectors"][j] @ proto))
                    box = (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h)
            new_boxes.append(LabelledBox(video=sample.path, time=entry["time"],
                                         class_name=name, box=box, source=FOUND,
                                         confidence=round(score, 4), verdict=PENDING))
            known.add(key)
            proposed[name] = proposed.get(name, 0) + 1
    # The search can take minutes; a box drawn in the player meanwhile is in
    # the file, not in what was read at the start. Add to the file as it is now.
    labels = store(project)
    labels.extend(new_boxes)
    labels.save()
    auto = cutoff.apply(project, labels)
    settled = settle(project, labels)
    _write_json(project.path(FIND_FILE), {"crops": crop_counts(project, labels)})
    return {"searched": len(regions), "proposed": proposed,
            "auto_accepted": auto["auto_accepted"], "taken_back": auto["taken_back"],
            "cutoffs": auto["cutoffs"], "samples_accepted": settled["accepted"],
            "questions": len(store(project).pending())}
