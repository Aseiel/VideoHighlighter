"""Boxes for an object project: proposed, checked, or drawn in the labeller.

An accepted object sample says "it is in here somewhere". A detector needs
where. Three sources, all ending in ``labels.json``
(``modules.vision.label_store``), all reviewed the same way:

* **Proposed** (``propose``). The stock detector finds everything it can in
  a few frames of the sample; each box is cropped, embedded, and scored
  against the class; the crop that stands out wins. When the class *is* one
  of the stock detector's labels, its own boxes are taken directly. From
  round 2, the project's own detector proposes (``source="model"``).
* **Drawn** in ``tools/labeler.py`` — the manual step that gave good results
  with a hundred samples. ``labeler_worklist`` says which clips still need it;
  ``import_labeler`` reads the export back.
* **Negatives**. A sample judged "none of these" becomes a frame with no
  boxes: what stops a detector firing on everything.

Proposals start ``pending`` and reach a dataset only once accepted on a sheet
(``next_sheet``, ``apply_verdicts``): a proposed box is a question.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Callable, Optional, Sequence

import numpy as np

from modules.teach import embed as embed_mod
from modules.teach.naming import PROMPTS, load_vocabulary
from modules.teach.project import (
    LABELS_FILE, NEGATIVE as SAMPLE_NEGATIVE, OBJECTS, Project,
)
from modules.vision.label_store import (
    ACCEPTED, NEGATIVE, PENDING, REJECTED, LabelledBox, LabelStore,
    from_labeler_export,
)

BOX_REVIEW_PREFIX = "boxes"
# A crop must beat the frame's other crops by this much to be the class's box.
STANDOUT = 0.02
MIN_AREA = 0.0015          # fraction of the frame; smaller is noise, not a thing
TILE_STANDOUT = 0.03       # a region must beat the frame's typical one by this
SHRINK_STEPS = 8
SHRINK_TOLERANCE = 0.01    # likeness a shrink may cost before it stops
AUTO_BOX_CONFIDENCE = 0.6  # a detector's own class at this confidence: no sheet
MIN_CROP_EXAMPLES = 3      # accepted boxes before crops replace the class name
MAX_REJECTED_PER_FRAME = 2 # proposals a frame may have rejected before the labeller
MAX_CROP_EXAMPLES = 60


def store(project: Project) -> LabelStore:
    return LabelStore(project.path(LABELS_FILE)).load()


def frame_times(duration: float, count: int) -> list:
    """``count`` moments spread across a sample, away from its cut edges."""
    count = max(1, int(count))
    return [round(duration * (k + 0.5) / count, 3) for k in range(count)]


def _read_at(path: str, moment: float):
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _labelled_keys(labels: LabelStore) -> set:
    """Frames already answered: a box waiting, accepted, or marked empty.
    A frame whose boxes were all rejected is not answered; see ``propose``."""
    return {(b.video, round(b.time, 3)) for b in labels.boxes if b.verdict != REJECTED}


def _rejected(labels: LabelStore) -> dict:
    out: dict = {}
    for b in labels.boxes:
        if b.verdict == REJECTED:
            out.setdefault((b.video, round(b.time, 3)), []).append(tuple(b.box))
    return out


def _crop_ready(labels: LabelStore) -> set:
    """Classes with enough accepted boxes to be matched by their crops."""
    counts: dict = {}
    for b in labels.accepted():
        if b.class_name and any(b.box):
            counts[b.class_name] = counts.get(b.class_name, 0) + 1
    return {name for name, n in counts.items() if n >= MIN_CROP_EXAMPLES}


def retryable(project: Project, labels: Optional[LabelStore] = None) -> list:
    """Frames of accepted samples whose proposals were all rejected and that
    ``propose`` would try again now (see there): the class has crops to match,
    and the frame has not used up its tries."""
    labels = labels or store(project)
    done = _labelled_keys(labels)
    rejected = _rejected(labels)
    ready = _crop_ready(labels)
    out = []
    for sample in project.accepted():
        if sample.label not in ready:
            continue
        for moment in frame_times(sample.duration, project.settings.boxes_per_sample):
            key = (sample.path, moment)
            if key not in done and 0 < len(rejected.get(key, ())) < MAX_REJECTED_PER_FRAME:
                out.append(key)
    return out


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _excluded(box, excluded) -> bool:
    return any(_iou(box, other) > 0.5 for other in excluded)


def propose(project: Project, detector, embedder, *,
            read_at: Optional[Callable] = None, model_detector=None) -> dict:
    """Propose boxes for accepted samples' frames that have none yet."""
    if project.task != OBJECTS:
        raise ValueError("boxes are for object projects")
    read_at = read_at or _read_at
    labels = store(project)
    done = _labelled_keys(labels)
    rejected = _rejected(labels)
    stock = set(load_vocabulary(OBJECTS))
    class_vectors, from_crops = _class_vectors(project, labels, embedder, read_at)

    added, empty, auto_accepted = 0, 0, 0
    for sample in project.samples:
        if sample.verdict == SAMPLE_NEGATIVE:
            moment = frame_times(sample.duration, 1)[0]
            if (sample.path, moment) not in done:
                labels.add(LabelledBox(video=sample.path, time=moment, class_name="",
                                       source="hand", verdict=NEGATIVE))
                done.add((sample.path, moment))
            continue
        if sample.verdict != ACCEPTED:
            continue
        sample.boxes_tried = True
        for moment in frame_times(sample.duration, project.settings.boxes_per_sample):
            key = (sample.path, moment)
            if key in done:
                continue
            # A frame whose proposals were all rejected gets another try, but
            # only once the class is matched by its accepted crops (the same
            # text-based guess would just come back), never at a rejected
            # region, and at most twice before it is left to the labeller.
            excluded = rejected.get(key, [])
            if excluded and (sample.label not in from_crops
                             or len(excluded) >= MAX_REJECTED_PER_FRAME):
                continue
            frame = read_at(sample.path, moment)
            if frame is None:
                continue
            box = None
            if model_detector is not None:
                box = _from_detector(model_detector, frame, sample.label, "model", excluded)
            if box is None and sample.label in stock:
                box = _from_detector(detector, frame, sample.label, "stock", excluded)
            vector = class_vectors[sample.label]
            if box is None:
                box = _by_clip(detector, embedder, frame, vector, excluded)
            if box is None:
                box = _by_tiles(embedder, frame, vector)
                if box is not None and _excluded(box[0], excluded):
                    box = None
            if box is None:
                empty += 1
                if excluded:
                    # A retry that found nothing new counts as one more
                    # rejection, so the frame runs out of tries and ``status``
                    # stops offering it.
                    labels.add(LabelledBox(video=sample.path, time=moment,
                                           class_name=sample.label, box=(0.0, 0.0, 0.0, 0.0),
                                           source="prompt", verdict=REJECTED))
                continue
            coords, confidence, source = box
            # A detector's own box for its own class, or last round's model,
            # at a confidence detectors rarely reach by accident: accepted
            # without a sheet. CLIP-picked regions always get a look.
            sure = (project.settings.auto_accept and source in ("model", "stock")
                    and confidence >= AUTO_BOX_CONFIDENCE)
            labels.add(LabelledBox(video=sample.path, time=moment,
                                   class_name=sample.label, box=coords,
                                   source="prompt" if source == "stock" else source,
                                   confidence=confidence,
                                   verdict=ACCEPTED if sure else PENDING))
            auto_accepted += sure
            done.add((sample.path, moment))
            added += 1
    labels.save()
    project.save()
    return {"proposed": added, "auto_accepted": auto_accepted,
            "frames_without_a_proposal": empty,
            "pending": len(labels.pending())}


def _class_vectors(project: Project, labels: LabelStore, embedder, read_at) -> tuple:
    """What each class looks like, as one CLIP vector.

    From the crops of its accepted boxes once there are a few — a picture of
    the thing matches pictures far better than its name does — and from its
    name and description until then. Crop vectors are cached, so each accepted
    box is embedded once, ever.
    """
    cache = embed_mod.VectorCache(project.root, getattr(embedder, "model_id", ""))
    out, from_crops = {}, set()
    for spec in project.classes:
        crops = []
        for box in labels.accepted():
            if box.class_name != spec.name or box.source == "hand" and not any(box.box):
                continue
            key = f"box:{box.video}@{box.time:.3f}:{','.join(f'{v:.4f}' for v in box.box)}"
            vector = cache.get(key)
            if vector is None:
                frame = read_at(box.video, box.time)
                if frame is None:
                    continue
                h, w = frame.shape[:2]
                x, y, bw, bh = box.pixels(w, h)
                crop = frame[int(y):int(y + bh), int(x):int(x + bw)]
                if not crop.size:
                    continue
                vector = embed_mod.unit(embedder.images([crop]))[0]
                cache.put(key, vector)
            crops.append(vector)
            if len(crops) >= MAX_CROP_EXAMPLES:
                break
        if len(crops) >= MIN_CROP_EXAMPLES:
            out[spec.name] = embed_mod.unit(np.mean(crops, axis=0))
            from_crops.add(spec.name)
        else:
            prompts = [PROMPTS[OBJECTS].format(spec.name)] + (
                [spec.description] if spec.description else [])
            out[spec.name] = embed_mod.unit(embedder.texts(prompts).mean(axis=0))
    cache.save()
    return out, from_crops


def _by_tiles(embedder, frame, class_vector):
    """For a thing no detector knows: the region of the frame that looks most
    like the class, then shrunk while it still does.

    Overlapping regions at two sizes (``llm.category_scoring.tile_rects``);
    the winner must stand out from the frame's typical region. Then each
    side is pulled in by a step as long as the crop's likeness does not drop
    by more than ``SHRINK_TOLERANCE``, so the box ends up around the thing
    rather than around a fixed fraction of the frame. Coarser than a
    detector's box, which is why these always go to review.
    """
    from llm.category_scoring import crop_tiles, tile_rects

    h, w = frame.shape[:2]
    tiles = tile_rects(w, h, 3, 0.5) + tile_rects(w, h, 4, 0.35)
    crops = [c for c in crop_tiles(frame, tiles)]
    sims = embed_mod.unit(embedder.images(crops)) @ class_vector
    best = int(np.argmax(sims))
    if float(sims[best] - np.percentile(sims, 40)) < TILE_STANDOUT:
        return None
    x1, y1, x2, y2 = tiles[best]
    score = float(sims[best])
    for _ in range(SHRINK_STEPS):
        dx, dy = max(1, int((x2 - x1) * 0.15)), max(1, int((y2 - y1) * 0.15))
        options = [(x1 + dx, y1, x2, y2), (x1, y1 + dy, x2, y2),
                   (x1, y1, x2 - dx, y2), (x1, y1, x2, y2 - dy)]
        options = [o for o in options if o[2] - o[0] >= 16 and o[3] - o[1] >= 16]
        if not options:
            break
        vs = embed_mod.unit(embedder.images(crop_tiles(frame, options))) @ class_vector
        pick = int(np.argmax(vs))
        if float(vs[pick]) < score - SHRINK_TOLERANCE:
            break
        (x1, y1, x2, y2), score = options[pick], max(score, float(vs[pick]))
    return (x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h), score, "category"


def _normalise(det, width: int, height: int) -> tuple:
    x1, y1 = max(0.0, det.x1) / width, max(0.0, det.y1) / height
    x2, y2 = min(float(width), det.x2) / width, min(float(height), det.y2) / height
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def _from_detector(detector, frame, class_name: str, source: str, excluded=()):
    height, width = frame.shape[:2]
    hits = [d for d in detector.detect(frame) if d.class_name == class_name
            and not _excluded(_normalise(d, width, height), excluded)]
    if not hits:
        return None
    best = max(hits, key=lambda d: d.confidence)
    return _normalise(best, width, height), float(best.confidence), source


def _by_clip(detector, embedder, frame, class_vector, excluded=()):
    """The detected region that looks most like the class, if one stands out."""
    height, width = frame.shape[:2]
    candidates = []
    for det in detector.detect(frame):
        box = _normalise(det, width, height)
        if box[2] * box[3] < MIN_AREA or _excluded(box, excluded):
            continue
        x1, y1 = int(det.x1), int(det.y1)
        crop = frame[max(0, y1):int(det.y2), max(0, x1):int(det.x2)]
        if crop.size:
            candidates.append((box, crop))
    if not candidates:
        return None
    vectors = embed_mod.unit(embedder.images([c for _, c in candidates]))
    sims = vectors @ class_vector
    order = np.argsort(-sims)
    lead = float(sims[order[0]] - sims[order[1]]) if len(order) > 1 else STANDOUT
    if lead < STANDOUT:
        return None
    return candidates[int(order[0])][0], float(sims[order[0]]), "prompt"


# ---------------------------------------------------------------------------
# Review of boxes
# ---------------------------------------------------------------------------

def _draw(frame, box, text: str):
    import cv2
    out = frame.copy()
    h, w = out.shape[:2]
    x, y, bw, bh = box
    cv2.rectangle(out, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)),
                  (0, 210, 255), max(2, w // 300))
    return out


def next_sheet(project: Project, size: int = 20, *, read_at: Optional[Callable] = None,
               renderer: Optional[Callable] = None, keep_tiles: bool = False) -> dict:
    from modules.teach import review

    read_at = read_at or _read_at
    labels = store(project)
    pending = sorted(labels.pending(), key=lambda b: b.confidence)[:size]
    if not pending:
        return {}
    renderer = renderer or review.render_sheet
    review_dir = project.path(review.REVIEW_DIR)
    os.makedirs(review_dir, exist_ok=True)
    numbers = [int(m.group(1)) for name in os.listdir(review_dir)
               for m in [re.match(BOX_REVIEW_PREFIX + r"-(\d+)\.json$", name)] if m]
    number = max(numbers, default=0) + 1
    tiles, captions, items = [], [], []
    for n, box in enumerate(pending, 1):
        frame = read_at(box.video, box.time)
        drawn = [_draw(frame, box.box, box.class_name)] if frame is not None else []
        tiles.append(review._tile(drawn, 220))
        captions.append(f"{box.class_name} ({box.source} {box.confidence:.2f})")
        items.append({"n": n, "video": box.video, "time": box.time,
                      "class_name": box.class_name, "box": list(box.box),
                      "caption": captions[-1]})
    image = os.path.join(review_dir, f"{BOX_REVIEW_PREFIX}-{number:04d}.jpg")
    renderer(tiles, captions, 4, image,
             f"boxes {number}: is the yellow box around the named thing, and tight?")
    record = {"sheet": number, "image": image, "created": time.time(), "items": items}
    with open(os.path.join(review_dir, f"{BOX_REVIEW_PREFIX}-{number:04d}.json"), "w",
              encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    if keep_tiles:
        record["tiles"] = tiles          # for the review window; not saved
    return record


def apply_verdicts(project: Project, number: int, *, accept: str = "",
                   reject: str = "", accept_rest: bool = False) -> dict:
    from modules.teach.review import parse_numbers

    with open(project.path("review", f"{BOX_REVIEW_PREFIX}-{number:04d}.json"),
              "r", encoding="utf-8") as handle:
        sheet = json.load(handle)
    labels = store(project)
    by_key = {(b.video, round(b.time, 3), b.class_name, tuple(round(v, 5) for v in b.box)): b
              for b in labels.boxes}
    decided = {n: ACCEPTED for n in parse_numbers(accept)}
    for n in parse_numbers(reject):
        if n in decided:
            return {"applied": 0, "errors": [f"tile {n} was given two verdicts"]}
        decided[n] = REJECTED
    if accept_rest:
        for item in sheet["items"]:
            decided.setdefault(item["n"], ACCEPTED)
    applied, errors = 0, []
    for item in sheet["items"]:
        verdict = decided.get(item["n"])
        if verdict is None:
            continue
        key = (item["video"], round(item["time"], 3), item["class_name"],
               tuple(round(v, 5) for v in item["box"]))
        box = by_key.get(key)
        if box is None:
            errors.append(f"tile {item['n']}: that box is no longer in labels.json")
            continue
        labels.set_verdict(box, verdict)
        applied += 1
    labels.save()
    return {"applied": applied, "errors": errors}


# ---------------------------------------------------------------------------
# The labeller, for what proposals could not do
# ---------------------------------------------------------------------------

def labeler_worklist(project: Project) -> list:
    """Accepted samples with no accepted box: open these in tools/labeler.py."""
    labels = store(project)
    boxed = {b.video for b in labels.accepted()}
    return [{"sample": s.id, "class": s.label, "path": s.path}
            for s in project.accepted() if s.path not in boxed]


def import_labeler(project: Project, paths: Sequence[str], accept: bool = False,
                   box_fraction: float = 0.12) -> dict:
    """Read labeller exports. A point becomes a box around it, pending unless
    ``accept`` — the person who clicked already looked."""
    labels = store(project)
    by_name = {os.path.basename(s.path): s.path for s in project.samples}
    names = set(project.class_names())
    added, unknown = 0, set()
    for path in paths:
        for box in from_labeler_export(path, box_fraction,
                                       verdict=ACCEPTED if accept else PENDING):
            box.video = by_name.get(os.path.basename(box.video), box.video)
            if box.class_name not in names:
                unknown.add(box.class_name)
                continue
            labels.add(box)
            added += 1
    labels.save()
    return {"imported": added,
            "skipped_unknown_classes": sorted(unknown)}
