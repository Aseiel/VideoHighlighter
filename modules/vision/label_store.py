"""Labelled examples on disk, and the COCO dataset built from them.

This is the join between "the user marked some things" and "a detector can be
trained", and it is deliberately ignorant of where a box came from. A box is a
box: hand-drawn, proposed by a taught category, prompted out of an
open-vocabulary detector, imported from the labeller, or emitted by the model
trained in the previous round. Each carries its ``source`` so the record can be
read afterwards, and nothing downstream branches on it.

That indifference is the point. The box sources available here differ enormously
in quality, and the good ones will change:

===================  ==========================================================
``hand``             a person dragged it. The best geometry there is.
``category``         the winning region from a taught category — a fixed
                     fraction of the frame at one of nine positions, so it
                     says *where to look*, not where the thing is.
``labeler``          imported from ``tools/labeler.py``, which stores *points*;
                     each becomes a fixed-size box around the click. Same
                     coarseness as ``category``, for the same reason.
``prompt``           an open-vocabulary detector's fitted box. Real geometry,
                     but the class needs a nameable noun.
``model``            the detector trained last round. Real geometry, no
                     vocabulary limit, and the reason a second round is cheaper
                     than the first.
===================  ==========================================================

Only ``accepted`` boxes reach a dataset. A proposal is a question, and the
answer is a person's.

The split is the subtle part
----------------------------

Labels are grouped into **segments** — runs of labels close together in time in
the same video — and whole segments go to train or to validation, never split
across.

A random split would be wrong in a way that flatters every measurement taken
afterwards. Frames a second apart in the same shot are near-identical; scatter
them and validation is scoring the model on pictures it trained on. The
validation loss then falls beautifully and means nothing, and the "never
promote a worse model" rule the training loop depends on is comparing two
numbers that are both fiction.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable, Optional, Sequence

# Labels closer together than this in the same video belong to the same
# segment, and therefore to the same side of the split. Generous on purpose:
# the cost of over-grouping is a slightly coarser split, while the cost of
# under-grouping is a validation set that quietly contains the training set.
SEGMENT_GAP_SECONDS = 5.0

# A segment is also closed once it spans this long, however continuous the
# labelling was. Without a cap, one steadily-labelled video is a single segment
# and can never be split at all — labels every third of a second produce one
# group of hundreds, `split_segments` refuses to break it, and the run gets no
# validation set while reporting nothing wrong. Half a minute is short enough
# to yield several groups from a few minutes of footage and long enough that
# neighbouring frames still travel together.
MAX_SEGMENT_SPAN_SECONDS = 30.0

# Fraction of segments held out for validation, when the caller does not say.
DEFAULT_VAL_FRACTION = 0.2

# How many segments to aim for when choosing the span automatically. Enough
# that a 20% holdout is more than one group, and that no single group carries a
# large share of the labels.
TARGET_SEGMENTS = 8

ACCEPTED = "accepted"
REJECTED = "rejected"
PENDING = "pending"

# A frame deliberately marked as containing none of the classes. Not the same
# as a frame with no labels, which merely has not been looked at.
NEGATIVE = "negative"


@dataclass
class LabelledBox:
    """One box, in one frame, of one video."""

    video: str
    time: float                     # seconds into the video
    class_name: str
    # Normalised (x, y, w, h) in [0,1], matching the ROI convention used by
    # the rest of the app. Normalised rather than pixels so a re-encode, a
    # resize, or a proxy file does not silently invalidate every label.
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    source: str = "hand"
    confidence: float = 1.0
    verdict: str = PENDING
    added: float = field(default_factory=time.time)

    def to_json(self) -> dict:
        data = asdict(self)
        data["box"] = [float(v) for v in self.box]
        return data

    @classmethod
    def from_json(cls, data: dict) -> "LabelledBox":
        return cls(
            video=str(data["video"]),
            time=float(data["time"]),
            class_name=str(data["class_name"]),
            box=tuple(float(v) for v in data.get("box", (0, 0, 0, 0))),
            source=str(data.get("source", "hand")),
            confidence=float(data.get("confidence", 1.0)),
            verdict=str(data.get("verdict", PENDING)),
            added=float(data.get("added", 0.0)),
        )

    @property
    def is_negative(self) -> bool:
        return self.verdict == NEGATIVE

    def pixels(self, width: int, height: int) -> tuple:
        """The box in pixels, clamped to the frame. → (x, y, w, h)"""
        x, y, w, h = self.box
        x0 = max(0.0, min(1.0, x)) * width
        y0 = max(0.0, min(1.0, y)) * height
        x1 = max(0.0, min(1.0, x + w)) * width
        y1 = max(0.0, min(1.0, y + h)) * height
        return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


class LabelStore:
    """The labels for one project, as user data.

    JSON, and outside the repository: these are the user's own class names and
    their own footage, which the project's conventions keep out of the code.
    """

    def __init__(self, path: str):
        self.path = path
        self.boxes: list = []

    # ── persistence ──────────────────────────────────────────────────────

    def load(self) -> "LabelStore":
        if not os.path.exists(self.path):
            return self
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.boxes = [LabelledBox.from_json(d) for d in data.get("boxes", [])]
        except Exception as exc:
            print(f"[labels] unreadable store ({exc}); starting empty")
        return self

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        payload = {"boxes": [b.to_json() for b in self.boxes]}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)   # atomic: a crash cannot truncate the store

    # ── building it up ───────────────────────────────────────────────────

    def add(self, box: LabelledBox) -> LabelledBox:
        self.boxes.append(box)
        return box

    def extend(self, boxes: Iterable) -> None:
        self.boxes.extend(boxes)

    def set_verdict(self, box: LabelledBox, verdict: str) -> None:
        box.verdict = verdict

    # ── reading it back ──────────────────────────────────────────────────

    def accepted(self) -> list:
        return [b for b in self.boxes if b.verdict == ACCEPTED]

    def negatives(self) -> list:
        return [b for b in self.boxes if b.verdict == NEGATIVE]

    def pending(self) -> list:
        return [b for b in self.boxes if b.verdict == PENDING]

    def class_names(self) -> list:
        """Accepted classes, in a stable order.

        Sorted, and derived only from accepted boxes: the category id a model
        is trained against comes from this list, so it must not move when a
        pending proposal is rejected.
        """
        return sorted({b.class_name for b in self.accepted()})

    def counts(self) -> dict:
        """Accepted boxes per class — what a UI shows against its target."""
        out: dict = {}
        for box in self.accepted():
            out[box.class_name] = out.get(box.class_name, 0) + 1
        return out


# ── splitting ────────────────────────────────────────────────────────────

def auto_span(boxes: Sequence, target: int = TARGET_SEGMENTS) -> float:
    """Choose a segment span that actually yields splittable groups.

    A fixed cap cannot serve both a ten-minute file and a twenty-second one:
    30 seconds gives a long file plenty of groups and a short one exactly one,
    which is no validation set at all. So the span is derived from how much
    footage was labelled, aiming for ``target`` groups, and capped at
    :data:`MAX_SEGMENT_SPAN_SECONDS` so a long file does not get needlessly
    fine ones.

    **This does not make a one-video validation set independent.** Segments cut
    from a single continuous shot are still neighbours, and a score measured
    across them is optimistic. It is a usable relative measure between rounds
    on the same footage, and it is not evidence the model generalises. A second
    video is what buys that.
    """
    per_video: dict = {}
    for box in boxes:
        low, high = per_video.get(box.video, (box.time, box.time))
        per_video[box.video] = (min(low, box.time), max(high, box.time))
    total = sum(high - low for low, high in per_video.values())
    if total <= 0:
        return MAX_SEGMENT_SPAN_SECONDS
    return max(1.0, min(MAX_SEGMENT_SPAN_SECONDS, total / max(1, target)))


def segments(boxes: Sequence, gap: float = SEGMENT_GAP_SECONDS,
             max_span: Optional[float] = None) -> list:
    """Group labels into runs that belong on the same side of a split.

    A segment continues while labels are in the same video, within ``gap``
    seconds of the previous one, and the segment has not yet spanned
    ``max_span``. Returns a list of lists, each ordered by time.

    ``max_span`` defaults to :func:`auto_span` over these same labels, because
    the right cap depends on how much footage was labelled — see there.
    """
    if max_span is None:
        max_span = auto_span(boxes)
    ordered = sorted(boxes, key=lambda b: (b.video, b.time))
    out: list = []
    current: list = []
    for box in ordered:
        continues = (
            current
            and box.video == current[-1].video
            and box.time - current[-1].time <= gap
            and box.time - current[0].time < max_span
        )
        if continues:
            current.append(box)
        else:
            if current:
                out.append(current)
            current = [box]
    if current:
        out.append(current)
    return out


def split_segments(groups: Sequence, val_fraction: float = DEFAULT_VAL_FRACTION,
                   seed: int = 0) -> tuple:
    """Assign whole segments to train or validation. → (train, val)

    Deterministic for a given seed, so re-assembling a dataset does not shuffle
    the split underneath a comparison between two rounds.

    With too few segments to hold any back, everything goes to train and the
    caller gets an empty validation set — stated plainly rather than silently
    borrowing a training segment, because a validation score measured on
    training data is worse than no score at all.
    """
    import random

    groups = list(groups)
    if len(groups) < 2 or val_fraction <= 0:
        return (groups, [])
    wanted = max(1, int(round(len(groups) * val_fraction)))
    wanted = min(wanted, len(groups) - 1)      # never leave training empty
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    val_ids = set(order[:wanted])
    train = [g for i, g in enumerate(groups) if i not in val_ids]
    val = [g for i, g in enumerate(groups) if i in val_ids]
    return (train, val)


# ── COCO assembly ────────────────────────────────────────────────────────

def coco_document(boxes: Sequence, class_names: Sequence,
                  frames: dict) -> dict:
    """Build the COCO dict for one split.

    ``frames`` maps ``(video, time)`` to ``(file_name, width, height)`` — the
    images already extracted. Kept as an argument rather than looked up here so
    this stays pure and the extraction stays testable separately.

    Category ids are 1-based and follow ``class_names``' order, which is the
    contract with the exported model's ``labels.json``.
    """
    category_id = {name: i + 1 for i, name in enumerate(class_names)}
    images, annotations = [], []
    image_id = {}

    for key, (file_name, width, height) in sorted(frames.items()):
        ident = len(images) + 1
        image_id[key] = ident
        images.append({"id": ident, "file_name": file_name,
                       "width": int(width), "height": int(height)})

    for box in boxes:
        key = (box.video, box.time)
        if key not in image_id or box.is_negative:
            continue
        if box.class_name not in category_id:
            continue
        _, width, height = frames[key]
        x, y, w, h = box.pixels(width, height)
        if w < 1.0 or h < 1.0:
            continue                      # a degenerate box teaches nothing
        annotations.append({
            "id": len(annotations) + 1,
            "image_id": image_id[key],
            "category_id": category_id[box.class_name],
            "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            "area": round(w * h, 2),
            "iscrowd": 0,
        })

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": str(n)}
                       for i, n in enumerate(class_names)],
    }


def build_dataset(store: LabelStore, out_dir: str,
                  val_fraction: float = DEFAULT_VAL_FRACTION,
                  seed: int = 0,
                  progress: Optional[Callable] = None) -> dict:
    """Extract the frames and write a COCO dataset ``train_yolox_run`` can read.

    Layout, matching ``yolox.data.COCODataset``::

        out_dir/annotations/train.json   out_dir/train/<frames>
        out_dir/annotations/val.json     out_dir/val/<frames>

    Returns a summary dict: counts per split, the class list, and the frames
    that could not be read.

    Negatives are carried through as images with no annotations. A detector
    trained only on frames that contain the classes never learns what they are
    not, and fires everywhere.
    """
    import cv2

    class_names = store.class_names()
    if not class_names:
        raise ValueError("No accepted labels — nothing to build a dataset from")

    usable = store.accepted() + store.negatives()
    train_groups, val_groups = split_segments(
        segments(usable), val_fraction=val_fraction, seed=seed)

    splits = {
        "train": [b for group in train_groups for b in group],
        "val": [b for group in val_groups for b in group],
    }

    os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)
    summary = {"classes": class_names, "unreadable": [], "splits": {}}
    total = sum(len({(b.video, b.time) for b in boxes})
                for boxes in splits.values())
    done = 0

    for split, boxes in splits.items():
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        frames: dict = {}

        # One capture per video, seeking forward through its labels in time
        # order — reopening per frame costs seconds each on long files.
        by_video: dict = {}
        for box in boxes:
            by_video.setdefault(box.video, set()).add(box.time)

        for video, times in by_video.items():
            capture = cv2.VideoCapture(video)
            if not capture.isOpened():
                summary["unreadable"].append(video)
                continue
            try:
                for moment in sorted(times):
                    capture.set(cv2.CAP_PROP_POS_MSEC, moment * 1000.0)
                    ok, frame = capture.read()
                    done += 1
                    if progress is not None:
                        progress(done, total)
                    if not ok or frame is None:
                        summary["unreadable"].append(f"{video}@{moment:.2f}")
                        continue
                    height, width = frame.shape[:2]
                    name = _frame_name(video, moment)
                    cv2.imwrite(os.path.join(split_dir, name), frame)
                    frames[(video, moment)] = (name, width, height)
            finally:
                capture.release()

        document = coco_document(boxes, class_names, frames)
        path = os.path.join(out_dir, "annotations", f"{split}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(document, fh, indent=2)
        summary["splits"][split] = {
            "images": len(document["images"]),
            "annotations": len(document["annotations"]),
        }

    print(f"[labels] dataset at {out_dir}: "
          + ", ".join(f"{k} {v['images']} images / {v['annotations']} boxes"
                      for k, v in summary["splits"].items()))
    if not summary["splits"].get("val", {}).get("images"):
        # Said out loud, because everything downstream quietly degrades: there
        # is no score to compare rounds by, and "never promote a worse model"
        # has nothing to test. Usually means the labels all sit inside one
        # segment — spread them across the footage, or lower max_span.
        print("[labels] WARNING: the validation split is empty. Training will "
              "report no validation loss, and rounds cannot be compared.")
    return summary


def _frame_name(video: str, moment: float) -> str:
    """A stable file name for one moment of one video.

    Includes the time in milliseconds so two labels a fraction of a second
    apart cannot collide, and the video's stem so two files with the same
    frame number do not overwrite each other.
    """
    stem = os.path.splitext(os.path.basename(video))[0]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return f"{safe}_{int(round(moment * 1000)):09d}.jpg"


# ── importing what already exists ────────────────────────────────────────

def from_labeler_export(path: str, box_fraction: float = 0.12,
                        verdict: str = PENDING) -> list:
    """Read a ``tools/labeler.py`` export into labelled boxes.

    The labeller stores **points**, not boxes, so each point becomes a box of
    ``box_fraction`` of the frame centred on it — the same construction
    ``training/train_yolox_dataset.py`` already applies, kept identical so the
    two paths cannot disagree about what a labelled point means.

    They arrive ``pending`` by default. A fixed-size box around a click is a
    proposal about position, not a statement about extent, and it is worth a
    person's glance before it becomes training data.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    video = str(data.get("video") or "")
    fps = float(data.get("fps") or 0.0) or 30.0
    width = float(data.get("frame_width") or 0.0)
    height = float(data.get("frame_height") or 0.0)
    if width <= 0 or height <= 0:
        raise ValueError(f"{path} has no frame size; cannot normalise its points")

    out = []
    for frame in data.get("keyframes", []):
        index = frame.get("frame_number", frame.get("frame", 0))
        moment = float(index) / fps
        for name, raw in (frame.get("points") or {}).items():
            points = raw if (raw and isinstance(raw[0], list)) else [raw]
            for point in points:
                x, y = float(point[0]), float(point[1])
                out.append(LabelledBox(
                    video=video,
                    time=moment,
                    class_name=str(name),
                    box=(x / width - box_fraction / 2,
                         y / height - box_fraction / 2,
                         box_fraction, box_fraction),
                    source="labeler",
                    confidence=1.0,
                    verdict=verdict,
                ))
    return out
