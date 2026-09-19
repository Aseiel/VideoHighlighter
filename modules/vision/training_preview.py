"""What a model finds, round by round, while it trains.

The live detection preview shows what a finished detector sees in a video.
This is its counterpart for a detector that is still learning: after each
round (epoch), the model as it stands is run over a handful of fixed frames it
is *not* trained on, and two things come out of that:

* a number a person can read — "found 5 of 8" — which is the "how is it going"
  line in the panel, and which moves in the direction that matters, unlike a
  loss value nobody outside this file can interpret;
* a picture — the frames with the person's own boxes in grey and the model's
  guesses in colour — which is the live view, for anyone who wants to watch the
  guesses tighten from round to round.

The frames are the validation split ``modules.vision.label_store.build_dataset``
wrote, so they were held out from training on purpose; when there is no
validation split, training frames are used and the view says so, because a
model finding things it was shown is not evidence of anything.

The same frames are used every round, so round 3 and round 20 are directly
comparable — the difference between them is the learning.

No Qt: the window (``modules/ui/training_preview.py``) turns the mosaic into a
pixmap. numpy and cv2 only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

PREVIEW_FRAMES = 6
CELL_SIZE = (320, 180)          # (w, h) of one frame in the mosaic
GAP = 6                         # pixels between frames, so they read as separate
MATCH_IOU = 0.4                 # loose on purpose: a first model boxes roughly
SCORE_THR = 0.3

GT_COLOUR = (170, 170, 170)     # BGR: the person's own box
HIT_COLOUR = (80, 200, 80)      # a guess that matches one of those
MISS_COLOUR = (60, 140, 255)    # a guess that matches nothing


@dataclass
class PreviewFrame:
    image: object                       # BGR ndarray
    truth: list                         # [(class_name, x1, y1, x2, y2)]
    held_out: bool = True               # False when taken from the training split


@dataclass
class RoundSnapshot:
    """One round of the live view, ready to hand across a thread boundary."""

    epoch: int
    total_epochs: int
    found: int
    expected: int
    false_alarms: int
    held_out: bool
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    mosaic_rgb: object = None           # HxWx3 uint8, or None when not rendered
    history: list = field(default_factory=list)   # [(epoch, found, expected)]

    def sentence(self) -> str:
        where = ("on frames it has not learned from" if self.held_out
                 else "on frames it learned from (no held-out frames yet)")
        if self.expected == 0:
            return f"Round {self.epoch} of {self.total_epochs}."
        text = (f"Round {self.epoch} of {self.total_epochs}: found "
                f"{self.found} of {self.expected} {where}")
        if self.false_alarms:
            text += f", {self.false_alarms} wrong guess{'es' if self.false_alarms != 1 else ''}"
        earlier = [h for h in self.history if h[0] < self.epoch]
        if earlier:
            first = earlier[0]
            text += f" (round {first[0]}: {first[1]} of {first[2]})"
        return text + "."


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def pick_frames(dataset_dir: str, limit: int = PREVIEW_FRAMES) -> list:
    """Up to ``limit`` frames with their true boxes, spread across the split.

    Prefers the validation split; frames with boxes first, then one frame with
    none (if any), because watching a model stay quiet on an empty frame is as
    informative as watching it find something.
    """
    import cv2

    for split, held_out in (("val", True), ("train", False)):
        ann_path = os.path.join(dataset_dir, "annotations", f"{split}.json")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        images = sorted(doc.get("images", []), key=lambda i: i.get("file_name", ""))
        if not images:
            continue
        names = {c["id"]: str(c["name"]) for c in doc.get("categories", [])}
        boxes: dict = {}
        for a in doc.get("annotations", []):
            x, y, w, h = a["bbox"]
            boxes.setdefault(a["image_id"], []).append(
                (names.get(a["category_id"], "?"), x, y, x + w, y + h))
        with_boxes = [i for i in images if boxes.get(i["id"])]
        without = [i for i in images if not boxes.get(i["id"])]
        chosen = _spread(with_boxes, limit - (1 if without else 0)) + _spread(without, 1)
        chosen = chosen[:limit]

        frames = []
        for info in chosen:
            image = cv2.imread(os.path.join(dataset_dir, split, info["file_name"]))
            if image is not None:
                frames.append(PreviewFrame(image, boxes.get(info["id"], []), held_out))
        if frames:
            return frames
    return []


def _spread(items: Sequence, n: int) -> list:
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(detections: Sequence, truth: Sequence, iou: float = MATCH_IOU) -> tuple:
    """(found, expected, false_alarms, hit_flags) for one frame.

    Greedy by confidence: each true box can be claimed once, by a guess of the
    same class that overlaps it enough. ``hit_flags`` marks which detections
    matched, for colouring.
    """
    order = sorted(range(len(detections)),
                   key=lambda i: -float(getattr(detections[i], "confidence", 0.0)))
    claimed = [False] * len(truth)
    hit = [False] * len(detections)
    for i in order:
        d = detections[i]
        box = (d.x1, d.y1, d.x2, d.y2)
        best, best_j = iou, -1
        for j, t in enumerate(truth):
            if claimed[j] or t[0] != d.class_name:
                continue
            overlap = _iou(box, t[1:])
            if overlap >= best:
                best, best_j = overlap, j
        if best_j >= 0:
            claimed[best_j] = True
            hit[i] = True
    found = sum(claimed)
    return found, len(truth), hit.count(False), hit


def evaluate(detector, frames: Sequence, score_thr: float = SCORE_THR) -> tuple:
    """Run ``detector`` over the frames → (found, expected, false_alarms, per_frame)
    where ``per_frame`` is ``[(detections, hit_flags)]`` for drawing."""
    found = expected = alarms = 0
    per_frame = []
    for f in frames:
        dets = [d for d in detector.detect(f.image) if d.confidence >= score_thr]
        fo, ex, fa, hits = match(dets, f.truth)
        found, expected, alarms = found + fo, expected + ex, alarms + fa
        per_frame.append((dets, hits))
    return found, expected, alarms, per_frame


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def render(frames: Sequence, per_frame: Sequence, caption: str = "",
           cell: tuple = CELL_SIZE) -> object:
    """One RGB mosaic: every preview frame, truth in grey, guesses in colour."""
    import cv2
    import numpy as np

    cw, ch = cell
    cols = 3 if len(frames) > 4 else max(1, min(2, len(frames)))
    rows = max(1, -(-len(frames) // cols))
    header, gap = 26, GAP
    canvas = np.full((header + rows * ch + (rows - 1) * gap,
                      cols * cw + (cols - 1) * gap, 3), 24, dtype=np.uint8)
    if caption:
        cv2.putText(canvas, _ascii(caption)[:90], (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (220, 220, 220), 1, cv2.LINE_AA)

    for idx, (frame, (dets, hits)) in enumerate(zip(frames, per_frame)):
        img = frame.image
        h0, w0 = img.shape[:2]
        scale = min(cw / w0, ch / h0)
        nw, nh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
        tile = np.full((ch, cw, 3), 12, dtype=np.uint8)
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        tile[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        def _pt(x, y):
            return int(ox + x * scale), int(oy + y * scale)

        for name, x1, y1, x2, y2 in frame.truth:
            cv2.rectangle(tile, _pt(x1, y1), _pt(x2, y2), GT_COLOUR, 1, cv2.LINE_AA)
        for d, ok in zip(dets, hits):
            colour = HIT_COLOUR if ok else MISS_COLOUR
            p1, p2 = _pt(d.x1, d.y1), _pt(d.x2, d.y2)
            cv2.rectangle(tile, p1, p2, colour, 2, cv2.LINE_AA)
            cv2.putText(tile, f"{_ascii(d.class_name)[:14]} {d.confidence:.2f}",
                        (p1[0] + 2, max(12, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, colour, 1, cv2.LINE_AA)
        if not frame.truth and not dets:
            cv2.putText(tile, "nothing here - correctly quiet", (6, ch - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, HIT_COLOUR, 1, cv2.LINE_AA)

        r, c = divmod(idx, cols)
        top, left = header + r * (ch + gap), c * (cw + gap)
        canvas[top:top + ch, left:left + cw] = tile

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def _ascii(text: str) -> str:
    """cv2's Hershey fonts draw ASCII only; anything else becomes '?'."""
    return str(text).encode("ascii", "replace").decode("ascii")


def snapshot(report, frames: Sequence, history: list, draw: bool) -> RoundSnapshot:
    """Evaluate one ``training.train_yolox_run.EpochReport`` → ``RoundSnapshot``.

    ``history`` is the caller's running list; this appends to it. Rendering is
    skipped when nobody is watching — the numbers are cheap, the picture is not
    free.
    """
    found, expected, alarms, per_frame = evaluate(report.detector, frames)
    history.append((report.epoch, found, expected))
    held_out = all(f.held_out for f in frames) if frames else True
    snap = RoundSnapshot(
        epoch=report.epoch, total_epochs=report.total_epochs,
        found=found, expected=expected, false_alarms=alarms, held_out=held_out,
        train_loss=report.train_loss, val_loss=report.val_loss,
        history=list(history),
    )
    if draw and frames:
        # Short on purpose: the full sentence sits under the picture, where it can wrap.
        snap.mosaic_rgb = render(
            frames, per_frame,
            caption=f"Round {snap.epoch} of {snap.total_epochs}: found {found} of {expected}")
    return snap
