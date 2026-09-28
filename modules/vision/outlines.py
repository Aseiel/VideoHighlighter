"""Outlines for detections: the shape inside a box, traced on demand.

Composition rules can be decided on outlines rather than boxes
(``modules/rules/shapes.py``), for the rules that ask (``outline: true``).
This module supplies them, and keeps the cost to what those rules need:

* **Only where it can matter.** An outline lies inside its box, so two shapes
  can only meet, overlap or contain each other where their boxes already
  do. A detection is traced only when its box meets a box of the class its
  rule pairs it with. Everywhere else the answer is already "no", from the
  boxes alone.
* **Once.** Outlines are stored in the analysis cache beside the boxes they
  belong to (``contours``, aligned with ``bboxes``), so changing a threshold
  and re-running costs nothing.
* **At the detector's rate.** Detection keeps one frame per second, so a
  one-hour video is at most 3,600 frames, before the filter above.

Two outliners, one interface (``outline(frame_bgr, boxes_px) -> [contour]``):

``GrabCutOutliner``  OpenCV's GrabCut, seeded with the box. No model, nothing
                     to download, ~50-150 ms per object on a CPU at the working
                     size used here. Good on objects that differ in colour or
                     texture from what is around them; weak where they do not.
``SamOutliner``      Segment Anything, prompted with the boxes, through the
                     ``transformers`` the app already bundles (Apache-2.0 code
                     and weights; the distilled SlimSAM checkpoints are a few
                     tens of MB). One image encoding per frame serves every box
                     in it. Much better on hard footage; needs the weights
                     fetched once, and a GPU to be quick.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from modules.rules.shapes import simplify

# Outlines are stored with this tolerance (fraction of the frame): about two
# pixels at 1080p, and a few dozen points for a typical object.
SIMPLIFY_TOLERANCE = 0.002
# Smallest part of the box an outline may cover before it is distrusted: a
# near-empty mask means the outliner found nothing, and an empty outline would
# make every rule on it silently false.
MIN_FILL = 0.05

DEFAULT_SAM_MODEL = "Zigeng/SlimSAM-uniform-77"


# ---------------------------------------------------------------------------
# masks -> outlines
# ---------------------------------------------------------------------------

def mask_to_contour(mask: np.ndarray, offset=(0, 0), frame_size=None) -> Optional[list]:
    """The largest outer contour of a binary mask, normalised to the frame.

    ``offset`` places a crop's mask in the frame; ``frame_size`` is
    ``(width, height)`` of the whole frame.
    """
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= 0:
        return None
    width, height = frame_size or (binary.shape[1], binary.shape[0])
    ox, oy = offset
    points = [((float(p[0][0]) + ox) / width, (float(p[0][1]) + oy) / height)
              for p in largest]
    points = simplify(points, SIMPLIFY_TOLERANCE)
    return [[round(x, 5), round(y, 5)] for x, y in points] if len(points) >= 3 else None


class GrabCutOutliner:
    """GrabCut inside each box. No model; see the module docstring."""

    name = "grabcut"

    def __init__(self, work_size: int = 192, iterations: int = 4, pad: float = 0.1):
        self.work_size, self.iterations, self.pad = work_size, iterations, pad

    def outline(self, frame_bgr, boxes_px: Sequence) -> list:
        return [self._one(frame_bgr, box) for box in boxes_px]

    def _one(self, frame, box) -> Optional[list]:
        import cv2

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        w, h = x2 - x1, y2 - y1
        if w < 4 or h < 4:
            return None
        pad = int(self.pad * max(w, h))
        X0, Y0 = max(0, x1 - pad), max(0, y1 - pad)
        X1, Y1 = min(width, x2 + pad), min(height, y2 + pad)
        crop = frame[Y0:Y1, X0:X1]
        scale = min(1.0, self.work_size / max(crop.shape[:2]))
        small = (cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else crop)
        rect = (int((x1 - X0) * scale), int((y1 - Y0) * scale),
                max(1, int(w * scale)), max(1, int(h * scale)))
        mask = np.zeros(small.shape[:2], np.uint8)
        bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, mask, rect, bg, fg, self.iterations, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return None
        found = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
        if found.sum() < MIN_FILL * rect[2] * rect[3]:
            return None
        full = cv2.resize(found, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
        return mask_to_contour(full, offset=(X0, Y0), frame_size=(width, height))


class SamOutliner:
    """Segment Anything prompted with the detector's boxes.

    Loaded on first use. ``model_id`` is any ``transformers`` SAM checkpoint;
    the default is a distilled one small enough to ship on demand. The weights
    are fetched by ``transformers`` into its cache the first time.
    """

    name = "sam"

    def __init__(self, model_id: str = DEFAULT_SAM_MODEL, device: str = "auto"):
        self.model_id, self.device = model_id, device
        self._model = self._processor = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import SamModel, SamProcessor

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = SamProcessor.from_pretrained(self.model_id)
            self._model = SamModel.from_pretrained(self.model_id).to(self.device).eval()

    def outline(self, frame_bgr, boxes_px: Sequence) -> list:
        if not len(boxes_px):
            return []
        import cv2
        import torch
        from PIL import Image

        self._load()
        height, width = frame_bgr.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=image,
                                 input_boxes=[[[float(v) for v in b] for b in boxes_px]],
                                 return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = self._model(**inputs, multimask_output=False)
        masks = self._processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0]           # [boxes, 1, H, W]
        result = []
        for box, mask in zip(boxes_px, masks):
            m = mask[0].numpy().astype(np.uint8)
            x1, y1, x2, y2 = box
            if m.sum() < MIN_FILL * max(1.0, (x2 - x1) * (y2 - y1)):
                result.append(None)
                continue
            result.append(mask_to_contour(m, frame_size=(width, height)))
        return result


def make_outliner(name: str = "grabcut"):
    if name == "grabcut":
        return GrabCutOutliner()
    if name == "sam":
        return SamOutliner()
    raise ValueError(f"outliner must be 'grabcut' or 'sam', not {name!r}")


# ---------------------------------------------------------------------------
# which detections, and the pass
# ---------------------------------------------------------------------------

def _boxes_meet(a: Sequence, b: Sequence, gap: float = 0.0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax - gap <= bx + bw and bx - gap <= ax + aw
            and ay - gap <= by + bh and by - gap <= ay + ah)


def wanted(entry: dict, pairs: Iterable) -> set:
    """Indices of the detections in one cache entry that need an outline.

    A detection needs one when its box meets (within the rule's gap) a box of
    the class its rule pairs it with — anywhere else, boxes already answer.
    """
    names = entry.get("objects") or []
    boxes = entry.get("bboxes") or []
    out = set()
    for source, region, gap in pairs:
        src = [i for i, n in enumerate(names) if n == source and i < len(boxes)]
        rgn = [j for j, n in enumerate(names) if n == region and j < len(boxes)]
        for i in src:
            for j in rgn:
                if i != j and _boxes_meet(boxes[i], boxes[j], gap):
                    out.update((i, j))
    return out


def read_frames_at(video_path: str, timestamps: Sequence[float]):
    """``(timestamp, frame)`` for each timestamp, in order, by seeking."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
            ok, frame = cap.read()
            yield ts, (frame if ok else None)
    finally:
        cap.release()


def add_outlines(video_path: str, bboxes: list, pairs: Sequence, *,
                 outliner=None, frame_reader: Optional[Callable] = None,
                 progress: Optional[Callable] = None, cancel=None) -> dict:
    """Trace the outlines ``pairs`` need, into ``bboxes`` in place.

    ``pairs`` is ``CompositionEngine.outline_pairs``. Detections that already
    carry an outline, or were already tried (stored as ``[]``), are skipped,
    so a re-run traces only what is new. Returns counts.
    """
    outliner = outliner or GrabCutOutliner()
    frame_reader = frame_reader or read_frames_at
    todo = {}
    for k, entry in enumerate(bboxes):
        contours = entry.get("contours") or []
        need = [i for i in sorted(wanted(entry, pairs))
                if not (i < len(contours) and contours[i] is not None)]
        if need:
            todo[k] = need
    traced = empty = 0
    order = sorted(todo, key=lambda k: float(bboxes[k].get("timestamp", 0)))
    stamps = [float(bboxes[k].get("timestamp", 0)) for k in order]
    for n, (k, (ts, frame)) in enumerate(zip(order, frame_reader(video_path, stamps))):
        if cancel is not None and cancel.is_set():
            break
        entry = bboxes[k]
        count = len(entry.get("objects") or [])
        contours = list(entry.get("contours") or [])
        contours += [None] * (count - len(contours))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        need = todo[k]
        boxes_px = []
        for i in need:
            x, y, w, h = entry["bboxes"][i]
            boxes_px.append((x * width, y * height, (x + w) * width, (y + h) * height))
        for i, outline in zip(need, outliner.outline(frame, boxes_px)):
            # [] records "tried, found nothing": the box stands in, and the
            # next run does not try again.
            contours[i] = outline or []
            traced += bool(outline)
            empty += not outline
        entry["contours"] = contours
        if progress:
            progress(n + 1, len(order))
    return {"frames": len(order), "traced": traced, "no_outline": empty,
            "outliner": getattr(outliner, "name", type(outliner).__name__)}
