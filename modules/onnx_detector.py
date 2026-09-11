"""Run a YOLO ONNX export through ONNX Runtime, so detection can use a GPU
that torch cannot reach.

Detection is the heaviest per-frame stage, so on a machine with no CUDA and no
Intel GPU it is the largest thing still on the processor. ``docs/AMD-GPU.md``
names this as the way out: an ONNX export under ONNX Runtime's DirectML
provider, which needs no torch of any particular version and therefore fits in
a packaged build (see :mod:`modules.ort_directml`).

Why the pre- and post-processing is here rather than borrowed
-------------------------------------------------------------
Ultralytics can load a ``.onnx`` itself, but its backend asks ONNX Runtime for
CUDA, CoreML or CPU and never for DirectML, so a model loaded that way runs on
the processor on exactly the machines this exists for. Its loader would also
try to *pip install* an onnxruntime distribution it cannot find by name, which
a frozen build cannot do.

Doing the two ends by hand is ~100 lines of well-trodden arithmetic, and it
buys something beyond the workaround: this file has no Ultralytics import at
all, so the Pro edition — where Ultralytics is barred by licence — can use it
with a different export in front of it.

The result objects are duck-typed to the shape the call sites already expect
from Ultralytics (``results[i].boxes``, ``box.cls``, ``box.conf``,
``box.xyxy[0].cpu().numpy()``), so a caller swaps the detector and changes
nothing else.
"""

from __future__ import annotations

import ast
import os
from typing import Optional

import numpy as np

from modules import ort_directml

# Ultralytics' letterbox fill. Not arbitrary: the model was trained with it, and
# padding with black instead measurably moves boxes near the edges.
PAD_VALUE = 114

DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45


# ---------------------------------------------------------------------------
# The Ultralytics-shaped result objects
# ---------------------------------------------------------------------------

class _Tensorish:
    """A numpy array wearing the two torch methods the call sites reach for.

    ``box.xyxy[0].cpu().numpy()`` is written all over this codebase against
    Ultralytics results. Rather than touch every call site, indexing returns
    another of these (as indexing a tensor returns a tensor) and ``.cpu()`` is
    the no-op it always was for data that never left the processor.
    """

    __slots__ = ("_a",)

    def __init__(self, values):
        self._a = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def tolist(self):
        return self._a.tolist()

    def __getitem__(self, index):
        got = self._a[index]
        return _Tensorish(got) if isinstance(got, np.ndarray) else got

    def __len__(self):
        return len(self._a)

    def __iter__(self):
        return iter(self._a)

    def __float__(self):
        return float(self._a)

    def __int__(self):
        return int(self._a)

    def __repr__(self):  # pragma: no cover - diagnostics only
        return f"_Tensorish({self._a!r})"


class _Box:
    """One detection, in the shape Ultralytics hands back."""

    __slots__ = ("cls", "conf", "xyxy")

    def __init__(self, cls_id, confidence, xyxy):
        self.cls = _Tensorish([float(cls_id)])
        self.conf = _Tensorish([float(confidence)])
        # A (1, 4) block, so `xyxy[0]` is the row — the way the call sites index.
        self.xyxy = _Tensorish([list(xyxy)])


class _Boxes:
    """Iterable of :class:`_Box`, and never None: callers test ``is not None``
    before iterating, so an empty frame must still give them something."""

    __slots__ = ("_boxes",)

    def __init__(self, boxes):
        self._boxes = list(boxes)

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, index):
        return self._boxes[index]


class _Result:
    __slots__ = ("boxes", "orig_shape")

    def __init__(self, boxes, orig_shape):
        self.boxes = boxes
        self.orig_shape = orig_shape


# ---------------------------------------------------------------------------
# The two ends
# ---------------------------------------------------------------------------

def letterbox(frame, imgsz=DEFAULT_IMGSZ):
    """Resize into a square, keeping the aspect ratio and padding the rest.

    Returns the image plus the scale and padding needed to map boxes back —
    the caller needs both, and recomputing them from the shapes afterwards is
    how off-by-one padding errors get in.

    OpenCV is imported here rather than at module scope so the arithmetic below
    (:func:`decode`, the result objects) stays importable and testable without
    it — the test suite shims cv2 away.
    """
    import cv2  # noqa: PLC0415 - see docstring

    height, width = frame.shape[:2]
    scale = min(imgsz / max(1, height), imgsz / max(1, width))
    new_w, new_h = int(round(width * scale)), int(round(height * scale))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), PAD_VALUE, dtype=np.uint8)
    pad_x, pad_y = (imgsz - new_w) // 2, (imgsz - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def _to_input(canvas):
    """HWC BGR uint8 → NCHW RGB float32 in 0..1, which is what the export wants.

    The channel swap is a numpy slice rather than ``cv2.cvtColor`` for the same
    reason the import above is lazy, and it is the same operation.
    """
    rgb = np.asarray(canvas)[:, :, ::-1]
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(np.expand_dims(chw, axis=0))


def _nms(rects, scores, iou_threshold):
    """Greedy non-maximum suppression over (x, y, w, h) boxes, best first.

    Hand-rolled rather than ``cv2.dnn.NMSBoxes`` so the whole decode path can be
    exercised without OpenCV installed. It is the standard algorithm and about
    fifteen lines; the alternative was a test that asserts against a mock.
    """
    if len(scores) == 0:
        return []
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 0] + rects[:, 2]
    y2 = rects[:, 1] + rects[:, 3]
    areas = np.maximum(0.0, rects[:, 2]) * np.maximum(0.0, rects[:, 3])
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        inter_w = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]))
        inter_h = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]))
        inter = inter_w * inter_h
        union = areas[best] + areas[rest] - inter
        # A zero-area box cannot overlap anything; guard rather than divide.
        overlap = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[overlap <= iou_threshold]
    return keep


def _rows(raw):
    """The model's output as (detections, 4 + classes).

    YOLOv8/11 exports emit (1, 4 + nc, anchors) — the transpose of what is
    convenient. Orientation is read from the shape rather than assumed: the
    anchor count (8400 at 640) dwarfs 4 + nc for any class list anyone ships,
    so the longer axis is the detections.
    """
    out = np.asarray(raw)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"unexpected detector output shape {np.asarray(raw).shape}")
    return out.T if out.shape[0] < out.shape[1] else out


def decode(raw, scale, pad_x, pad_y, orig_shape, conf=DEFAULT_CONF,
           iou=DEFAULT_IOU):
    """Model output → boxes in the original frame's pixels.

    Kept separate from the session so it can be tested without a runtime, an
    ONNX file, or a GPU.
    """
    rows = _rows(raw)
    if rows.size == 0:
        return _Boxes([])

    xywh, scores = rows[:, :4], rows[:, 4:]
    if scores.size == 0:
        return _Boxes([])

    class_ids = scores.argmax(axis=1)
    confidences = scores[np.arange(scores.shape[0]), class_ids]
    keep = confidences >= conf
    xywh, class_ids, confidences = xywh[keep], class_ids[keep], confidences[keep]
    if len(confidences) == 0:
        return _Boxes([])

    # cx, cy, w, h (input pixels) → x, y, w, h, which is what NMSBoxes takes.
    x = xywh[:, 0] - xywh[:, 2] / 2
    y = xywh[:, 1] - xywh[:, 3] / 2
    rects = np.stack([x, y, xywh[:, 2], xywh[:, 3]], axis=1)

    kept = _nms(rects, confidences, float(iou))
    if not kept:
        return _Boxes([])

    height, width = orig_shape[:2]
    boxes = []
    for i in kept:
        bx, by, bw, bh = rects[i]
        # Undo the letterbox, then clamp: a box may legitimately be predicted
        # a few pixels outside the frame, and a negative coordinate reaches
        # OpenCV's drawing calls as a wrap-around rectangle.
        x1 = (bx - pad_x) / scale
        y1 = (by - pad_y) / scale
        x2 = (bx + bw - pad_x) / scale
        y2 = (by + bh - pad_y) / scale
        x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
        y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
        boxes.append(_Box(class_ids[i], confidences[i], (x1, y1, x2, y2)))
    return _Boxes(boxes)


# ---------------------------------------------------------------------------
# The detector
# ---------------------------------------------------------------------------

def _names_from_metadata(sess) -> dict:
    """Class names as the export recorded them.

    Ultralytics writes ``names`` into the ONNX metadata as the repr of a dict.
    Read with ``literal_eval``, never ``eval``: this is a file that may have
    come from anywhere, and a label list is not worth a code-execution hole.
    """
    try:
        meta = sess.get_modelmeta().custom_metadata_map or {}
    except Exception:  # noqa: BLE001 - a model without metadata is usable
        return {}
    raw = meta.get("names")
    if not raw:
        return {}
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}
    if isinstance(parsed, dict):
        return {int(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, (list, tuple)):
        return {i: str(v) for i, v in enumerate(parsed)}
    return {}


class OnnxDetector:
    """A YOLO ONNX export, called like an Ultralytics model.

    ``detector(frame, verbose=False, imgsz=640)`` returns a one-element list of
    results, because that is what the call sites iterate.
    """

    def __init__(self, model_path, session=None, imgsz=DEFAULT_IMGSZ,
                 conf=DEFAULT_CONF, iou=DEFAULT_IOU, names=None):
        self.model_path = str(model_path)
        self.session = session if session is not None else ort_directml.session(model_path)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self._input_name = self.session.get_inputs()[0].name
        self.names = names if names is not None else _names_from_metadata(self.session)

    @property
    def backend(self) -> str:
        """Which provider the session actually got — asked, not assumed."""
        return ort_directml.session_backend(self.session)

    def __call__(self, frame, verbose=False, imgsz=None, conf=None, **_ignored):
        canvas, scale, pad_x, pad_y = letterbox(frame, imgsz or self.imgsz)
        outputs = self.session.run(None, {self._input_name: _to_input(canvas)})
        boxes = decode(outputs[0], scale, pad_x, pad_y, frame.shape,
                       conf=self.conf if conf is None else float(conf),
                       iou=self.iou)
        return [_Result(boxes, frame.shape[:2])]

    # Ultralytics' own alias, for a call site that spells it out.
    predict = __call__


def load(model_path, **kwargs) -> Optional[OnnxDetector]:
    """An :class:`OnnxDetector`, or None when this machine cannot run one.

    None rather than an exception: every caller's answer to "no DirectML here"
    is the detector it already had, and that is a fallback, not an error.
    """
    if not ort_directml.available():
        return None
    if not os.path.exists(str(model_path)):
        return None
    try:
        return OnnxDetector(model_path, **kwargs)
    except Exception as e:  # noqa: BLE001 - a broken export must not stop a run
        print(f"⚠️ ONNX detector unavailable: {type(e).__name__}: {e}")
        return None
