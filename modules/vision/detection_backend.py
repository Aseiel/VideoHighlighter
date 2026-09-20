"""Object detector backend — YOLOX (Apache-2.0) detector runtime.

WHY THIS EXISTS
    Both editions detect objects with YOLOX. An AGPL detector would make every
    model trained on top of it AGPL-encumbered too, and models people train and
    share have to be usable by anyone, in any build. This module provides the
    permissively-licensed detector, producing the output the rest of the
    pipeline (object detection, the composition engine, etc.) already consumes.
    Kept byte-identical between the two editions.

STATUS:  ✅ WORKING with official YOLOX COCO weights.
    The pre/post-processing follows YOLOX's official OpenVINO/ONNX demo
    (github.com/Megvii-BaseDetection/YOLOX → demo/OpenVINO/python) and has
    been verified against real footage (person boxes in the AR live preview).

    Install models with tools/get_yolox_model.py — it downloads the official
    Apache-2.0 ONNX exports and converts them to OpenVINO IR under
    models/yolox/, where find_default_yolox_ir() discovers them
    (prefer="large" for object detection, prefer="small" for the live
    person-ROI loop). A model a user trains lands in models/custom/ and is
    picked up through build_object_detector(mode="custom" | "mixed").
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Protocol, Sequence


@dataclass
class Detection:
    """One detected object, in pixel coordinates of the ORIGINAL frame."""
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector(Protocol):
    """The contract the rest of the app codes against. The free build can
    adapt its detector to this; pro wires YoloxOpenVINODetector. Keeping every
    call site on this interface is what lets the two editions swap the
    detector without touching the pipeline."""

    def detect(self, frame_bgr) -> list[Detection]:
        """Run detection on one BGR (OpenCV) frame."""
        ...


# --- YOLOX config (adjust to your exported model) --------------------------
INPUT_SIZE = (640, 640)   # (h, w) fallback when the IR has a dynamic input
STRIDES = (8, 16, 32)     # YOLOX FPN strides
SCORE_THR = 0.30          # tune against free YOLO on real footage
NMS_THR = 0.45

# Default location tools/get_yolox_model.py drops converted IRs into.
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "yolox"


def find_default_yolox_ir(prefer: str = "large") -> str | None:
    """Find a converted YOLOX IR under models/yolox/.

    prefer="large" picks the most accurate model available (object detection);
    prefer="small" picks the fastest (live person detection during AR).
    Returns the .xml path as str, or None when nothing is installed.
    """
    if not DEFAULT_MODEL_DIR.is_dir():
        return None
    # size rank, small -> large; unknown names rank in the middle
    rank = {"nano": 0, "tiny": 1, "s": 2, "m": 3, "l": 4, "x": 5}

    def _key(p: Path) -> int:
        suffix = p.stem.split("_")[-1].lower()
        return rank.get(suffix, 2)

    candidates = sorted(DEFAULT_MODEL_DIR.glob("*.xml"), key=_key)
    if not candidates:
        return None
    chosen = candidates[0] if prefer == "small" else candidates[-1]
    return str(chosen)


class YoloxOpenVINODetector:
    """YOLOX inference via OpenVINO. Implements the Detector protocol.

    Lazy-imports numpy/cv2/openvino so merely importing this module is cheap
    and never pulls heavy deps until a detector is actually constructed.
    """

    def __init__(self, ov_model_xml: str, class_names: Sequence[str],
                 device: str = "GPU", input_size: tuple[int, int] | None = None,
                 score_thr: float = SCORE_THR, nms_thr: float = NMS_THR):
        import openvino as ov  # lazy
        self.class_names = list(class_names)
        self.score_thr = float(score_thr)
        self.nms_thr = float(nms_thr)
        core = ov.Core()
        model = core.read_model(ov_model_xml)
        try:
            self._compiled = core.compile_model(model, device)
        except Exception:
            self._compiled = core.compile_model(model, "CPU")
        self._out = self._compiled.output(0)
        # Input size: explicit arg > static IR shape > module default.
        # (yolox nano/tiny export at 416x416, s/m/l/x at 640x640.)
        if input_size is not None:
            self.input_size = (int(input_size[0]), int(input_size[1]))
        else:
            self.input_size = INPUT_SIZE
            try:
                shape = self._compiled.input(0).partial_shape
                if shape[2].is_static and shape[3].is_static:
                    self.input_size = (shape[2].get_length(), shape[3].get_length())
            except Exception:
                pass

    # -- preprocessing: letterbox to input_size, pad 114, CHW float32 --------
    def _preprocess(self, frame_bgr):
        import cv2
        import numpy as np
        in_h, in_w = self.input_size
        h0, w0 = frame_bgr.shape[:2]
        r = min(in_h / h0, in_w / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.ones((in_h, in_w, 3), dtype=np.float32) * 114.0
        padded[:nh, :nw] = resized
        padded = padded.transpose(2, 0, 1)          # HWC -> CHW
        padded = np.ascontiguousarray(padded[None], dtype=np.float32)  # add batch
        return padded, r

    # -- postprocessing: decode grids/strides, then per-class NMS ------------
    def _decode(self, outputs):
        import numpy as np
        # outputs: [1, n_anchors, 5 + num_classes]  (cx,cy,w,h,obj, cls...)
        in_h, in_w = self.input_size
        grids, expanded_strides = [], []
        for stride in STRIDES:
            gh, gw = in_h // stride, in_w // stride
            xv, yv = np.meshgrid(np.arange(gw), np.arange(gh))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded_strides.append(np.full((1, grid.shape[1], 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    def _infer(self, blob):
        return self._compiled([blob])[self._out]

    def detect(self, frame_bgr) -> list[Detection]:
        import numpy as np
        blob, r = self._preprocess(frame_bgr)
        raw = self._infer(blob)
        preds = self._decode(np.array(raw))[0]     # [n_anchors, 5+C]

        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:5] * preds[:, 5:]        # obj * class
        # cxcywh -> xyxy (in padded 640 space), then undo letterbox scale
        boxes = np.empty_like(boxes_cxcywh)
        boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes /= r

        dets: list[Detection] = []
        for c in range(scores.shape[1]):
            cls_scores = scores[:, c]
            keep_mask = cls_scores > self.score_thr
            if not keep_mask.any():
                continue
            cboxes, cscores = boxes[keep_mask], cls_scores[keep_mask]
            for i in _nms(cboxes, cscores, self.nms_thr):
                x1, y1, x2, y2 = cboxes[i]
                name = self.class_names[c] if c < len(self.class_names) else str(c)
                dets.append(Detection(c, name, float(cscores[i]),
                                      float(x1), float(y1), float(x2), float(y2)))
        return dets


class YoloxOnnxRuntimeDetector(YoloxOpenVINODetector):
    """The same YOLOX model, run by ONNX Runtime instead of OpenVINO.

    OpenVINO's GPU plugin is Intel-only, so on an AMD or NVIDIA card the
    OpenVINO detector runs on the processor. ONNX Runtime's DirectML provider
    reaches any DX12 GPU. Pre-processing and decoding are inherited unchanged —
    it is the same raw-grid export — so only the inference call differs.

    The caller supplies the ``InferenceSession`` (it knows which providers this
    machine has); this class imports no runtime of its own.
    """

    def __init__(self, session, class_names: Sequence[str],
                 score_thr: float = SCORE_THR, nms_thr: float = NMS_THR):
        self.class_names = list(class_names)
        self.score_thr = float(score_thr)
        self.nms_thr = float(nms_thr)
        self._session = session
        inp = session.get_inputs()[0]
        self._input_name = inp.name
        shape = list(getattr(inp, "shape", []) or [])
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = (shape[2], shape[3])
        else:
            self.input_size = INPUT_SIZE

    def _infer(self, blob):
        return self._session.run(None, {self._input_name: blob})[0]


class ModernYoloOnnxDetector:
    """Runtime for a modern single-stage YOLO detect head (the transposed,
    objectness-free output layout used by recent AGPL YOLO trainers), loaded via
    OpenVINO from its exported .onnx or IR. Implements the Detector protocol so
    it drops into every call site YoloxOpenVINODetector serves.

    WHY THIS EXISTS
        A model trained with the common AGPL YOLO toolkit has an output tensor
        layout that differs from YOLOX, so YoloxOpenVINODetector._decode can't
        read it. This class carries ONLY the postprocessing for that published
        output tensor [1, 4+num_classes, num_anchors] — a clean-room decoder with
        no third-party training-framework code and no such import. It lets the
        pro app run a model a user supplies LOCALLY.

    LICENSING
        That YOLO toolkit (and its trained weights) are AGPL. This *code* is
        clean and shippable, but such a model file is AGPL-encumbered — do NOT
        bundle one in a distributed pro build. Point the app at a user-supplied
        model at runtime instead (config: yolox_model_xml). For a fully shippable
        custom detector, retrain with the Apache-2.0 YOLOX path
        (training/train_yolox.py).
    """

    def __init__(self, ov_model_xml: str, class_names: Sequence[str],
                 device: str = "GPU", input_size: tuple[int, int] | None = None,
                 score_thr: float = SCORE_THR, nms_thr: float = NMS_THR):
        import openvino as ov  # lazy
        self.class_names = list(class_names)
        self.score_thr = float(score_thr)
        self.nms_thr = float(nms_thr)
        core = ov.Core()
        model = core.read_model(ov_model_xml)
        try:
            self._compiled = core.compile_model(model, device)
        except Exception:
            self._compiled = core.compile_model(model, "CPU")
        self._out = self._compiled.output(0)
        if input_size is not None:
            self.input_size = (int(input_size[0]), int(input_size[1]))
        else:
            self.input_size = INPUT_SIZE
            try:
                shape = self._compiled.input(0).partial_shape
                if shape[2].is_static and shape[3].is_static:
                    self.input_size = (shape[2].get_length(), shape[3].get_length())
            except Exception:
                pass

    # -- preprocessing: letterbox to input_size, RGB, /255 (modern-YOLO style) --
    def _preprocess(self, frame_bgr):
        import cv2
        import numpy as np
        in_h, in_w = self.input_size
        h0, w0 = frame_bgr.shape[:2]
        r = min(in_h / h0, in_w / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = resized                       # top-left pad (offset 0,0)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # HWC->CHW, normalise
        return np.ascontiguousarray(blob[None], dtype=np.float32), r

    def detect(self, frame_bgr) -> list[Detection]:
        import numpy as np
        blob, r = self._preprocess(frame_bgr)
        raw = self._compiled([blob])[self._out]
        # modern-YOLO output: [1, 4+num_classes, num_anchors] -> [num_anchors, 4+nc]
        preds = np.array(raw)[0].T
        boxes_cxcywh = preds[:, :4]
        scores = preds[:, 4:]                             # already sigmoid'd, no objectness

        # cxcywh (in letterboxed input space) -> xyxy, then undo the resize ratio
        boxes = np.empty_like(boxes_cxcywh)
        boxes[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        boxes[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        boxes[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        boxes[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
        boxes /= r

        dets: list[Detection] = []
        for c in range(scores.shape[1]):
            cls_scores = scores[:, c]
            keep_mask = cls_scores > self.score_thr
            if not keep_mask.any():
                continue
            cboxes, cscores = boxes[keep_mask], cls_scores[keep_mask]
            for i in _nms(cboxes, cscores, self.nms_thr):
                x1, y1, x2, y2 = cboxes[i]
                name = self.class_names[c] if c < len(self.class_names) else str(c)
                dets.append(Detection(c, name, float(cscores[i]),
                                      float(x1), float(y1), float(x2), float(y2)))
        return dets


def create_detector(model_xml: str, class_names: Sequence[str], device: str = "GPU",
                    score_thr: float = SCORE_THR, nms_thr: float = NMS_THR):
    """Build the right Detector for a model file by inspecting its output layout,
    so both YOLOX raw-grid exports and modern single-stage YOLO detect exports work:

        YOLOX      : output [1, num_anchors, 5+num_classes]  (channels last, objectness)
        modern YOLO: output [1, 4+num_classes, num_anchors]  (channels first, no obj)

    The anchor dimension always dwarfs the channel dimension, so whichever of the
    two trailing dims is larger is the anchor axis — which tells us the layout.
    Falls back to YOLOX when the shape is dynamic/unknown (the historical default).
    """
    import openvino as ov  # lazy
    try:
        shape = ov.Core().read_model(model_xml).output(0).partial_shape
        d1, d2 = shape[1], shape[2]
        if d1.is_static and d2.is_static and d1.get_length() < d2.get_length():
            return ModernYoloOnnxDetector(model_xml, class_names, device=device,
                                          score_thr=score_thr, nms_thr=nms_thr)
    except Exception:
        pass
    return YoloxOpenVINODetector(model_xml, class_names, device=device,
                                 score_thr=score_thr, nms_thr=nms_thr)


class MixedDetector:
    """Runs several detectors on each frame and concatenates their detections.

    Backs the 'Mixed' object mode (e.g. COCO + a custom model) so general
    objects and custom classes are detected together. Implements the Detector
    protocol; overlays key on class-name strings, so merging is just append as
    long as the two label sets don't collide.
    """

    def __init__(self, detectors: Sequence["Detector"]):
        self._detectors = [d for d in detectors if d is not None]
        names: list[str] = []
        for d in self._detectors:
            names.extend(getattr(d, "class_names", []) or [])
        self.class_names = list(dict.fromkeys(names))  # de-dup, keep order

    def detect(self, frame_bgr) -> list[Detection]:
        out: list[Detection] = []
        for d in self._detectors:
            out.extend(d.detect(frame_bgr))
        return out


def build_object_detector(mode: str = "coco", custom_model_xml: str = "",
                          device: str = "GPU", default_prefer: str = "large",
                          coco_labels_file: str = "yolo_objects_labels.json",
                          score_thr: float = SCORE_THR, nms_thr: float = NMS_THR,
                          log=print, auto_install: bool = False):
    """Resolve the object detector for a UI mode. Returns (detector, class_names),
    or (None, []) when nothing usable is installed.

        coco   → default YOLOX (models/yolox) with COCO labels
        custom → the user's custom model (names read from its metadata / sidecar)
        mixed  → both, merged via MixedDetector

    `default_prefer` picks the COCO model size: "small" for the live loop,
    "large" for the offline batch pass. Falls back to COCO if a custom model is
    requested but unusable.

    `auto_install` downloads the stock YOLOX models on first use when none are
    present (a source checkout); a frozen build ships them and never downloads.
    """
    mode = (mode or "coco").lower()

    def _coco():
        xml = find_default_yolox_ir(prefer=default_prefer)
        if not xml and auto_install:
            import sys
            if not getattr(sys, "frozen", False):
                try:
                    from modules.vision import yolox_models
                    log("⬇️ First run: fetching the YOLOX object detector (Apache-2.0)…")
                    yolox_models.install(log=log)
                    xml = find_default_yolox_ir(prefer=default_prefer)
                except Exception as e:
                    log(f"⚠️ Could not fetch the YOLOX detector: {e}")
        if not xml:
            log("⚠️ No default YOLOX IR under models/yolox/ (run tools/get_yolox_model.py)")
            return None, []
        names = load_class_names(coco_labels_file)
        if not names:
            log(f"⚠️ COCO labels not found: {coco_labels_file}")
            return None, []
        return create_detector(xml, names, device=device,
                               score_thr=score_thr, nms_thr=nms_thr), names

    def _custom():
        if not custom_model_xml or not os.path.exists(custom_model_xml):
            log(f"⚠️ Custom object model not found: {custom_model_xml or '(none)'}")
            return None, []
        names = names_from_model(custom_model_xml) or load_class_names(
            os.path.join(os.path.dirname(custom_model_xml), "labels.json"))
        if not names:
            log(f"⚠️ No class names for custom model: {custom_model_xml}")
            return None, []
        return create_detector(custom_model_xml, names, device=device,
                               score_thr=score_thr, nms_thr=nms_thr), names

    if mode == "custom":
        det, names = _custom()
        if det is None:
            log("↩️ Falling back to COCO object detector")
            return _coco()
        return det, names

    if mode == "mixed":
        cd, cn = _coco()
        ud, un = _custom()
        if cd and ud:
            return MixedDetector([cd, ud]), list(dict.fromkeys(cn + un))
        return (cd, cn) if cd else (ud, un)

    return _coco()


def names_from_model(model_path: str | Path) -> list[str]:
    """Read class names embedded in a detector model, so the user only has to
    pick the model file (no separate labels file).

    Common YOLO trainers write names into ONNX ``metadata_props`` as
    ``{0: 'a', 1: 'b'}``; OpenVINO IR converted from one keeps them in ``rt_info``.
    Returns [] when no embedded names are found (caller falls back to a labels
    file / COCO default).
    """
    p = Path(model_path)
    try:
        if p.suffix.lower() == ".onnx":
            import onnx  # lazy
            meta = {mp.key: mp.value for mp in onnx.load(str(p)).metadata_props}
            raw = meta.get("names")
            if raw:
                import ast
                parsed = ast.literal_eval(raw)  # "{0: 'a', 1: 'b'}" -> dict
                if isinstance(parsed, dict):
                    return [str(parsed[k]) for k in sorted(parsed)]
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed]
        else:
            import openvino as ov  # lazy
            model = ov.Core().read_model(str(p))
            for key in ("names", "class_names", "labels"):
                try:
                    if model.has_rt_info(key):
                        raw = model.get_rt_info(key).astype(str)
                        import ast
                        parsed = ast.literal_eval(raw)
                        if isinstance(parsed, dict):
                            return [str(parsed[k]) for k in sorted(parsed)]
                        if isinstance(parsed, (list, tuple)):
                            return [str(x) for x in parsed]
                except Exception:
                    continue
    except Exception:
        pass
    return []


def _labels_path(source: str | Path) -> Path:
    """Where a labels file named like ``yolo_objects_labels.json`` actually is.

    A bare filename means *the app's own* labels file, not one in whatever
    directory the process happens to be in. That distinction only shows up in
    the packaged build: it works from the user-data dir (see
    ``app_paths.use_writable_cwd``) while its data ships under ``_MEIPASS``, so
    every relative lookup here resolved beside the exe, found nothing, and the
    COCO detector was dropped for want of names — with the model itself found,
    because ``DEFAULT_MODEL_DIR`` is absolute. From source both are the project
    root, which is why it never showed there.

    A path that exists as given wins, so an explicit file the user picked (or a
    relative one in a source checkout) is still used exactly as before.
    """
    path = Path(source)
    if path.exists() or path.parent != Path("."):
        return path
    from modules.system.app_paths import data_file  # lazy: keeps this import-cheap
    return Path(data_file(path.name))


def load_class_names(source: str | Path | Sequence[str] | None) -> list[str]:
    """Load class names from a list or a JSON labels file.

    Supported JSON shapes:
    - ["person", "car"]
    - {"0": "person", "1": "car"}
    - {"class": {"0": "person", "1": "car"}}
    """
    if source is None:
        return []
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, Path)):
        return [str(item) for item in source]

    path = _labels_path(source)
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict):
        classes = data.get("class", data)
        if isinstance(classes, dict):
            def _sort_key(value: str) -> tuple[int, str]:
                text = str(value)
                return (0, int(text)) if text.isdigit() else (1, text)

            return [str(classes[key]) for key in sorted(classes, key=_sort_key)]

    return []


def _nms(boxes, scores, iou_thr):
    """Standard single-class NMS. boxes: [n,4] xyxy, scores: [n]."""
    import numpy as np
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


# --- Training adapter (legacy .predict() API for model_training) ------------

class _PredictBox:
  """Minimal box object for training code that expects result.boxes tensors.

  ``xyxy`` is a [1, 4] tensor, so ``b.xyxy[0]`` is the four coordinates — the
  shape every caller unpacks (``x1, y1, x2, y2 = map(int, b.xyxy[0])``). It was
  once a list wrapping the [1, 4] tensor, and that unpack raised on every box.
  """

  def __init__(self, xyxy, conf: float, cls_id: int = 0):
    import torch
    self.xyxy = torch.tensor(xyxy, dtype=torch.float32).reshape(1, 4)
    self.conf = torch.tensor([conf], dtype=torch.float32)
    self.cls = torch.tensor([cls_id], dtype=torch.int64)


class _PredictResult:
  def __init__(self, boxes: list[_PredictBox]):
    self.boxes = boxes
    self.keypoints = None


class YoloxPeopleDetector:
  """YOLOX person detector with a .predict() surface for training pipelines.

  Training code historically called ``detector.predict(frame_rgb, classes=[0])``.
  Frames are expected in **RGB** (as in model_training); they are converted to
  BGR before YOLOX inference.
  """

  names = {0: "person"}

  def __init__(self, model_xml: str | None = None, device: str = "GPU",
               score_thr: float = 0.40):
    model_xml = model_xml or find_default_yolox_ir(prefer="small")
    if not model_xml:
      raise FileNotFoundError(
        "No YOLOX IR found. Run: python tools/get_yolox_model.py"
      )
    self.model_xml = model_xml
    self.device = device
    self._detector = YoloxOpenVINODetector(
      model_xml, class_names=["person"], device=device, score_thr=score_thr,
    )

  def to(self, device):
    return self

  def predict(self, frame, conf=0.40, classes=None, verbose=False, **kwargs):
    import cv2
    import numpy as np

    img = frame
    if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes: list[_PredictBox] = []
    for det in self._detector.detect(img):
      if classes is not None and det.class_id not in classes:
        continue
      if det.confidence < conf:
        continue
      xyxy = np.array([[det.x1, det.y1, det.x2, det.y2]], dtype=np.float32)
      boxes.append(_PredictBox(xyxy, det.confidence, det.class_id))
    return [_PredictResult(boxes)]
