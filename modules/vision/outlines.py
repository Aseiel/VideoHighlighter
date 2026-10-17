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
``OpenVinoSamOutliner`` the same SlimSAM, exported by ``tools/export_sam.py``
                     and run by OpenVINO, so nothing imports ``transformers``:
                     that import alone took ~22 s, which every first outline
                     in a session waited for. Loads in a second or two; the
                     image encoder runs on an Intel GPU when there is one
                     (~75 ms a frame on an Arc, ~620 ms on a CPU) and the box
                     decoder on the CPU (~20 ms a box; OpenVINO's GPU plugin
                     crashes on it).

``sam_outliner()`` picks the best SAM there is: OpenVINO when the export is
installed and reproduces its probe, else ``transformers``.
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
# The OpenVINO export (tools/export_sam.py): pinned, so a rebuild gives the
# same model, and checked at load against a probe it was exported with.
SAM_EXPORT_ID = "slimsam-uniform-77"
SAM_SOURCE_REVISION = "79c09c1ce6b4ae51f00634ed171d9b8e888f6911"
SAM_DIRNAME = SAM_EXPORT_ID
SAM_ENCODER_FILE = "encoder.onnx"
SAM_DECODER_FILE = "decoder.onnx"
SAM_META_FILE = "outliner.json"
SAM_FORMAT = 1
SAM_DIR_ENV = "VH_SAM_DIR"
# SAM's own preprocessing (SamProcessor): longest side to 1024, ImageNet
# normalisation, padded to a square; masks come back at a quarter of that.
SAM_INPUT = 1024
SAM_MASK = 256
SAM_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SAM_STD = np.array([0.229, 0.224, 0.225], np.float32)
# A route must trace the probe this close to how PyTorch traced it at export.
SAM_PROBE_MIN_IOU = 0.95
# What the SAM processor returns for post-processing only: moved, never cast.
_SIZE_KEYS = ("original_sizes", "reshaped_input_sizes")


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
                self.device = pick_device(torch)
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
        inputs = to_device(inputs, self.device)
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


def pick_device(torch) -> str:
    """CUDA, then an Intel GPU (``torch.xpu``, in builds that have it), then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return "cpu"


def to_device(inputs, device) -> dict:
    """The processor's tensors on ``device``, floating ones as float32 first.

    The processor returns some prompts (``input_boxes``) as float64, which an
    Intel GPU cannot hold at all, and the model runs in float32 anyway. The
    sizes post-processing reads are moved as they are.
    """
    out = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if key not in _SIZE_KEYS and value.is_floating_point():
                value = value.float()
            value = value.to(device)
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# SAM through OpenVINO (no transformers at run time)
# ---------------------------------------------------------------------------

def sam_preprocess(frame_bgr: np.ndarray) -> tuple:
    """A frame as SAM's encoder takes it: RGB, longest side to ``SAM_INPUT``,
    normalised, padded at the bottom and right. Returns ``(pixels [1,3,S,S],
    (resized_h, resized_w))``."""
    import cv2

    h, w = frame_bgr.shape[:2]
    scale = SAM_INPUT / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (nw, nh), interpolation=interp), cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 255.0 - SAM_MEAN) / SAM_STD
    pixels = np.zeros((SAM_INPUT, SAM_INPUT, 3), np.float32)
    pixels[:nh, :nw] = norm
    return pixels.transpose(2, 0, 1)[None], (nh, nw)


def sam_boxes(boxes_px: Sequence, frame_size: tuple, resized: tuple) -> np.ndarray:
    """Pixel boxes ``(x1, y1, x2, y2)`` in the frame -> in the resized frame."""
    (h, w), (nh, nw) = frame_size, resized
    scale = np.array([nw / w, nh / h, nw / w, nh / h], np.float32)
    return np.asarray(boxes_px, np.float32).reshape(-1, 4) * scale


def sam_mask(logits: np.ndarray, resized: tuple, frame_size: tuple) -> np.ndarray:
    """One low-resolution mask's logits -> a binary mask the size of the frame,
    as SamProcessor post-processes: up to the padded input, crop the padding,
    then to the frame."""
    import cv2

    (nh, nw), (h, w) = resized, frame_size
    full = cv2.resize(logits.astype(np.float32), (SAM_INPUT, SAM_INPUT),
                      interpolation=cv2.INTER_LINEAR)[:nh, :nw]
    return (cv2.resize(full, (w, h), interpolation=cv2.INTER_LINEAR) > 0.0).astype(np.uint8)


def sam_probe() -> tuple:
    """The image and box every route must outline the same way: a dark disc
    with a lighter bar across it, on seeded noise. ``(frame_bgr, box_px)``."""
    import cv2

    rng = np.random.default_rng(1234)
    frame = rng.integers(90, 170, (360, 480, 3), dtype=np.uint8)
    cv2.circle(frame, (200, 180), 90, (40, 60, 160), -1)
    cv2.rectangle(frame, (150, 160), (330, 200), (220, 200, 60), -1)
    return frame, (100.0, 80.0, 340.0, 280.0)


def outline_iou(a: Sequence, b: Sequence, size: int = 256) -> float:
    """IoU of two normalised outlines, drawn on a grid."""
    import cv2

    if not a or not b:
        return 0.0
    ma, mb = np.zeros((size, size), np.uint8), np.zeros((size, size), np.uint8)
    cv2.fillPoly(ma, [np.array([[x * size, y * size] for x, y in a], np.int32)], 1)
    cv2.fillPoly(mb, [np.array([[x * size, y * size] for x, y in b], np.int32)], 1)
    union = int((ma | mb).sum())
    return float((ma & mb).sum()) / union if union else 0.0


def sam_dir_candidates() -> list:
    """Where the export may be, most specific first: the override, then the
    frame encoder's places (beside the exe, per user, the bundle, the source
    checkout), each with this model's folder name."""
    import os

    from modules.vision import frame_encoder as fe

    out = []
    override = os.environ.get(SAM_DIR_ENV, "").strip()
    if override:
        out.append(override)
    fe_override = os.environ.get(fe.DIR_ENV, "").strip()
    for folder in fe.model_dir_candidates():
        if fe_override and folder == fe_override:
            continue
        out.append(os.path.join(os.path.dirname(folder), SAM_DIRNAME))
    return out


def find_sam_dir() -> Optional[str]:
    import os

    for folder in sam_dir_candidates():
        if all(os.path.isfile(os.path.join(folder, f))
               for f in (SAM_ENCODER_FILE, SAM_DECODER_FILE, SAM_META_FILE)):
            return folder
    return None


def read_sam_meta(folder: str) -> dict:
    """outliner.json, checked against what this app was built for."""
    import json
    import os

    with open(os.path.join(folder, SAM_META_FILE), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("id") != SAM_EXPORT_ID:
        raise ValueError(f"{SAM_META_FILE} is for {meta.get('id')!r}, this app uses {SAM_EXPORT_ID!r}")
    if int(meta.get("format", 0)) > SAM_FORMAT:
        raise ValueError(f"{SAM_META_FILE} format {meta.get('format')} is newer than this app")
    if len(meta.get("probe_outline") or []) < 3:
        raise ValueError(f"{SAM_META_FILE} has no probe outline")
    return meta


class OpenVinoSamOutliner:
    """SlimSAM run by OpenVINO from the ``tools/export_sam.py`` export.

    Same interface and the same masks as ``SamOutliner`` (checked at load
    against the outline PyTorch drew for ``sam_probe()``), without importing
    ``transformers`` or ``torch``. The image encoder goes to an Intel GPU when
    OpenVINO sees one, else the CPU; the box decoder always runs on the CPU.
    A GPU that traces the probe wrongly is dropped for the CPU.
    """

    name = "sam"

    def __init__(self, folder: Optional[str] = None, device: str = "auto"):
        self.folder, self.device = folder, device
        self._encoder = self._decoder = None

    @staticmethod
    def _compile(path: str, device: str):
        import openvino as ov

        from modules.vision import frame_encoder as fe

        core = ov.Core()
        cache = fe._compile_cache_dir() if device.startswith("GPU") else None
        if cache:
            core.set_property({"CACHE_DIR": cache})
        # fp32 on every device. The GPU's default fp16 overflows in SAM's image
        # encoder: every embedding came back NaN (measured on an Arc), and the
        # probe check below caught it. fp32 there costs ~95 ms a frame.
        return core.compile_model(path, device, {"INFERENCE_PRECISION_HINT": "f32"})

    def _load(self):
        if self._encoder is not None:
            return
        import os

        from modules.vision import frame_encoder as fe

        folder = self.folder or find_sam_dir()
        if not folder:
            raise FileNotFoundError(f"the SAM export (models/{SAM_DIRNAME}) is not installed")
        meta = read_sam_meta(folder)
        self._decoder = self._compile(os.path.join(folder, SAM_DECODER_FILE), "CPU")
        wanted = self.device
        if wanted == "auto":
            wanted = fe._openvino_gpu()[0] or "CPU"
        devices = [wanted] + (["CPU"] if wanted != "CPU" else [])
        last = None
        for device in devices:
            try:
                self._encoder = self._compile(os.path.join(folder, SAM_ENCODER_FILE), device)
                frame, box = sam_probe()
                got = self.outline(frame, [box])[0]
                iou = outline_iou(got, meta["probe_outline"])
                if iou < SAM_PROBE_MIN_IOU:
                    raise ValueError(f"traces the probe at IoU {iou:.3f} against PyTorch's")
                self.device, self.folder = f"OpenVINO {device}", folder
                return
            except Exception as exc:          # noqa: BLE001 - try the next device
                print(f"[outlines] SAM encoder on OpenVINO {device} unusable: {exc}")
                self._encoder, last = None, exc
        raise RuntimeError(f"no OpenVINO device could run the SAM export ({last})")

    def outline(self, frame_bgr, boxes_px: Sequence) -> list:
        if not len(boxes_px):
            return []
        if self._encoder is None:
            self._load()
        h, w = frame_bgr.shape[:2]
        pixels, resized = sam_preprocess(frame_bgr)
        embeddings = np.asarray(self._encoder(pixels)[0], np.float32)
        out = self._decoder([embeddings, sam_boxes(boxes_px, (h, w), resized)])
        logits = np.asarray(out[0], np.float32)          # [boxes, 256, 256]
        result = []
        for box, low in zip(boxes_px, logits):
            mask = sam_mask(low, resized, (h, w))
            x1, y1, x2, y2 = box
            if mask.sum() < MIN_FILL * max(1.0, (x2 - x1) * (y2 - y1)):
                result.append(None)
                continue
            result.append(mask_to_contour(mask, frame_size=(w, h)))
        return result


def sam_outliner(device: str = "auto"):
    """The best SAM there is: the OpenVINO export when it is installed and
    traces its probe right (loads in seconds), else ``transformers`` (whose
    import alone takes ~20 s)."""
    if find_sam_dir():
        try:
            outliner = OpenVinoSamOutliner(device=device)
            outliner._load()
            return outliner
        except Exception as exc:               # noqa: BLE001 - fall back to transformers
            print(f"[outlines] OpenVINO SAM unusable ({exc}); using transformers")
    return SamOutliner()


def make_outliner(name: str = "grabcut"):
    if name == "grabcut":
        return GrabCutOutliner()
    if name == "sam":
        return sam_outliner()
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
