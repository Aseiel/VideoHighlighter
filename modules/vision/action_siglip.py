"""
action_siglip.py — action recognition with the shared frame encoder and a
taught action head.

Replaces the Intel encoder/decoder and R3D paths. What it does per window of
the video (WINDOW_S long, one every STRIDE_S):

    4 frames spread over the window
      -> people found by YOLOX on those frames
      -> one crop per person (the 2 largest; the union of their boxes over the
         window, 20 % margin), or the whole frame when nobody is found
      -> the frame encoder (SigLIP2) on every crop's 4 frames
      -> the action head: one sigmoid score per action
      -> an action is detected when its score reaches its trust threshold;
         a window's score for an action is its best crop's

Why crops rather than whole frames: heads are trained on the dataset cropper's
one-action clips. Measured on unseen footage, whole frames sorted 16 % of
windows against 40 % for per-person crops and 43 % for the real cropper, which
costs ~7 s per 5-s window (docs/plans/2026-10-03-automatic-sorting.md).

A head is a folder with ``head.onnx`` and ``head.json`` as written by
``model_training.action_head.train``. Its classes are the user's own; nothing
here names any of them.

Returns the same thing run_action_detection() does, so the pipeline needs no
second code path: ``[(timestamp, frame_id, action_id, score, name), ...]`` and
a list of ``{timestamp, action_name, confidence, bbox, model_type}`` boxes.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

HEAD_MODEL = "head.onnx"
HEAD_META = "head.json"
HEAD_KIND = "action-head"
HEAD_DIR_ENV = "VH_ACTION_HEAD_DIR"

WINDOW_S = 5.0        # the dataset's clip length: the head learned that time scale
STRIDE_S = 2.5        # windows overlap by half
PEOPLE_PER_WINDOW = 2
CROP_MARGIN = 0.20
PERSON_CONF = 0.40
ENCODE_BATCH = 32     # crops' frames per encoder call, across windows
MODEL_TYPE = "siglip2"

LogFn = Callable[[str], None]


# ── heads ────────────────────────────────────────────────────────────────────

def _head_dirs() -> List[str]:
    dirs = []
    env = os.environ.get(HEAD_DIR_ENV)
    if env:
        dirs.append(env)
    try:
        from modules.system import app_paths
        root = app_paths.action_models_dir()
        if os.path.isdir(root):
            dirs += sorted((os.path.join(root, d) for d in os.listdir(root)),
                           key=lambda p: os.path.getmtime(p), reverse=True)
    except Exception:  # noqa: BLE001 - no app paths, no managed heads
        pass
    return dirs


def read_head_meta(folder: str) -> dict:
    with open(os.path.join(folder, HEAD_META), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("kind") != HEAD_KIND:
        raise ValueError(f"{HEAD_META} is not an action head (kind={meta.get('kind')!r})")
    if not os.path.isfile(os.path.join(folder, HEAD_MODEL)):
        raise ValueError(f"{HEAD_MODEL} is missing")
    return meta


def find_heads(encoder_id: Optional[str] = None) -> List[str]:
    """Head folders that can run here, newest first; ``VH_ACTION_HEAD_DIR``
    wins. With ``encoder_id``, only heads trained on that encoder."""
    found = []
    for folder in _head_dirs():
        try:
            meta = read_head_meta(folder)
        except Exception:  # noqa: BLE001 - not a head, or a broken one
            continue
        if encoder_id and meta.get("encoder") != encoder_id:
            continue
        found.append(folder)
    return found


def available() -> bool:
    """True when the frame encoder and a head trained on it are installed."""
    try:
        from modules.vision import frame_encoder
        return frame_encoder.is_installed() and bool(find_heads(frame_encoder.ENCODER_ID))
    except Exception:  # noqa: BLE001 - anything missing means "not here"
        return False


def installed_head_classes() -> Optional[tuple]:
    """``(head name, classes)`` of the head a run would use, or None. For the
    action picker: with this backend the user chooses among their own classes."""
    try:
        from modules.vision import frame_encoder
        heads = find_heads(frame_encoder.ENCODER_ID)
        if not heads:
            return None
        meta = read_head_meta(heads[0])
        return os.path.basename(os.path.normpath(heads[0])), list(meta.get("classes", []))
    except Exception:  # noqa: BLE001
        return None


class ActionHead:
    """head.onnx + head.json: frame vectors in, one score per action out."""

    def __init__(self, folder: str):
        import onnxruntime as ort

        self.folder = folder
        self.meta = read_head_meta(folder)
        self.classes: List[str] = list(self.meta["classes"])
        self.frames = int(self.meta.get("frames", 4))
        self.encoder_id = self.meta.get("encoder")
        self.thresholds = [None if t is None else float(t)
                           for t in self.meta.get("trust_thresholds", [None] * len(self.classes))]
        index = {c: i for i, c in enumerate(self.classes)}
        self.pairs = []  # (a, b, threshold) for trusted pairs
        for p in self.meta.get("pair_thresholds", []):
            a, b = p.get("actions", (None, None))
            if p.get("threshold") is not None and a in index and b in index:
                self.pairs.append((index[a], index[b], float(p["threshold"])))
        self._session = ort.InferenceSession(os.path.join(folder, HEAD_MODEL),
                                             providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name

    @property
    def name(self) -> str:
        return os.path.basename(os.path.normpath(self.folder))

    @property
    def trusted(self) -> List[str]:
        return [c for c, t in zip(self.classes, self.thresholds) if t is not None]

    def scores(self, features: np.ndarray) -> np.ndarray:
        """[N, frames, dims] -> [N, classes] sigmoid scores."""
        logits = self._session.run(None, {self._input: features.astype(np.float32)})[0]
        return 1.0 / (1.0 + np.exp(-logits))

    def detected(self, scores: np.ndarray) -> np.ndarray:
        """[N, classes] bool: trusted actions at or over their threshold, plus
        both actions of a trusted pair whose lower score reaches the pair's."""
        th = np.array([np.inf if t is None else t for t in self.thresholds])
        out = scores >= th
        for a, b, t in self.pairs:
            both = np.minimum(scores[:, a], scores[:, b]) >= t
            out[both, a] = True
            out[both, b] = True
        return out


# ── windows and crops ────────────────────────────────────────────────────────

def window_frames(total_frames: int, fps: float, k: int,
                  window_s: float = WINDOW_S, stride_s: float = STRIDE_S) -> List[List[int]]:
    """Frame numbers for every window: ``k`` per window, centres of k equal
    parts, as the trainer samples a clip. A video shorter than one window is
    one window."""
    fps = fps if fps and fps > 0 else 25.0
    win = max(1, int(round(window_s * fps)))
    step = max(1, int(round(stride_s * fps)))
    total = max(1, int(total_frames))
    starts = list(range(0, max(1, total - win + 1), step))
    if total > win and starts[-1] + win < total:
        starts.append(total - win)          # the tail gets a window too
    out = []
    for s in starts:
        length = min(win, total - s)
        out.append([min(total - 1, s + int((i + 0.5) * length / k)) for i in range(k)])
    return out


def window_seconds(windows: Sequence[Sequence[int]], w: int, fps: float,
                   last_second: int) -> range:
    """The whole seconds window ``w`` reports on. Neighbouring windows split
    the time between them halfway between their centres, half-open, so a
    second is never claimed twice and none is skipped; the first window starts
    at 0 and the last runs to the end. One detection per second lets the
    pipeline's 1.3 s grouping join neighbours into one sequence."""
    def centre(i):
        return (windows[i][0] + windows[i][-1]) / 2 / fps

    lo = 0 if w == 0 else int(np.ceil((centre(w - 1) + centre(w)) / 2))
    hi = (last_second if w == len(windows) - 1
          else int(np.ceil((centre(w) + centre(w + 1)) / 2)))
    return range(lo, max(lo, hi))


def _area(b) -> float:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def person_regions(boxes_per_frame: Sequence[Sequence[tuple]], width: int, height: int,
                   people: int = PEOPLE_PER_WINDOW, margin: float = CROP_MARGIN) -> List[tuple]:
    """One fixed region per person for the whole window.

    The ``people`` largest boxes of the frame that sees the most people are the
    anchors; each anchor takes, on every frame, the box that overlaps it most,
    and its region is the union of those boxes plus ``margin`` on each side.
    No people anywhere: the whole frame, once.
    """
    frames = [list(b) for b in boxes_per_frame]
    if not any(frames):
        return [(0, 0, width, height)]
    seed = max(frames, key=len)
    anchors = sorted(seed, key=_area, reverse=True)[:people]
    regions = []
    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        for fb in frames:
            if not fb:
                continue
            best = max(fb, key=lambda b: _iou(anchor, b))
            if _iou(anchor, best) > 0.1:
                x1, y1 = min(x1, best[0]), min(y1, best[1])
                x2, y2 = max(x2, best[2]), max(y2, best[3])
        mx, my = margin * (x2 - x1), margin * (y2 - y1)
        regions.append((int(max(0, x1 - mx)), int(max(0, y1 - my)),
                        int(min(width, x2 + mx)), int(min(height, y2 + my))))
    return regions


# ── the run ──────────────────────────────────────────────────────────────────

def _read_frames(video_path: str, wanted: Iterable[int], on_frame) -> int:
    """Decode front to back once and hand each wanted frame to ``on_frame``.
    Returns how many frames were read."""
    import cv2

    wanted = sorted(set(wanted))
    cap = cv2.VideoCapture(video_path)
    index = 0
    try:
        for target in wanted:
            while index < target:
                if not cap.grab():
                    return index
                index += 1
            ok, frame = cap.read()
            index += 1
            if not ok:
                return index
            if on_frame(target, frame) is False:
                return index
    finally:
        cap.release()
    return index


def run_action_detection_siglip(video_path: str, *, head: Optional[ActionHead] = None,
                                encoder=None, detector=None, device: str = "AUTO",
                                interesting_actions: Optional[Sequence[str]] = None,
                                progress_callback=None, cancel_flag=None,
                                log: LogFn = print, window_s: float = WINDOW_S,
                                stride_s: float = STRIDE_S):
    """Timed action detections for ``video_path`` (see the module docstring).

    ``head``/``encoder``/``detector`` default to the newest installed head, the
    frame encoder on the best route here, and YOLOX on ``device``. Returns
    ``(detections, bboxes)``; ``([], [])`` when something needed is missing,
    after saying what in ``log``.
    """
    import cv2

    from modules.vision import frame_encoder

    if head is None:
        heads = find_heads(frame_encoder.ENCODER_ID)
        if not heads:
            log("⚠️ Action recognition: no trained action head found "
                f"(train one, or set {HEAD_DIR_ENV})")
            return [], []
        head = ActionHead(heads[0])
    if encoder is None:
        encoder = frame_encoder.load(log=log)
        if encoder is None:
            return [], []
    if head.encoder_id and head.encoder_id != encoder.encoder_id:
        log(f"⚠️ Action head {head.name} was trained on {head.encoder_id}, "
            f"not {encoder.encoder_id}; skipping action recognition")
        return [], []
    if detector is None:
        from modules.vision.detection_backend import YoloxPeopleDetector
        detector = YoloxPeopleDetector(device=device, score_thr=PERSON_CONF)

    wanted_names = None
    if interesting_actions:
        wanted_names = {a.strip().lower() for a in interesting_actions if a and a.strip()}
        untrusted = sorted(a for a in wanted_names
                           if a in {c.lower() for c in head.classes}
                           and a not in {c.lower() for c in head.trusted})
        unknown = sorted(a for a in wanted_names if a not in {c.lower() for c in head.classes})
        if untrusted:
            log(f"ℹ️ Not reported, the head's held-out evidence is too thin to trust: "
                f"{', '.join(untrusted)}")
        if unknown:
            log(f"ℹ️ Not in action head {head.name}: {', '.join(unknown)}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if total <= 0 or width <= 0:
        log(f"⚠️ Action recognition: cannot read {os.path.basename(video_path)}")
        return [], []

    windows = window_frames(total, fps, head.frames, window_s, stride_s)
    owner = {}  # frame -> windows that use it
    for w, frames in enumerate(windows):
        for f in frames:
            owner.setdefault(f, []).append(w)
    remaining = {w: len(set(frames)) for w, frames in enumerate(windows)}
    pending = {}            # frame -> (bgr, boxes) until its windows are done
    queue = []              # (window, region, [4 crops]) waiting for the encoder
    window_scores = {}      # window -> (best score per class, region per class)
    log(f"🎬 Action recognition: {head.name} ({len(head.trusted)} of "
        f"{len(head.classes)} actions trusted) on {encoder.label}, "
        f"{len(windows)} windows of {window_s:g} s")

    def flush():
        if not queue:
            return
        feats = encoder.encode_bgr([c for _, _, crops in queue for c in crops])
        feats = feats.reshape(len(queue), head.frames, -1)
        scores = head.scores(feats)
        for (w, region, _), s in zip(queue, scores):
            best, where = window_scores.get(w, (None, None))
            if best is None:
                window_scores[w] = (s.copy(), [region] * len(s))
            else:
                better = s > best
                best[better] = s[better]
                for i in np.flatnonzero(better):
                    where[i] = region
        queue.clear()

    def on_frame(index, frame):
        if cancel_flag is not None and cancel_flag.is_set():
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.predict(rgb, conf=PERSON_CONF, classes=[0], verbose=False)
        boxes = [tuple(int(v) for v in b.xyxy[0]) for r in result for b in r.boxes]
        pending[index] = (frame, boxes)
        for w in owner[index]:
            remaining[w] -= 1
            if remaining[w] == 0:
                frames = windows[w]
                regions = person_regions([pending[f][1] for f in frames], width, height)
                for region in regions:
                    x1, y1, x2, y2 = region
                    queue.append((w, region, [pending[f][0][y1:y2, x1:x2] for f in frames]))
                if len(queue) * head.frames >= ENCODE_BATCH:
                    flush()
                if progress_callback:
                    progress_callback(w + 1, len(windows), "Action recognition",
                                      f"{(w + 1) * 100 // len(windows)}%")
        # A frame is kept only while a window still needs it.
        for f in [f for f in pending if all(remaining[w] == 0 for w in owner[f])]:
            del pending[f]
        return True

    _read_frames(video_path, owner.keys(), on_frame)
    flush()

    detections, bboxes = [], []
    last_second = int(np.ceil(total / fps))
    for w in sorted(window_scores):
        scores, where = window_scores[w]
        hits = head.detected(scores[None, :])[0]
        seconds = window_seconds(windows, w, fps, last_second)
        for i in np.flatnonzero(hits):
            name = head.classes[i]
            if wanted_names is not None and name.lower() not in wanted_names:
                continue
            x1, y1, x2, y2 = where[i]
            box = [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]
            for sec in seconds:
                if sec < 0 or sec * fps >= total:
                    continue
                detections.append((float(sec), int(sec * fps), int(i), float(scores[i]), name))
                bboxes.append({"timestamp": float(sec), "action_name": name,
                               "confidence": float(scores[i]), "bbox": box,
                               "model_type": MODEL_TYPE})
    detections.sort(key=lambda d: (d[0], -d[3]))
    log(f"✅ Action recognition: {len(detections)} detections "
        f"({len({d[4] for d in detections})} actions) in {len(windows)} windows")
    return detections, bboxes
