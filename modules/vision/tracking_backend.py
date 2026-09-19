"""Permissive person tracking.

YOLOX (Apache-2.0) detection + IoU association. Used by identity tagging and
the avoid pipeline (compute_forbidden). Kept byte-identical between editions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from modules.vision.detection_backend import YoloxOpenVINODetector, find_default_yolox_ir

# GUI / legacy size tokens -> YOLOX IR filename suffix (yolox_<suffix>.xml)
_SIZE_ALIASES = {
    "n": "nano",
    "nano": "nano",
    "tiny": "tiny",
    "s": "s",
    "m": "m",
    "l": "l",
    "x": "x",
}


def resolve_yolox_ir(model_size: str = "n") -> str | None:
    """Pick a YOLOX IR for tracking. Prefers the requested size, then any installed."""
    from pathlib import Path

    suffix = _SIZE_ALIASES.get(str(model_size).lower(), str(model_size).lower())
    explicit = Path(__file__).resolve().parents[2] / "models" / "yolox" / f"yolox_{suffix}.xml"
    if explicit.is_file():
        return str(explicit)
    return find_default_yolox_ir(prefer="small")


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class TrackedPerson:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


class _IoUPersonTracker:
    """Associate per-frame person detections with persistent track ids."""

    def __init__(self, iou_threshold: float = 0.3, max_lost_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self._tracks: dict[int, dict] = {}
        self._next_id = 0

    def update(self, detections: list[tuple[tuple[float, float, float, float], float]]
               ) -> list[TrackedPerson]:
        if not detections:
            for tid in list(self._tracks):
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost_frames:
                    del self._tracks[tid]
            return []

        assigned: dict[int, int] = {}  # det_idx -> track_id
        used_tracks: set[int] = set()

        for det_idx, (box, _conf) in enumerate(detections):
            best_iou = 0.0
            best_tid: int | None = None
            for tid, data in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou = _iou(box, data["box"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                assigned[det_idx] = best_tid
                used_tracks.add(best_tid)
                self._tracks[best_tid]["box"] = box
                self._tracks[best_tid]["lost"] = 0
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"box": box, "lost": 0}
                assigned[det_idx] = tid
                used_tracks.add(tid)

        for tid in list(self._tracks):
            if tid not in used_tracks:
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self.max_lost_frames:
                    del self._tracks[tid]

        out: list[TrackedPerson] = []
        for det_idx, (box, conf) in enumerate(detections):
            tid = assigned[det_idx]
            x1, y1, x2, y2 = box
            out.append(TrackedPerson(
                track_id=tid,
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                conf=float(conf),
            ))
        return out


class YoloxPersonTracker:
    """YOLOX person detector + IoU tracker for offline identity / avoid passes."""

    def __init__(self, model_xml: str | None = None, model_size: str = "n",
                 device: str = "GPU", person_conf: float = 0.25):
        model_xml = model_xml or resolve_yolox_ir(model_size)
        if not model_xml:
            raise FileNotFoundError(
                "No YOLOX IR found for tracking. Run: python tools/get_yolox_model.py"
            )
        self.model_xml = model_xml
        self.person_conf = float(person_conf)
        self._detector = YoloxOpenVINODetector(
            model_xml,
            class_names=["person"],
            device=device,
            score_thr=self.person_conf,
        )
        self._tracker = _IoUPersonTracker()

    def iter_frames(self, video_path: str, vid_stride: int = 1,
                    person_conf: float | None = None) -> Iterator[tuple]:
        """Yield (frame_bgr, real_frame_idx, list[TrackedPerson]) per processed frame."""
        import cv2

        conf = self.person_conf if person_conf is None else float(person_conf)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frame_idx = 0
        processed = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % max(1, vid_stride) != 0:
                    frame_idx += 1
                    continue

                detections: list[tuple[tuple[float, float, float, float], float]] = []
                for det in self._detector.detect(frame):
                    if det.class_id != 0:
                        continue
                    if det.confidence < conf:
                        continue
                    detections.append((
                        (det.x1, det.y1, det.x2, det.y2),
                        det.confidence,
                    ))

                tracked = self._tracker.update(detections)
                yield frame, frame_idx, tracked
                processed += 1
                frame_idx += 1
        finally:
            cap.release()
