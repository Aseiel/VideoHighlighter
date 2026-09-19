"""Run an installed community model on frames and return timeline-ready results.

    runner = CommunityModelRunner(installed_model, providers=app_providers)
    hits = runner.detect(frame_bgr)          # object_detection / image_classification
    hits = runner.classify_clip(frames_bgr)  # action_recognition

Each hit is a Hit(label, score, box) with box=None for classification. Frames
are numpy uint8 arrays in OpenCV's BGR order, as the rest of the app uses.

Detectors run through ``modules.vision.detection_backend.YoloxOnnxRuntimeDetector`` —
the same letterbox, decode and NMS as a model the user trained themselves, so a
community model and an own model can never disagree about the same frame
because of glue code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .hub import InstalledModel, verify_installed

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Hit:
    label: str
    score: float
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2 in frame pixels


class ModelCheckFailed(RuntimeError):
    pass


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class CommunityModelRunner:
    def __init__(self, model: InstalledModel, providers: Sequence[str] | None = None,
                 verify: bool = True):
        import onnxruntime as ort

        if verify:
            report = verify_installed(model)
            if not report.ok:
                raise ModelCheckFailed(f"{model.repo_id} failed checks:\n{report.text()}")
        self.model = model
        self.m = model.manifest
        available = set(ort.get_available_providers())
        chosen = [p for p in (providers or []) if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model.model_path), providers=chosen)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.dtype = {"tensor(float16)": np.float16, "tensor(uint8)": np.uint8}.get(inp.type, np.float32)
        self._detector = None
        if self.m.task == "object_detection":
            from modules.vision.detection_backend import YoloxOnnxRuntimeDetector
            self._detector = YoloxOnnxRuntimeDetector(
                self.session, self.m.labels, score_thr=self.m.confidence_threshold)

    @property
    def signal_name(self) -> str:
        return f"community:{self.m.name}"

    # ---------------------------------------------------------- preprocessing
    def _prep_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Plain resize for classifiers. Returns HWC."""
        import cv2

        i = self.m.input
        img = frame_bgr if i.color == "BGR" else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (i.width, i.height), interpolation=cv2.INTER_AREA)
        if i.channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if i.color == "RGB" else cv2.COLOR_BGR2GRAY)[..., None]
        return img

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        if self.dtype == np.uint8:
            return img.astype(np.uint8)
        x = img.astype(np.float32)
        n = self.m.input.normalize
        if n == "0-1":
            x /= 255.0
        elif n == "minus1-1":
            x = x / 127.5 - 1.0
        elif n == "imagenet":
            x = (x / 255.0 - IMAGENET_MEAN[: x.shape[-1]]) / IMAGENET_STD[: x.shape[-1]]
        return x.astype(self.dtype)

    def _layout(self, x: np.ndarray) -> np.ndarray:
        # x is (..., H, W, C)
        if self.m.input.layout == "NCHW":
            x = np.moveaxis(x, -1, -3)
        return np.ascontiguousarray(x[None])

    # ------------------------------------------------------------------ run
    def detect(self, frame_bgr: np.ndarray, threshold: float | None = None) -> list[Hit]:
        if self.m.task == "action_recognition":
            raise ValueError("Use classify_clip() for action recognition models.")
        thr = self.m.confidence_threshold if threshold is None else threshold

        if self._detector is not None:
            self._detector.score_thr = float(thr)
            h, w = frame_bgr.shape[:2]
            hits = [Hit(d.class_name, float(d.confidence),
                        (max(0.0, d.x1), max(0.0, d.y1), min(float(w), d.x2), min(float(h), d.y2)))
                    for d in self._detector.detect(frame_bgr)]
            return sorted(hits, key=lambda h_: -h_.score)

        img = self._prep_frame(frame_bgr)
        out = self.session.run(None, {self.input_name: self._layout(self._normalize(img))})[0]
        return self._decode_scores(out[0], thr)

    def classify_clip(self, frames_bgr: Sequence[np.ndarray],
                      threshold: float | None = None) -> list[Hit]:
        if self.m.task != "action_recognition":
            raise ValueError("classify_clip() is only for action recognition models.")
        need = self.m.input.frames
        if len(frames_bgr) != need:
            idx = np.linspace(0, len(frames_bgr) - 1, need).round().astype(int)
            frames_bgr = [frames_bgr[i] for i in idx]
        clip = np.stack([self._normalize(self._prep_frame(f)) for f in frames_bgr])  # T,H,W,C
        if self.m.input.layout == "NCHW":
            clip = np.moveaxis(clip, -1, 1)  # T,C,H,W
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(clip[None])})[0]
        thr = self.m.confidence_threshold if threshold is None else threshold
        return self._decode_scores(out[0], thr)

    def _decode_scores(self, row: np.ndarray, thr: float) -> list[Hit]:
        probs = _softmax(row.astype(np.float32)) if self.m.output_format == "logits" else row
        order = np.argsort(probs)[::-1]
        return [Hit(self.m.labels[int(i)], float(probs[i])) for i in order if probs[i] >= thr]
