"""
clip_quality.py — cheap sharpness scoring for candidate highlight clips.

Motion scoring happily selects clips that are exciting but unwatchable: a
whip-pan or a defocused lens produces huge frame deltas and near-zero visual
information. The classic cheap detector for this is the variance of the
Laplacian — blur removes high-frequency edges, so the second derivative of a
blurry frame is nearly flat and its variance collapses.

This module is only measurement. The pipeline decides what to do with the
score (a later phase wires `quality_gate` / `quality_threshold` from
gui_config into clip selection); nothing here knows about scoring weights or
segments.

Design constraints
==================
- `sample_sharpness` runs per candidate clip during scoring, so it must stay
  cheap: a handful of seeked frames, downscaled before the Laplacian.
- The MEDIAN across samples is used, not the mean: one black or corrupt frame
  must not drag an otherwise sharp clip below the gate.
- Missing data never penalizes. An unreadable clip returns None, and
  `is_blurry(None, ...)` is False — we only reject clips we positively
  measured as blurry.
"""

from __future__ import annotations

from statistics import median
from typing import Optional

import cv2


def laplacian_sharpness(gray_frame) -> float:
    """Variance of the Laplacian of a grayscale frame (higher = sharper)."""
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def sample_sharpness(video_path: str, t_start: float, t_end: float,
                     samples: int = 3, max_dim: int = 640) -> Optional[float]:
    """Median Laplacian sharpness over frames sampled from [t_start, t_end].

    Seeks to `samples` evenly spaced timestamps (sample midpoints, so the last
    read never lands on the possibly-absent frame at exactly t_end), downscales
    each frame so max(h, w) <= max_dim, and scores its grayscale Laplacian
    variance. Returns None when no frame in the window is readable.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        scores = []
        n = max(1, int(samples))
        span = max(0.0, float(t_end) - float(t_start))
        for i in range(n):
            t = float(t_start) + span * (i + 0.5) / n
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            scale = max_dim / float(max(h, w, 1))
            if scale < 1.0:
                frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                                   interpolation=cv2.INTER_AREA)
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scores.append(laplacian_sharpness(gray))
        if not scores:
            return None
        return float(median(scores))
    finally:
        cap.release()


def is_blurry(score: Optional[float], threshold: float) -> bool:
    """True when a measured score falls below the threshold.

    None (nothing readable) is False: missing data never penalizes a clip.
    """
    if score is None:
        return False
    return float(score) < float(threshold)
