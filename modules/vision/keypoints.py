"""Keypoints for body-part rules, estimated on demand (RTMPose).

The counterpart of ``outlines.py`` for rules naming a body part
(``modules/rules/body_parts.py``). Same economy: only people whose box meets
a box of the class the rule relates them to are estimated, each once, and
the result is kept in the analysis cache as ``keypoints`` aligned with
``bboxes`` (per detection: None, ``[]`` for tried-and-nothing, or
normalised ``[[x, y, score]] * 17``).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from modules.vision.outlines import read_frames_at, wanted


def add_keypoints(video_path: str, bboxes: list, pairs: Sequence, estimator, *,
                  frame_reader: Optional[Callable] = None,
                  progress: Optional[Callable] = None, cancel=None) -> dict:
    """Estimate the keypoints ``pairs`` need into ``bboxes``, in place.

    ``pairs`` is ``CompositionEngine.keypoint_pairs``: ``(person class, other
    class, gap)``. Only the person side is estimated.
    """
    frame_reader = frame_reader or read_frames_at
    people = {p for p, _, _ in pairs}
    todo = {}
    for k, entry in enumerate(bboxes):
        names = entry.get("objects") or []
        have = entry.get("keypoints") or []
        need = [i for i in sorted(wanted(entry, pairs))
                if i < len(names) and names[i] in people
                and not (i < len(have) and have[i] is not None)]
        if need:
            todo[k] = need
    done = empty = 0
    order = sorted(todo, key=lambda k: float(bboxes[k].get("timestamp", 0)))
    stamps = [float(bboxes[k].get("timestamp", 0)) for k in order]
    for n, (k, (ts, frame)) in enumerate(zip(order, frame_reader(video_path, stamps))):
        if cancel is not None and cancel.is_set():
            break
        if frame is None:
            continue
        entry = bboxes[k]
        count = len(entry.get("objects") or [])
        stored = list(entry.get("keypoints") or [])
        stored += [None] * (count - len(stored))
        height, width = frame.shape[:2]
        need = todo[k]
        boxes_px = []
        for i in need:
            x, y, w, h = entry["bboxes"][i]
            boxes_px.append((x * width, y * height, (x + w) * width, (y + h) * height))
        poses = estimator.estimate(frame, boxes_px)
        for i, pose in zip(need, poses):
            points = getattr(pose, "keypoints", None)
            if points is None or len(points) == 0:
                stored[i] = []
                empty += 1
                continue
            stored[i] = [[round(float(p[0]) / width, 5), round(float(p[1]) / height, 5),
                          round(float(p[2]), 3)] for p in points]
            done += 1
        entry["keypoints"] = stored
        if progress:
            progress(n + 1, len(order))
    return {"frames": len(order), "people": done, "no_pose": empty}
