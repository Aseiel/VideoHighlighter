"""Body parts as rule classes: ``person.left_wrist``, ``person.hand``.

A box says where a person is; the question a composition rule usually means
is about a *part* of them: a hand at something, a foot on something. A
person's box is the worst possible stand-in for their hand. This lets a rule
name the part, and the engine sees one point per visible keypoint (RTMPose,
COCO-17, which the app already runs for the cropper):

    - {source: person.hand, region: <thing>, relation: touches, max_gap: 0.01}

``<class>.<part>`` where part is a COCO-17 keypoint name, or a group that
matches either side (``hand`` = both wrists). A point works with every
relation: ``inside`` = the point is in the region, ``touches`` = within
``max_gap`` of it. Keypoints below ``MIN_SCORE`` are treated as not visible.
Nothing here names subject matter; the part names are the pose model's own.
"""
from __future__ import annotations

from typing import Optional

from modules.vision.pose_backend import KEYPOINT_NAMES

MIN_SCORE = 0.3

GROUPS = {
    "hand": ("left_wrist", "right_wrist"),
    "wrist": ("left_wrist", "right_wrist"),
    "foot": ("left_ankle", "right_ankle"),
    "ankle": ("left_ankle", "right_ankle"),
    "knee": ("left_knee", "right_knee"),
    "elbow": ("left_elbow", "right_elbow"),
    "shoulder": ("left_shoulder", "right_shoulder"),
    "hip": ("left_hip", "right_hip"),
    "head": ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
}


def split(name: str) -> tuple:
    """``"person.left_wrist"`` -> ``("person", ("left_wrist",))``;
    a plain class -> ``(name, ())``."""
    base, dot, part = str(name).rpartition(".")
    if not dot or not base:
        return name, ()
    part = part.strip().lower()
    if part in KEYPOINT_NAMES:
        return base, (part,)
    if part in GROUPS:
        return base, GROUPS[part]
    return name, ()


def is_part(name: str) -> bool:
    return bool(split(name)[1])


def points(keypoints, parts, min_score: float = MIN_SCORE) -> list:
    """Visible ``(x, y, score)`` of ``parts`` from one person's ``[17][3]``."""
    out = []
    if not keypoints:
        return out
    for part in parts:
        i = KEYPOINT_NAMES.index(part)
        if i < len(keypoints):
            x, y, score = keypoints[i][:3]
            if score is not None and float(score) >= min_score:
                out.append((float(x), float(y), float(score)))
    return out


def part_detections(entry: dict, wanted: set) -> dict:
    """Point detections for the ``wanted`` part-classes in one cache entry.

    ``entry['keypoints']`` is aligned with ``objects``: per detection, None or
    a normalised ``[[x, y, score], ...]`` in COCO-17 order.
    """
    result: dict = {}
    names = entry.get("objects") or []
    keypoints = entry.get("keypoints") or []
    for ref in wanted:
        base, parts = split(ref)
        for i, cls in enumerate(names):
            if cls != base or i >= len(keypoints):
                continue
            for x, y, score in points(keypoints[i], parts):
                result.setdefault(ref, []).append(
                    {"box": [x, y, 0.0, 0.0], "conf": score, "point": True})
    return result


def base_class(name: str) -> str:
    return split(name)[0]


def required_parts(refs) -> dict:
    """``{base class: {keypoint names}}`` a set of rule classes needs."""
    out: dict = {}
    for ref in refs:
        base, parts = split(ref)
        if parts:
            out.setdefault(base, set()).update(parts)
    return out


def describe(name: str) -> Optional[str]:
    base, parts = split(name)
    return f"{base}: {', '.join(parts)}" if parts else None
