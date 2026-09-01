"""
shot_type.py — what kind of shot is this, in framing terms.

Why this exists
===============
The scorer ranks moments. It has no idea that half a shoot is the camera
pointed back at the person carrying it and the other half is what they were
looking at — and an edit that does not know the difference cuts six selfies
together and then six landscapes, which reads as two edits stuck end to end.
Alternating them is most of what makes a sequence feel arranged.

So this measures *framing*, not subject matter: how much of the frame a face
occupies, how much the picture moves, how bright and how sharp it is. A shot
where a face fills a good share of the frame is a close shot whoever is in it;
a shot with no face is a wide one whatever it is of. The module holds no
opinion about content and its categories describe the camera, which is the
distinction that survives being pointed at anything.

Why these signals
=================
**Face fraction** separates a held-at-arm's-length shot from a person standing
in a landscape far better than face *presence* does — on a wide action-camera
lens a distant figure and a selfie both "have a face", and only the size tells
them apart.

**Motion** is measured between sampled frames rather than within them, so it
reflects how much the shot changes rather than how fast anything in it moves.
That is the property an editor cares about when deciding what can sit next to
what.

**Sharpness** (variance of Laplacian) catches the shots that look wrong on a
big screen for reasons nobody articulates. It is reported rather than acted on
here; the caller decides whether softness disqualifies a shot.

Sampling
========
Frames are read at even intervals rather than decoded in full: eight frames
answer "what kind of shot is this" as well as eight hundred, and the whole
point is to classify a folder of clips in seconds rather than minutes. Results
cache next to the clip, keyed on size and mtime, because this runs every time
a reel is planned.

Public API
==========
    KINDS
    ShotType
    classify(path, ...) -> ShotType
    classify_all(paths, ...) -> dict[str, ShotType]
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

# Framing categories. Named for what the camera is doing, never for what is in
# front of it — see the module docstring.
KIND_CLOSE = "close_subject"
KIND_SUBJECT = "subject"
KIND_WIDE = "wide"
KINDS = (KIND_CLOSE, KIND_SUBJECT, KIND_WIDE)

# Share of the frame a face has to occupy to count as a close shot. Measured on
# action-camera footage: a camera held at arm's length puts a face around 2-6%
# of a wide frame, while somebody standing in the scene is well under 0.5%.
# 1.2% sits in the empty space between those two clusters rather than on either
# edge of one.
CLOSE_FACE_FRACTION = 0.012

# A face has to appear in this share of sampled frames before the shot counts
# as being *of* somebody. Below it, the detector found a face in one frame of
# eight, which on this kind of footage is usually a rock.
FACE_PRESENCE = 0.34

# Detector confidence. YuNet's default is 0.9; slightly lower catches faces at
# the edge of a fisheye frame, which is exactly where a helmet-mounted camera
# puts them.
FACE_CONFIDENCE = 0.75

SAMPLES = 8
CACHE_VERSION = 1


@dataclass
class ShotType:
    path: str
    kind: str = KIND_WIDE
    face_fraction: float = 0.0     # largest face area / frame area
    face_presence: float = 0.0     # share of sampled frames holding a face
    motion: float = 0.0            # mean change between samples, 0..1
    brightness: float = 0.0        # mean luma, 0..1
    sharpness: float = 0.0         # variance of Laplacian, unnormalised
    duration: float = 0.0

    @property
    def has_subject(self) -> bool:
        return self.kind in (KIND_CLOSE, KIND_SUBJECT)


def _cache_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".shot.json"


def _load_cached(path: str) -> ShotType | None:
    """Cached classification, or None when it is missing or stale.

    Keyed on size and mtime rather than a hash: the files are gigabytes, and a
    clip that changed without changing either is not a case this needs to
    survive.
    """
    cache = _cache_path(path)
    try:
        with open(cache, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        stat = os.stat(path)
        if (data.get("version") != CACHE_VERSION
                or data.get("size") != stat.st_size
                or abs(float(data.get("mtime", 0)) - stat.st_mtime) > 1.0):
            return None
        return ShotType(**data["shot"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_cached(shot: ShotType) -> None:
    try:
        stat = os.stat(shot.path)
        with open(_cache_path(shot.path), "w", encoding="utf-8") as fh:
            json.dump({"version": CACHE_VERSION, "size": stat.st_size,
                       "mtime": stat.st_mtime, "shot": asdict(shot)}, fh)
    except OSError:
        pass   # an unwritable folder costs the cache, not the classification


def _face_detector(width: int, height: int):
    """YuNet, sized to the frames it will be shown, or None when unavailable.

    Returning None rather than raising is deliberate: without a detector every
    shot classifies as wide, which is a worse edit and still an edit. Faces are
    an improvement to selection, not a precondition for it.
    """
    try:
        import cv2

        from modules.app_paths import data_file

        model = data_file(os.path.join(
            "video_ai_editor", "models", "face_detection_yunet_2023mar.onnx"))
        if not os.path.exists(model):
            return None
        return cv2.FaceDetectorYN.create(
            model, "", (width, height), FACE_CONFIDENCE, 0.3, 5000)
    except Exception:
        return None


def classify(path: str, *, samples: int = SAMPLES, use_cache: bool = True,
             log_fn=print) -> ShotType:
    """Measure one clip's framing.

    Never raises for an unreadable clip — it comes back as a wide shot with
    zeroed measurements, so one bad file cannot stop a reel being planned.
    """
    if use_cache:
        cached = _load_cached(path)
        if cached is not None:
            return cached

    shot = ShotType(path=path)
    try:
        import cv2
        import numpy as np
    except Exception:
        return shot

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        log_fn(f"⚠️ Could not open {os.path.basename(path)} to classify it")
        return shot

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        shot.duration = total / fps if total else 0.0
        if total <= 0:
            return shot

        # Skip the first and last tenth: a cut often begins or ends mid-motion,
        # and both ends are the least representative part of a shot.
        first, last = int(total * 0.1), int(total * 0.9)
        picks = [first + round(i * (last - first) / max(1, samples - 1))
                 for i in range(samples)]

        detector = None
        frames: list = []
        faces_seen = 0
        largest = 0.0

        for index in picks:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            # Detection and the difference maths both run on a small copy: a
            # 5312-wide frame costs a lot to process and answers nothing a
            # 640-wide one does not.
            scale = 640.0 / max(1, width)
            small = cv2.resize(frame, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))

            if detector is None:
                detector = _face_detector(small.shape[1], small.shape[0])
            if detector is not None:
                try:
                    _, found = detector.detect(small)
                except Exception:
                    found = None
                if found is not None and len(found):
                    faces_seen += 1
                    area = small.shape[0] * small.shape[1]
                    biggest = max(float(f[2]) * float(f[3]) for f in found)
                    largest = max(largest, biggest / area)

        if not frames:
            return shot

        shot.face_presence = faces_seen / len(frames)
        shot.face_fraction = largest
        shot.brightness = float(np.mean(frames[-1])) / 255.0
        shot.sharpness = float(cv2.Laplacian(frames[-1], cv2.CV_64F).var())
        if len(frames) > 1:
            diffs = [float(np.mean(cv2.absdiff(a, b))) / 255.0
                     for a, b in zip(frames, frames[1:])]
            shot.motion = float(np.mean(diffs))

        if shot.face_presence >= FACE_PRESENCE:
            shot.kind = (KIND_CLOSE if shot.face_fraction >= CLOSE_FACE_FRACTION
                         else KIND_SUBJECT)
        else:
            shot.kind = KIND_WIDE
    finally:
        capture.release()

    if use_cache:
        _save_cached(shot)
    return shot


def classify_all(paths, *, samples: int = SAMPLES, use_cache: bool = True,
                 log_fn=print) -> dict:
    """Classify a list of clips, reporting the mix once at the end.

    The summary line is the useful output when this runs inside a longer job:
    "9 wide, 8 close" tells you what kind of reel is possible, and a folder
    that comes back all-wide is usually a missing detector rather than a shoot
    with nobody in it.
    """
    out: dict[str, ShotType] = {}
    for path in paths or []:
        out[path] = classify(path, samples=samples, use_cache=use_cache,
                             log_fn=log_fn)
    if out:
        counts: dict[str, int] = {}
        for shot in out.values():
            counts[shot.kind] = counts.get(shot.kind, 0) + 1
        log_fn("🎞️ Shots: " + ", ".join(
            f"{n} {kind.replace('_', ' ')}" for kind, n in sorted(counts.items())))
    return out
