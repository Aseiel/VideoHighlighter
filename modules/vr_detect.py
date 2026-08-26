"""Work out whether a video packs two eyes into one frame.

Side-by-side footage — 180° VR, and stereo 3D generally — stores the left and
right eye next to each other inside a single frame. Every view of a frame in
this app already knows how to show just the left one (`modules.video_regions`
for analysis, the player's crop, the thumbnail cache); what none of them had
was a way to find out that they should, short of the user ticking a box.

Untold, a filmstrip is at its worst on exactly this footage. Drawn whole, each
thumbnail is the same moment twice at half the size; drawn into a slot narrower
than the frame, the painter centre-crops, and the centre of a side-by-side
frame is the seam between the eyes — so the one part of the picture guaranteed
to show nothing is the part that survives.

The checkbox stays the final word: this only decides where it starts, and a
wrong guess is one click from being corrected.

Two questions, asked in this order because the first is free:

* **Could this be side-by-side?** Two eyes make a frame twice as wide as one
  eye, and an eye is a shape somebody actually shoots — square (fisheye or
  equirectangular 180°), 4:3, or 16:9. That rejects nearly everything without
  decoding a pixel, and in particular it rejects a 2.39:1 scope film, whose
  halves would be 1.19:1 — a shape no camera produces.

* **Is it?** The geometry gate lets through the one family it cannot tell apart
  from side-by-side: a monoscopic 360° panorama is 2:1 as well, and so is a
  film shot 2:1. So the halves are compared, shrunk to 8×8 first — coarse
  enough that parallax washes out, which matters more than it sounds, because
  a subject close to the camera moves a long way between the eyes. Measured
  over seven VR files the halves then correlate 0.74–1.00, while a panorama's
  reach 0.51 and unrelated pictures sit near zero.

Correlation alone is not enough: a centred, symmetric composition — mirrored
letterbox bars, a face in the middle of a blurred background — pushes a flat
frame's halves to 0.95 on its own. So a sample must also beat its own control,
the right half *mirrored*. Two eyes are a copy, not a reflection, so stereo
wins that comparison; a symmetric flat frame cannot, because a reflection is
exactly what its halves are.

Neither test is trusted on one frame — both directions have believable-looking
single frames — so three are taken across the running time and the majority
decides. A sample carrying no detail (a fade, a black frame, a title card) is
discarded rather than believed, because every such frame's halves match.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import NamedTuple

import cv2
import numpy as np

try:
    from modules.app_paths import ffmpeg_exe
except Exception:  # pragma: no cover - OpenCV sampling still works without it
    ffmpeg_exe = None


# Shapes a single eye is ever delivered in. Anything whose halves are not one
# of these is not two eyes.
EYE_SHAPES = (1.0, 4 / 3, 16 / 9)
EYE_SHAPE_TOLERANCE = 0.05

# Narrower than this and the frame cannot hold two eyes of any of those shapes.
MIN_FULL_ASPECT = 1.9

# Where to look, as fractions of the running time. Spread out, because one
# moment can be misleading in both directions — a symmetric title card matches
# when the video is flat, a hard cut between the eyes' exposure does not when
# it is stereo.
SAMPLE_POINTS = (0.2, 0.5, 0.8)

# Each half is grabbed at this size and compared at COMPARE_EDGE. The grab is
# the bigger of the two so the detail check below has something to look at;
# the comparison is deliberately coarse, because parallax is local and the
# question is whether the two halves are the same *scene*.
SAMPLE_EDGE = 64
COMPARE_EDGE = 8

# Standard deviation below which a sample is discarded as carrying no evidence.
MIN_DETAIL = 6.0

# Correlation between the halves above which they are the same scene, and how
# far it must beat the mirrored control. Both sit below the worst honest pair
# measured (0.74, and a 0.08 margin) and above the best flat one (0.51, and a
# margin of about zero).
MIN_CORRELATION = 0.65
MIN_MIRROR_MARGIN = 0.05

# Names that say it outright. Only consulted when no frame could be read at
# all, so a mislabelled file never overrules what the pixels show.
_NAME_MARKERS = re.compile(
    r"(?:^|[_\-. ])(lr|sbs|3dh|mkx200|mkx220)(?:$|[_\-. ])", re.IGNORECASE)


class Layout(NamedTuple):
    """What one frame of a video contains, and why we think so."""

    side_by_side: bool
    width: int
    height: int
    reason: str

    @property
    def eye_aspect(self) -> float:
        """Aspect ratio of the picture a viewer should be shown."""
        if self.width <= 0 or self.height <= 0:
            return 0.0
        width = self.width / 2 if self.side_by_side else self.width
        return width / self.height


_results: dict = {}


def probe(video_path) -> Layout:
    """Decide whether `video_path` is side-by-side, cached per file version.

    Never raises: an unreadable video is reported as a plain one, because the
    cost of being wrong that way is a filmstrip that shows too much, while the
    other way it is a filmstrip that silently hides half of every frame.
    """
    path = str(video_path or "")
    key = _file_key(path)
    cached = _results.get(key)
    if cached is not None:
        return cached
    try:
        result = _probe(path)
    except Exception as e:                       # pragma: no cover - defensive
        result = Layout(False, 0, 0, f"layout probe failed: {e}")
    _results[key] = result
    return result


def forget(video_path=None) -> None:
    """Drop cached results (all of them, or one file's). For tests."""
    if video_path is None:
        _results.clear()
    else:
        _results.pop(_file_key(str(video_path)), None)


# ── internals ────────────────────────────────────────────────────────────

def _file_key(path: str) -> tuple:
    """Identify a file by content, not just name, so a replaced file re-probes."""
    try:
        st = os.stat(path)
        return (path, st.st_size, int(st.st_mtime))
    except OSError:
        return (path, 0, 0)


def _probe(path: str) -> Layout:
    width, height, duration = _read_metadata(path)
    if width <= 0 or height <= 0:
        return Layout(False, 0, 0, "could not read the video's size")

    aspect = width / height
    shape = f"{width}×{height} ({aspect:.2f}:1)"

    if aspect < MIN_FULL_ASPECT:
        return Layout(False, width, height,
                      f"{shape} — too narrow to hold two eyes")
    if not _is_eye_shaped(aspect / 2):
        return Layout(False, width, height,
                      f"{shape} — halves would be {aspect / 2:.2f}:1, "
                      f"not a shape an eye comes in")

    matched, correlation, samples = _halves_match(path, duration)
    if samples == 0:
        # Nothing readable to judge by. The name is the only evidence left, and
        # weak evidence only gets to confirm what the geometry already allows.
        if _NAME_MARKERS.search(os.path.basename(path)):
            return Layout(True, width, height,
                          f"{shape}, no frame could be read — going by the name")
        return Layout(False, width, height,
                      f"{shape}, but no frame could be read to check the halves")

    if matched:
        return Layout(True, width, height,
                      f"{shape}, halves are the same scene (r={correlation:.2f} "
                      f"over {samples} frames)")
    return Layout(False, width, height,
                  f"{shape}, but the halves are different pictures "
                  f"(r={correlation:.2f} over {samples} frames)")


def _is_eye_shaped(aspect: float) -> bool:
    return any(abs(aspect - shape) <= shape * EYE_SHAPE_TOLERANCE
               for shape in EYE_SHAPES)


def _read_metadata(path: str) -> tuple:
    """Size and duration from the container — metadata only, no frame decoded."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return (0, 0, 0.0)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    return (width, height, duration)


def _halves_match(path: str, duration: float) -> tuple:
    """Compare the two halves of a few frames. → (matched, correlation, samples).

    Stops as soon as two samples agree, since that settles a best-of-three and
    a skipped sample is a whole keyframe decode not paid for — on 8K footage
    the difference is felt at startup.
    """
    scores = []
    passes = failures = 0
    for fraction in SAMPLE_POINTS:
        halves = _sample_halves(path, duration * fraction if duration > 0 else 0.0)
        if halves is None:
            continue
        left, right = halves
        if min(float(left.std()), float(right.std())) < MIN_DETAIL:
            continue                       # a fade or a black frame proves nothing
        left = _shrink(left)
        right = _shrink(right)
        score = _correlation(left, right)
        mirrored = _correlation(left, np.fliplr(right))
        scores.append(score)
        if score >= MIN_CORRELATION and score - mirrored >= MIN_MIRROR_MARGIN:
            passes += 1
        else:
            failures += 1
        if passes >= 2 or failures >= 2:
            break

    if not scores:
        return (False, 0.0, 0)
    return (passes > failures, float(np.median(scores)), len(scores))


def _shrink(half):
    """Down to the size the comparison is made at, averaging as it goes."""
    return cv2.resize(half, (COMPARE_EDGE, COMPARE_EDGE),
                      interpolation=cv2.INTER_AREA)


def _correlation(left, right) -> float:
    a = left.astype(np.float32).ravel()
    b = right.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _sample_halves(path: str, seconds: float):
    """The two halves of one frame, each as a SAMPLE_EDGE² grey square."""
    frame = _grab_via_ffmpeg(path, seconds)
    if frame is None:
        frame = _grab_via_opencv(path, seconds)
    if frame is None:
        return None
    return (frame[:, :SAMPLE_EDGE], frame[:, SAMPLE_EDGE:SAMPLE_EDGE * 2])


def _grab_via_ffmpeg(path: str, seconds: float):
    """One keyframe, decoded and squashed inside ffmpeg.

    The same trick the thumbnail cache uses (`-noaccurate_seek` plus
    `-skip_frame nokey`), for the same reason: on the high-resolution footage
    this question is actually asked about, decoding a run of 8K frames to land
    on an exact timestamp costs seconds, and any keyframe near the mark answers
    the question just as well.
    """
    exe = ffmpeg_exe() if ffmpeg_exe is not None else None
    if not exe:
        return None
    cmd = [
        exe, "-nostdin", "-v", "error",
        "-ss", f"{max(0.0, seconds):.3f}",
        "-noaccurate_seek",
        "-skip_frame", "nokey",
        "-i", path,
        "-frames:v", "1",
        "-vf", f"scale={SAMPLE_EDGE * 2}:{SAMPLE_EDGE}",
        "-pix_fmt", "gray",
        "-f", "rawvideo", "-",
    ]
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           timeout=30, creationflags=creationflags)
    except Exception:
        return None
    expected = SAMPLE_EDGE * SAMPLE_EDGE * 2
    if r.returncode != 0 or len(r.stdout) < expected:
        return None
    return np.frombuffer(r.stdout[:expected], dtype=np.uint8).reshape(
        SAMPLE_EDGE, SAMPLE_EDGE * 2)


def _grab_via_opencv(path: str, seconds: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        if seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (SAMPLE_EDGE * 2, SAMPLE_EDGE),
                      interpolation=cv2.INTER_AREA)
