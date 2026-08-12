"""
shot_look.py — do two clips look the same, closely enough to not sit together.

Why this exists
===============
:mod:`modules.shot_place` stops a reel using one *spot* twice, and that fixes
the commonest repetition: standing still and pressing record. It cannot fix the
other one. A landmark is visible from a long way off, so two clips filmed five
hundred metres apart can both be of the same lake, and by position they are
plainly different places — which they are. The viewer does not care where the
camera was. They see the same picture twice.

So this compares the pictures. It is deliberately the *last* thing tried rather
than the first, because on the footage this was built for it very nearly does
not work at all.

Why the obvious version fails
=============================
On a shoot that is one colour — a moor, a beach, a snowfield — a single
descriptor cannot tell "the same view" from "the same kind of view". Measured
against known pairs on a real shoot, each of these was tried on its own and
each failed:

- a coarse colour signature called thirty of sixty-six pairs near-duplicates;
- a perceptual hash, a normalised greyscale layout at three resolutions, an
  HSV histogram and ORB feature matching each ranked at most one of four known
  pairs into their top four.

The colour histogram's worst false positive was two shots of a footpath at
0.966 — more similar, it said, than the two shots of the same reservoir. The
greyscale layout's worst was a hillside against a hazy moor at 0.882.

What works, and how little of it
===============================
The two descriptors fail on different pairs, so the smaller of the two is high
only when the layout and the palette both match — which is close to what a
person means by "that is the same shot again".

How well that works was measured over all two hundred and ten pairs of a
twenty-one clip shoot, against pairs marked by hand as repeats. It is worth
being blunt about the answer, because it decides what this module is allowed
to be used for:

===========  ==================  ===================
Threshold    repeats it finds    pairs it invents
===========  ==================  ===================
0.88         1 of 8              0
0.86         1 of 8              5
0.84         2 of 8              7
0.80         3 of 8              13
===========  ==================  ===================

Below the default the false pairs arrive faster than the true ones. Only the
single clearest repeat is separated from the field at all — two shots of one
reservoir at 0.906, against 0.879 for the next pair, which is of two different
things. The rest of the repeats a person would name sit at 0.79 and below,
mixed in with pairs that share nothing but a hillside.

So the claim here is deliberately small: **this finds the one or two shots in
a reel that are nearly the same picture, and nothing subtler.** It is used to
stop those sitting next to each other and for nothing else — not to drop a
clip, not to rank a reel. Repetition that position can see belongs to
:mod:`modules.shot_place`, which is exact; this covers only the case position
is blind to, and covers it partially.

Position and appearance are genuinely independent, which is why both exist:
two clips filmed eleven metres apart score 0.217 here, because the camera was
pointed the other way.

Public API
==========
    Look
    look(path, ...) -> Look
    look_all(paths, ...) -> dict[str, Look]
    similarity(a, b) -> float
    same_view(a, b, threshold=SAME_VIEW) -> bool
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

# Layout, as an NxN grid of brightness. Small on purpose: at eight it describes
# where the light and dark masses are, which is what "the same view" means, and
# stops short of the detail that makes two frames of one clip disagree.
GRID = 8

# Colour, as a hue/saturation histogram. Value is left out so the same view
# under a passing cloud still matches itself.
HUE_BINS = 24
SATURATION_BINS = 8

# Frames read per clip. The descriptors are averaged over them so a single
# frame with somebody walking through it does not decide the answer.
SAMPLES = 6

# Above this, the layout and the palette both match closely enough to call two
# clips the same view. Set where the table in the module docstring stops
# inventing pairs: at 0.88 nothing false is reported, and every step below it
# admits false pairs faster than true ones. Tuned for precision rather than
# recall on purpose — a reel that keeps a repeat is disappointing, and a reel
# that drops a good shot because a hillside matched a hillside is worse, since
# nobody watching can tell why.
SAME_VIEW = 0.88

CACHE_VERSION = 1


@dataclass
class Look:
    """One clip's appearance, as two normalised descriptors."""
    path: str
    structure: list = field(default_factory=list)   # GRID*GRID, zero-mean
    colour: list = field(default_factory=list)      # hue x saturation, L2
    measured: bool = False


def _cache_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".look.json"


def _load_cached(path: str) -> Look | None:
    try:
        with open(_cache_path(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        stat = os.stat(path)
        if (data.get("version") != CACHE_VERSION
                or data.get("size") != stat.st_size
                or abs(float(data.get("mtime", 0)) - stat.st_mtime) > 1.0):
            return None
        return Look(**data["look"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_cached(item: Look) -> None:
    try:
        stat = os.stat(item.path)
        with open(_cache_path(item.path), "w", encoding="utf-8") as fh:
            json.dump({"version": CACHE_VERSION, "size": stat.st_size,
                       "mtime": stat.st_mtime, "look": asdict(item)}, fh)
    except OSError:
        pass


def look(path: str, *, samples: int = SAMPLES, use_cache: bool = True,
         log_fn=print) -> Look:
    """Measure one clip's appearance.

    Never raises. An unreadable clip, or a machine with no OpenCV, comes back
    unmeasured — and :func:`similarity` reports 0 for anything unmeasured, so
    the caller simply learns nothing rather than being told two clips differ.
    """
    if use_cache:
        cached = _load_cached(path)
        if cached is not None:
            return cached

    item = Look(path=path)
    try:
        import cv2
        import numpy as np
    except Exception:
        return item

    capture = None
    try:
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return item
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return item

        # The middle of the clip: both ends are the least representative part
        # of a shot, the head especially — see modules.shot_window for what is
        # usually happening there.
        first, last = int(total * 0.15), int(total * 0.85)
        picks = {first + round(i * (last - first) / max(1, samples - 1))
                 for i in range(samples)}

        grids, hues = [], []
        for index in sorted(picks):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            grid = cv2.cvtColor(
                cv2.resize(frame, (GRID, GRID), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY).astype(np.float32)
            grids.append(grid.flatten())
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hues.append(cv2.calcHist([hsv], [0, 1], None,
                                     [HUE_BINS, SATURATION_BINS],
                                     [0, 180, 0, 256]).flatten())
        if not grids:
            return item

        structure = np.mean(grids, axis=0)
        # Zero-mean and unit-variance, so the comparison is of layout rather
        # than of exposure: the same view metered differently is still it.
        structure = (structure - structure.mean()) / (structure.std() or 1.0)
        colour = np.sum(hues, axis=0)

        item.structure = [float(v) for v in structure / GRID]
        item.colour = [float(v) for v in colour / (np.linalg.norm(colour) or 1.0)]
        item.measured = True
    except Exception as exc:
        log_fn(f"⚠️ Could not measure how {os.path.basename(path)} looks ({exc})")
        return Look(path=path)
    finally:
        try:
            if capture is not None:
                capture.release()
        except Exception:
            pass

    if use_cache and item.measured:
        _save_cached(item)
    return item


def similarity(a: Look, b: Look) -> float:
    """How alike two clips are, 0..1, or 0 when either was not measured.

    The smaller of the layout and the colour agreement, which is the whole
    idea: each descriptor has false positives the other does not, so taking
    the weaker of the two only reports a match when both agree. Reporting the
    mean instead brings back exactly the false positives this exists to avoid.
    """
    if not (a and b and a.measured and b.measured):
        return 0.0
    if len(a.structure) != len(b.structure) or len(a.colour) != len(b.colour):
        return 0.0
    structure = sum(x * y for x, y in zip(a.structure, b.structure))
    colour = sum(x * y for x, y in zip(a.colour, b.colour))
    return min(structure, colour)


def same_view(a: Look, b: Look, threshold: float = SAME_VIEW) -> bool:
    """Whether two clips are close enough to read as a repeat."""
    return similarity(a, b) >= threshold


def look_all(paths, *, use_cache: bool = True, log_fn=print) -> dict:
    """Measure a list of clips, reporting how many pairs read as repeats."""
    out: dict[str, Look] = {}
    for path in paths or []:
        out[path] = look(path, use_cache=use_cache, log_fn=log_fn)

    measured = [v for v in out.values() if v.measured]
    if len(measured) > 1:
        pairs = sum(1 for i, a in enumerate(measured)
                    for b in measured[i + 1:] if same_view(a, b))
        log_fn(f"👁️ Compared {len(measured)} clip(s)"
               + (f"; {pairs} pair(s) look like the same view"
                  if pairs else "; none look alike"))
    return out
