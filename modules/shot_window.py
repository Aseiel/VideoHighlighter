"""
shot_window.py — which part of a clip is worth putting on screen.

Why this exists
===============
Every other stage answers "which clips". None of them answers "which *seconds*
of the clip", and the reel planner had no way to ask: it started each shot at
the beginning of its source and took the length the pace called for. On real
footage that is the wrong second to start on more often than not. A shot begins
while the camera is still being pulled out of a pocket, raised onto a mount, or
swung round to point at the thing — and the reel opens on a blurred lurch that
settles just as the cut ends.

The clip is not bad. Its *first* seconds are, and the fix is to start later
rather than to throw the clip away.

What "settled" means here
=========================
Not "still" — an action camera is never still, and a reel of static shots is
its own kind of dead. What separates a shot being *carried* from a shot being
*used* is whether the frame-to-frame movement is coherent:

**Coherence** is the response of a phase correlation between consecutive
frames: how well one frame explains the next as a rigid shift of the whole
picture. A camera on a mount panning fast still scores high — the picture moves
as one. A camera in a hand being repositioned does not, because the lens is
also rotating, the exposure is chasing, and half the frame is something that
was not in shot a moment ago. Measured on a real shoot this is the signal that
separates the two states cleanly: the same clip runs 0.03–0.35 while the camera
is being placed and 0.85–0.99 once it is.

**Jerk** — how much the movement itself changes between samples — catches what
coherence alone does not. A deliberate pan has a nearly constant shift; a hand
searching for framing changes direction constantly.

**Sharpness** is measured against the clip's own upper range rather than an
absolute, because the variance of a Laplacian depends as much on what is in
front of the lens as on the focus. Half a stop of motion blur reads as a large
drop *within one clip* and as nothing at all across a folder.

**Exposure drift** catches the auto-exposure ramp that follows a camera coming
out of a bag into daylight. The picture is sharp and steady and still visibly
wrong for a second and a half.

None of these is decisive alone, so the score is a weighted mean pulled down
towards its worst term: a window that is perfect except that a hand crosses the
lens is not a good window, and averaging alone would say it was.

What it does not do
===================
It has no opinion on whether the content is interesting — that is the scorer's
job, and second-guessing it here would mean two parts of the program quietly
disagreeing about what a highlight is. This only says which seconds are
*usable*, which is a property of the camera rather than of the subject.

Public API
==========
    Sample / Window / ClipWindows
    profile(path, ...) -> ClipWindows
    profile_all(paths, ...) -> dict[str, ClipWindows]
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

# Frames measured per second of clip. Eight is enough to catch a lurch that
# lasts a fifth of a second and cheap enough to run over a folder: the whole
# measurement is a grayscale phase correlation on a 256-wide copy.
SAMPLES_PER_SECOND = 8.0

# Everything is measured on a copy this wide. Camera movement is a property of
# the whole frame, and a 5312-wide GoPro frame answers the question no better
# than a postcard does — it just costs sixty times as much.
WORK_WIDTH = 256

# Global movement, in frame-widths per second, at which the "calm" term hits
# zero. 0.9 is a fast whip-pan; a mounted camera on a moving bike sits near
# 0.15, which is why calm is the weakest of the four terms — punishing motion
# hard would select the dullest second of every clip.
SHIFT_FULL = 0.9

# Change in that movement, same units. A deliberate pan holds its speed and
# scores near zero here however fast it is; a hand hunting for framing does not.
JERK_FULL = 1.6

# Sharpness is scored against this share of the clip's own best. Below it the
# picture is soft *for this clip*, which is what motion blur looks like.
SHARP_REFERENCE = 0.75

# Brightness change per second at which the exposure term hits zero — an
# auto-exposure ramp coming out of a pocket moves several times this fast.
EXPOSURE_FULL = 0.55

# Luma outside this range is a picture nobody can see: a lens cap, a pocket, or
# a blown sky. Scored as a hard floor rather than a curve.
DARK = 0.06
BLOWN = 0.97

# How the four terms combine. Coherence dominates because it is the one that
# actually separates "being carried" from "being used"; the others break ties
# and veto.
WEIGHTS = {"coherence": 0.42, "sharpness": 0.28, "steady": 0.20, "exposure": 0.10}

# How far the worst sample in a window pulls the mean down. At 0 a window is
# its average and one unusable half-second is invisible; at 1 it is only ever
# as good as its worst frame, which rejects every real shot. 0.35 lets a window
# survive a blink and not a lurch.
WORST_PULL = 0.35

# A window has to clear this share of the clip's best window before the clip
# counts as having settled at all. Reported, not enforced — see ClipWindows.
SETTLE_LEVEL = 0.75

# Shortest measurable clip. Below two samples there is no movement to measure,
# since every signal here is a difference between frames.
MIN_MEASURABLE = 3.0 / SAMPLES_PER_SECOND

CACHE_VERSION = 1


@dataclass
class Sample:
    """One measurement, at ``t`` seconds into the clip."""
    t: float
    coherence: float = 0.0     # 0..1, how rigidly one frame explains the next
    shift: float = 0.0         # frame-widths per second of global movement
    jerk: float = 0.0          # change in that, per second
    sharpness: float = 0.0     # variance of Laplacian, unnormalised
    brightness: float = 0.0    # mean luma, 0..1
    drift: float = 0.0         # change in brightness per second
    usable: float = 0.0        # the four terms combined, 0..1


@dataclass
class Window:
    """A stretch of a clip and how usable it is.

    ``score`` is comparable across clips: it is the mean of the samples inside
    the window pulled towards the worst of them, and every term feeding it is
    either already normalised or normalised against the clip's own range.
    """
    start: float = 0.0
    duration: float = 0.0
    score: float = 0.0

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class ClipWindows:
    """Everything measured about one clip, and the windows it can offer.

    Constructed even for a clip that could not be opened — with no samples, in
    which case :meth:`best` hands back the head of the clip and the caller
    behaves exactly as it did before this module existed. Measurement failing
    should cost the improvement, never the reel.
    """
    path: str
    duration: float = 0.0
    samples: list[Sample] = field(default_factory=list)
    measured: bool = False

    @property
    def interval(self) -> float:
        return 1.0 / SAMPLES_PER_SECOND

    @property
    def settle(self) -> float:
        """When the clip first becomes as good as it is going to get.

        The earliest half-second that scores :data:`SETTLE_LEVEL` of the best
        half-second in the clip. Scored as a window rather than sample by
        sample on purpose: requiring every individual sample to clear the bar
        means one blink in an otherwise good stretch disqualifies it, and the
        clip then reports that it settles at zero — the one answer that is
        certainly wrong for a clip that opens badly enough to be asked about.
        """
        if not self.samples:
            return 0.0
        span = 0.5
        scores = [(s.t, self.score_at(s.t, span)) for s in self.samples
                  if s.t <= max(0.0, self.duration - span) + 1e-6]
        if not scores:
            return 0.0
        best = max(score for _, score in scores)
        if best <= 0:
            return 0.0
        for t, score in scores:
            if score >= best * SETTLE_LEVEL:
                return t
        return 0.0

    @property
    def head_penalty(self) -> float:
        """How much worse the first second is than the clip's best.

        The number this module exists for: 0 means the clip starts as well as
        it ever does, and 0.6 means five sixths of what the opening shows is
        the camera being got ready. Reported so a log line can say why an
        in-point moved.
        """
        if not self.samples:
            return 0.0
        head = [s.usable for s in self.samples if s.t < 1.0]
        best = max((s.usable for s in self.samples), default=0.0)
        if not head or best <= 0:
            return 0.0
        return max(0.0, best - (sum(head) / len(head)))

    def score_at(self, start: float, duration: float) -> float:
        """How usable ``duration`` seconds from ``start`` are."""
        inside = [s for s in self.samples
                  if start - 1e-6 <= s.t < start + duration + 1e-6]
        if not inside:
            return 0.0
        mean = sum(s.usable for s in inside) / len(inside)
        worst = min(s.usable for s in inside)
        return mean * (1.0 - WORST_PULL) + worst * WORST_PULL

    def best(self, want: float, *, after: float = 0.0) -> Window:
        """The best ``want``-second window starting at or after ``after``.

        Falls back to the requested position rather than to nothing: a caller
        asking for four seconds of a three-second clip should get the three
        seconds, because a short shot is a far smaller problem than a missing
        one.
        """
        want = max(0.0, float(want))
        after = max(0.0, float(after))
        room = max(0.0, self.duration - after)
        if room <= 0:
            return Window(start=min(after, self.duration), duration=0.0)
        length = min(want, room)
        if not self.samples or length <= 0:
            return Window(start=after, duration=length)

        latest = self.duration - length
        starts = [s.t for s in self.samples if after - 1e-6 <= s.t <= latest + 1e-6]
        if not starts:
            return Window(start=min(after, max(0.0, latest)), duration=length)

        # Earliest wins a tie, which keeps a clip whose quality is flat cutting
        # in shooting order rather than wandering to its end for no reason.
        best_start = starts[0]
        best_score = self.score_at(best_start, length)
        for start in starts[1:]:
            score = self.score_at(start, length)
            if score > best_score + 1e-9:
                best_start, best_score = start, score
        return Window(start=best_start, duration=length, score=best_score)


def _cache_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".window.json"


def _load_cached(path: str) -> ClipWindows | None:
    """Cached measurements, or None when missing or stale. Keyed on size and
    mtime for the same reason :mod:`modules.shot_type` is: the files are
    gigabytes and hashing them would cost more than measuring them."""
    try:
        with open(_cache_path(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        stat = os.stat(path)
        if (data.get("version") != CACHE_VERSION
                or data.get("size") != stat.st_size
                or abs(float(data.get("mtime", 0)) - stat.st_mtime) > 1.0):
            return None
        return ClipWindows(
            path=path, duration=float(data["duration"]),
            samples=[Sample(**s) for s in data["samples"]],
            measured=bool(data.get("measured", True)))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_cached(clip: ClipWindows) -> None:
    try:
        stat = os.stat(clip.path)
        with open(_cache_path(clip.path), "w", encoding="utf-8") as fh:
            json.dump({"version": CACHE_VERSION, "size": stat.st_size,
                       "mtime": stat.st_mtime, "duration": clip.duration,
                       "measured": clip.measured,
                       "samples": [asdict(s) for s in clip.samples]}, fh)
    except OSError:
        pass   # an unwritable folder costs the cache, not the measurement


def _score(samples: list[Sample]) -> None:
    """Fill in ``usable`` on every sample, in place.

    Sharpness is the only term normalised against the clip rather than against
    a constant, and it has to be: the variance of a Laplacian on a hedge and on
    a wall differ by an order of magnitude at identical focus, so an absolute
    threshold measures the subject and not the picture.
    """
    if not samples:
        return
    ranked = sorted(s.sharpness for s in samples)
    # The 85th percentile rather than the maximum: one accidentally crisp frame
    # should not make every other frame in the clip look soft.
    reference = ranked[min(len(ranked) - 1, int(len(ranked) * 0.85))]
    floor = reference * SHARP_REFERENCE

    for s in samples:
        coherence = min(1.0, max(0.0, s.coherence))
        calm = 1.0 - min(1.0, s.shift / SHIFT_FULL)
        steady = 1.0 - min(1.0, s.jerk / JERK_FULL)
        # Movement that holds its speed is fine; movement that keeps changing
        # is not. Weighted towards steadiness for that reason.
        steady = steady * 0.7 + calm * 0.3
        sharp = min(1.0, s.sharpness / floor) if floor > 0 else 1.0
        exposure = 1.0 - min(1.0, s.drift / EXPOSURE_FULL)
        if s.brightness < DARK or s.brightness > BLOWN:
            exposure = 0.0

        terms = {"coherence": coherence, "sharpness": sharp,
                 "steady": steady, "exposure": exposure}
        mean = sum(WEIGHTS[k] * v for k, v in terms.items())
        s.usable = mean * (1.0 - WORST_PULL) + min(terms.values()) * WORST_PULL


def profile(path: str, *, use_cache: bool = True, log_fn=print) -> ClipWindows:
    """Measure one clip and return the windows it can offer.

    Never raises. A clip that cannot be opened, or an OpenCV that is not
    installed, comes back unmeasured — and an unmeasured clip behaves the way
    everything did before this module existed, which is the right failure for
    something that only ever improves a choice.
    """
    if use_cache:
        cached = _load_cached(path)
        if cached is not None:
            return cached

    clip = ClipWindows(path=path)
    try:
        import cv2
        import numpy as np
    except Exception:
        return clip

    try:
        capture = cv2.VideoCapture(path)
        opened = bool(capture.isOpened())
    except Exception:
        return clip
    if not opened:
        log_fn(f"⚠️ Could not open {os.path.basename(path)} to measure it")
        return clip

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        clip.duration = total / fps if total else 0.0
        if clip.duration < MIN_MEASURABLE:
            return clip

        step = max(1, int(round(fps / SAMPLES_PER_SECOND)))
        samples: list[Sample] = []
        previous = None          # float32 grayscale, for phase correlation
        last_shift = None
        last_brightness = None
        window = None            # Hanning window, sized once

        # Read forward and decode every ``step``-th frame. Seeking per sample
        # is the obvious way to write this and is both slower on long GOPs and
        # inexact — a seek to frame 40 on GoPro footage can land on 36, which
        # puts the measurements on the wrong timestamps.
        index = 0
        while index < total:
            if not capture.grab():
                break
            if index % step == 0:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    height, width = frame.shape[:2]
                    scale = WORK_WIDTH / max(1.0, float(width))
                    small = cv2.resize(
                        frame, (max(8, int(width * scale)), max(8, int(height * scale))),
                        interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    current = np.float32(gray)
                    if window is None:
                        window = cv2.createHanningWindow(
                            (current.shape[1], current.shape[0]), cv2.CV_32F)

                    t = index / fps
                    sample = Sample(t=t)
                    sample.sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    sample.brightness = float(gray.mean()) / 255.0
                    dt = step / fps

                    if previous is not None:
                        try:
                            (dx, dy), response = cv2.phaseCorrelate(
                                previous, current, window)
                        except Exception:
                            dx = dy = 0.0
                            response = 0.0
                        # Frame-widths per second, so the numbers mean the same
                        # thing at any sample rate or working size.
                        moved = math.hypot(dx, dy) / max(1.0, current.shape[1])
                        sample.shift = moved / dt
                        sample.coherence = max(0.0, float(response))
                        if last_shift is not None:
                            sample.jerk = abs(sample.shift - last_shift) / dt
                        last_shift = sample.shift
                        if last_brightness is not None:
                            sample.drift = abs(
                                sample.brightness - last_brightness) / dt
                    last_brightness = sample.brightness

                    previous = current
                    samples.append(sample)
            index += 1

        # The first sample has no predecessor, so its movement terms are unmeasured
        # zeros rather than measurements of stillness. Copying the second
        # sample's is closer to true than scoring the opening frame as perfectly
        # steady — which is exactly the frame this module exists to distrust.
        if len(samples) > 1:
            samples[0].coherence = samples[1].coherence
            samples[0].shift = samples[1].shift
            samples[0].jerk = samples[1].jerk
            samples[0].drift = samples[1].drift

        clip.samples = samples
        clip.measured = len(samples) > 1
        _score(clip.samples)
    except Exception as exc:
        # The contract this module is used under is that measuring can only
        # ever improve a choice, so a decoder that returns something nobody
        # expected has to cost the measurement and nothing else. Half-filled
        # samples are dropped rather than scored: a clip measured up to the
        # point it broke would be ranked against clips measured in full.
        log_fn(f"⚠️ Could not measure {os.path.basename(path)} ({exc}); "
               f"its shots will start at the top of the clip")
        clip.samples = []
        clip.measured = False
    finally:
        try:
            capture.release()
        except Exception:
            pass

    if use_cache and clip.measured:
        _save_cached(clip)
    return clip


def profile_all(paths, *, use_cache: bool = True, log_fn=print) -> dict:
    """Measure a list of clips, reporting once on what was found.

    The summary names the clips whose openings are worst, because that is the
    line that explains an in-point the user did not choose: "three clips open
    on the camera being placed" is actionable, and a per-clip dump of eight
    numbers is not.
    """
    out: dict[str, ClipWindows] = {}
    for path in paths or []:
        out[path] = profile(path, use_cache=use_cache, log_fn=log_fn)

    measured = [c for c in out.values() if c.measured]
    if measured:
        late = [c for c in measured if c.settle > 0.4]
        log_fn(f"🔍 Measured {len(measured)} clip(s) for camera settling"
               + (f"; {len(late)} start on the camera still being placed"
                  if late else "; all of them start clean"))
    return out
