"""Subtitle times that respect shot changes, the way subtitlers time them.

A subtitle that appears a few frames after a cut, or hangs a few frames into
the next shot, reads as a flicker: the eye sees the picture change and the text
change separately. Professional timing therefore lines the two up. The rules
here are the common ones (they are in the major streaming style guides, and in
the defaults of the subtitle editors that read shot-change lists):

* **In-time.** A cue that starts within ``SNAP_FRAMES`` of a cut starts on the
  cut, before or after - the text appears with the new picture.
* **Out-time.** A cue that ends within ``SNAP_FRAMES`` of a cut ends
  ``GAP_FRAMES`` before it. When pulling it back would leave it shorter than
  it can be read, it is held past the cut instead, by at least
  ``SNAP_FRAMES``, so it no longer looks like a flicker.
* **Gaps.** Two cues closer than ``SNAP_FRAMES`` are joined to a
  ``GAP_FRAMES`` gap: a blink of empty screen between two lines is noise.
  The gap is kept, so the viewer still sees that a new line has begun.

No cue moves by more than ``SNAP_FRAMES``, and none crosses its neighbour.
Times come back on frame boundaries of the video's own frame rate. The cut
list is the one ``modules/segments/shot_boundaries.py`` produces; nothing
here reads video.
"""
from __future__ import annotations

import bisect
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Half a second at 24-25 fps: the window the style guides use.
SNAP_FRAMES = 12
# What separates two cues, and a cue from the cut it ends before.
GAP_FRAMES = 2
# Shorter than this a line cannot be read; the SRT writer uses the same.
MIN_SECONDS = 0.8


def _nearest(cuts: Sequence[int], frame: int) -> Optional[int]:
    i = bisect.bisect_left(cuts, frame)
    best = None
    for c in cuts[max(0, i - 1):i + 1]:
        if best is None or abs(c - frame) < abs(best - frame):
            best = c
    return best


def snap(cues: List[Dict], cut_times: Sequence[float], fps: float,
         snap_frames: int = SNAP_FRAMES, gap_frames: int = GAP_FRAMES,
         min_seconds: float = MIN_SECONDS) -> Tuple[List[Dict], Dict[str, int]]:
    """``(cues, counts)``: the cues (dicts with ``start``/``end`` in seconds,
    in order) re-timed by the rules above, as new dicts, and how many starts,
    ends and gaps moved. ``fps`` is the video's frame rate."""
    if not cues or fps <= 0:
        return [dict(c) for c in cues], {"starts": 0, "ends": 0, "gaps": 0}
    cuts = sorted({int(round(t * fps)) for t in cut_times})
    min_frames = int(round(min_seconds * fps))
    s = [int(round(float(c["start"]) * fps)) for c in cues]
    e = [max(int(round(float(c["end"]) * fps)), s[i] + 1) for i, c in enumerate(cues)]
    counts = {"starts": 0, "ends": 0, "gaps": 0}
    n = len(cues)

    for i in range(n):
        # In-time onto a cut, not past this one's end. When the line before
        # still runs over that cut, it ends before the cut instead - the two
        # lines then change with the picture - unless that leaves it too short.
        c = _nearest(cuts, s[i]) if cuts else None
        if (c is None or c == s[i] or abs(c - s[i]) > snap_frames
                or e[i] - c < min(min_frames, e[i] - s[i])):
            continue
        if i and c < e[i - 1] + gap_frames:
            prev_end = c - gap_frames
            if prev_end - s[i - 1] < min(min_frames, e[i - 1] - s[i - 1]):
                continue
            e[i - 1] = prev_end
        s[i] = c
        counts["starts"] += 1

    for i in range(n):
        ceiling = s[i + 1] - gap_frames if i + 1 < n else None
        c = _nearest(cuts, e[i]) if cuts else None
        if c is None or abs(c - e[i]) > snap_frames or c <= s[i]:
            continue
        before = c - gap_frames
        long_enough = before - s[i] >= min(min_frames, e[i] - s[i])
        held = c + snap_frames
        if before != e[i] and long_enough and (ceiling is None or before <= ceiling):
            e[i] = before
            counts["ends"] += 1
        elif not long_enough and held > e[i] and (ceiling is None or held <= ceiling):
            e[i] = held
            counts["ends"] += 1

    for i in range(n - 1):
        gap = s[i + 1] - e[i]
        if gap != gap_frames and gap < snap_frames:
            target = s[i + 1] - gap_frames
            # Closing a gap never carries a line over a cut it ended before.
            crossed = [c for c in cuts if e[i] < c < s[i + 1]]
            if crossed:
                target = min(target, crossed[0] - gap_frames)
            if target > s[i] and target != e[i]:
                e[i] = target
                counts["gaps"] += 1

    out = []
    for i, cue in enumerate(cues):
        out.append(dict(cue, start=s[i] / fps, end=max(e[i], s[i] + 1) / fps))
    return out, counts


def shot_changes(video_path: str, log_fn: Callable[[str], None] = print,
                 progress_fn=None, should_cancel=None) -> Optional[Tuple[List[float], float]]:
    """``(cut_times, fps)`` for ``video_path``, or None when it cannot be read
    (the subtitles are then written as they were)."""
    try:
        from modules.segments import shot_boundaries
        shots = shot_boundaries.detect_shots(video_path, progress_fn=progress_fn,
                                             should_cancel=should_cancel, log_fn=log_fn)
    except Exception as e:  # noqa: BLE001 - snapping is not worth the subtitles
        if type(e).__name__ == "ShotDetectionCancelled":
            raise
        print(f"shot_snap: no shot list for {video_path}: {type(e).__name__}: {e}")
        log_fn("⚠️ Could not find the shot changes; subtitles are timed without them")
        return None
    return list(shots.cut_times), shots.fps_num / shots.fps_den
