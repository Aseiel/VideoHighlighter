"""Cut a whole video into contiguous chapters, using its own shot structure.

Highlight selection picks a *sparse ranked* handful of moments and lets the rest
of the video fall away. A chapter list is the opposite shape: an exhaustive
partition where every second belongs to exactly one chapter, in order, with no
gaps and no ranking. None of the region-merging in ``modules/auto_segments.py``
applies, so this is a parallel path over the same signals rather than a mode of
that one.

The method, and why it is this one
----------------------------------

The obvious approach -- Foote novelty on the per-second CLIP embeddings -- fails
badly on edited video, and the way it fails is instructive. A dialogue exchange
cuts between two faces every two seconds. Per-second novelty therefore spikes at
every one of those cuts, dozens of times, inside what any viewer would call a
single scene. The signal is real; it is just measuring shot changes, and shot
changes are not chapter changes.

Two corrections turn it into something that works:

**Boundaries can only fall on real cuts.** A chapter never begins mid-shot, so
the candidate set is the video's own scene list rather than every second. On a
feature film that is a few thousand candidates instead of a few hundred
thousand, and every one of them is a place a boundary could legitimately go.
This is why having scene detection already makes movies *easier* than the
unstructured case, not harder.

**Comparison happens between runs of shots, not across a single cut.** Each shot
is collapsed to one signature vector (the mean of its sampled seconds), and a
candidate cut is scored by how unlike the previous ``W`` shots the next ``W``
shots are. Shot-reverse-shot averages out inside such a window -- alternating
between two faces has a stable mean -- so the dialogue scene reads as one
plateau, and the score only rises where the footage genuinely stops looking like
what came before: a new location, a new palette, a new set of faces.

That is a checkerboard kernel evaluated on a shot lattice instead of a time
lattice, which is the whole trick.

Signals are optional, and it degrades in a defined order
--------------------------------------------------------

Visual signatures need the CLIP index, which needs the optional OpenVINO stack.
When embeddings are unavailable the same operators run on whatever else the
caller passes -- an audio quiet-gap curve, per-second motion -- and failing
everything, ``chapterize`` falls back to grouping shots into runs of roughly the
target length. The result is always a valid partition; only its reasons get
weaker, and ``method`` on each chapter records which one applied.

Naming
------

Nothing here invents chapter titles. A vector is a point, not a name, and this
repo does not ship a built-in vocabulary to match points against. What each
chapter carries instead is measured description -- how long it runs, how many
shots, how fast it cuts, and how far it sits from the video's own average look.
Callers that *do* have a vocabulary (user-taught categories, a transcript) can
label the partition afterwards; the partition does not depend on them.

Everything below the embeddings is numpy over small arrays, so the logic that
decides what the user sees is testable on synthetic input in milliseconds -- see
``tests/test_chapters.py``. ``chapterize`` returns plain lists and dicts, so the
result is JSON-serialisable and can cross a thread boundary into Qt untouched.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

# How many shots either side of a candidate cut get averaged into the "before"
# and "after" vectors. Six is chosen against shot-reverse-shot: a dialogue
# exchange alternating between two setups needs several of each on both sides
# before the window means stop tracking the alternation. Too small and every
# cut in a conversation scores; too large and a genuinely short scene is
# straddled by the window and never separates from its neighbours.
DEFAULT_SHOT_WINDOW = 6

# Nothing shorter than this becomes its own chapter. A chapter list is a
# navigation aid, and a viewer cannot navigate to a 20-second entry in a list of
# three hundred. This is also what stops a burst of high-novelty cuts (a montage,
# an action sequence) from shattering into fragments.
DEFAULT_MIN_CHAPTER_SECONDS = 90.0

# A run this long with no boundary is suspicious even if nothing scored: a
# single locked-off shot can hold a low novelty score for a very long time. Past
# this, the strongest interior candidate is taken regardless of its score.
DEFAULT_MAX_CHAPTER_SECONDS = 900.0

# Scales median absolute deviation onto the standard-deviation scale, so the
# threshold below reads on the familiar one. Same constant, same reason, as
# modules/reaction_bursts.py.
MAD_TO_SIGMA = 1.4826

# How far above the video's own typical novelty a cut must score to be called a
# boundary. Expressed in robust z units rather than an absolute cosine distance
# because the scale of "typical novelty" is a property of the edit -- a talky
# chamber piece and a travelogue differ by more than any fixed number survives.
DEFAULT_Z = 1.5

# Below this many shots there is no structure to find and windowed comparison is
# meaningless; the caller gets one chapter covering the whole video.
MIN_SHOTS_FOR_STRUCTURE = 4


# ---------------------------------------------------------------------------
# Operators -- pure numpy, no I/O, no model
# ---------------------------------------------------------------------------
def l2_normalize(a: np.ndarray) -> np.ndarray:
    """Row-wise unit norm, safe on zero rows."""
    a = np.asarray(a, dtype=np.float32)
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)


def shot_signatures(timestamps: Sequence[float], embeddings: np.ndarray,
                    scenes: Sequence[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each shot to one unit vector: the mean of its sampled seconds.

    Returns ``(signatures, keep)`` where ``keep`` indexes the scenes that had at
    least one sampled frame. Scenes shorter than the sampling interval land
    between samples and have no signature; dropping them here rather than
    substituting a neighbour's vector keeps a fast cutting passage from inventing
    novelty it did not measure.
    """
    timestamps = np.asarray(timestamps, dtype=np.float64)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    sigs, keep = [], []
    for i, (start, end) in enumerate(scenes):
        sel = (timestamps >= float(start)) & (timestamps < float(end))
        if not sel.any():
            continue
        sigs.append(embeddings[sel].mean(axis=0))
        keep.append(i)
    if not sigs:
        dim = embeddings.shape[1] if embeddings.ndim == 2 else 512
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=int)
    return l2_normalize(np.stack(sigs)), np.asarray(keep, dtype=int)


def boundary_novelty(signatures: np.ndarray,
                     window: int = DEFAULT_SHOT_WINDOW) -> np.ndarray:
    """How unlike the preceding shots the following shots are, at every cut.

    ``novelty[i]`` scores the cut *entering* shot ``i``, so index 0 is always 0 --
    there is no cut before the first shot. The value is one minus the cosine
    between the mean of up to ``window`` shots on each side, i.e. 0 for "the film
    carries on looking the same" and rising toward 2 for a complete change.

    Truncating the window at the ends rather than padding matters: padding with
    zeros or edge-repeats manufactures a large distance at the first and last
    cuts, which would place a spurious boundary a few shots into the film every
    time.
    """
    signatures = np.asarray(signatures, dtype=np.float32)
    n = len(signatures)
    novelty = np.zeros(n, dtype=np.float32)
    if n < 2:
        return novelty
    window = max(1, int(window))
    for i in range(1, n):
        before = signatures[max(0, i - window):i].mean(axis=0)
        after = signatures[i:min(n, i + window)].mean(axis=0)
        bn = float(np.linalg.norm(before))
        an = float(np.linalg.norm(after))
        if bn < 1e-12 or an < 1e-12:
            continue
        novelty[i] = 1.0 - float(before @ after) / (bn * an)
    return novelty


def robust_threshold(values: np.ndarray, z: float = DEFAULT_Z) -> float:
    """Median plus ``z`` robust deviations -- the bar a cut must clear.

    Median and MAD rather than mean and standard deviation because the quantity
    being described is contaminated by exactly the thing being looked for: real
    boundaries are high outliers, and they would inflate a mean-based bar enough
    to hide themselves.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * MAD_TO_SIGMA
    return median + z * max(mad, 1e-6)


def pick_boundaries(novelty: np.ndarray, starts: Sequence[float],
                    min_gap: float = DEFAULT_MIN_CHAPTER_SECONDS,
                    max_run: float = DEFAULT_MAX_CHAPTER_SECONDS,
                    z: float = DEFAULT_Z,
                    target: Optional[int] = None,
                    duration: Optional[float] = None) -> list[int]:
    """Choose which cuts become chapter boundaries. Returns shot indices, sorted.

    Strongest-first with suppression, rather than a left-to-right scan: a scan
    commits to the first cut that clears the bar and then blocks the far better
    one eight seconds later, which in practice puts boundaries on the establishing
    shot *before* the scene change instead of on the change itself.

    ``target``, when given, replaces the adaptive threshold with a count -- what
    a "give me about 12 chapters" control needs. ``min_gap`` is honoured either
    way, so a request for more chapters than the video has room for returns
    fewer rather than adjacent duplicates.
    """
    novelty = np.asarray(novelty, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.float64)
    n = len(novelty)
    if n < 2:
        return []

    candidates = np.argsort(novelty)[::-1]
    if target is None:
        bar = robust_threshold(novelty[1:], z=z)
        candidates = [i for i in candidates if i >= 1 and novelty[i] >= bar]
    else:
        candidates = [i for i in candidates if i >= 1]

    chosen: list[int] = []
    video_start = float(starts[0])
    for i in candidates:
        if target is not None and len(chosen) >= max(0, int(target) - 1):
            break
        # The opening and closing chapters are bounded by the video, not by a
        # neighbouring boundary, so they need checking separately -- an empty
        # `chosen` satisfies the pairwise test vacuously and would otherwise let
        # a boundary land seconds into the film. That is not hypothetical: the
        # truncated window at the first cut compares one shot against six, which
        # scores high on any video whose opening shot is atypical.
        if (starts[i] - video_start) < min_gap:
            continue
        if duration is not None and (duration - starts[i]) < min_gap:
            continue
        if all(abs(starts[i] - starts[j]) >= min_gap for j in chosen):
            chosen.append(int(i))

    chosen = _fill_long_runs(chosen, novelty, starts, min_gap, max_run, duration)
    return sorted(chosen)


def _fill_long_runs(chosen: list[int], novelty: np.ndarray, starts: np.ndarray,
                    min_gap: float, max_run: float,
                    duration: Optional[float]) -> list[int]:
    """Split any stretch longer than ``max_run`` at its best interior cut.

    Runs until nothing is over-long, because one split can leave both halves
    still over-long on a very static hour.
    """
    if max_run <= 0 or len(novelty) < 2:
        return chosen
    end = float(duration) if duration is not None else float(starts[-1])
    guard = 0
    while guard < 64:
        guard += 1
        edges = sorted(chosen)
        spans = []
        prev_time, prev_idx = float(starts[0]), 0
        for idx in edges + [None]:
            here = end if idx is None else float(starts[idx])
            spans.append((prev_time, here, prev_idx, idx))
            prev_time, prev_idx = here, idx
        over = [s for s in spans if (s[1] - s[0]) > max_run]
        if not over:
            break
        added = False
        for start_t, end_t, lo, hi in over:
            interior = [
                i for i in range(1, len(novelty))
                if start_t + min_gap <= starts[i] <= end_t - min_gap
                and i not in chosen
            ]
            if not interior:
                continue
            best = max(interior, key=lambda i: novelty[i])
            chosen.append(int(best))
            added = True
        if not added:
            break
    return chosen


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    """``H:MM:SS`` -- the form a chapter list is read in."""
    seconds = max(0, int(round(float(seconds))))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def describe_pace(shots: int, seconds: float) -> str:
    """Name the cutting rate. Mechanism only -- says nothing about subject.

    The bands are the conventional ones for edited video: under ~4 seconds per
    shot reads as fast, over ~15 reads as held.
    """
    if shots <= 0 or seconds <= 0:
        return "unmeasured"
    per_shot = seconds / shots
    if per_shot < 4.0:
        return "fast-cut"
    if per_shot < 15.0:
        return "steady"
    return "held"


def chapterize(duration: float,
               scenes: Sequence[tuple],
               timestamps: Optional[Sequence[float]] = None,
               embeddings: Optional[np.ndarray] = None,
               window: int = DEFAULT_SHOT_WINDOW,
               min_chapter: float = DEFAULT_MIN_CHAPTER_SECONDS,
               max_chapter: float = DEFAULT_MAX_CHAPTER_SECONDS,
               z: float = DEFAULT_Z,
               target: Optional[int] = None,
               log_fn=print) -> list[dict]:
    """Partition ``[0, duration]`` into chapters. Returns plain JSON-safe dicts.

    ``scenes`` is the ``(start, end)`` list from
    ``modules/motion_scene_detect_optimized.py``. ``timestamps``/``embeddings``
    are a CLIP index's arrays; without them the shot-length fallback applies.
    """
    duration = float(duration)
    scenes = [(float(a), float(b)) for a, b in (scenes or [])]
    scenes.sort(key=lambda s: s[0])

    if duration <= 0:
        return []
    if len(scenes) < MIN_SHOTS_FOR_STRUCTURE:
        return [_chapter(1, 0.0, duration, len(scenes), 0.0, "single")]

    method = "visual"
    if embeddings is None or timestamps is None or len(np.atleast_1d(timestamps)) == 0:
        starts = np.asarray([s[0] for s in scenes], dtype=np.float64)
        novelty = np.zeros(len(scenes), dtype=np.float32)
        method = "shot-length"
        edges = _even_shot_split(starts, duration, min_chapter, max_chapter, target)
    else:
        sigs, keep = shot_signatures(timestamps, embeddings, scenes)
        if len(sigs) < MIN_SHOTS_FOR_STRUCTURE:
            log_fn("📖 Chapters: too few sampled shots for visual structure")
            starts = np.asarray([s[0] for s in scenes], dtype=np.float64)
            novelty = np.zeros(len(scenes), dtype=np.float32)
            method = "shot-length"
            edges = _even_shot_split(starts, duration, min_chapter, max_chapter, target)
        else:
            starts = np.asarray([scenes[i][0] for i in keep], dtype=np.float64)
            novelty = boundary_novelty(sigs, window=window)
            edges = pick_boundaries(novelty, starts, min_gap=min_chapter,
                                    max_run=max_chapter, z=z, target=target,
                                    duration=duration)

    cut_times = [0.0] + [float(starts[i]) for i in edges] + [duration]
    scene_starts = np.asarray([s[0] for s in scenes], dtype=np.float64)

    chapters = []
    for n, (a, b) in enumerate(zip(cut_times[:-1], cut_times[1:]), start=1):
        if b - a <= 0:
            continue
        shots = int(((scene_starts >= a) & (scene_starts < b)).sum())
        score = float(novelty[edges[n - 2]]) if n >= 2 and edges else 0.0
        chapters.append(_chapter(n, a, b, shots, score, method))

    log_fn(f"📖 Chapters: {len(chapters)} over {format_timestamp(duration)} ({method})")
    return chapters


def _even_shot_split(starts: np.ndarray, duration: float, min_chapter: float,
                     max_chapter: float, target: Optional[int]) -> list[int]:
    """Fallback partition: cut at the shot nearest each evenly spaced mark.

    Deliberately not a fixed-time split. Landing every boundary on a real cut is
    the one guarantee worth keeping when there is nothing to measure, because a
    chapter that begins mid-shot is visibly wrong in a way an arbitrary-but-clean
    one is not.
    """
    if target and target > 1:
        count = int(target)
    else:
        count = max(1, int(round(duration / max(min_chapter * 3, 1.0))))
        if max_chapter > 0:
            count = max(count, int(np.ceil(duration / max_chapter)))
    if count <= 1:
        return []
    chosen: list[int] = []
    for k in range(1, count):
        mark = duration * k / count
        idx = int(np.argmin(np.abs(starts - mark)))
        if idx >= 1 and idx not in chosen:
            if all(abs(starts[idx] - starts[j]) >= min_chapter for j in chosen):
                chosen.append(idx)
    return sorted(chosen)


def _chapter(number: int, start: float, end: float, shots: int,
             score: float, method: str) -> dict:
    seconds = max(0.0, end - start)
    return {
        "number": number,
        "start": round(float(start), 2),
        "end": round(float(end), 2),
        "duration": round(seconds, 2),
        "timestamp": format_timestamp(start),
        # No vocabulary ships here, so the title is positional and the caller is
        # expected to overwrite it once it has something to name the span with.
        "title": f"Chapter {number}",
        "shots": shots,
        "pace": describe_pace(shots, seconds),
        # How strongly the cut that opened this chapter separated from what came
        # before. 0.0 on the first chapter, which no cut opened.
        "boundary_score": round(float(score), 4),
        "method": method,
    }


# ---------------------------------------------------------------------------
# Cached visual signatures
# ---------------------------------------------------------------------------
def cached_index_arrays(video_path: str, cache_dir: str = "./cache"):
    """``(timestamps, embeddings)`` if an index is *already* cached, else nones.

    Deliberately never builds one. Encoding a feature film's frames is minutes of
    work, and a chapter list is a by-product of a run the user asked for
    something else from -- silently tripling that run's cost to improve a section
    they did not request is the wrong trade. So the good boundaries appear on any
    video that has been searched before, and everything else falls back to shot
    structure, which costs nothing and is honest about being weaker (`method`
    records which applied).

    Staleness is handled by the cache path itself: `cache_path_for` keys on the
    video's path, size and mtime, so a re-encode simply misses. The model/interval
    check `load_or_build` also does is skipped on purpose -- it needs the CLIP
    model instantiated, which is the expensive import this function exists to
    avoid, and every comparison here is between embeddings *within* one index.

    The import is guarded because the CLIP stack is optional; without it this
    returns nones rather than taking the report down.
    """
    try:
        from llm.clip_index import ClipFrameIndex, cache_path_for
    except Exception as exc:
        print(f"📖 Chapters: no CLIP stack ({exc}); using shot structure")
        return None, None

    try:
        path = cache_path_for(video_path, cache_dir)
    except OSError:
        return None, None          # the video moved or vanished
    if not os.path.exists(path):
        print("📖 Chapters: no cached visual index; using shot structure")
        return None, None
    try:
        index = ClipFrameIndex.load(path)
    except Exception as exc:
        print(f"⚠️  Chapters: unreadable visual index ({exc}); using shot structure")
        return None, None
    print(f"📖 Chapters: reusing {len(index)} cached frame signatures")
    return index.timestamps, index.embeddings


def chapters_for_video(video_path: str,
                       scenes: Sequence[tuple],
                       duration: float,
                       cache_dir: str = "./cache",
                       log_fn=print,
                       **kw) -> list[dict]:
    """The one call a caller needs: cached signatures if any, then partition.

    Exists so a call site is an import and one line. The pipeline and the free
    edition's copy of it are the same file, and every extra line of integration
    there is a line that has to be ported by hand and can drift.
    """
    timestamps, embeddings = cached_index_arrays(video_path, cache_dir)
    return chapterize(duration, scenes, timestamps, embeddings,
                      log_fn=log_fn, **kw)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def to_ffmetadata(chapters: Sequence[dict]) -> str:
    """An ffmpeg metadata file -- the portable way to attach chapters.

    Written back into a container with::

        ffmpeg -i in.mp4 -i chapters.txt -map_metadata 1 -codec copy out.mp4

    which needs no MKVToolNix and no remux beyond a stream copy. Timebase is
    milliseconds; ffmpeg wants integers, and a chapter boundary is not a
    sub-millisecond quantity.
    """
    lines = [";FFMETADATA1"]
    for ch in chapters:
        start_ms = int(round(float(ch["start"]) * 1000))
        end_ms = int(round(float(ch["end"]) * 1000))
        if end_ms <= start_ms:
            continue
        title = str(ch.get("title", "")).replace("\\", "\\\\")
        for special in ("=", ";", "#", "\n"):
            title = title.replace(special, "\\" + special)
        lines += ["", "[CHAPTER]", "TIMEBASE=1/1000",
                  f"START={start_ms}", f"END={end_ms}", f"title={title}"]
    return "\n".join(lines) + "\n"


def to_youtube(chapters: Sequence[dict]) -> str:
    """The description-box format: one ``H:MM:SS Title`` per line."""
    return "\n".join(f"{ch['timestamp']} {ch.get('title', '')}".rstrip()
                     for ch in chapters)


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------
def main():
    """Chapterize one video from the command line.

        python -m modules.chapters --video "D:\\clips\\a.mp4" --ffmetadata out.txt

    Both heavy imports live in here rather than at module scope: the operators
    above are pure numpy, and keeping them importable without cv2/torch/OpenVINO
    is what lets the test suite run in a 5 MB environment.
    """
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Cut a video into chapters.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="CLIP sampling interval, seconds")
    ap.add_argument("--target", type=int, default=None,
                    help="roughly how many chapters to produce")
    ap.add_argument("--min-chapter", type=float, default=DEFAULT_MIN_CHAPTER_SECONDS)
    ap.add_argument("--max-chapter", type=float, default=DEFAULT_MAX_CHAPTER_SECONDS)
    ap.add_argument("--window", type=int, default=DEFAULT_SHOT_WINDOW)
    ap.add_argument("--z", type=float, default=DEFAULT_Z)
    ap.add_argument("--device", default="AUTO")
    ap.add_argument("--no-visual", action="store_true",
                    help="skip CLIP and partition on shot structure alone")
    ap.add_argument("--ffmetadata", help="write an ffmpeg chapter file here")
    ap.add_argument("--json", help="write the chapter list here")
    args = ap.parse_args()

    from modules.motion_scene_detect_optimized import detect_scenes_motion_optimized

    print(f"🎬 Detecting shots in {os.path.basename(args.video)} ...")
    scenes, _, _ = detect_scenes_motion_optimized(args.video, debug=False)
    if not scenes:
        print("No shots detected; nothing to chapterize.")
        return
    duration = float(scenes[-1][1])
    print(f"🎬 {len(scenes)} shots over {format_timestamp(duration)}")

    timestamps = embeddings = None
    if not args.no_visual:
        try:
            from llm.clip_index import cache_path_for, load_or_build

            index = load_or_build(args.video, cache_path_for(args.video),
                                  interval=args.interval, device=args.device)
            timestamps, embeddings = index.timestamps, index.embeddings
        except Exception as e:
            print(f"⚠️  No visual signatures ({e}); falling back to shot structure")

    chapters = chapterize(duration, scenes, timestamps, embeddings,
                          window=args.window, min_chapter=args.min_chapter,
                          max_chapter=args.max_chapter, z=args.z,
                          target=args.target)

    print()
    for ch in chapters:
        print(f"{ch['timestamp']}  {ch['title']:<14} "
              f"{format_timestamp(ch['duration']):>8}  "
              f"{ch['shots']:>4} shots  {ch['pace']:<10} "
              f"novelty {ch['boundary_score']:.3f}")

    if args.ffmetadata:
        with open(args.ffmetadata, "w", encoding="utf-8") as fh:
            fh.write(to_ffmetadata(chapters))
        print(f"\n📝 {args.ffmetadata}")
        print(f'   ffmpeg -i "{args.video}" -i "{args.ffmetadata}" '
              f'-map_metadata 1 -codec copy out.mp4')
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(chapters, fh, indent=2)
        print(f"📝 {args.json}")


if __name__ == "__main__":
    main()
