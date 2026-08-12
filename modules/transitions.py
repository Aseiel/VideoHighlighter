"""
transitions.py — join clips with a transition between them, not just a cut.

Why this exists
===============
:mod:`modules.combine_videos` normalises every clip to one format and then
joins them with the concat demuxer and ``-c copy``. That is the right design
for what it does — the stream copy is only safe *because* of the normalise
pass, and it makes a long reel cheap. But a demuxer can only butt one clip
against the next, so every join is a hard cut, and a reel of hard cuts reads
as a slideshow no matter how well the moments were chosen.

A transition needs the two clips to overlap, which means decoding both and
re-encoding the result: it cannot be a stream copy. So this is a separate
path rather than a flag on the old one, and the old one stays the fast default
for the all-cuts case (this module delegates to it, rather than doing a
pointless re-encode).

How the timing works
====================
``xfade`` places the overlap by absolute offset into the running result, so the
offsets have to be accumulated rather than taken per clip::

    acc = d0
    for each following clip i with transition t:
        offset_i = acc - t
        acc      = acc + d_i - t

The reel therefore ends up *shorter* than the sum of its clips by the total of
every transition — which is worth knowing when a script asked for 90 seconds
and got 84.

A transition also cannot be longer than either clip it sits between; ffmpeg
fails outright rather than clamping. Since clips here are often 4-6 seconds and
a user can ask for a 2 second dissolve, every duration is clamped against both
neighbours before it reaches the filtergraph. Clamping quietly is right: the
alternative is failing a twenty-minute render over a tenth of a second.

Beat timing
===========
``duration_for_bars`` turns a music analysis into a transition length, so a
crossfade can be exactly half a bar instead of an arbitrary 0.5 s. That is what
makes a transition feel placed rather than applied.

Delivery size
=============
``width``/``height`` override the canvas. The engine's clips come off the
camera at whatever it shot — 5.3K on a modern GoPro, which makes a two-minute
reel about 1.5 GB. Rendering the reel at 1080p is a delivery decision, not a
quality loss in the source, and it belongs here because this is the only stage
that re-encodes everything anyway.

Public API
==========
    TRANSITIONS                      -> the names this module accepts
    build_reel(clips, output, ...)   -> str
    duration_for_bars(analysis, ...) -> float
    plan_transitions(n, ...)         -> list[Transition]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

from modules.app_paths import ffmpeg_exe

# Names the UI and the script format use, mapped to ffmpeg xfade transitions.
# Deliberately a curated subset: xfade ships dozens, most of which (pixelize,
# squeeze, hlwind) read as a video-editor demo rather than as film grammar.
# "cut" is not an xfade at all — it means no overlap, and is handled before the
# filtergraph is built.
TRANSITIONS: dict[str, str] = {
    "cut": "",
    "crossfade": "fade",
    "dissolve": "dissolve",
    "dip_to_black": "fadeblack",
    "dip_to_white": "fadewhite",
    "wipe_left": "wipeleft",
    "wipe_right": "wiperight",
    "wipe_up": "wipeup",
    "wipe_down": "wipedown",
    "slide_left": "slideleft",
    "slide_right": "slideright",
    "slide_up": "slideup",
    "slide_down": "slidedown",
    "smooth_left": "smoothleft",
    "smooth_right": "smoothright",
    "circle_open": "circleopen",
    "circle_close": "circleclose",
    "radial": "radial",
}

DEFAULT_DURATION = 0.5

# A transition may not eat more than this fraction of either neighbouring clip.
# At 0.5 a pair of 1-second clips could dissolve for a full second and leave
# nothing of either on screen alone; a third keeps a recognisable middle.
MAX_CLIP_FRACTION = 1.0 / 3.0

# Below this the overlap is shorter than a couple of frames and reads as a
# glitch rather than a transition, so it degrades to a clean cut.
MIN_DURATION = 0.08

# One timebase for everything entering the blend. The value hardly matters —
# 90000 is the MP4 convention and divides the common frame rates — but every
# xfade input agreeing on it does: the filter compares them and refuses to
# configure when they differ, which is how a reel of mixed copied and
# re-encoded runs fails with no output at all.
VIDEO_TIMEBASE = "1/90000"
AUDIO_TIMEBASE = "1/48000"


class ReelCancelled(RuntimeError):
    """Raised between steps when cancel_check() says stop. ffmpeg is never
    killed mid-file, so nothing partial escapes the temp directory."""


@dataclass
class Transition:
    """One join between clip ``index`` and clip ``index + 1``."""
    index: int
    kind: str = "crossfade"
    duration: float = DEFAULT_DURATION

    @property
    def is_cut(self) -> bool:
        return self.kind == "cut" or self.duration < MIN_DURATION


def normalise_kind(kind: str) -> str:
    """Accept a transition name, or raise with the list of real ones.

    Raising beats silently falling back to a cut: a script that asks for
    ``wipe_lefy`` and renders twenty minutes of hard cuts gives the user no way
    to find out why.
    """
    key = (kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in TRANSITIONS:
        raise ValueError(
            f"unknown transition {kind!r} — expected one of "
            f"{', '.join(sorted(TRANSITIONS))}")
    return key


def duration_for_bars(analysis, *, bars: float = 0.5,
                      fallback: float = DEFAULT_DURATION) -> float:
    """Transition length as a fraction of a musical bar.

    Half a bar is the useful default: long enough to read as a blend, short
    enough that the incoming clip is fully visible by the next downbeat. Falls
    back when there is no usable tempo, so a track that could not be analysed
    still produces a sane reel.
    """
    try:
        interval = float(getattr(analysis, "beat_interval", 0.0) or 0.0)
        meter = int(getattr(analysis, "meter", 4) or 4)
    except (TypeError, ValueError):
        return fallback
    if interval <= 0 or meter <= 0:
        return fallback
    return max(MIN_DURATION, interval * meter * float(bars))


def plan_transitions(count: int, *, kind: str = "crossfade",
                     duration: float = DEFAULT_DURATION,
                     every: int = 1, other: str = "cut") -> list[Transition]:
    """Transitions for ``count`` clips — that is ``count - 1`` joins.

    ``every`` places the named transition on every Nth join and ``other`` on
    the rest, which is how a reel gets a dip to black at each section change
    without dissolving through every single cut.
    """
    kind = normalise_kind(kind)
    other = normalise_kind(other)
    step = max(1, int(every))
    return [
        Transition(index=i,
                   kind=kind if i % step == 0 else other,
                   duration=duration)
        for i in range(max(0, count - 1))
    ]


def _probe_duration(path: str) -> float:
    from modules.video_probe import probe_video
    return float(probe_video(path)["duration"])


def _clamp(transitions, durations) -> list[Transition]:
    """Shrink any transition that would outrun the clips it joins.

    ffmpeg errors rather than clamping, and the failure arrives after the
    normalise pass has already spent minutes, so this happens up front.
    """
    out: list[Transition] = []
    for t in transitions:
        if t.index + 1 >= len(durations):
            continue
        room = min(durations[t.index], durations[t.index + 1]) * MAX_CLIP_FRACTION
        duration = min(float(t.duration), room)
        kind = t.kind if duration >= MIN_DURATION else "cut"
        out.append(Transition(index=t.index, kind=kind, duration=duration))
    return out


def _runs(transitions, count: int) -> tuple[list[list[int]], list["Transition"]]:
    """Split clip indices into runs joined by hard cuts, plus the blended
    transitions between those runs.

    ``[a -cut- b -crossfade- c -cut- d]`` becomes ``[[a, b], [c, d]]`` and one
    crossfade. The point is that everything inside a run can be joined by the
    concat *demuxer* — outside any filtergraph — leaving xfade to see nothing
    but plain files.
    """
    runs: list[list[int]] = []
    between: list[Transition] = []
    current = [0]
    for i, t in enumerate(transitions):
        if t.is_cut:
            current.append(i + 1)
        else:
            runs.append(current)
            between.append(t)
            current = [i + 1]
    if current:
        runs.append(current)
    return runs, between


def _filtergraph(transitions, durations, fps: int) -> tuple[str, str, float]:
    """The xfade/acrossfade chain over already-joined runs, and the duration.

    Every input here is a whole run, so there are no cuts left to express and
    the chain is uniform. That uniformity is the reason runs are pre-joined:
    the concat *filter* cannot feed xfade — ffmpeg fails with "Could not open
    encoder before EOF" and writes nothing — so a graph that mixed the two
    worked only as long as no reel happened to start with a hard cut.

    Video and audio are chained in step: xfade positions its overlap by an
    absolute offset into the running result, while acrossfade simply joins the
    tail of one to the head of the next, so only the video side accumulates.
    """
    parts: list[str] = []
    # xfade refuses two inputs whose timebases differ — "First input link main
    # timebase (1/15360) do not match ... (1/90000)" — and a run that was
    # copied through keeps a different one from a run that was re-encoded. So
    # every input is pinned to one timebase and frame rate before it can reach
    # a filter that cares.
    for i in range(len(transitions) + 1):
        parts.append(f"[{i}:v]settb={VIDEO_TIMEBASE},fps={fps},"
                     f"format=yuv420p,setsar=1[x{i}]")
        parts.append(f"[{i}:a]aresample=48000,asettb={AUDIO_TIMEBASE}[y{i}]")

    v_label, a_label = "x0", "y0"
    acc = durations[0]

    for i, t in enumerate(transitions):
        nxt = i + 1
        out_v, out_a = f"v{nxt}", f"a{nxt}"
        offset = max(0.0, acc - t.duration)
        parts.append(
            f"[{v_label}][x{nxt}]xfade=transition={TRANSITIONS[t.kind]}"
            f":duration={t.duration:.3f}:offset={offset:.3f}[{out_v}]")
        parts.append(
            f"[{a_label}][y{nxt}]acrossfade=d={t.duration:.3f}"
            f":c1=tri:c2=tri[{out_a}]")
        acc += durations[nxt] - t.duration
        v_label, a_label = out_v, out_a

    parts.append(f"[{v_label}]fps={fps},format=yuv420p[vout]")
    parts.append(f"[{a_label}]aresample=48000[aout]")
    return ";".join(parts), "", acc


def _join_run(paths: list[str], out: str, fps: int) -> str:
    """Join a run of hard-cut clips into one continuous file for xfade.

    Deliberately a re-encode rather than the ``-c copy`` the plain combiner
    uses. A stream-copied concat is several segments in one container with
    restarting timestamps, and xfade cannot read that any better than it can
    read the concat filter: same "Could not open encoder before EOF", same
    empty output. Re-encoding produces one continuous stream with monotonic
    timestamps, which is the thing xfade actually requires.

    The cost is one extra encode of the clips that are joined by cuts. That is
    real, and it is why reels with no transitions at all never reach this code
    — they take the stream-copy path in build_reel instead.
    """
    if len(paths) == 1:
        shutil.copy2(paths[0], out)
        return out
    listing = out + ".txt"
    with open(listing, "w", encoding="utf-8") as fh:
        for path in paths:
            p = os.path.abspath(path).replace("\\", "/").replace("'", "'\\''")
            fh.write(f"file '{p}'\n")
    result = subprocess.run(
        [ffmpeg_exe(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", listing,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
         "-fps_mode", "cfr", "-r", str(fps),
         "-video_track_timescale", "90000",
         "-fflags", "+genpts", "-avoid_negative_ts", "make_zero", out],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=1800)
    if result.returncode != 0 or not os.path.exists(out):
        err = (result.stderr or "").strip()[-400:] or "unknown error"
        raise RuntimeError(f"joining a run of cuts failed: {err}")
    return out


def build_reel(clips, output, *, transitions=None, kind: str = "crossfade",
               duration: float = DEFAULT_DURATION, width: int = 0,
               height: int = 0, fps: int = 0, crf: int = 20,
               preset: str = "medium", music=None, music_optional: bool = False,
               log_fn=print, progress_fn=None, cancel_check=None) -> str:
    """Join ``clips`` into ``output`` with transitions between them.

    ``transitions`` is a list of :class:`Transition` (one per join); when it is
    omitted every join uses ``kind`` and ``duration``. ``width``/``height``
    override the delivery canvas — leave them at 0 to keep the largest input's
    size. ``music`` is the same dict :func:`modules.combine_videos.combine_videos`
    takes.

    ``music_optional`` decides what a failure in the music step costs. Off
    (the default) it fails the whole call, which is what a caller who asked for
    music specifically should get. On, the finished but silent reel is shipped
    with a warning — correct for a long automatic run, where the expensive work
    is everything before this point and throwing it away over an audio filter
    would be absurd.

    When every join is a cut this delegates to ``combine_videos``, whose
    stream-copy concat is both faster and lossless. There is no reason to run
    a filtergraph to produce a result the demuxer already gives away.

    Returns ``output``. Raises ValueError on bad input, ReelCancelled on
    cancellation, RuntimeError when ffmpeg fails.
    """
    valid = [c for c in (clips or []) if c and os.path.exists(c)]
    for c in (clips or []):
        if c and not os.path.exists(c):
            log_fn(f"⚠️ Skipping missing input: {c}")
    if not valid:
        raise ValueError("No valid input files to build a reel from")

    if transitions is None:
        transitions = plan_transitions(len(valid), kind=kind, duration=duration)
    else:
        transitions = [
            Transition(index=t.index, kind=normalise_kind(t.kind),
                       duration=float(t.duration))
            for t in transitions
        ]

    durations = [_probe_duration(c) for c in valid]
    transitions = _clamp(transitions, durations)

    if len(valid) == 1 or all(t.is_cut for t in transitions):
        log_fn("🎬 Every join is a cut — using the stream-copy combiner")
        from modules.combine_videos import CombineCancelled, combine_videos
        try:
            return combine_videos(valid, output, log_fn=log_fn,
                                  progress_fn=progress_fn,
                                  cancel_check=cancel_check, music=music)
        except CombineCancelled as exc:
            raise ReelCancelled(str(exc)) from exc
        except Exception:
            if not (music_optional and music and music.get("path")):
                raise
            # Same bargain as below: keep the reel, lose the music. Retried
            # rather than salvaged because the combiner stages internally and
            # never leaves the silent reel where this function can reach it.
            log_fn("⚠️ Music could not be applied; rebuilding without it")
            return combine_videos(valid, output, log_fn=log_fn,
                                  progress_fn=progress_fn,
                                  cancel_check=cancel_check, music=None)

    # Reuse the combiner's canvas + normalise: the uniformity xfade needs is
    # exactly the uniformity concat needed, and that code already handles
    # rotation baking, pillarboxing and silent-track synthesis.
    from modules.combine_videos import _normalize, _target_canvas

    canvas_w, canvas_h, canvas_fps = _target_canvas(valid, log_fn)
    width = int(width) or canvas_w
    height = int(height) or canvas_h
    fps = int(fps) or canvas_fps
    width, height = max(2, width - width % 2), max(2, height - height % 2)

    named = ", ".join(sorted({t.kind for t in transitions if not t.is_cut}))
    log_fn(f"🎬 Building a reel of {len(valid)} clips at {width}x{height} @ {fps}fps "
           f"({named})")

    temp_dir = tempfile.mkdtemp(prefix="vh_reel_")
    _, ext = os.path.splitext(output)
    staged = os.path.join(temp_dir, f"reel{ext or '.mp4'}")
    output = os.path.abspath(output)
    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)

    try:
        normalized = []
        for i, src in enumerate(valid):
            if cancel_check is not None and cancel_check():
                raise ReelCancelled("cancelled")
            if progress_fn:
                try:
                    progress_fn(i, len(valid) + 1, "Reel",
                                f"normalizing {i + 1}/{len(valid)}")
                except Exception:
                    pass
            log_fn(f"⚙️ Normalizing {i + 1}/{len(valid)}: {os.path.basename(src)}")
            dst = os.path.join(temp_dir, f"n{i:03d}.mp4")
            _normalize(src, dst, width, height, fps, log_fn)
            normalized.append(dst)

        # Re-probe: normalising resamples to a constant frame rate, so the
        # durations shift by a frame or two and the xfade offsets must be built
        # from what the filter will actually receive.
        durations = [_probe_duration(p) for p in normalized]
        transitions = _clamp(transitions, durations)

        if cancel_check is not None and cancel_check():
            raise ReelCancelled("cancelled")

        # Join every run of hard cuts first, so the filtergraph below only ever
        # sees whole runs. See _runs() for why mixing the concat filter into an
        # xfade chain does not work.
        groups, blended = _runs(transitions, len(normalized))
        pieces: list[str] = []
        for i, group in enumerate(groups):
            if len(group) > 1:
                log_fn(f"🔗 Joining {len(group)} clips cut hard together")
            pieces.append(_join_run([normalized[n] for n in group],
                                       os.path.join(temp_dir, f"run{i:03d}.mp4"), fps))

        if progress_fn:
            try:
                progress_fn(len(valid), len(valid) + 1, "Reel", "blending")
            except Exception:
                pass

        run_durations = [_probe_duration(p) for p in pieces]
        blended = _clamp([Transition(index=i, kind=t.kind, duration=t.duration)
                          for i, t in enumerate(blended)], run_durations)
        graph, _, expected = _filtergraph(blended, run_durations, fps)
        lost = sum(t.duration for t in blended)
        log_fn(f"🔗 Blending — {len(blended)} transition(s) across "
               f"{len(pieces)} run(s), {lost:.1f}s absorbed, ~{expected:.1f}s out")

        cmd = [ffmpeg_exe(), "-y", "-v", "error"]
        for path in pieces:
            cmd += ["-i", path]
        cmd += [
            "-filter_complex", graph,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            staged,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=3600)
        if result.returncode != 0 or not os.path.exists(staged) or os.path.getsize(staged) == 0:
            err = (result.stderr or "").strip()[-800:] or "unknown error"
            raise RuntimeError(f"Reel build failed: {err}")

        if music and music.get("path"):
            if cancel_check is not None and cancel_check():
                raise ReelCancelled("cancelled")
            from modules import music_track
            with_music = os.path.join(temp_dir, f"reel_music{ext or '.mp4'}")
            try:
                music_track.apply_music(
                    staged, music["path"], with_music,
                    mode=music.get("mode", "replace"),
                    music_volume=float(music.get("volume", 0.8)),
                    log_fn=log_fn)
                staged = with_music
            except Exception as exc:
                if not music_optional:
                    raise
                log_fn(f"⚠️ Music could not be applied ({exc}); "
                       f"keeping the reel without it")

        # Only now does the user-visible path change: a cancel or a music
        # failure above leaves the previous output untouched rather than
        # replacing it with a music-less stand-in.
        shutil.move(staged, output)
        log_fn(f"✅ Reel saved: {output}")
        return output
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
