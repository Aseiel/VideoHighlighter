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

Masks and softness
==================
Every transition with an edge — wipes, irises, barn doors, blinds, clock
sweeps, film grain — is one expression over a *mask*: a field giving each pixel
the moment it changes hands. That makes a new shape a one-line addition, and it
makes every shape feather-able, which is the difference between a wipe with a
hard stair-stepped edge and one that looks like it was made on purpose. See
:data:`MASKS` and :func:`mask_expression`.

Public API
==========
    TRANSITIONS / CURATED / FAMILIES -> the names this module accepts
    MASKS / EASINGS                  -> the shapes and the curves
    mask_expression(kind, ...)       -> str
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
    # The rest of what this ffmpeg's xfade offers. Kept out of the curated set
    # above — which is what the UI shows first — but available by name, because
    # "no reason to expose it" and "no reason to forbid it" are different.
    "circle_crop": "circlecrop",
    "rect_crop": "rectcrop",
    "distance": "distance",
    "vert_open": "vertopen",
    "vert_close": "vertclose",
    "horz_open": "horzopen",
    "horz_close": "horzclose",
    "pixelize": "pixelize",
    "diag_tl": "diagtl",
    "diag_tr": "diagtr",
    "diag_bl": "diagbl",
    "diag_br": "diagbr",
    "hl_slice": "hlslice",
    "hr_slice": "hrslice",
    "vu_slice": "vuslice",
    "vd_slice": "vdslice",
    "blur": "hblur",
    "fade_grays": "fadegrays",
    "wipe_tl": "wipetl",
    "wipe_tr": "wipetr",
    "wipe_bl": "wipebl",
    "wipe_br": "wipebr",
    "squeeze_h": "squeezeh",
    "squeeze_v": "squeezev",
    "zoom_in": "zoomin",
    "fade_fast": "fadefast",
    "fade_slow": "fadeslow",
    "smooth_up": "smoothup",
    "smooth_down": "smoothdown",
    "wind_left": "hlwind",
    "wind_right": "hrwind",
    "wind_up": "vuwind",
    "wind_down": "vdwind",
    "cover_left": "coverleft",
    "cover_right": "coverright",
    "cover_up": "coverup",
    "cover_down": "coverdown",
    "reveal_left": "revealleft",
    "reveal_right": "revealright",
    "reveal_up": "revealup",
    "reveal_down": "revealdown",
}

# Masks
# =====
# A mask is the shape of the join, written as an *arrival time*: for every
# pixel, the fraction of the transition at which that pixel stops showing the
# outgoing clip and starts showing the incoming one. A left-to-right wipe is
# ``X/W`` — the left edge arrives at 0, the right edge at 1. A circular iris is
# the distance from the centre. Everything below is that one idea with a
# different field.
#
# Writing it this way buys two things a list of hard-coded transitions cannot.
# Any new shape is one line, and — because the field is a number rather than a
# branch — every shape can be *feathered*: instead of a pixel flipping when
# progress passes its arrival time, it ramps over a band of them, which is the
# difference between a wipe with a jagged stair-stepped edge and one with a
# soft edge. See :func:`mask_expression`.
#
# The fields are written in X, Y, W and H because that is all an xfade custom
# expression can see, and they are *inlined* rather than assigned to a slot
# with ``st()``. That is not a style choice. ffmpeg evaluates this expression
# on several slices of the frame at once and the ``st()``/``ld()`` registers
# are shared between those threads, so a field stored in slot 0 by one row is
# read back by another mid-write: measured, an st()/ld() version of the soft
# wipe below produced 0.48, 0.07, 0.11, 0.00, 0.72 … across a frame that should
# have ramped smoothly from 1 to 0. The inlined form measures 1.00, 0.73, 0.49,
# 0.24, 0.00. Sub-expressions therefore appear more than once here on purpose.
_RADIUS = "(hypot(X/W-0.5,Y/H-0.5)/0.70711)"
# Clockwise from twelve. The +2*PI/mod pair turns atan2's -PI..PI into 0..2*PI
# without naming the angle twice, which at two trig calls per pixel is worth
# the trick.
_ANGLE = "(mod(atan2(X/W-0.5,0.5-Y/H)+2*PI,2*PI)/(2*PI))"
# A hash, not a random: an expression is evaluated per pixel with no memory, so
# the grain has to be a repeatable function of position or it would crawl.
_HASH = "(mod(abs(sin(X*12.9898+Y*78.233))*43758.5453,1))"

MASKS: dict[str, str] = {
    # Straight edges.
    "wipe_left": "(1-X/W)",
    "wipe_right": "(X/W)",
    "wipe_up": "(1-Y/H)",
    "wipe_down": "(Y/H)",
    "wipe_tl": "((X/W+Y/H)/2)",
    "wipe_tr": "((1-X/W+Y/H)/2)",
    "wipe_bl": "((X/W+1-Y/H)/2)",
    "wipe_br": "((2-X/W-Y/H)/2)",
    # Shapes opening from, or closing onto, the middle of the frame.
    "circle_open": _RADIUS,
    "circle_close": f"(1-{_RADIUS})",
    "iris_open": _RADIUS,
    "iris_close": f"(1-{_RADIUS})",
    "diamond_open": "(abs(X/W-0.5)+abs(Y/H-0.5))",
    "diamond_close": "(1-abs(X/W-0.5)-abs(Y/H-0.5))",
    "box_open": "(max(abs(X/W-0.5),abs(Y/H-0.5))*2)",
    "box_close": "(1-max(abs(X/W-0.5),abs(Y/H-0.5))*2)",
    # Barn doors: one axis only, so the incoming shot arrives as a widening
    # band rather than as a shape.
    "barn_open": "(abs(X/W-0.5)*2)",
    "barn_close": "(1-abs(X/W-0.5)*2)",
    "barn_up": "(abs(Y/H-0.5)*2)",
    "barn_down": "(1-abs(Y/H-0.5)*2)",
    # A hand sweeping round the dial.
    "clock": _ANGLE,
    "clock_back": f"(1-{_ANGLE})",
    # Bands that all travel together. The count is baked into the name rather
    # than exposed as a parameter: six and fourteen are two different looks,
    # and every value between them is the same look slightly off.
    "blinds": "(mod(Y*6/H,1))",
    "blinds_fine": "(mod(Y*14/H,1))",
    "blinds_v": "(mod(X*6/W,1))",
    "blinds_v_fine": "(mod(X*14/W,1))",
    # Alternating squares, leaning left-to-right so it reads as a direction
    # rather than as a flicker.
    "checker": "((mod(floor(X*8/W)+floor(Y*8/H),2)+X/W)/2)",
    # Every pixel on its own schedule — the film-lab dissolve, and the one
    # shape that does not read as a graphic.
    "grain": _HASH,
    # A dissolve that starts in the middle and spreads.
    "grain_iris": f"(({_RADIUS}+{_HASH})/2)",
    # An iris with a wave in its edge.
    "ripple": f"(clip({_RADIUS}+0.09*sin({_RADIUS}*28),0,1))",
    "spiral": f"(clip({_RADIUS}*0.55+{_ANGLE}*0.45,0,1))",
}

# Masks with no xfade equivalent, which therefore always render as a custom
# expression — including at linear easing with no feather, where a mask that
# *does* have a built-in would hand the job back to ffmpeg's own.
MASK_ONLY: frozenset = frozenset(MASKS) - frozenset(TRANSITIONS)

for _name in sorted(MASK_ONLY):
    # Reachable by name like any other transition. The empty string marks
    # "there is no built-in for this" — see _filtergraph.
    TRANSITIONS[_name] = ""

# The two that are a straight mix of the whole frame rather than a shape, and
# so have no mask and cannot be feathered: there is no edge to soften.
BLENDS: frozenset = frozenset({"crossfade", "dissolve"})

# How wide the soft edge is, as a fraction of the transition. 0 is a hard edge
# — every pixel flips the instant progress passes it — which is what xfade does
# and what makes an unfeathered wipe look like a slide show. 0.25 is a good
# general default when softness is wanted.
DEFAULT_FEATHER = 0.0
MAX_FEATHER = 1.0

# Below this the band is thinner than a pixel's worth of progress and the
# expression divides by something near zero, so it degrades to the hard edge it
# is indistinguishable from anyway.
MIN_FEATHER = 0.01

# The ones worth offering first. The rest are reachable by name but a list of
# eighty is a menu nobody reads, and a good few of them (pixelize, squeeze,
# hlwind) read as a video-editor demo rather than as film grammar.
CURATED = (
    "cut", "crossfade", "dissolve", "dip_to_black", "dip_to_white",
    "wipe_left", "wipe_right", "wipe_up", "wipe_down",
    "slide_left", "slide_right", "slide_up", "slide_down",
    "cover_left", "cover_up", "reveal_left", "reveal_up",
    "smooth_left", "smooth_right", "circle_open", "circle_close",
    "zoom_in", "blur", "radial",
    # Shapes, which only exist as masks.
    "iris_open", "iris_close", "diamond_open", "box_open",
    "barn_open", "barn_close", "clock",
    "blinds", "blinds_v", "checker", "grain", "grain_iris", "ripple",
)

# Grouped for a menu, so a picker can show "shapes" apart from "wipes" instead
# of one alphabetical wall. Every name here is a key of TRANSITIONS; anything
# not listed is still reachable by typing it.
FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Cut", ("cut",)),
    ("Fades", ("crossfade", "dissolve", "dip_to_black", "dip_to_white",
               "fade_grays", "fade_fast", "fade_slow", "blur")),
    ("Wipes", ("wipe_left", "wipe_right", "wipe_up", "wipe_down",
               "wipe_tl", "wipe_tr", "wipe_bl", "wipe_br")),
    ("Shapes", ("iris_open", "iris_close", "circle_open", "circle_close",
                "diamond_open", "diamond_close", "box_open", "box_close",
                "clock", "clock_back", "ripple", "spiral", "radial")),
    ("Bands", ("barn_open", "barn_close", "barn_up", "barn_down",
               "blinds", "blinds_fine", "blinds_v", "blinds_v_fine",
               "checker")),
    ("Grain", ("grain", "grain_iris")),
    ("Slides", ("slide_left", "slide_right", "slide_up", "slide_down",
                "cover_left", "cover_right", "cover_up", "cover_down",
                "reveal_left", "reveal_right", "reveal_up", "reveal_down",
                "smooth_left", "smooth_right", "smooth_up", "smooth_down",
                "zoom_in", "squeeze_h", "squeeze_v")),
)

# Easing curves, written in terms of T: elapsed fraction of the transition,
# 0 at the start and 1 at the end.
#
# This is the difference between a transition that looks applied and one that
# looks designed. xfade moves linearly: the blend starts and stops at the same
# rate it runs at, so the eye catches both ends. Every motion system in use —
# CSS, iOS, Material — eases instead, and for the same reason.
#
# T is a placeholder, substituted at build time, because xfade's own progress
# variable P runs the other way — from 1 down to 0. Writing these in P directly
# is the obvious thing to do and produces transitions that play backwards: the
# first version of this ran blue-to-red on a red-to-blue cut, and looked
# entirely plausible until the pixels were measured.
EASINGS: dict[str, str] = {
    "linear": "T",
    # Slow start, full speed out — the incoming shot arrives with intent.
    "ease_in": "(T*T*T)",
    # Fast start, gentle landing. The safest general-purpose choice.
    "ease_out": "(1-pow(1-T,3))",
    # Slow at both ends. Reads as deliberate.
    "ease_in_out": "(if(lt(T,0.5), 4*T*T*T, 1-pow(-2*T+2,3)/2))",
    # Smoothstep — gentler than cubic, closest to a hand-drawn fade.
    "smooth": "(T*T*(3-2*T))",
    # Most of the move happens immediately, then it settles. Good on fast cuts
    # where a symmetric ease would waste the little time there is.
    "snap": "pow(T,0.45)",
}

# What T becomes in a real expression. See the note above.
PROGRESS = "(1-P)"

DEFAULT_EASING = "linear"

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
    easing: str = DEFAULT_EASING
    # How wide the mask's soft edge is, as a fraction of the transition. Only
    # means anything for a transition that has an edge — see MASKS.
    feather: float = DEFAULT_FEATHER

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


def normalise_easing(easing: str) -> str:
    """Accept an easing name, or raise with the real ones."""
    key = (easing or "linear").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in EASINGS:
        raise ValueError(f"unknown easing {easing!r} — expected one of "
                         f"{', '.join(sorted(EASINGS))}")
    return key


def normalise_feather(feather) -> float:
    """A softness in 0..1, or raise. Anything below :data:`MIN_FEATHER` comes
    back as 0, which is the hard edge it would be indistinguishable from."""
    try:
        value = float(feather or 0.0)
    except (TypeError, ValueError):
        raise ValueError(f"feather must be a number between 0 and 1, got "
                         f"{feather!r}") from None
    if value != value or value < 0 or value > MAX_FEATHER:
        raise ValueError(f"feather must be between 0 and {MAX_FEATHER:g}, got "
                         f"{feather!r}")
    return 0.0 if value < MIN_FEATHER else value


def mask_expression(kind: str, easing: str = DEFAULT_EASING,
                    feather: float = DEFAULT_FEATHER) -> str:
    """An xfade ``custom`` expression for ``kind``, or "" to use the built-in.

    xfade's custom mode gives an expression the progress ``P`` and the two
    source pixels ``A`` and ``B`` at the current coordinate — and nothing at
    any other coordinate. So a fade, which mixes the two pixels in front of it,
    and any mask, which only asks whether this pixel's turn has come, can both
    be written; a slide, which needs the pixel a hundred columns over, cannot.
    Those keep their built-in linear form rather than being faked.

    Three things can put a transition on this path: a mask with no built-in at
    all, an easing curve, or a feather. Otherwise "" is returned and ffmpeg's
    own implementation does the work, which is both faster and exactly what was
    asked for.

    The feathered form is the interesting one::

        A + (B-A) * clip((progress*(1+f) - arrival) / f, 0, 1)

    A pixel whose arrival time is well past the progress gets 0 and stays on
    A; one well behind gets 1 and is fully B; the band of width ``f`` between
    them ramps. Scaling progress by ``(1+f)`` is what guarantees the frame is
    fully handed over at the end rather than a feather's width short of it.
    """
    kind = normalise_kind(kind)
    easing = normalise_easing(easing)
    feather = normalise_feather(feather)
    progress = EASINGS[easing].replace("T", PROGRESS)

    if kind == "cut":
        return ""
    if kind in BLENDS:
        # No edge to soften, so feather is not applicable — only easing can
        # take a fade off the built-in path.
        return "" if easing == "linear" else f"A+(B-A)*{progress}"

    field = MASKS.get(kind)
    if not field:
        # A built-in with no mask (slides, zooms, winds). Easing cannot be
        # applied to these — see the docstring — so they keep their own form
        # rather than silently becoming something else.
        return ""
    if easing == "linear" and not feather and kind not in MASK_ONLY:
        return ""

    if not feather:
        return f"if(gte({progress},{field}),B,A)"
    return (f"A+(B-A)*clip(({progress}*{1.0 + feather:.4f}-{field})"
            f"/{feather:.4f},0,1)")


# The name this had before masks existed. Kept because a caller asking for an
# eased expression is asking for exactly this.
eased_expression = mask_expression


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
                     every: int = 1, other: str = "cut",
                     easing: str = DEFAULT_EASING,
                     feather: float = DEFAULT_FEATHER) -> list[Transition]:
    """Transitions for ``count`` clips — that is ``count - 1`` joins.

    ``every`` places the named transition on every Nth join and ``other`` on
    the rest, which is how a reel gets a dip to black at each section change
    without dissolving through every single cut.
    """
    kind = normalise_kind(kind)
    other = normalise_kind(other)
    feather = normalise_feather(feather)
    step = max(1, int(every))
    return [
        Transition(index=i,
                   kind=kind if i % step == 0 else other,
                   duration=duration, easing=easing, feather=feather)
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
        out.append(Transition(index=t.index, kind=kind, duration=duration,
                              easing=getattr(t, "easing", DEFAULT_EASING),
                              feather=getattr(t, "feather", DEFAULT_FEATHER)))
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
        expr = mask_expression(t.kind, getattr(t, "easing", DEFAULT_EASING),
                               getattr(t, "feather", DEFAULT_FEATHER))
        if expr:
            # ' is the filtergraph's own quote, so the expression is wrapped in
            # it and must not contain one; no easing or mask does.
            spec = f"transition=custom:expr='{expr}'"
        else:
            # A mask-only name should never reach here without an expression,
            # but a built-in that is missing from this ffmpeg would fail deep
            # inside the encode rather than here, so it falls back instead.
            spec = f"transition={TRANSITIONS[t.kind] or 'fade'}"
        parts.append(
            f"[{v_label}][x{nxt}]xfade={spec}"
            f":duration={t.duration:.3f}:offset={offset:.3f}[{out_v}]")
        parts.append(
            f"[{a_label}][y{nxt}]acrossfade=d={t.duration:.3f}"
            f":c1=tri:c2=tri[{out_a}]")
        acc += durations[nxt] - t.duration
        v_label, a_label = out_v, out_a

    parts.append(f"[{v_label}]fps={fps},format=yuv420p[vout]")
    parts.append(f"[{a_label}]aresample=48000[aout]")
    return ";".join(parts), "", acc


def _normalize_filled(src: str, dst: str, width: int, height: int, fps: int,
                      log_fn=print) -> None:
    """Normalise like :func:`modules.combine_videos._normalize`, but *fill* the
    canvas by cropping instead of padding to fit.

    Both are right for different jobs. Padding preserves the whole frame and is
    correct for a film, where losing the edges of a composed shot is worse than
    a black bar. Filling is correct for a vertical Reel, where 16:9 footage
    padded into 9:16 is a small strip in the middle of a mostly-black screen —
    which no one watches. The sides are lost either way; this loses them to the
    crop rather than to the letterbox.
    """
    from modules.combine_videos import _has_audio

    with_silence = not _has_audio(src, log_fn)
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},setsar=1,fps={fps},setpts=N/FRAME_RATE/TB")
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", src]
    if with_silence:
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000"]
    cmd += [
        "-map", "0:v:0", "-map", "1:a:0" if with_silence else "0:a:0",
        "-vf", vf, "-af", "aresample=48000,asetpts=N/SR/TB",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-g", str(fps * 2), "-keyint_min", str(fps), "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-fps_mode", "cfr", "-max_muxing_queue_size", "1024",
        "-fflags", "+genpts", "-avoid_negative_ts", "make_zero",
    ]
    if with_silence:
        cmd += ["-shortest"]
    cmd += [dst]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=900)
    if result.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        err = (result.stderr or "").strip()[-500:] or "unknown error"
        raise RuntimeError(f"Normalization failed for {os.path.basename(src)}: {err}")


def _font_path() -> str:
    """A bold sans on this machine, or "" — the raw path.

    Kept separate from the escaped form because the fitting below has to hand
    it to a font library, which wants the path as the filesystem spells it.
    """
    for path in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.exists(path):
            return path
    return ""


def _escape_path(path: str) -> str:
    """A path drawtext will accept as an option value. On Windows it contains
    a drive colon, which ffmpeg's parser reads as the end of the value unless
    it is backslash-escaped — and the resulting error names the font rather
    than the syntax."""
    return path.replace("\\", "/").replace(":", r"\:")


# Share of the frame width a caption may occupy. Short-form is watched on a
# phone held at arm's length; text running to the very edge reads as broken
# even when every glyph is on screen.
TEXT_WIDTH = 0.86

# Lines a caption may wrap to before it is shrunk instead. Four lines of a
# hook is already more reading than anybody does in two seconds.
TEXT_LINES = 3


def fit_caption(text: str, width: int, height: int,
                font_path: str = "") -> tuple[list, int]:
    """Wrap ``text`` to the frame and pick a size for it: (lines, points).

    drawtext has no notion of a text box. It draws one line at whatever size
    it is given and centres it, so a caption wider than the frame simply hangs
    off both sides — which is what "21 clips. One morning." did at 105 points
    on a 1080-wide reel, losing a word at each end of every render.

    Measured with a real font rather than by counting characters: proportional
    glyphs vary by a factor of four between an 'i' and a 'W', so a character
    count either wraps far too early on narrow text or not at all on wide.
    Falls back to a conservative estimate when no font library is available,
    because a caption that wraps a little early is a great deal better than one
    that runs off the screen.
    """
    words = " ".join((text or "").split())
    if not words:
        return [], 0

    limit = max(1.0, width * TEXT_WIDTH)
    size = max(18, int(height * 0.055))
    measure = _measurer(font_path)

    while size >= 16:
        lines = _wrap(words, limit, size, measure)
        if len(lines) <= TEXT_LINES and all(
                measure(line, size) <= limit for line in lines):
            return lines, size
        size = int(size * 0.92)
    return _wrap(words, limit, 16, measure), 16


def _measurer(font_path: str):
    """A function giving the rendered width of a string at a size, in pixels."""
    try:
        from PIL import ImageFont

        cache: dict = {}

        def measure(line: str, size: int) -> float:
            font = cache.get(size)
            if font is None:
                font = ImageFont.truetype(font_path, size)
                cache[size] = font
            return float(font.getlength(line))

        if font_path:
            measure("M", 20)     # fail here rather than mid-wrap
            return measure
    except Exception:
        pass
    # No font library, or a font it will not open. 0.58 em per character is a
    # deliberate over-estimate for a bold sans, so this wraps early rather
    # than late.
    return lambda line, size: len(line) * size * 0.58


def _wrap(words: str, limit: float, size: int, measure) -> list:
    """Greedy word wrap. A single word too wide for the frame is left on its
    own line, where the caller's shrink loop deals with it."""
    lines: list[str] = []
    current = ""
    for word in words.split(" "):
        candidate = f"{current} {word}".strip()
        if current and measure(candidate, size) > limit:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _font_file() -> str:
    """A font drawtext can load, or "" when none is found.

    Returned already escaped for a filter argument: on Windows the path
    contains a drive colon, which ffmpeg's parser reads as the end of an
    option value unless it is backslash-escaped, and the resulting error names
    the font rather than the syntax.
    """
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path.replace("\\", "/").replace(":", r"\:")
    return ""


def _escape_text(text: str) -> str:
    """Escape a caption for drawtext's own mini-language.

    Backslash first or it would double-escape what the later rules add. The
    characters that matter are the ones that end an option (``:``), separate
    filters (``,``) or quote a value — an apostrophe in "I didn't" is enough to
    make ffmpeg fail with a syntax error about something else entirely.
    """
    out = text.replace("\\", r"\\\\")
    for ch in (":", ",", "'", "%", "[", "]", ";"):
        out = out.replace(ch, "\\" + ch)
    # Newlines survive: they are how a wrapped caption gets its line breaks,
    # and drawtext reads a literal one in the value as exactly that.
    # Collapsing them to spaces — which this used to do — undoes the wrapping
    # and puts the caption back off the side of the frame.
    return out


def burn_text(src: str, dst: str, text: str, *, height: int, width: int = 0,
              position: str = "lower", log_fn=print) -> str:
    """Burn ``text`` into ``src``, or copy it through when that is not possible.

    Degrading to a copy is deliberate. A caption is worth a lot on a reel and
    is still not worth failing a finished render for: no font on the machine,
    or a string drawtext will not take, should cost the words rather than the
    film.

    The caption is wrapped and sized to fit ``width`` — see :func:`fit_caption`
    for why that cannot be left to drawtext, which has no text box and will
    happily draw a line twice as wide as the frame.
    """
    raw_font = _font_path()
    if not text.strip() or not raw_font:
        if not text.strip():
            shutil.copy2(src, dst)
            return dst
        log_fn("⚠️ No usable font found; on-screen text skipped")
        shutil.copy2(src, dst)
        return dst

    font = _escape_path(raw_font)
    if not width:
        # Nothing was said about the frame, so assume the narrow case rather
        # than the wide one: guessing 16:9 on a vertical reel is how the text
        # ran off the screen in the first place.
        width = int(height * 9 / 16)
    lines, size = fit_caption(text, width, height, raw_font)
    if not lines:
        shutil.copy2(src, dst)
        return dst
    if len(lines) > 1:
        log_fn(f"🔤 Caption wrapped to {len(lines)} lines at {size}px to fit "
               f"{width}px")

    pad = max(12, int(height * 0.03))
    y = f"h-th-{pad * 3}" if position == "lower" else str(pad * 2)
    drawn = "\n".join(lines)
    graph = (
        f"drawtext=fontfile='{font}':text='{_escape_text(drawn)}'"
        f":fontcolor=white:fontsize={size}"
        f":x=(w-tw)/2:y={y}"
        f":box=1:boxcolor=black@0.55:boxborderw={max(6, size // 4)}"
        f":line_spacing={max(4, size // 6)}"
    )
    result = subprocess.run(
        [ffmpeg_exe(), "-y", "-v", "error", "-i", src, "-vf", graph,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-c:a", "copy", dst],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900)
    if result.returncode != 0 or not os.path.exists(dst):
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        log_fn(f"⚠️ Could not draw text ({tail[0]}); the clip keeps its picture")
        shutil.copy2(src, dst)
    return dst


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
               texts=None, fill: str = "pad", easing: str = DEFAULT_EASING,
               feather: float = DEFAULT_FEATHER, motion: str = "none",
               motions=None,
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
        transitions = plan_transitions(len(valid), kind=kind, duration=duration,
                                       easing=easing, feather=feather)
    else:
        transitions = [
            Transition(index=t.index, kind=normalise_kind(t.kind),
                       duration=float(t.duration),
                       easing=normalise_easing(getattr(t, "easing", easing)),
                       feather=normalise_feather(
                           getattr(t, "feather", feather)))
            for t in transitions
        ]

    durations = [_probe_duration(c) for c in valid]
    transitions = _clamp(transitions, durations)

    # The stream-copy combiner is faster and lossless, but it decides its own
    # canvas, always pads, and knows nothing about captions. Delegating to it
    # when any of those were asked for silently returns something else — which
    # is exactly what a vertical reel of hard cuts is: every join is a cut, so
    # the shortcut fired and the 1080x1920 crop and the on-screen text both
    # vanished from a render that otherwise looked fine.
    plain = (not int(width) and not int(height) and not int(fps)
             and fill != "crop" and not any((texts or {}).values()))
    if plain and (len(valid) == 1 or all(t.is_cut for t in transitions)):
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
    softest = max((t.feather for t in transitions if not t.is_cut), default=0.0)
    log_fn(f"🎬 Building a reel of {len(valid)} clips at {width}x{height} @ {fps}fps "
           f"({named}{f', soft edge {softest:.0%}' if softest else ''})")

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
            if fill == "crop":
                _normalize_filled(src, dst, width, height, fps, log_fn)
            else:
                _normalize(src, dst, width, height, fps, log_fn)
            # A move on the ends of the clip, which is the half of a
            # transition a mask cannot express — see modules.motion.
            wanted = (motions or {}).get(i, motion) if motions else motion
            if wanted and wanted != "none":
                from modules.motion import apply_motion
                moved = os.path.join(temp_dir, f"m{i:03d}.mp4")
                dst = apply_motion(
                    dst, moved, wanted, duration=_probe_duration(dst),
                    width=width, height=height, fps=fps,
                    head=i > 0, tail=i < len(valid) - 1, log_fn=log_fn)

            caption = (texts or {}).get(i, "") if texts else ""
            if caption.strip():
                # After normalising, so the text is drawn at delivery size and
                # scales with it rather than being resized along with the
                # picture.
                lettered = os.path.join(temp_dir, f"t{i:03d}.mp4")
                log_fn(f"🔤 Text on clip {i + 1}: {caption[:48]}")
                dst = burn_text(dst, lettered, caption, height=height,
                                width=width, log_fn=log_fn)
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
        blended = _clamp([Transition(index=i, kind=t.kind, duration=t.duration,
                                     easing=getattr(t, "easing", easing),
                                     feather=getattr(t, "feather", feather))
                          for i, t in enumerate(blended)], run_durations)

        if not blended:
            # Every join was a cut, and we are only here because the delivery
            # size, the fill or a caption ruled out the stream-copy shortcut.
            # The single run those clips joined into is already the reel; a
            # filtergraph with nothing to blend would only re-encode it again.
            log_fn(f"🔗 {len(valid)} clips, all cuts — no blending needed")
            shutil.move(pieces[0], staged)
        else:
            lost = sum(t.duration for t in blended)
            log_fn(f"🔗 Blending — {len(blended)} transition(s) across "
                   f"{len(pieces)} run(s), {lost:.1f}s absorbed, ~{sum(run_durations) - lost:.1f}s out")
            graph, _, _ = _filtergraph(blended, run_durations, fps)
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
                                    encoding="utf-8", errors="replace",
                                    timeout=3600)
            if result.returncode != 0:
                err = (result.stderr or "").strip()[-800:] or "unknown error"
                raise RuntimeError(f"Reel build failed: {err}")

        if not os.path.exists(staged) or os.path.getsize(staged) == 0:
            raise RuntimeError("Reel build produced no output")

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
