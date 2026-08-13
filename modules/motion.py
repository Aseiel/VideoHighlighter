"""
motion.py — make the cut itself move, rather than dissolving one shape into
another.

Why this exists
===============
:mod:`modules.transitions` can draw eighty different joins and they all feel
the same, which is not a failure of imagination — it is what the mechanism is.
Every one of them is a *mask*: a shape deciding which of two frozen frames each
pixel shows. An iris and a wipe and blinds are the same event wearing different
clothes, because in all three the two clips sit perfectly still while a shape
moves over them.

What makes short-form video feel cut rather than assembled is the opposite: the
picture moves. A shot punches in on the last few frames before the cut; the
next one lands already large and settles; the frame kicks sideways on a beat.
None of that is expressible as a mask, so none of it was reachable.

So this is the other half. A motion is applied to the *ends of a clip* rather
than to the join, which means it composes with everything the mask system
already does — a soft iris with a punch on either side of it is both.

What ffmpeg will and will not do
================================
Measured rather than assumed, because the obvious approach does not work:

- ``crop`` with a time-varying **size** fails outright. Its ``w`` and ``h`` are
  evaluated once when the filter is configured, so ``crop=w='iw/(1+t)'`` errors
  with "Failed to configure input pad" rather than zooming.
- ``zoompan`` does vary size over time, and its ramp is linear in ``it`` — a
  0.4 zoom over two seconds measures 1.02, 1.10, 1.20 and 1.32 at quarter
  points against the 1.02, 1.12, 1.24, 1.38 asked for, the shortfall being the
  measurement losing antialiased edges rather than the filter.
- ``crop``'s **x** and **y** *are* evaluated per frame, so a shake is an
  over-scaled picture with a wandering crop origin.
- ``rotate`` takes an expression in ``t``.
- ``rgbashift`` cannot ramp, but it takes ``enable=``, which is all a glitch
  wants — a glitch that fades in is not a glitch.

Every motion below is built from those four and nothing else.

Public API
==========
    MOTIONS / MOTION_LABELS
    normalise_motion(name) -> str
    motion_filter(name, ...) -> str
    apply_motion(src, dst, name, ...) -> str
"""

from __future__ import annotations

import os
import shutil
import subprocess

from modules.app_paths import ffmpeg_exe

# How long the move takes, in seconds at each end of a clip. Short: this is a
# punctuation mark, not a shot. Beyond about a third of a second it stops
# reading as an accent on the cut and starts reading as a camera move nobody
# made.
WINDOW = 0.22

# How far each motion travels at full strength. Tuned to be felt rather than
# watched — a 60% zoom is a special effect, an 18% one is an edit.
ZOOM = 0.18
SHAKE = 0.035        # share of the frame the picture wanders
ROLL = 0.075         # radians, about four degrees

# The picture is scaled up before anything that moves it, so a shake or a roll
# never exposes the edge of the frame. Just enough to cover the largest travel
# above.
OVERSCAN = 1.14

MOTIONS: tuple = ("none", "punch", "pull", "shake", "roll", "glitch")

MOTION_LABELS: dict = {
    "none": "Still",
    "punch": "Punch in",
    "pull": "Pull back",
    "shake": "Shake",
    "roll": "Roll",
    "glitch": "Glitch",
}


def normalise_motion(name: str) -> str:
    """Accept a motion name, or raise with the real ones."""
    key = (name or "none").strip().lower().replace("-", "_").replace(" ", "_")
    if key not in MOTIONS:
        raise ValueError(f"unknown motion {name!r} — expected one of "
                         f"{', '.join(MOTIONS)}")
    return key


def _ramps(duration: float, window: float, head: bool, tail: bool) -> str:
    """An expression, 0 to 1, that rises at whichever ends were asked for.

    ``it`` is the input frame's timestamp in seconds, which is the one clock
    available to every filter used here and does not depend on the frame rate.
    """
    window = max(0.01, min(window, duration / 2.0 if duration else window))
    parts = []
    if head:
        # 1 at the very start, falling to 0 once the window has passed.
        parts.append(f"max(0,1-{{t}}/{window:.3f})")
    if tail and duration > 0:
        start = max(0.0, duration - window)
        parts.append(f"max(0,({{t}}-{start:.3f})/{window:.3f})")
    if not parts:
        return "0"
    return "(" + "+".join(parts) + ")"


def motion_filter(name: str, *, duration: float, width: int, height: int,
                  fps: float = 30.0, head: bool = False, tail: bool = False,
                  window: float = WINDOW, strength: float = 1.0) -> str:
    """The filter chain for ``name``, or "" when there is nothing to do.

    ``head`` and ``tail`` say which ends of the clip the motion happens at —
    normally the ones with a join next to them, so a clip in the middle of a
    reel moves at both and the first and last move only on the inside.
    """
    name = normalise_motion(name)
    if name == "none" or not (head or tail) or strength <= 0:
        return ""

    # zoompan reads `it`; crop and rotate read `t`. Same clock, different name.
    zoom_ramp = _ramps(duration, window, head, tail).format(t="it")
    frame_ramp = _ramps(duration, window, head, tail).format(t="t")
    width, height = max(2, int(width)), max(2, int(height))

    if name in ("punch", "pull"):
        # Punch: largest at the join and settling away from it, so the cut
        # lands on the biggest frame. Pull is the same move backwards.
        amount = ZOOM * strength
        expression = (f"1+{amount:.4f}*{zoom_ramp}" if name == "punch"
                      else f"{1 + amount:.4f}-{amount:.4f}*{zoom_ramp}")
        # zoompan sets its own output rate and defaults to 25, which would
        # quietly change the frame rate of every clip it touched; it is given
        # the clip's own instead. And nothing may follow it in this chain — an
        # fps filter appended after zoompan stops its expression advancing at
        # all, which measured as a punch that never zoomed.
        return (f"zoompan=z='{expression}':d=1:s={width}x{height}"
                f":fps={float(fps):.6g}"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'")

    if name in ("shake", "glitch"):
        travel = SHAKE * strength * min(width, height)
        # Two different frequencies so the wander does not read as a circle.
        chain = (
            f"scale=ceil(iw*{OVERSCAN}/2)*2:ceil(ih*{OVERSCAN}/2)*2,"
            f"crop={width}:{height}"
            f":x='(iw-{width})/2+{travel:.2f}*sin(46*t)*{frame_ramp}'"
            f":y='(ih-{height})/2+{travel * 0.8:.2f}*cos(38*t)*{frame_ramp}'"
        )
        if name == "glitch":
            # No ramp: rgbashift cannot take one, and a glitch that fades in
            # is not a glitch. It is simply on for the window.
            shift = max(2, int(0.006 * width * strength))
            when = _glitch_window(duration, window, head, tail)
            chain += (f",rgbashift=rh={shift}:bh=-{shift}:gv={shift // 2}"
                      f":enable='{when}'")
        return chain

    if name == "roll":
        angle = ROLL * strength
        return (
            f"scale=ceil(iw*{OVERSCAN}/2)*2:ceil(ih*{OVERSCAN}/2)*2,"
            f"rotate='{angle:.4f}*{frame_ramp}':c=black@0:ow=iw:oh=ih,"
            f"crop={width}:{height}"
        )
    return ""


def _glitch_window(duration: float, window: float, head: bool, tail: bool) -> str:
    """When the colour split is switched on — an ``enable`` condition."""
    spans = []
    if head:
        spans.append(f"lt(t,{window:.3f})")
    if tail and duration > 0:
        spans.append(f"gt(t,{max(0.0, duration - window):.3f})")
    return "+".join(spans) if spans else "0"


def apply_motion(src: str, dst: str, name: str, *, duration: float,
                 width: int, height: int, fps: float = 30.0,
                 head: bool = False, tail: bool = False, window: float = WINDOW,
                 strength: float = 1.0, crf: int = 18, log_fn=print) -> str:
    """Re-encode ``src`` with the motion applied, or copy it through.

    Copying through on failure is the same bargain the rest of the render
    makes: a motion is worth a lot and is not worth failing a finished reel
    for. The audio is copied rather than touched, since nothing here is
    audible.
    """
    chain = motion_filter(name, duration=duration, width=width, height=height,
                          fps=fps, head=head, tail=tail, window=window,
                          strength=strength)
    if not chain:
        shutil.copy2(src, dst)
        return dst

    result = subprocess.run(
        [ffmpeg_exe(), "-y", "-v", "error", "-i", src, "-vf", chain,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
         "-pix_fmt", "yuv420p", "-c:a", "copy", dst],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900)
    if result.returncode != 0 or not os.path.exists(dst) \
            or os.path.getsize(dst) == 0:
        tail_line = (result.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        log_fn(f"⚠️ Could not apply the {name} motion ({tail_line[0]}); "
               f"the clip keeps its picture")
        shutil.copy2(src, dst)
    return dst
