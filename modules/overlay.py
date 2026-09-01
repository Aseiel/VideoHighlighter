"""
overlay.py — animated graphics drawn over the reel.

Why this exists
===============
A cut list can only ever rearrange what the camera saw. Everything a viewer
knows about a route beyond the pictures — how far it ran, how much of it was
uphill, where in it this particular shot happened — is in the GPS track and
nowhere in the footage. Drawing it is the difference between a montage of nice
landscapes and a record of a specific day.

It also answers the complaint that every transition looks the same, though not
in the way that sounds. Transitions are the join between two shots and there
are only so many ways to make one; what actually makes short-form video feel
designed is what is drawn *on* it. An elevation profile that fills as the reel
plays gives the whole thing a spine, and it is the same spine for every shot,
which is what makes a sequence read as one piece.

How it draws
============
Frames are drawn with Pillow and handed to ffmpeg as raw RGBA on a pipe, which
composites them over the video with ``overlay``. Pillow is already a dependency
and permissively licensed, so this adds nothing to what a user has to install,
and every element is a plain function of time that can be tested by looking at
the pixels it produced.

Piped rather than written out as files: a 24-second vertical reel is 720 frames
of 1080x1920 RGBA, which is 6 GB on disk and about a hundred megabytes at a
time in memory.

Everything is a function of one number
======================================
Each element draws itself for a given moment, expressed as ``t`` in seconds
into the finished reel. Nothing holds state between frames. That is what makes
an animation reproducible, testable a frame at a time, and — for later — safe
to let something else author, since an element is a small declaration rather
than a program.

Public API
==========
    Style / Scene
    ElevationProfile / RouteMap / Readout / Ticker
    ELEMENTS
    render_overlay(scene, elements, output, ...)  -> str
    burn_overlay(src, dst, scene, elements, ...)  -> str
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field

from modules.app_paths import ffmpeg_exe

# Everything is positioned and sized as a fraction of the frame, never in
# pixels, so one description works on a vertical reel and a wide one alike.
@dataclass
class Style:
    """How an element is drawn.

    Defaults are a white line with a soft shadow, which is the one combination
    that stays legible over footage the drawing knows nothing about — a thin
    bright line disappears into a bright sky, and the shadow is what stops it.
    """
    colour: tuple = (255, 255, 255, 235)
    accent: tuple = (255, 90, 90, 255)      # the marker, and anything "now"
    shade: tuple = (0, 0, 0, 90)            # the fill under a profile
    width: float = 0.004                    # stroke, as a share of the frame
    shadow: bool = True
    font: float = 0.026                     # text size, as a share of height


@dataclass
class Box:
    """Where an element sits, as fractions of the frame: left, top, width,
    height. Fractions rather than pixels so the same layout works on any
    delivery size."""
    x: float = 0.06
    y: float = 0.78
    w: float = 0.88
    h: float = 0.13

    def pixels(self, width: int, height: int) -> tuple:
        return (int(self.x * width), int(self.y * height),
                int(self.w * width), int(self.h * height))


@dataclass
class Scene:
    """Everything the elements need to know about this reel.

    ``marks`` is the important one: for each cut, when it starts on screen and
    how far along the route it was filmed. That mapping is what lets a profile
    show *where this shot happened* rather than just how long the reel has been
    running — and it exists only because :mod:`modules.shot_place` already
    placed every clip on the track.
    """
    duration: float = 0.0
    width: int = 1080
    height: int = 1920
    # (time on screen, progress along the route 0..1) for each cut.
    marks: list = field(default_factory=list)
    elevations: list = field(default_factory=list)
    length: float = 0.0        # metres
    climb: float = 0.0         # metres

    def progress_at(self, t: float) -> float:
        """How far through the reel we are, 0..1 — a plain progress bar.

        This drives everything that *accumulates*: the filled part of the
        profile, the traced part of the route, the counters. It runs with the
        clock rather than with the footage, and that is a deliberate
        separation from :meth:`marker_at`.

        The first version tied both to where each shot was filmed, and it read
        as broken. A reel is ordered as a story — the hook is whichever shot
        is most striking and the payoff is whichever ends it, and neither has
        anything to do with the route — so the counters ran 12.6 km, 9.3 km,
        20.0 km, 3.2 km. A distance that goes backwards is worse than no
        distance at all.
        """
        if self.duration <= 0:
            return 0.0
        return min(1.0, max(0.0, t / self.duration))

    def marker_at(self, t: float) -> float:
        """Where along the route the shot on screen *now* was filmed, 0..1.

        This drives the dot, and only the dot. It jumps, and that is right:
        a pointer that moves to each new shot reads as "this bit happened
        here", where the same jumping applied to a bar reads as a bug.

        Eased over a fraction of a second rather than cut, because a dot that
        teleports is hard to follow across a wide profile.
        """
        if not self.marks:
            return self.progress_at(t)
        travel = 0.35
        previous = self.marks[0][1]
        for start, progress in self.marks:
            if t < start:
                break
            if t < start + travel:
                return _ease(previous, progress, (t - start) / travel)
            previous = progress
        return previous


ELEMENTS: dict = {}


def _register(cls):
    ELEMENTS[cls.key] = cls
    return cls


def _ease(a: float, b: float, k: float) -> float:
    """Cubic ease-out between two values. The same curve the transitions use,
    and for the same reason: a marker that starts and stops at the speed it
    travels reads as a jump."""
    k = min(1.0, max(0.0, k))
    return a + (b - a) * (1 - pow(1 - k, 3))


def _font(size: int):
    """A bold sans at ``size``, or Pillow's built-in when none is installed."""
    from PIL import ImageFont

    from modules.transitions import _font_path

    path = _font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _line(draw, points, colour, width, shadow=True):
    """A polyline, with a soft dark copy behind it when asked.

    The shadow is not decoration. These are drawn over footage this module has
    never seen, and a white line over a bright sky is invisible; an offset dark
    copy costs one more pass and makes the line readable over anything.
    """
    if len(points) < 2:
        return
    if shadow:
        draw.line([(x + width, y + width) for x, y in points],
                  fill=(0, 0, 0, 120), width=width, joint="curve")
    draw.line(points, fill=colour, width=width, joint="curve")


# Points an element may draw. A polyline is being rendered into a box a few
# hundred pixels wide, so anything past roughly one point per pixel is detail
# nobody can see and every frame pays for: a four-hour track is fifteen
# thousand points, and drawing all of them took 3.9 seconds *per frame* — over
# half an hour for a twenty-second reel. Decimated, the same element draws in
# under a millisecond and looks identical.
MAX_POINTS = 600


def _thin(points: list, limit: int = MAX_POINTS) -> list:
    """Evenly drop points until at most ``limit`` remain, keeping the last."""
    if len(points) <= limit:
        return list(points)
    step = len(points) / float(limit)
    out = [points[int(i * step)] for i in range(limit)]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


@dataclass
class Element:
    """One thing drawn on the reel. Subclasses implement :meth:`draw`.

    Geometry that does not depend on ``t`` is built once and cached against
    the frame size, because it is the same drawing on every one of the several
    hundred frames a reel needs.
    """
    box: Box = field(default_factory=Box)
    style: Style = field(default_factory=Style)

    def __post_init__(self):
        self._shape = None
        self._shape_for = None

    def shape(self, scene: Scene) -> list:
        """The element's static polyline in screen coordinates."""
        key = (scene.width, scene.height)
        if self._shape_for != key:
            self._shape = self.build(scene)
            self._shape_for = key
        return self._shape

    def build(self, scene: Scene) -> list:  # pragma: no cover
        return []

    def draw(self, draw, scene: Scene, t: float) -> None:  # pragma: no cover
        raise NotImplementedError


@_register
@dataclass
class ElevationProfile(Element):
    """The shape of the route, filling in as the reel plays.

    The one element that is worth having on its own. An elevation profile is
    instantly readable — it is a picture of hills — and because the reel's cuts
    are placed along it, it doubles as a progress bar that means something.
    """
    key = "elevation"
    label = "Elevation profile"

    def build(self, scene: Scene) -> list:
        heights = scene.elevations
        if len(heights) < 2:
            return []
        x0, y0, w, h = self.box.pixels(scene.width, scene.height)
        low, high = min(heights), max(heights)
        span = (high - low) or 1.0

        # One point per column at most: the profile is being drawn into a box
        # about a thousand pixels wide, and a fifteen-thousand-point track has
        # no more to say at that size.
        columns = max(2, min(w, MAX_POINTS))
        shape = []
        for i in range(columns):
            share = i / (columns - 1)
            height = heights[min(len(heights) - 1,
                                 int(share * (len(heights) - 1)))]
            shape.append((x0 + share * w, y0 + h - (height - low) / span * h))
        return shape

    def draw(self, draw, scene: Scene, t: float) -> None:
        shape = self.shape(scene)
        if len(shape) < 2:
            return
        x0, y0, w, h = self.box.pixels(scene.width, scene.height)
        stroke = max(2, int(self.style.width * scene.width))
        reached = x0 + scene.progress_at(t) * w

        # The part already travelled is filled; the rest is the outline, so the
        # profile reads as a whole route with a position in it rather than as
        # something still being drawn.
        done = [(x, y) for x, y in shape if x <= reached]
        if len(done) > 1:
            draw.polygon(done + [(done[-1][0], y0 + h), (x0, y0 + h)],
                         fill=self.style.shade)
        _line(draw, shape, self.style.colour, stroke, self.style.shadow)
        if len(done) > 1:
            _line(draw, done, self.style.accent, stroke, self.style.shadow)

        # And where the shot on screen was actually filmed. Separate from the
        # fill above, which is the reel's own progress — see Scene.marker_at.
        at = min(len(shape) - 1,
                 int(scene.marker_at(t) * (len(shape) - 1)))
        _dot(draw, shape[at], stroke * 2.4, self.style.accent)


@_register
@dataclass
class RouteMap(Element):
    """The route seen from above, with a marker on the shot being shown.

    Drawn to fit its box with the aspect ratio kept, because a route squashed
    to fill a wide box is a different shape, and the shape is the whole point.
    """
    key = "route"
    label = "Route map"

    # (latitude, longitude) per track point — set by build_scene.
    path: list = field(default_factory=list)

    def build(self, scene: Scene) -> list:
        if len(self.path) < 2:
            return []
        x0, y0, w, h = self.box.pixels(scene.width, scene.height)
        lats = [p[0] for p in self.path]
        lons = [p[1] for p in self.path]
        # Longitude degrees are shorter than latitude ones away from the
        # equator, so a route drawn on raw degrees leans. cos(latitude) is the
        # correction, and at 53 degrees north it is a factor of 0.6 — visible
        # at a glance.
        scale_x = math.cos(math.radians(sum(lats) / len(lats)))
        span_x = (max(lons) - min(lons)) * scale_x or 1e-9
        span_y = (max(lats) - min(lats)) or 1e-9
        fit = min(w / span_x, h / span_y)
        pad_x = (w - span_x * fit) / 2
        pad_y = (h - span_y * fit) / 2

        return [(x0 + pad_x + (lon - min(lons)) * scale_x * fit,
                 y0 + h - pad_y - (lat - min(lats)) * fit)
                for lat, lon in self.path]

    def draw(self, draw, scene: Scene, t: float) -> None:
        shape = self.shape(scene)
        if len(shape) < 2:
            return
        stroke = max(2, int(self.style.width * scene.width * 0.8))
        _line(draw, shape, self.style.colour, stroke, self.style.shadow)
        traced = min(len(shape) - 1,
                     int(scene.progress_at(t) * (len(shape) - 1)))
        if traced > 1:
            _line(draw, shape[:traced + 1], self.style.accent, stroke, False)
        at = min(len(shape) - 1, int(scene.marker_at(t) * (len(shape) - 1)))
        _dot(draw, shape[at], stroke * 2.6, self.style.accent)


@_register
@dataclass
class Readout(Element):
    """A number that counts up as the reel plays — distance, climb, or both.

    Counted rather than stated: a figure that ticks reads as something being
    covered, and a figure that simply sits there reads as a caption.
    """
    key = "readout"
    label = "Distance and climb"

    show: tuple = ("distance", "climb")

    def draw(self, draw, scene: Scene, t: float) -> None:
        x0, y0, w, h = self.box.pixels(scene.width, scene.height)
        size = max(12, int(self.style.font * scene.height))
        font = _font(size)
        share = scene.progress_at(t)

        parts = []
        if "distance" in self.show and scene.length:
            parts.append(f"{scene.length * share / 1000:.1f} km")
        if "climb" in self.show and scene.climb:
            parts.append(f"{scene.climb * share:,.0f} m up")
        if not parts:
            return

        text = "   ".join(parts)
        if self.style.shadow:
            draw.text((x0 + 2, y0 + 2), text, font=font, fill=(0, 0, 0, 150))
        draw.text((x0, y0), text, font=font, fill=self.style.colour)


@_register
@dataclass
class Ticker(Element):
    """A thin bar of the reel's own progress, marked where every cut falls.

    The one element that describes the edit rather than the route, and the
    cheapest way to make a reel feel deliberate: the marks show the rhythm the
    cuts are keeping.
    """
    key = "ticker"
    label = "Cut ticker"

    def draw(self, draw, scene: Scene, t: float) -> None:
        if not scene.duration:
            return
        x0, y0, w, h = self.box.pixels(scene.width, scene.height)
        thickness = max(2, int(self.style.width * scene.width * 0.7))
        middle = y0 + h // 2

        draw.line([(x0, middle), (x0 + w, middle)],
                  fill=(255, 255, 255, 70), width=thickness)
        for start, _ in scene.marks:
            at = x0 + int(w * min(1.0, start / scene.duration))
            draw.line([(at, middle - thickness * 2), (at, middle + thickness * 2)],
                      fill=(255, 255, 255, 110), width=max(1, thickness // 2))
        done = x0 + int(w * min(1.0, t / scene.duration))
        draw.line([(x0, middle), (done, middle)],
                  fill=self.style.accent, width=thickness)


def _dot(draw, point, radius, colour):
    x, y = point
    radius = max(2, int(radius))
    draw.ellipse([x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1],
                 fill=(0, 0, 0, 110))
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=colour)


def build_scene(edl, track, places, *, width: int = 0, height: int = 0) -> Scene:
    """Work out what the elements need from a cut list and a track.

    Returns an empty scene when there is nothing to draw, so a caller can ask
    for overlays unconditionally and get a plain reel when the footage carries
    no position — which is the same bargain every other measurement here makes.
    """
    from modules.transitions import MAX_CLIP_FRACTION, MIN_DURATION

    scene = Scene(width=int(width or edl.width or 1080),
                  height=int(height or edl.height or 1920))
    if track:
        scene.elevations = track.elevations
        scene.length = track.length
        scene.climb = track.climb

    # Where each cut lands on screen. The transitions overlap, so a cut starts
    # earlier than the sum of the ones before it — the same arithmetic
    # Edl.duration does, and it has to agree with it or every mark drifts.
    at = 0.0
    for index, cut in enumerate(edl.cuts):
        progress = 0.0
        place = (places or {}).get(cut.source)
        if track and place is not None and getattr(place, "when", None):
            progress = track.progress_at(place.when)
        scene.marks.append((at, progress))

        at += cut.duration
        if index + 1 < len(edl.cuts) and cut.transition != "cut":
            room = min(cut.duration, edl.cuts[index + 1].duration) * MAX_CLIP_FRACTION
            held = min(float(cut.transition_duration), room)
            if held >= MIN_DURATION:
                at -= held
    scene.duration = at
    return scene


def make_elements(names, scene: Scene, track=None) -> list:
    """Build the named elements, laid out so they do not sit on top of each
    other. Unknown names are skipped rather than raising: a preset naming an
    element a later build removed should cost that element, not the render."""
    chosen = [n for n in (names or []) if n in ELEMENTS]
    out: list = []
    # Stacked up from the bottom of the frame, above where a caption sits.
    bottom = 0.90
    for name in chosen:
        if name == "elevation":
            bottom -= 0.11
            out.append(ElevationProfile(box=Box(0.06, bottom, 0.88, 0.10)))
        elif name == "route":
            out.append(RouteMap(box=Box(0.70, 0.06, 0.24, 0.14),
                                path=_thin([(p[1], p[2]) for p in track.points])
                                if track else []))
        elif name == "readout":
            bottom -= 0.05
            out.append(Readout(box=Box(0.06, bottom, 0.88, 0.04)))
        elif name == "ticker":
            bottom -= 0.03
            out.append(Ticker(box=Box(0.06, bottom, 0.88, 0.02)))
    return out


def frames(scene: Scene, elements, fps: int):
    """Yield each overlay frame as RGBA bytes.

    A generator so the caller can push straight into a pipe without ever
    holding more than one frame: a 24-second vertical reel is 720 frames of
    8 MB each.
    """
    from PIL import Image, ImageDraw

    total = max(1, int(round(scene.duration * fps)))
    for index in range(total):
        t = index / fps
        image = Image.new("RGBA", (scene.width, scene.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image, "RGBA")
        for element in elements:
            try:
                element.draw(draw, scene, t)
            except Exception:
                # One element that cannot draw itself must not cost the other
                # three, and certainly not the render.
                continue
        yield image.tobytes()


def burn_overlay(src: str, dst: str, scene: Scene, elements, *, fps: int = 30,
                 crf: int = 20, preset: str = "medium", log_fn=print,
                 cancel_check=None) -> str:
    """Draw ``elements`` over ``src`` and write ``dst``.

    Degrades to a copy on any failure, for the same reason :func:`burn_text`
    does: graphics are worth a lot on a reel and are not worth failing a
    finished render for.
    """
    if not elements:
        shutil.copy2(src, dst)
        return dst
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        log_fn("⚠️ Pillow is not available, so the graphics were skipped")
        shutil.copy2(src, dst)
        return dst

    names = ", ".join(getattr(e, "label", type(e).__name__) for e in elements)
    log_fn(f"🎨 Drawing {names} over {scene.duration:.0f}s at "
           f"{scene.width}x{scene.height}")

    command = [
        ffmpeg_exe(), "-y", "-v", "error",
        "-i", src,
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{scene.width}x{scene.height}", "-r", str(fps), "-i", "-",
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto[v]",
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-movflags", "+faststart", dst,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames(scene, elements, fps):
            if cancel_check is not None and cancel_check():
                process.kill()
                raise RuntimeError("cancelled")
            process.stdin.write(frame)
        process.stdin.close()
    except (BrokenPipeError, OSError):
        # ffmpeg gave up early; its stderr below says why.
        pass
    _, stderr = process.communicate(timeout=1800)

    if process.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        detail = (stderr or b"").decode("utf-8", "replace").strip()[-300:]
        log_fn(f"⚠️ Could not draw the graphics ({detail or 'unknown error'}); "
               f"the reel keeps its picture")
        shutil.copy2(src, dst)
    return dst
