"""
shot_place.py — when and where a clip was taken, and which clips share a spot.

Why this exists
===============
A reel that shows the same view three times reads as one shot with hiccups,
and no amount of picking better *moments* fixes it, because the moments really
are good — they are just of the same thing. :mod:`modules.shot_type` knows how
a clip is framed and :mod:`modules.shot_window` knows which seconds of it are
steady. Neither has any way to notice that four of them are the same valley.

Measured on a real shoot, that is exactly what goes wrong: of twelve shots in a
reel, four *pairs* had been filmed from within 130 m of each other, three of
those pairs less than twenty seconds apart. Standing still and pressing record
twice is the single most common way a montage repeats itself.

Why not compare the pictures
============================
The obvious approach is to measure how similar two clips look, and it was tried
first. On a shoot that is one colour — a moor, a beach, a ski slope — it does
not work. A coarse colour signature called thirty of sixty-six pairs
near-duplicates; a perceptual hash, a normalised greyscale layout, an HSV
histogram and ORB feature matching were each checked against known-identical
pairs and none of them separated those from the rest. "The same lake from a
different angle" is a judgement about the world, and the pixels genuinely do
not carry it on footage where everything is green.

Where the camera was, on the other hand, is a fact, and it is already written
down. So this measures position and time and leaves the pictures alone.

Where the answer comes from
===========================
Three sources, in order of how much they can be trusted:

1. **The clip's own location tag.** Exact, and present on rather less footage
   than you would hope — on the shoot above, thirteen of twenty-one source
   files had one and the derived highlight clips had almost none.
2. **A GPS track**, if the caller supplies one. Any watch or phone that can
   export GPX will do; the clip's timestamp is looked up against the track.
   This is what fills in the other eight.
3. **Time alone.** Always available, and enough on its own for the case that
   matters most: two clips a few seconds apart are the same spot whatever
   else is true.

Timestamps
==========
The recording time lives on the *source* file. A clip that has been through the
engine is a re-encode and has lost it — ffmpeg writes its own tags — so a
``..._highlight.mp4`` is traced back to the file it came from before being
asked. Failing that, the file's own modification time stands in, which is
right often enough to be worth having and wrong often enough not to be trusted
over a real tag.

Public API
==========
    Place / Track
    read_track(path) -> Track
    locate(paths, track=None) -> dict[str, Place]
    group(places, ...) -> dict[str, int]
    distance(a, b) -> float
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
from dataclasses import dataclass

# Two clips shot within this far of each other are of the same place. 150 m is
# chosen from the shoot above: the pairs that are genuinely one setup sit at
# 30-130 m, and the next-closest pair — two different vantage points on the
# same valley, which are worth having both of — is 640 m. The gap between
# those is wide enough that the threshold is not balanced on an edge.
PLACE_METRES = 150.0

# The fallback when position is unknown for either clip. Three minutes is long
# enough to cover pressing record twice and short enough that it does not
# swallow a whole leg of a route.
PLACE_MINUTES = 3.0

# How far from a track point a timestamp may fall before the lookup is
# refused. A gap larger than this means the clip was shot before the watch
# started, after it stopped, or on another day entirely.
TRACK_TOLERANCE_SECONDS = 120.0

# "+53.5765-001.9497/" — the ISO 6709 string ffmpeg writes into the location
# tag. Altitude, when present, is a third signed group and is ignored.
_ISO6709 = re.compile(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")

EARTH_RADIUS = 6371000.0

# Metres a climb has to accumulate to before it counts as one. Chosen against
# a watch's own figure for the same route: 2 m gives 1030, 3 m gives 1007, 5 m
# gives 945, against the 1012 the watch reported.
CLIMB_THRESHOLD = 3.0


@dataclass
class Place:
    """Where and when one clip was taken."""
    path: str
    when: dt.datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    # "tag", "track", "time" or "" — how much the answer can be trusted, and
    # the thing to put in a log line when a grouping looks surprising.
    source: str = ""

    @property
    def located(self) -> bool:
        return self.latitude is not None and self.longitude is not None


@dataclass
class Track:
    """A GPS track: ``(time, latitude, longitude, elevation)`` in time order.

    Elevation is carried because it is the one series a viewer recognises
    instantly — an elevation profile *is* the shape of the route, and drawing
    it is most of what :mod:`modules.overlay` exists for. It is the fourth
    element rather than a parallel list so a point stays one thing.
    """
    points: list = None
    path: str = ""

    def __post_init__(self):
        if self.points is None:
            self.points = []
        self._distances = None

    def __bool__(self) -> bool:
        return bool(self.points)

    @property
    def elevations(self) -> list:
        """Metres above sea level at each point, 0 where the file had none."""
        return [(p[3] if len(p) > 3 and p[3] is not None else 0.0)
                for p in self.points]

    @property
    def distances(self) -> list:
        """Metres travelled by each point, cumulative from the start.

        Computed once and kept: a four-hour track is fifteen thousand points
        and every overlay frame asks for this.
        """
        if self._distances is None:
            running = 0.0
            out = [0.0]
            for a, b in zip(self.points, self.points[1:]):
                running += _haversine(a[1], a[2], b[1], b[2])
                out.append(running)
            self._distances = out
        return self._distances

    @property
    def length(self) -> float:
        """How far the route runs, in metres."""
        return self.distances[-1] if self.points else 0.0

    @property
    def climb(self) -> float:
        """Total ascent in metres.

        Accumulated with hysteresis rather than by summing every positive
        step, because a barometer standing still still wanders by centimetres
        and a track sampled once a second has fifteen thousand chances to add
        that up. Summing raw positives gives 1281 m on a route the watch
        recorded as 1012; discarding steps under a metre individually gives 3,
        since at one hertz almost every step *is* under a metre. Banking the
        run only once it clears :data:`CLIMB_THRESHOLD` gives 1007.
        """
        total = 0.0
        running = 0.0
        heights = self.elevations
        for a, b in zip(heights, heights[1:]):
            running += b - a
            if running >= CLIMB_THRESHOLD:
                total += running
                running = 0.0
            elif running < 0:
                running = 0.0
        return total

    def index_at(self, when: dt.datetime,
                 tolerance: float = TRACK_TOLERANCE_SECONDS):
        """Which point is nearest ``when``, or None when out of range."""
        if not self.points or when is None:
            return None
        when = _as_utc(when)
        best = min(range(len(self.points)),
                   key=lambda i: abs((self.points[i][0] - when).total_seconds()))
        if abs((self.points[best][0] - when).total_seconds()) > tolerance:
            return None
        return best

    def progress_at(self, when: dt.datetime) -> float:
        """How far into the route ``when`` falls, 0..1 by *distance*.

        Distance rather than time, because a route is a shape rather than a
        schedule: an hour spent climbing one hill should not put the marker
        halfway along a profile it barely moved through.
        """
        index = self.index_at(when)
        if index is None or self.length <= 0:
            return 0.0
        return self.distances[index] / self.length

    def at(self, when: dt.datetime,
           tolerance: float = TRACK_TOLERANCE_SECONDS):
        """Position at ``when``, or None when the track does not cover it.

        Nearest point rather than interpolated: a track is sampled about once
        a second, so the nearest one is within a couple of metres of the
        interpolated answer and this module's thresholds are in the hundreds.
        """
        if not self.points or when is None:
            return None
        when = _as_utc(when)
        best = min(self.points, key=lambda p: abs((p[0] - when).total_seconds()))
        if abs((best[0] - when).total_seconds()) > tolerance:
            return None
        return best[1], best[2]


def _as_utc(when: dt.datetime) -> dt.datetime:
    """A timezone-aware UTC datetime. A naive one is read as UTC, which is what
    ffmpeg writes and what GPX requires."""
    if when.tzinfo is None:
        return when.replace(tzinfo=dt.timezone.utc)
    return when.astimezone(dt.timezone.utc)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Metres between two coordinates."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS * math.asin(math.sqrt(min(1.0, h)))


def distance(a: Place, b: Place) -> float:
    """Metres between two located places, or ``inf`` when either is not."""
    if not (a.located and b.located):
        return float("inf")
    return _haversine(a.latitude, a.longitude, b.latitude, b.longitude)


def read_track(path: str, *, log_fn=print) -> Track:
    """Read a GPX file into a :class:`Track`.

    GPX rather than the watch's own format on purpose: every maker exports it,
    which keeps this working for anyone rather than for one brand. A file that
    cannot be read comes back empty and the caller carries on with whatever the
    clips themselves say — a missing track should cost the clips that had no
    location tag, not the reel.
    """
    track = Track(path=path)
    if not path or not os.path.exists(path):
        return track
    try:
        import xml.etree.ElementTree as ET

        points = []
        latitude = longitude = None
        for event, element in ET.iterparse(path, events=("end",)):
            # GPX is namespaced and the namespace differs by version, so tags
            # are matched on their local name.
            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "trkpt":
                latitude = element.get("lat")
                longitude = element.get("lon")
                stamp = None
                height = None
                for child in element:
                    name = child.tag.rsplit("}", 1)[-1]
                    if name == "time":
                        stamp = (child.text or "").strip()
                    elif name == "ele":
                        try:
                            height = float((child.text or "").strip())
                        except ValueError:
                            height = None
                if stamp and latitude and longitude:
                    when = _parse_iso(stamp)
                    if when is not None:
                        points.append((when, float(latitude), float(longitude),
                                       height))
                element.clear()
        points.sort(key=lambda p: p[0])
        track.points = points
        if points:
            log_fn(f"🗺️ Track: {len(points)} points, "
                   f"{points[0][0]:%H:%M} to {points[-1][0]:%H:%M} UTC")
        else:
            log_fn(f"⚠️ {os.path.basename(path)} has no timed track points")
    except Exception as exc:
        log_fn(f"⚠️ Could not read the GPS track ({exc}); "
               f"clips will be placed by what they carry themselves")
    return track


def _parse_iso(text: str) -> dt.datetime | None:
    """A GPX or ffmpeg timestamp. Both are ISO 8601; the Z suffix predates
    fromisoformat's willingness to take it on older Pythons."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return _as_utc(dt.datetime.fromisoformat(text.replace("Z", "+00:00")))
    except ValueError:
        return None


def _source_file(path: str) -> str:
    """The camera file a derived clip came from, or ``path`` unchanged.

    The engine writes ``NAME_highlight.mp4`` beside ``NAME.MP4``. Re-encoding
    drops the recording time, so the original is where the question has to be
    asked — and the original is the one file we can be sure of finding, since
    the naming is the engine's own.
    """
    directory, name = os.path.split(path)
    stem, _ = os.path.splitext(name)
    for suffix in ("_highlight",):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        return path
    for extension in (".MP4", ".mp4", ".MOV", ".mov", ".LRV"):
        candidate = os.path.join(directory, stem + extension)
        if os.path.exists(candidate) and candidate != path:
            return candidate
    return path


def _tags(path: str) -> dict:
    """The container tags, via one ffprobe call. Empty on any failure."""
    import json
    import subprocess

    from modules.video_probe import ffprobe_exe

    try:
        result = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-show_entries", "format_tags",
             "-of", "json", path],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=30)
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout or "{}")
        return (data.get("format") or {}).get("tags") or {}
    except Exception:
        return {}


def _read_one(path: str) -> Place:
    """Time and position from the file itself, following a derived clip back
    to the camera file it came from."""
    place = Place(path=path)
    source = _source_file(path)
    tags = _tags(source)
    if source != path and not tags:
        tags = _tags(path)

    for key in ("creation_time", "com.apple.quicktime.creationdate", "date"):
        when = _parse_iso(str(tags.get(key, "")))
        if when is not None:
            place.when = when
            place.source = "time"
            break
    if place.when is None:
        # Better than nothing and worse than a tag: a copied file keeps its
        # modification time on most systems and loses it on some.
        try:
            place.when = _as_utc(
                dt.datetime.fromtimestamp(os.path.getmtime(source)))
            place.source = "time"
        except OSError:
            pass

    for key in ("location", "location-eng",
                "com.apple.quicktime.location.ISO6709"):
        match = _ISO6709.match(str(tags.get(key, "")).strip())
        if match:
            place.latitude = float(match.group(1))
            place.longitude = float(match.group(2))
            place.source = "tag"
            break
    return place


def locate(paths, *, track: Track = None, log_fn=print) -> dict:
    """Where and when each clip was taken.

    Never raises and never returns nothing: a clip whose time and position are
    both unknown comes back as an empty :class:`Place`, and :func:`group` puts
    every such clip in a place of its own — which is the answer that changes
    the edit least.
    """
    places: dict[str, Place] = {}
    for path in paths or []:
        try:
            places[path] = _read_one(path)
        except Exception:
            places[path] = Place(path=path)

    if track:
        filled = 0
        for place in places.values():
            if place.located or place.when is None:
                continue
            found = track.at(place.when)
            if found:
                place.latitude, place.longitude = found
                place.source = "track"
                filled += 1
        if filled:
            log_fn(f"🗺️ Placed {filled} clip(s) on the track that carried no "
                   f"GPS of their own")

    located = sum(1 for p in places.values() if p.located)
    timed = sum(1 for p in places.values() if p.when is not None)
    if places:
        log_fn(f"📍 {located} of {len(places)} clip(s) have a position, "
               f"{timed} have a time")
    return places


def group(places: dict, *, metres: float = PLACE_METRES,
          minutes: float = PLACE_MINUTES, log_fn=print) -> dict:
    """Assign each clip a place number, so a reel can spread across them.

    Clips are walked in time order and each either joins the place being built
    or starts a new one. Position decides when both clips have it; time decides
    when either does not. That order matters — two clips a hundred metres apart
    are one setup however long the gap, and two clips ten seconds apart are one
    setup even with no idea where either was.

    Greedy against the place's *first* member rather than transitively linked,
    because a route where you stop every hundred metres would otherwise chain
    into a single place stretching for miles.
    """
    ordered = sorted(
        places.values(),
        key=lambda p: (p.when or dt.datetime.max.replace(tzinfo=dt.timezone.utc),
                       p.path))
    numbers: dict[str, int] = {}
    anchor: Place | None = None
    previous: Place | None = None
    number = -1

    for place in ordered:
        if anchor is None:
            number += 1
            anchor = place
        else:
            gap = float("inf")
            if place.when is not None and previous is not None \
                    and previous.when is not None:
                gap = abs((place.when - previous.when).total_seconds()) / 60.0
            if place.located and anchor.located:
                same = distance(place, anchor) <= metres
            else:
                same = gap <= minutes
            if not same:
                number += 1
                anchor = place
        numbers[place.path] = number
        previous = place

    if numbers:
        count = len(set(numbers.values()))
        log_fn(f"📍 {len(numbers)} clip(s) from {count} place(s)"
               + ("" if count == len(numbers)
                  else " — some were shot from the same spot"))
    return numbers
