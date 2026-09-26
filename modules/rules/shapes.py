"""Geometry for composition rules: boxes, and the outlines inside them.

A rule used to ask one question: is the centre of this box inside that box?
Boxes are what the detector gives, and for upright, compact things they are
close enough. They are wrong in the ways that matter most for composition:

* a person reaching sideways has a box twice their width, mostly empty, and
  anything in the empty part is "inside the person";
* two diagonal things (a bat, a leaning board) have boxes that overlap
  heavily while the things themselves never touch;
* "touching" cannot be said at all: two boxes that share an edge may hold
  things a metre apart.

An outline (a polygon, normalised to the frame like the boxes) answers all
three. This module holds the relations, working on either: a detection
without an outline is its box as a four-point polygon, so rules written for
boxes keep working unchanged, and an outline, where one exists, makes the same
rule more exact.

Relations
---------
``inside``   (default) the source's centre lies inside the region.
             Box-and-box is the engine's original test, bit for bit.
``overlaps`` at least ``min_overlap`` of the source's area lies inside the
             region (0.5 = mostly inside).
``touches``  the two shapes meet, or come within ``max_gap`` (a fraction of
             the frame width) of each other.

Areas are measured by drawing both shapes onto a small grid fitted around
them, so precision follows the objects' size rather than the frame's: no
geometry library, and a polygon from any mask tracer works as it comes.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

INSIDE = "inside"
OVERLAPS = "overlaps"
TOUCHES = "touches"
RELATIONS = (INSIDE, OVERLAPS, TOUCHES)

# Grid the overlap is measured on. 128 cells across the two shapes' joint
# extent: an error of well under a percent of the smaller shape's area.
_GRID = 128


def box_polygon(box: Sequence[float]) -> list:
    """``[x, y, w, h]`` (normalised) -> its four corners."""
    x, y, w, h = (float(v) for v in box)
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def as_polygon(det: dict) -> list:
    """The detection's outline when it has one, else its box."""
    outline = det.get("contour")
    if outline and len(outline) >= 3:
        return [(float(p[0]), float(p[1])) for p in outline]
    return box_polygon(det["box"])


def has_outline(det: dict) -> bool:
    outline = det.get("contour")
    return bool(outline) and len(outline) >= 3


def area(poly: Sequence) -> float:
    """Shoelace area (absolute)."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(poly, list(poly[1:]) + [poly[0]]):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def centroid(poly: Sequence) -> tuple:
    """Area centroid; the vertex mean for a degenerate (zero-area) polygon."""
    a = 0.0
    cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(poly, list(poly[1:]) + [poly[0]]):
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-12:
        n = len(poly)
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    a *= 0.5
    return (cx / (6.0 * a), cy / (6.0 * a))


def contains(poly: Sequence, point: Sequence) -> bool:
    """Even-odd ray casting; a point on an edge counts as inside."""
    px, py = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if _on_segment(px, py, x1, y1, x2, y2):
            return True
        if (y1 > py) != (y2 > py):
            x_at = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_at:
                inside = not inside
    return inside


def _on_segment(px, py, x1, y1, x2, y2, eps=1e-9) -> bool:
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (min(x1, x2) - eps <= px <= max(x1, x2) + eps
            and min(y1, y2) - eps <= py <= max(y1, y2) + eps)


def _bounds(*polys) -> tuple:
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _raster(poly: Sequence, bounds: tuple, grid: int):
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = bounds
    sx = (grid - 1) / max(x1 - x0, 1e-9)
    sy = (grid - 1) / max(y1 - y0, 1e-9)
    image = Image.new("1", (grid, grid), 0)
    ImageDraw.Draw(image).polygon([((x - x0) * sx, (y - y0) * sy) for x, y in poly],
                                  fill=1, outline=1)
    return image


def overlap_fraction(source: Sequence, region: Sequence, grid: int = _GRID) -> float:
    """Share of ``source``'s area that lies inside ``region``, 0..1."""
    import numpy as np

    sb, rb = _bounds(source), _bounds(region)
    if sb[2] < rb[0] or rb[2] < sb[0] or sb[3] < rb[1] or rb[3] < sb[1]:
        return 0.0
    bounds = _bounds(source)       # the grid fits the source: its area is the unit
    s = np.asarray(_raster(source, bounds, grid), dtype=bool)
    r = np.asarray(_raster(region, bounds, grid), dtype=bool)
    total = int(s.sum())
    return float((s & r).sum()) / total if total else 0.0


def _segments_intersect(a1, a2, b1, b2) -> bool:
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    o1, o2 = orient(a1, a2, b1), orient(a1, a2, b2)
    o3, o4 = orient(b1, b2, a1), orient(b1, b2, a2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and _on_segment(*b1, *a1, *a2)) or (o2 == 0 and _on_segment(*b2, *a1, *a2))
            or (o3 == 0 and _on_segment(*a1, *b1, *b2)) or (o4 == 0 and _on_segment(*a2, *b1, *b2)))


def _point_segment(p, a, b) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def gap(a: Sequence, b: Sequence) -> float:
    """Shortest distance between two polygons; 0 when they meet or overlap."""
    if contains(a, b[0]) or contains(b, a[0]):
        return 0.0
    edges_a = list(zip(a, list(a[1:]) + [a[0]]))
    edges_b = list(zip(b, list(b[1:]) + [b[0]]))
    best = float("inf")
    for a1, a2 in edges_a:
        for b1, b2 in edges_b:
            if _segments_intersect(a1, a2, b1, b2):
                return 0.0
            best = min(best, _point_segment(a1, b1, b2), _point_segment(a2, b1, b2),
                       _point_segment(b1, a1, a2), _point_segment(b2, a1, a2))
    return best


def relates(source: dict, region: dict, relation: str = INSIDE,
            min_overlap: float = 0.5, max_gap: float = 0.0) -> bool:
    """Whether detection ``source`` stands in ``relation`` to ``region``.

    Detections are the engine's ``{'box': [x,y,w,h], 'contour': [[x,y],...]?}``.
    """
    if relation == INSIDE:
        if not has_outline(source) and not has_outline(region):
            # The original test, exactly: no rule written for boxes may change
            # its answer because this module exists.
            sx, sy, sw, sh = source["box"]
            rx, ry, rw, rh = region["box"]
            cx, cy = sx + sw / 2, sy + sh / 2
            return rx <= cx <= rx + rw and ry <= cy <= ry + rh
        return contains(as_polygon(region), centroid(as_polygon(source)))
    if relation == OVERLAPS:
        return overlap_fraction(as_polygon(source), as_polygon(region)) >= min_overlap
    if relation == TOUCHES:
        return gap(as_polygon(source), as_polygon(region)) <= max_gap
    raise ValueError(f"relation must be one of {RELATIONS}, not {relation!r}")


def simplify(points: Sequence, tolerance: float) -> list:
    """Ramer-Douglas-Peucker: drop points closer than ``tolerance`` to the
    line through their neighbours. Keeps stored outlines to a few dozen points."""
    pts = [tuple(p) for p in points]
    if len(pts) < 4:
        return pts

    def rdp(seq):
        if len(seq) < 3:
            return seq
        far, index = 0.0, 0
        for i in range(1, len(seq) - 1):
            d = _point_segment(seq[i], seq[0], seq[-1])
            if d > far:
                far, index = d, i
        if far <= tolerance:
            return [seq[0], seq[-1]]
        return rdp(seq[:index + 1])[:-1] + rdp(seq[index:])

    closed = rdp(pts + [pts[0]])[:-1]
    return closed if len(closed) >= 3 else pts


def union_bounds(dets: Sequence[dict]) -> Optional[list]:
    """Smallest box ``[x, y, w, h]`` around every shape given."""
    polys = [as_polygon(d) for d in dets]
    if not polys:
        return None
    x0, y0, x1, y1 = _bounds(*polys)
    return [x0, y0, x1 - x0, y1 - y0]
