"""Buffer: offset a convex polygon outward; the area grows by perimeter times d plus a disk.

Buffering a shape means growing it by a fixed distance in every
direction, the operation behind a no-build zone around a river, a
service radius around a road, or a safety margin around an obstacle.
For a convex polygon the buffered region has a clean description: it
is the Minkowski sum of the polygon with a disk of radius d, whose
boundary is the polygon's edges pushed outward by d, joined at each
vertex by a circular arc that turns through the vertex's exterior
angle. Its area follows from that picture exactly, and the formula is
the thing worth measuring. The original polygon contributes its own
area; each edge pushed out by d sweeps a rectangle of the edge's
length times d, and those sum to the perimeter times d; and the
corner arcs, whose exterior angles sum to a full turn for any convex
polygon, together make up exactly one full disk of radius d, area pi d
squared. So the buffered area is the area plus the perimeter times d
plus pi d squared, Steiner's formula, and it holds for every convex
polygon regardless of shape. The module builds the offset boundary by
pushing each edge along its outward normal and connecting the pushed
edges with arcs approximated by a fan of points, so the polygonal
approximation's area falls a little short of the formula, by an amount
that shrinks as the arc resolution rises, which the survey measures;
at a coarse resolution the shortfall is visible, at a fine one it is
negligible, and it is always a shortfall, never an excess, because an
inscribed fan lies inside its arc. The finding worth stating is that a
convex buffer's exact area is Steiner's three-term sum, and the
polygonal offset approaches it from below as the arc resolution grows,
so the formula is the check and the fan is the approximation. This
module offsets a convex polygon and computes the Steiner area, and a
survey measures the approximation against the formula.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _check_convex_ccw(polygon: list[Point]) -> None:
    if polygon is None or len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    n = len(polygon)
    for i in range(n):
        if _cross(polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n]) <= 0:
            raise Invalid("polygon must be strictly convex and counterclockwise")


def perimeter(polygon: list[Point]) -> float:
    n = len(polygon)
    total = 0.0
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def polygon_area(polygon: list[Point]) -> float:
    n = len(polygon)
    total = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def steiner_area(polygon: list[Point], d: float) -> float:
    _check_convex_ccw(polygon)
    if d < 0:
        raise Invalid("buffer distance must not be negative")
    return polygon_area(polygon) + perimeter(polygon) * d + math.pi * d * d


def offset(polygon: list[Point], d: float, arc_steps: int = 8) -> list[Point]:
    _check_convex_ccw(polygon)
    if d < 0:
        raise Invalid("buffer distance must not be negative")
    if arc_steps < 1:
        raise Invalid("arc_steps must be at least one")
    n = len(polygon)
    out: list[Point] = []
    for i in range(n):
        prev_pt, cur, nxt = polygon[i - 1], polygon[i], polygon[(i + 1) % n]
        # outward normals of the incoming and outgoing edges
        in_dx, in_dy = cur[0] - prev_pt[0], cur[1] - prev_pt[1]
        out_dx, out_dy = nxt[0] - cur[0], nxt[1] - cur[1]
        a0 = math.atan2(-in_dx, in_dy)  # normal of the incoming edge (rotated -90)
        a1 = math.atan2(-out_dx, out_dy)
        sweep = (a1 - a0) % (2 * math.pi)
        for k in range(arc_steps + 1):
            ang = a0 + sweep * k / arc_steps
            out.append((cur[0] + d * math.cos(ang), cur[1] + d * math.sin(ang)))
    return out
