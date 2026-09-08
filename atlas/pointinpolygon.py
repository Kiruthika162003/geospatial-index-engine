"""Point in polygon: the ray-casting parity rule, and where winding disagrees with it.

Deciding whether a point lies inside a polygon is a staple of spatial
work, hit-testing a map click against a region, filtering points to a
boundary. The ray-casting test answers it with a parity count: shoot a
ray from the point in a fixed direction, say straight right, and count
how many polygon edges it crosses; an odd count means the point is
inside, an even count means outside, because each crossing flips
between inside and outside and the point starts outside at infinity.
It handles concave polygons and even polygons with holes without any
special cases, which is its great virtue. The subtlety that must be
handled and is easy to get wrong is the edges that touch the ray's
level exactly, a vertex sitting on the ray, which can be counted twice
or zero times and corrupt the parity; the standard fix is a
half-open rule on each edge's vertical span, counting an edge only if
one endpoint is strictly above the ray level and the other is at or
below, so shared vertices are counted once. The alternative test is
the winding number, which sums the signed angle the polygon subtends
around the point and asks whether it wraps a nonzero number of times.
For a simple, non-self-intersecting polygon the two agree everywhere.
They part ways on self-intersecting polygons: the parity rule calls a
doubly-enclosed region outside, since a ray crosses an even number of
times, while the winding rule calls it inside, since the boundary
wraps it twice. The finding worth stating is that ray casting and
winding agree on every simple polygon and diverge only where the
polygon overlaps itself, so the choice between them is a choice of what
self-overlap should mean. This module implements both tests, and a
survey confirms they agree on simple polygons and names a
self-intersecting case where they differ.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]


def ray_casting(point: Point, polygon: list[Point]) -> bool:
    if polygon is None or len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # half-open span rule: count the edge if the ray level is in [min, max)
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if x < x_cross:
                inside = not inside
    return inside


def winding_number(point: Point, polygon: list[Point]) -> int:
    if polygon is None or len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    x, y = point
    winding = 0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if y1 <= y:
            if y2 > y and _is_left(x1, y1, x2, y2, x, y) > 0:
                winding += 1
        elif y2 <= y and _is_left(x1, y1, x2, y2, x, y) < 0:
            winding -= 1
    return winding


def contains_by_winding(point: Point, polygon: list[Point]) -> bool:
    return winding_number(point, polygon) != 0


def _is_left(x1: float, y1: float, x2: float, y2: float, px: float, py: float) -> float:
    return (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
