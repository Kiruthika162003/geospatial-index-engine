"""Polylabel: the pole of inaccessibility, the interior point farthest from every edge.

Placing a label on a polygon, a country on a map, wants a point that
is comfortably inside, with room around it, not the centroid, because
the centroid of a concave or ring-shaped polygon can lie outside the
polygon entirely or jammed against a narrow part. The pole of
inaccessibility is the better anchor: the interior point that is
farthest from the polygon's boundary, the center of the largest circle
that fits inside. Finding it exactly is hard, but a subdivision search
finds it to any tolerance. Cover the polygon's bounding box with cells,
and for each cell compute the distance from its center to the polygon
boundary, positive if the center is inside and negative if outside, and
an upper bound on how large that distance could get anywhere in the
cell, the center distance plus the cell's half-diagonal. Keep the cells
in a priority queue ordered by that upper bound, and repeatedly take
the most promising cell, split it into four, and push the children,
tracking the best actual center distance seen. A cell whose upper bound
cannot beat the best is never split, so the search homes in on the pole
without examining the whole area, stopping when the best cell's
potential improvement drops below the tolerance. The key contrast with
the centroid is not speed but correctness of placement: the pole is
guaranteed to be strictly inside the polygon and as far from any edge
as possible, so on a C-shape or an annulus, where the centroid falls in
the hollow, the pole sits in the solid part. The finding worth stating
is that the pole of inaccessibility stays inside and away from the
edges where the centroid can wander outside, which is exactly why label
placement uses it. This module computes the pole to a tolerance, and a
survey exhibits a concave polygon whose centroid lies outside while the
pole lies safely within.
"""

from __future__ import annotations

import heapq
import math

from atlas.errors import Invalid

Point = tuple[float, float]


def _point_to_segment(px: float, py: float, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dx == 0 and dy == 0:
        return math.hypot(px - a[0], py - a[1])
    t = ((px - a[0]) * dx + (py - a[1]) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))


def _signed_distance(px: float, py: float, polygon: list[Point]) -> float:
    # distance to the boundary, positive inside, negative outside
    inside = False
    min_d = math.inf
    n = len(polygon)
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        if (a[1] > py) != (b[1] > py):
            x_cross = a[0] + (py - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
            if px < x_cross:
                inside = not inside
        min_d = min(min_d, _point_to_segment(px, py, a, b))
    return min_d if inside else -min_d


def pole_of_inaccessibility(polygon: list[Point], precision: float = 0.01) -> Point:
    if polygon is None or len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    if precision <= 0:
        raise Invalid("precision must be positive")
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    cell = min(max_x - min_x, max_y - min_y)
    if cell == 0:
        raise Invalid("polygon is degenerate")
    h = cell / 2

    def make(cx: float, cy: float, half: float) -> tuple[float, float, float, float]:
        d = _signed_distance(cx, cy, polygon)
        return (-(d + half * math.sqrt(2)), cx, cy, half)  # neg upper bound for a max-heap

    heap: list[tuple[float, float, float, float]] = []
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            heapq.heappush(heap, make(x + h, y + h, h))
            y += cell
        x += cell
    best_x = (min_x + max_x) / 2
    best_y = (min_y + max_y) / 2
    best_d = _signed_distance(best_x, best_y, polygon)
    while heap:
        neg_bound, cx, cy, half = heapq.heappop(heap)
        bound = -neg_bound
        d = _signed_distance(cx, cy, polygon)
        if d > best_d:
            best_d, best_x, best_y = d, cx, cy
        if bound - best_d <= precision:
            continue
        q = half / 2
        for ox, oy in ((-q, -q), (q, -q), (-q, q), (q, q)):
            heapq.heappush(heap, make(cx + ox, cy + oy, q))
    return (best_x, best_y)
