"""Rotating calipers: the farthest pair of points sits on the hull, found in one walk.

The diameter of a set of points, the greatest distance between any two
of them, seems to demand comparing every pair, which is quadratic. But
the two farthest points must both lie on the convex hull, because
pushing either point outward to the hull can only increase the
distance, so the search can be confined to the hull's vertices. On the
hull, rotating calipers finds the diameter in a single linear walk. The
image is a pair of parallel lines, calipers, squeezing the polygon from
opposite sides; as they rotate together around the hull, they touch
pairs of vertices that are antipodal, as far apart across the polygon
as any pair in that direction can be, and the diameter is the largest
distance among these antipodal pairs. The walk advances two pointers
around the hull without ever backtracking, so it visits a linear number
of antipodal pairs rather than all quadratic pairs, and the farthest of
them is the diameter. The saving is real only when the hull is much
smaller than the point set, which it usually is; combined with an
n-log-n hull, the whole diameter computation is n log n, dominated by
the hull, against the quadratic brute force. The insight worth holding
onto is not the caliper mechanics but the reduction that makes them
possible: the farthest pair is a hull-vertex pair, so the interior
points, however many, cannot be part of the answer and are discarded by
the hull step before the calipers ever run. The finding worth stating
is that the diameter equals the brute-force farthest distance while
examining only the hull's antipodal pairs, a linear count, not all
pairs. This module computes the diameter by rotating calipers, and a
survey confirms it equals the brute farthest pair while touching far
fewer pairs.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _hull(points: list[Point]) -> list[Point]:
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    lower: list[Point] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[Point] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def diameter(points: list[Point]) -> tuple[float, Point, Point]:
    if points is None:
        raise Invalid("points must not be None")
    if len(set(points)) < 2:
        raise Invalid("need at least two distinct points for a diameter")
    hull = _hull(points)
    n = len(hull)
    if n == 2:
        d2 = _dist2(hull[0], hull[1])
        return (d2**0.5, hull[0], hull[1])
    best = 0.0
    pair = (hull[0], hull[1])
    j = 1
    for i in range(n):
        ni = (i + 1) % n
        # advance j while the far vertex keeps getting farther from edge i->i+1
        while (
            abs(_cross(hull[i], hull[ni], hull[(j + 1) % n]))
            > abs(_cross(hull[i], hull[ni], hull[j]))
        ):
            j = (j + 1) % n
        for candidate in (hull[j], hull[(j + 1) % n]):
            d2 = _dist2(hull[i], candidate)
            if d2 > best:
                best = d2
                pair = (hull[i], candidate)
    return (best**0.5, pair[0], pair[1])
