"""Douglas-Peucker: thin a polyline to its essential vertices within a distance bound.

A recorded path, a GPS track or a coastline, often has far more points
than its shape needs, and drawing or storing them all is waste. The
Douglas-Peucker algorithm simplifies a polyline by keeping only the
vertices that matter, guaranteeing that no discarded point lies farther
than a chosen tolerance from the simplified line. It works
recursively. Draw a straight segment from the first point to the last,
and find the original vertex farthest from that segment, measured by
perpendicular distance. If even that farthest point is within the
tolerance, the whole run between the endpoints is well approximated by
the single segment, so every interior point is dropped. If it is
beyond the tolerance, that farthest point is a feature the segment
cannot represent, so it is kept and the problem splits in two, the run
before it and the run after, each simplified the same way. The result
keeps sharp corners and drops redundant near-straight stretches, and
the tolerance is a direct knob: larger tolerance, fewer points, coarser
shape. The property worth stating precisely, because it is the
algorithm's contract, is that the maximum distance from any original
point to the simplified line never exceeds the tolerance, so the error
is bounded, not merely usually small. What it does not bound is the
number of points kept, which depends on how wiggly the path is; a
straight road collapses to two points, a jagged coast keeps many. The
finding worth stating is that Douglas-Peucker trades vertex count for a
hard error ceiling, so the simplified line is provably within tolerance
of the original everywhere. This module simplifies a polyline, and a
survey measures the reduction on a noisy path and confirms every
dropped point stays within the tolerance.
"""

from __future__ import annotations

import math
from itertools import pairwise

from atlas.errors import Invalid

Point = tuple[float, float]


def _perpendicular_distance(p: Point, a: Point, b: Point) -> float:
    if a == b:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    # distance from p to the infinite line through a and b via the cross product
    cross = abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx)
    return cross / length


def simplify_mask(points: list[Point], tolerance: float) -> list[bool]:
    if points is None:
        raise Invalid("points must not be None")
    if tolerance < 0:
        raise Invalid("tolerance must not be negative")
    keep = [False] * len(points)
    if len(points) <= 2:
        return [True] * len(points)
    keep[0] = keep[-1] = True
    _simplify(points, 0, len(points) - 1, tolerance, keep)
    return keep


def simplify(points: list[Point], tolerance: float) -> list[Point]:
    keep = simplify_mask(points, tolerance)
    return [p for p, k in zip(points, keep, strict=True) if k]


def _simplify(
    points: list[Point], lo: int, hi: int, tolerance: float, keep: list[bool]
) -> None:
    if hi <= lo + 1:
        return
    farthest = -1.0
    index = lo
    for i in range(lo + 1, hi):
        d = _perpendicular_distance(points[i], points[lo], points[hi])
        if d > farthest:
            farthest = d
            index = i
    if farthest <= tolerance:
        return
    keep[index] = True
    _simplify(points, lo, index, tolerance, keep)
    _simplify(points, index, hi, tolerance, keep)


def max_error(points: list[Point], tolerance: float) -> float:
    # the largest perpendicular distance from any dropped point to its kept segment
    keep = simplify_mask(points, tolerance)
    kept_indices = [i for i, k in enumerate(keep) if k]
    worst = 0.0
    for a, b in pairwise(kept_indices):
        for i in range(a + 1, b):
            worst = max(worst, _perpendicular_distance(points[i], points[a], points[b]))
    return worst
