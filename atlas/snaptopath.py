"""Snap to path: the nearest point on a polyline, which is rarely a vertex.

Matching a GPS fix to a road, a sample to a track, or a click to a
drawn route is a snap: find the point on the polyline nearest to a
query point, the distance to it, and where along the path it sits. The
tempting shortcut snaps to the nearest vertex, which is a hash lookup
away with a spatial index and needs no geometry. The right answer
snaps to the nearest point on the nearest segment, which may lie
anywhere along a segment's length, and the two differ by an amount the
survey measures rather than assumes. For each segment, the foot of the
perpendicular from the query is found by projecting the query onto the
segment's direction and clamping the projection parameter to the
segment's extent, so a query beyond either end snaps to that endpoint
and a query beside the segment snaps to its interior; the nearest such
foot over all segments is the snap, and its along-path position is the
cumulative length up to its segment plus the clamped parameter times
that segment's length. Vertex snapping is wrong in proportion to
segment length: a query sitting beside the middle of a long straight
segment is far from both of that segment's vertices but adjacent to
the segment itself, so vertex snapping overstates the distance by up
to half the segment's length and misplaces the along-path position by
the same. On a sparsely-vertexed path that error dominates; on a
densely-sampled one it shrinks toward zero, which is why some systems
densify a path before vertex snapping. The finding worth stating is
that segment snapping never reports a larger distance than vertex
snapping, and the gap between them is bounded by half the longest
segment and reaches it beside a long straight run, so vertex snapping
is an approximation whose error is set by the path's coarseness. This
module snaps to the nearest segment and exposes the vertex snap beside
it, and a survey measures the gap against the half-segment bound.
"""

from __future__ import annotations

import math
from itertools import pairwise

from atlas.errors import Invalid

Point = tuple[float, float]


def _foot(p: Point, a: Point, b: Point) -> tuple[Point, float, float]:
    # nearest point on segment ab, its distance, and the clamped parameter t in [0, 1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return a, math.hypot(p[0] - a[0], p[1] - a[1]), 0.0
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2
    t = max(0.0, min(1.0, t))
    foot = (a[0] + t * dx, a[1] + t * dy)
    return foot, math.hypot(p[0] - foot[0], p[1] - foot[1]), t


def snap(point: Point, path: list[Point]) -> tuple[Point, float, float]:
    # returns (nearest point on the path, distance to it, distance along the path)
    if path is None or len(path) < 2:
        raise Invalid("a path needs at least two vertices")
    best: tuple[Point, float, float] | None = None
    along = 0.0
    for a, b in pairwise(path):
        foot, dist, t = _foot(point, a, b)
        seg_len = math.hypot(b[0] - a[0], b[1] - a[1])
        if best is None or dist < best[1]:
            best = (foot, dist, along + t * seg_len)
        along += seg_len
    return best


def snap_to_vertex(point: Point, path: list[Point]) -> tuple[Point, float]:
    if path is None or not path:
        raise Invalid("a path needs at least one vertex")
    nearest = min(path, key=lambda v: math.hypot(point[0] - v[0], point[1] - v[1]))
    return nearest, math.hypot(point[0] - nearest[0], point[1] - nearest[1])


def longest_segment(path: list[Point]) -> float:
    if path is None or len(path) < 2:
        raise Invalid("a path needs at least two vertices")
    return max(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(path))
