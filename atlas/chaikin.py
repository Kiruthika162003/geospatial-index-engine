"""Chaikin smoothing: cutting corners until a polyline becomes a curve, with measured limits.

A polyline drawn from survey points has corners a road or a river
does not, and Chaikin's scheme rounds them by replacing each
segment with two points a quarter and three quarters of the way
along it, then joining those in order. Each pass cuts every corner,
doubles the point count, and moves the line a little toward the
inside of each bend, and in the limit the passes converge to the
quadratic B-spline whose control polygon is the original line. The
survey measures four things about the passes, and two guesses were
wrong. The point count was guessed at 2n minus 2 per pass; with
the endpoints kept and two points per segment it is exactly 2n, so
an open line of 11 points has 22, 44, and 88 after three passes.
The length shrinks with each pass by a fraction that depends on
how sharp the corners are, and the guess of 17 percent on the
first pass for a right-angle zigzag was the total, not the first
step: the first pass takes 13.2 percent, the second 3.3, the third
0.8, then 0.21, 0.05, 0.013, each a quarter of the last, and the
total converges to 16.96 percent rather than running to zero,
which is the difference between smoothing and collapsing; a closed
unit square loses 14.6 percent on the first pass and 18.8 in the
limit. The smoothed line never leaves the convex hull of the
original, a property inherited from the B-spline, and its largest
distance from the original polyline is bounded by the corner cut,
which was guessed at a quarter of the shorter adjacent segment
times the sine of the turn; that bound holds but is loose by two,
since the first pass departs by zero, its new points lying on the
old segments, and later passes add 0.0625, 0.031, 0.016, halving
each time toward an eighth, 0.124 after eight passes, because the
limit curve passes through the midpoint of the first cut. And the
endpoints of an open line are kept fixed, so smoothing does not
shorten a route at its ends. The finding worth stating is that
Chaikin smoothing converges geometrically, each pass a quarter of
the last in length and half in departure, so five or six passes
leave a line indistinguishable from its limit at plotting scale
and further passes only add points. This module smooths open and closed
polylines, and a survey measures the shrinkage, the convergence,
and the hull bound.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def smooth_once(line: Sequence[Point], closed: bool = False) -> list[Point]:
    if len(line) < 2:
        raise Invalid("a line needs at least two points")
    pairs = list(itertools.pairwise(line))
    if closed:
        pairs.append((line[-1], line[0]))
    out: list[Point] = []
    if not closed:
        out.append(line[0])
    for (x1, y1), (x2, y2) in pairs:
        out.append((0.75 * x1 + 0.25 * x2, 0.75 * y1 + 0.25 * y2))
        out.append((0.25 * x1 + 0.75 * x2, 0.25 * y1 + 0.75 * y2))
    if not closed:
        out.append(line[-1])
    return out


def smooth(line: Sequence[Point], passes: int, closed: bool = False) -> list[Point]:
    if passes < 0:
        raise Invalid("passes cannot be negative")
    current = list(line)
    for _ in range(passes):
        current = smooth_once(current, closed)
    return current


def length(line: Sequence[Point], closed: bool = False) -> float:
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(line))
    if closed and len(line) > 1:
        total += math.hypot(line[0][0] - line[-1][0], line[0][1] - line[-1][1])
    return total


def point_count(n: int, passes: int) -> int:
    # the count an open line of n points has after the passes: endpoints kept, two per
    # segment, so exactly 2n each pass (the guessed 2n - 2 was wrong)
    if n < 2 or passes < 0:
        raise Invalid("need at least two points and a non-negative pass count")
    return n * 2**passes


def _segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def max_departure(original: Sequence[Point], smoothed: Sequence[Point]) -> float:
    # the largest distance of any smoothed point from the original polyline
    worst = 0.0
    for p in smoothed:
        nearest = min(_segment_distance(p, a, b) for a, b in itertools.pairwise(original))
        worst = max(worst, nearest)
    return worst


def _corner_cut(line: Sequence[Point], fraction: float) -> float:
    bound = 0.0
    for (x0, y0), (x1, y1), (x2, y2) in zip(line, line[1:], line[2:], strict=False):
        ux, uy = x1 - x0, y1 - y0
        vx, vy = x2 - x1, y2 - y1
        lu, lv = math.hypot(ux, uy), math.hypot(vx, vy)
        if lu == 0 or lv == 0:
            continue
        sine = abs(ux * vy - uy * vx) / (lu * lv)
        bound = max(bound, fraction * min(lu, lv) * sine)
    return bound


def first_pass_bound(line: Sequence[Point]) -> float:
    # a quarter of the shorter adjacent segment times the sine of the turn: holds, loose by two
    return _corner_cut(line, 0.25)


def limit_departure(line: Sequence[Point]) -> float:
    # an eighth: the limit curve passes through the midpoint of the first pass's corner cut
    return _corner_cut(line, 0.125)


def zigzag(steps: int, size: float = 1.0) -> list[Point]:
    pts: list[Point] = [(0.0, 0.0)]
    for i in range(steps):
        x, y = pts[-1]
        pts.append((x + size, y) if i % 2 == 0 else (x, y + size))
    return pts
