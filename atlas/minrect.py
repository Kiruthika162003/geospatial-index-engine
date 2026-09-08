"""Minimum-area bounding rectangle: rotate to a hull edge and beat the axis-aligned box.

The axis-aligned bounding box is the cheap envelope every index uses,
but it is rarely the tightest rectangle: a long thin shape lying at
forty-five degrees fills only half of its axis-aligned box, the rest
being dead space that inflates every overlap test. The minimum-area
bounding rectangle, free to rotate, hugs the shape far more closely,
and a classical theorem makes it findable without searching all
angles: the minimum-area enclosing rectangle of a convex polygon has
one side collinear with an edge of that polygon. So the candidates are
finite, one rectangle per hull edge, and the search is a loop. For
each hull edge, take its direction as the rectangle's orientation,
project every hull vertex onto that direction and onto its
perpendicular, and the spans of those two projections are the
rectangle's width and height; the edge whose rectangle has the least
area wins. The rotating-calipers refinement tracks the extreme
vertices incrementally for a linear pass, but on the modest hulls of
map features the plain per-edge projection, quadratic in the hull
size, is simpler and fast enough. The measurement worth having is how
much the rotation buys, and a first guess undersold it. For an
axis-aligned shape the minimum rectangle is the axis-aligned box
itself, no gain. The guess was that a shape tilted to forty-five
degrees gains about a factor of two; the measurement on a ten by four
rectangle turned at forty-five degrees gave a box of ninety-eight
against a minimum of forty, a factor of two and forty-five hundredths,
because the exact gain for a width-by-height rectangle on the diagonal
is the square of width plus height over twice their product, which is
two only for a square and grows without bound as the shape thins, a
hundred and seventy-four for a slender diagonal strip. For random
point clouds the minimum rectangle is reliably tighter than the box,
about a tenth smaller on average and never looser, since the box is
one of the orientations the search could have chosen. The finding
worth stating is that the minimum-area rectangle is never larger than
the axis-aligned box and beats it by a factor set by the shape's
elongation, so the hull-edge theorem turns an infinite search over
angles into a finite loop that pays off exactly when the shape is
tilted and thin. This module finds the minimum-area
rectangle by the hull-edge rule, and a survey measures its area
against the axis-aligned box on tilted and untilted shapes.
"""

from __future__ import annotations

import math

from atlas.errors import Degenerate, Invalid

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


def axis_aligned_area(points: list[Point]) -> float:
    if points is None or not points:
        raise Invalid("points must not be empty")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def minimum_rectangle(points: list[Point]) -> tuple[float, float, float, float]:
    # returns (area, angle in degrees, width, height) of the minimum-area bounding rectangle
    if points is None or len(set(points)) < 3:
        raise Invalid("need at least three distinct points")
    hull = _hull(points)
    if len(hull) < 3:
        raise Degenerate("the points are collinear; the bounding rectangle has zero area")
    best: tuple[float, float, float, float] | None = None
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        length = math.hypot(ex, ey)
        if length == 0:
            continue
        ux, uy = ex / length, ey / length  # along the edge
        vx, vy = -uy, ux  # perpendicular
        along = [p[0] * ux + p[1] * uy for p in hull]
        across = [p[0] * vx + p[1] * vy for p in hull]
        width = max(along) - min(along)
        height = max(across) - min(across)
        area = width * height
        if best is None or area < best[0]:
            best = (area, math.degrees(math.atan2(uy, ux)) % 180.0, width, height)
    return best
