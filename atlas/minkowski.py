"""Minkowski sum: the shape a convex body sweeps when dragged round another, with area laws.

The Minkowski sum of two shapes is every point that is a point of
the first plus a point of the second, and it answers two map
questions at once: the region a vehicle of a given footprint
cannot enter without hitting an obstacle is the obstacle summed
with the footprint reflected, and the buffer of a polygon is the
polygon summed with a disc. For convex polygons the sum is
computed by merging their edges in angular order, since the
sum's boundary is made of the edges of both, each used once, in
the order of their directions; the result is convex, has at most
as many vertices as the two together, and is computed in linear
time after the two are put in counterclockwise order from their
lowest points. The survey measures the identities the sum must
obey. The area of the sum of two convex polygons is the area of
the first plus the area of the second plus twice their mixed
area, and for a polygon summed with itself the mixed area is its
own area, so the sum of P with P is 2P and has four times the
area: a triangle of area 6 summed with itself gave 24 with a
mixed area of 6, exactly. The guess for the sum with a square of
side s was the perimeter times half the side plus the square's
area, and it was wrong: the triangle with vertices (0,0), (4,0),
(1,3) summed with the unit square gave area 14.0 where the guess
said 12.70, because twice the mixed area with an axis-aligned
square is the side times the sum of the polygon's extents in x
and y, 1 times 4 plus 3, not half its perimeter; the hexagon of
radius 2 gave 18.856, its area 10.392 plus 1 times 4 plus 3.464
plus 1, confirming the corrected law. The vertex count of the sum
of an n-gon and an m-gon is n plus m when no two edges are
parallel and fewer when some are: the triangle and the square
gave 6 of a possible 7, one edge pair being parallel, the hexagon
and the square 8 of 10, and a square with a square 4, not 8. The
sum was commutative to the vertex and translating a summand
translated the sum within 4e-16. And the sum of a polygon with a
regular polygon of many sides approximates its buffer: the
triangle summed with a regular polygon of radius 1.5 fell short
of the true buffer area, area plus perimeter times r plus pi r
squared, by 3.14 percent with 8 sides, 0.64 with 16, 0.19 with
32, 0.043 with 64, and 0.0024 with 256, quartering per doubling
of the side count as the inscribed polygon closes on the disc.
The finding worth stating is that the convex Minkowski sum obeys
the mixed-area law exactly, P plus P quadrupling the area and a
square adding the side times the extents, and approximates a
buffer with a 64-gon to within 0.04 percent, so the buffer is a
Minkowski sum in disguise. This
module computes convex Minkowski sums, and a survey measures the
area laws, the vertex counts, and the buffer approximation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Degenerate, Invalid
from atlas.shoelace import signed_area

Point = tuple[float, float]


def _counterclockwise_from_lowest(polygon: Sequence[Point]) -> list[Point]:
    if len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    pts = list(polygon)
    if signed_area(pts) < 0:
        pts.reverse()
    if signed_area(pts) == 0:
        raise Degenerate("the polygon has no area")
    start = min(range(len(pts)), key=lambda i: (pts[i][1], pts[i][0]))
    return pts[start:] + pts[:start]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def is_convex(polygon: Sequence[Point]) -> bool:
    pts = list(polygon)
    n = len(pts)
    sign = 0
    for i in range(n):
        c = _cross(pts[i], pts[(i + 1) % n], pts[(i + 2) % n])
        if c != 0:
            if sign == 0:
                sign = 1 if c > 0 else -1
            elif (c > 0) != (sign > 0):
                return False
    return True


def minkowski_sum(p: Sequence[Point], q: Sequence[Point]) -> list[Point]:
    if not is_convex(p) or not is_convex(q):
        raise Invalid("the edge-merge sum needs convex polygons")
    a = _counterclockwise_from_lowest(p)
    b = _counterclockwise_from_lowest(q)
    a.extend([a[0], a[1]])
    b.extend([b[0], b[1]])
    out: list[Point] = []
    i = j = 0
    while i < len(a) - 2 or j < len(b) - 2:
        out.append((a[i][0] + b[j][0], a[i][1] + b[j][1]))
        cross = (a[i + 1][0] - a[i][0]) * (b[j + 1][1] - b[j][1]) - (a[i + 1][1] - a[i][1]) * (
            b[j + 1][0] - b[j][0]
        )
        if cross >= 0 and i < len(a) - 2:
            i += 1
        if cross <= 0 and j < len(b) - 2:
            j += 1
    return _dedupe(out)


def _dedupe(points: list[Point]) -> list[Point]:
    # drop repeated and collinear vertices
    out: list[Point] = []
    for p in points:
        if not out or math.dist(out[-1], p) > 1e-12:
            out.append(p)
    if len(out) > 1 and math.dist(out[0], out[-1]) < 1e-12:
        out.pop()
    cleaned: list[Point] = []
    n = len(out)
    for i in range(n):
        if abs(_cross(out[i - 1], out[i], out[(i + 1) % n])) > 1e-12:
            cleaned.append(out[i])
    return cleaned


def area(polygon: Sequence[Point]) -> float:
    return abs(signed_area(list(polygon)))


def perimeter(polygon: Sequence[Point]) -> float:
    n = len(polygon)
    return sum(math.dist(polygon[i], polygon[(i + 1) % n]) for i in range(n))


def mixed_area(p: Sequence[Point], q: Sequence[Point]) -> float:
    # V(P, Q) from area(P + Q) = area(P) + 2 V(P, Q) + area(Q)
    return (area(minkowski_sum(p, q)) - area(p) - area(q)) / 2.0


def regular(sides: int, radius: float) -> list[Point]:
    if sides < 3:
        raise Invalid("a polygon needs at least three sides")
    return [
        (radius * math.cos(2 * math.pi * k / sides), radius * math.sin(2 * math.pi * k / sides))
        for k in range(sides)
    ]


def buffer_area(polygon: Sequence[Point], radius: float) -> float:
    # the exact area of a convex polygon's buffer: area + perimeter r + pi r^2
    return area(polygon) + perimeter(polygon) * radius + math.pi * radius * radius
