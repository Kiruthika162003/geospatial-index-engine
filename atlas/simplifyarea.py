"""Simplifying a rough ring drops a third of its perimeter for a percent of area; none crossed.

Douglas and Peucker on a closed ring keeps the first vertex and
drops the rest within a tolerance of the chords. A clean circle of
360 vertices and radius 100, area 31,414.3 as the inscribed polygon
law says, keeps 104, 40, 32, 24, 16, 8 and 4 vertices at tolerances
0.1, 0.5, 1, 2, 5, 10 and 30 and loses 0.068, 0.49, 0.64, 1.53,
2.55, 9.96 and 36.3 percent of its area, since every chord of a
convex ring cuts inside it, while the perimeter falls only 0.017 to
10 percent. The guess that a rough ring loses area in proportion to
the vertices dropped was wrong: with radius noise of 2 the
simplifier keeps 349, 290, 241, 159, 41, 11 and 7 vertices and the
area moves by 0.003, 0.01, -0.08, -0.11, 1.29, -0.50 and -10.7
percent while the perimeter falls 0.001, 0.25, 1.37, 8.4, 32.0,
41.3 and 43.1 percent, because the dropped vertices are the noise's
own zigzag, whose excursions in and out cancel in area but add to
length. Noise 0.5 reads the same way, 72 vertices at tolerance 1
for 0.17 percent of area and 4.0 percent of perimeter.

The guess that an aggressive tolerance makes rings cross themselves
was wrong for forty rough rings of noise 3: none crossed at
tolerances of 1, 3, 5, 10 or 20, down to 7.8 vertices a ring. An
eight-pointed star keeps all 16 vertices through a tolerance of 30,
since each spike stands 52 off the chord between its neighbours,
and at 60 collapses to 4 vertices with 63 percent more area, the
square through the outer points.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid
from atlas.vectortile import simplify

Point = tuple[float, float]
Ring = list[Point]


def area(ring: Ring) -> float:
    if len(ring) < 3:
        raise Invalid("a ring needs three vertices")
    total = 0.0
    for i, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(i + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def perimeter(ring: Ring) -> float:
    return sum(math.dist(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring)))


def simplify_ring(ring: Ring, tolerance: float) -> Ring:
    # keep the first vertex fixed, run the line simplifier over the closed walk, drop the repeat
    if len(ring) < 4:
        return list(ring)
    closed = [*ring, ring[0]]
    kept = simplify(closed, tolerance)
    return kept[:-1]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def self_intersects(ring: Ring) -> bool:
    n = len(ring)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(ring[i], ring[(i + 1) % n], ring[j], ring[(j + 1) % n]):
                return True
    return False


def report(ring: Ring, tolerance: float) -> dict[str, float]:
    kept = simplify_ring(ring, tolerance)
    return {
        "vertices": float(len(kept)),
        "area_change": area(kept) / area(ring) - 1 if len(kept) >= 3 else -1.0,
        "perimeter_change": perimeter(kept) / perimeter(ring) - 1 if len(kept) >= 3 else -1.0,
        "crossed": float(len(kept) >= 4 and self_intersects(kept)),
    }


def noisy_circle(n: int, radius: float, noise: float, rng: random.Random) -> Ring:
    if n < 3:
        raise Invalid("a ring needs three vertices")
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        r = radius + rng.gauss(0, noise)
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def star(n: int, outer: float, inner: float) -> Ring:
    out = []
    for k in range(2 * n):
        r = outer if k % 2 == 0 else inner
        a = math.pi * k / n
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def crossing_rate(rings: list[Ring], tolerance: float) -> float:
    if not rings:
        raise Invalid("no rings")
    return sum(report(r, tolerance)["crossed"] for r in rings) / len(rings)


def polygon_area_law(n: int, radius: float) -> float:
    # a regular n-gon inscribed in a circle
    return 0.5 * n * radius * radius * math.sin(2 * math.pi / n)
