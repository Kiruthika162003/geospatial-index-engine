"""A polygon grid casts a ray for 5 percent of queries at 64 cells a side, 19 times faster.

A point-in-polygon grid classes every cell of a square grid over the
polygon's box as wholly inside, wholly outside, or crossed by the
boundary, and only a query landing in a crossed cell casts a ray.
For a 64-gon circle of radius 100 the crossed cells number 12, 28,
60, 124 and 252 at 4, 8, 16, 32 and 64 cells a side, fractions
0.75, 0.4375, 0.234, 0.121 and 0.0615 of the grid, about a quarter
above the law of perimeter times cells over extent from 8 cells on,
0.393 to 0.049, since a curve crossing a cell obliquely touches its
neighbours too; at 4 cells the law's 0.785 overshoots the 0.75.
Of 20,000 uniform query points, 62, 36, 19.5, 10.0 and 5.15 percent
cast a ray, which tracks the law from 8 cells on rather than the
cell count, since
the points are uniform over the box and not over the cells. Every
answer matched a plain ray cast, 12,996 inside for the 64-gon and
13,013 for a 512-gon.

The guess that the grid's gain is modest for a simple polygon was
wrong for a detailed one: on the 512-gon the 20,000 queries take
0.97 seconds by ray casting alone and 0.052 seconds through the
64-cell grid, 19 times faster, at a build cost of 0.41 seconds; the
64-gon runs 0.015 seconds through the same grid. An eight-pointed
star, perimeter 1178, is crossed in every cell at 4 a side and in
0.625, 0.375, 0.199 and 0.1035 of the cells at 8 to 64, casting rays
for 83, 52, 31, 16 and 8.4 percent of the queries.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Ring = list[Point]


def ray_cast(ring: Ring, x: float, y: float) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < cross:
                inside = not inside
    return inside


def _segments_cross_box(ring: Ring, x0: float, y0: float, x1: float, y1: float) -> bool:
    n = len(ring)
    for i in range(n):
        ax, ay = ring[i]
        bx, by = ring[(i + 1) % n]
        if max(ax, bx) < x0 or min(ax, bx) > x1 or max(ay, by) < y0 or min(ay, by) > y1:
            continue
        if x0 <= ax <= x1 and y0 <= ay <= y1:
            return True
        if x0 <= bx <= x1 and y0 <= by <= y1:
            return True
        for ex0, ey0, ex1, ey1 in (
            (x0, y0, x1, y0),
            (x1, y0, x1, y1),
            (x1, y1, x0, y1),
            (x0, y1, x0, y0),
        ):
            if _cross(ax, ay, bx, by, ex0, ey0, ex1, ey1):
                return True
    return False


def _orient(ax, ay, bx, by, cx, cy) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _cross(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    d1, d2 = _orient(cx, cy, dx, dy, ax, ay), _orient(cx, cy, dx, dy, bx, by)
    d3, d4 = _orient(ax, ay, bx, by, cx, cy), _orient(ax, ay, bx, by, dx, dy)
    return d1 * d2 < 0 and d3 * d4 < 0


class PolygonGrid:
    INSIDE, OUTSIDE, BOUNDARY = 1, 0, 2

    def __init__(self, ring: Ring, cells: int) -> None:
        if len(ring) < 3:
            raise Invalid("a polygon needs three vertices")
        if cells < 1:
            raise Invalid("the grid needs at least one cell a side")
        self.ring = ring
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        self.cells = cells
        self.dx = (self.max_x - self.min_x) / cells or 1.0
        self.dy = (self.max_y - self.min_y) / cells or 1.0
        self.kind = [[self.OUTSIDE] * cells for _ in range(cells)]
        for i in range(cells):
            for j in range(cells):
                x0, y0 = self.min_x + i * self.dx, self.min_y + j * self.dy
                x1, y1 = x0 + self.dx, y0 + self.dy
                if _segments_cross_box(ring, x0, y0, x1, y1):
                    self.kind[i][j] = self.BOUNDARY
                elif ray_cast(ring, (x0 + x1) / 2, (y0 + y1) / 2):
                    self.kind[i][j] = self.INSIDE

    def counts(self) -> tuple[int, int, int]:
        flat = [k for row in self.kind for k in row]
        return flat.count(self.INSIDE), flat.count(self.OUTSIDE), flat.count(self.BOUNDARY)

    def contains(self, x: float, y: float) -> tuple[bool, bool]:
        # the answer and whether a ray had to be cast
        if not (self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y):
            return False, False
        i = min(int((x - self.min_x) / self.dx), self.cells - 1)
        j = min(int((y - self.min_y) / self.dy), self.cells - 1)
        kind = self.kind[i][j]
        if kind == self.BOUNDARY:
            return ray_cast(self.ring, x, y), True
        return kind == self.INSIDE, False


def query_batch(grid: PolygonGrid, points: list[Point]) -> tuple[int, int]:
    if not points:
        raise Invalid("at least one point is needed")
    inside = rays = 0
    for x, y in points:
        answer, cast = grid.contains(x, y)
        inside += answer
        rays += cast
    return inside, rays


def circle(n: int, radius: float = 100.0) -> Ring:
    return [
        (radius * math.cos(2 * math.pi * k / n), radius * math.sin(2 * math.pi * k / n))
        for k in range(n)
    ]


def star(n: int, outer: float = 100.0, inner: float = 40.0) -> Ring:
    ring = []
    for k in range(2 * n):
        r = outer if k % 2 == 0 else inner
        angle = math.pi * k / n
        ring.append((r * math.cos(angle), r * math.sin(angle)))
    return ring


def random_points(count: int, rng: random.Random, span: float = 110.0) -> list[Point]:
    return [(rng.uniform(-span, span), rng.uniform(-span, span)) for _ in range(count)]


def boundary_fraction_law(cells: int, perimeter: float, extent: float) -> float:
    # cells crossed by a curve of length L on an n by n grid over extent E: about L n / E
    return min(1.0, perimeter * cells / extent / (cells * cells))
