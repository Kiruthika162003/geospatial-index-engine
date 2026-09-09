"""A hash grid examines (2r + c) squared over pi r squared points a hit; the best cell is cost.

A spatial hash drops each point into the bucket of the cell that
holds it, and a radius query scans every bucket the query square
touches. On 20,000 uniform points over a 1000 square, density 0.02,
the candidates examined per hit read 1.57 to 1.63 at a cell of a
quarter radius, 1.96 to 2.01 at half, 2.81 to 2.83 at the radius,
4.84 to 5.01 at twice and 10.4 to 11.6 at four times, for radii of 5
to 40, against the law (2r + c) squared over pi r squared of 1.61,
1.99, 2.86, 5.09 and 11.46; the buckets touched read 81, 25, 9, 4 and
2.2 to 2.3 against (1 + 2r/c) squared of 81, 25, 9, 4 and 2.25. The
guess that the cheapest cell is the radius itself was wrong on both
sides: counting candidates alone, the smallest cell always wins,
since a quarter-radius cell examines the fewest points, and counting
a bucket lookup as one distance test, the best cell is twice the
radius at radius 5, the radius at 10 and half the radius at 40,
because the number of points a cell holds grows with the radius
while the number of buckets does not. The cost law names the same
cell as the measurement in all nine cases tried.

Buckets of a 10-cell over the uniform points hold 2.30 points on
average and 9 at most, with 2763, 2708, 1826, 917, 336, 122, 27 and 7
buckets holding 1 to 8, the Poisson counts for a mean of 2; twenty
Gaussian clusters put 20,000 points into 1883 buckets of 10.6 on
average and 86 at most, and a radius 10 query there examines 9.2,
11.5, 16.3, 27.6 and 63.0 candidates for 5.71 hits at cells of 2.5
to 40. The hash returned exactly the brute-force set on 100 queries.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Key = tuple[int, int]


class SpatialHash:
    def __init__(self, cell: float) -> None:
        if cell <= 0:
            raise Invalid("the cell size must be positive")
        self.cell = cell
        self.buckets: dict[Key, list[Point]] = {}
        self.size = 0

    def key(self, p: Point) -> Key:
        return math.floor(p[0] / self.cell), math.floor(p[1] / self.cell)

    def insert(self, p: Point) -> None:
        self.buckets.setdefault(self.key(p), []).append(p)
        self.size += 1

    def within(self, centre: Point, radius: float) -> tuple[list[Point], int, int]:
        if radius < 0:
            raise Invalid("the radius must not be negative")
        kx0, ky0 = self.key((centre[0] - radius, centre[1] - radius))
        kx1, ky1 = self.key((centre[0] + radius, centre[1] + radius))
        found = []
        candidates = 0
        cells = 0
        r2 = radius * radius
        for kx in range(kx0, kx1 + 1):
            for ky in range(ky0, ky1 + 1):
                cells += 1
                for p in self.buckets.get((kx, ky), ()):
                    candidates += 1
                    if (p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2 <= r2:
                        found.append(p)
        return found, candidates, cells

    def occupancy(self) -> tuple[float, int]:
        if not self.buckets:
            raise Invalid("the hash is empty")
        sizes = [len(b) for b in self.buckets.values()]
        return sum(sizes) / len(sizes), max(sizes)


def build(points: list[Point], cell: float) -> SpatialHash:
    grid = SpatialHash(cell)
    for p in points:
        grid.insert(p)
    return grid


def brute_within(points: list[Point], centre: Point, radius: float) -> list[Point]:
    r2 = radius * radius
    return [p for p in points if (p[0] - centre[0]) ** 2 + (p[1] - centre[1]) ** 2 <= r2]


Cost = tuple[float, float, float]


def query_cost(grid: SpatialHash, queries: list[Point], radius: float) -> Cost:
    if not queries:
        raise Invalid("at least one query is needed")
    hits = candidates = cells = 0
    for q in queries:
        found, seen, touched = grid.within(q, radius)
        hits += len(found)
        candidates += seen
        cells += touched
    n = len(queries)
    return hits / n, candidates / n, cells / n


def cost_curve(
    points: list[Point], queries: list[Point], radius: float, cells: list[float]
) -> dict[float, Cost]:
    return {cell: query_cost(build(points, cell), queries, radius) for cell in cells}


def best_cell(curve: dict[float, Cost], cell_cost: float = 0.0) -> float:
    if cell_cost < 0:
        raise Invalid("the cell cost must not be negative")
    return min(curve, key=lambda c: curve[c][1] + cell_cost * curve[c][2])


def expected_cells(radius: float, cell: float) -> float:
    return (1 + 2 * radius / cell) ** 2


def expected_candidates(density: float, radius: float, cell: float) -> float:
    return density * (cell + 2 * radius) ** 2


def candidate_ratio(radius: float, cell: float) -> float:
    return (cell + 2 * radius) ** 2 / (math.pi * radius * radius)


def cheapest_cell(density: float, radius: float, cell_cost: float, cells: list[float]) -> float:
    if not cells:
        raise Invalid("offer at least one cell size")
    return min(
        cells,
        key=lambda c: expected_candidates(density, radius, c)
        + cell_cost * expected_cells(radius, c),
    )


def uniform(n: int, rng: random.Random, side: float = 1000.0) -> list[Point]:
    return [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]


def clustered(
    n: int, clusters: int, rng: random.Random, spread: float = 20.0, side: float = 1000.0
) -> list[Point]:
    centres = [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(clusters)]
    out = []
    for i in range(n):
        cx, cy = centres[i % clusters]
        out.append((rng.gauss(cx, spread), rng.gauss(cy, spread)))
    return out


def bucket_histogram(grid: SpatialHash) -> dict[int, int]:
    hist: dict[int, int] = {}
    for bucket in grid.buckets.values():
        hist[len(bucket)] = hist.get(len(bucket), 0) + 1
    return dict(sorted(hist.items()))
