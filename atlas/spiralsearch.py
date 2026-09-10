"""Stopping at the first ring with a point is wrong 31 percent of the time at 3 points a cell.

A grid index answers a nearest-neighbour query by searching the
query's cell and then the rings of cells around it, and the tempting
rule is to stop at the first ring that holds any point. On 2000
uniform points over a 1000 square with 1000 queries that rule
returns the wrong point 4.6, 5.9, 14.7, 31.1 and 18.1 percent of the
time at cells of 5, 10, 20, 40 and 80, holding 0.05, 0.2, 0.8, 3.2
and 12.8 points each, since a point in the next ring can lie closer
than one in the ring that stopped the search, and worst of all when
the query's own cell almost always holds a point that is not the
nearest. The guess that a denser cell makes the rule safer was wrong
until the cell is crowded: the error peaks at 3.2 points a cell and
falls only to 18 percent at 12.8. The safe rule keeps searching
while the best distance found exceeds the reach the rings so far
guarantee, k cells, and is never wrong; it costs 2.72, 1.61, 1.08,
1.0 and 1.0 rings against the naive 1.95, 0.98, 0.43, 0.03 and 0.0,
always at least one, since a point in the query's own cell never
rules out a closer one next door. The naive rule's mean ring
matches the Poisson first-hit law, 1.98, 0.99, 0.45, 0.04 and 0.0,
and ring k holds 8k cells.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Key = tuple[int, int]


class GridIndex:
    def __init__(self, points: list[Point], cell: float) -> None:
        if cell <= 0:
            raise Invalid("the cell must be positive")
        self.cell = cell
        self.buckets: dict[Key, list[Point]] = {}
        for p in points:
            self.buckets.setdefault(self.key(p), []).append(p)

    def key(self, p: Point) -> Key:
        return math.floor(p[0] / self.cell), math.floor(p[1] / self.cell)

    def ring(self, centre: Key, k: int) -> list[Key]:
        cx, cy = centre
        if k == 0:
            return [centre]
        out = []
        for dx in range(-k, k + 1):
            out.append((cx + dx, cy - k))
            out.append((cx + dx, cy + k))
        for dy in range(-k + 1, k):
            out.append((cx - k, cy + dy))
            out.append((cx + k, cy + dy))
        return out

    def nearest_naive_stop(
        self, query: Point, max_rings: int = 1000
    ) -> tuple[Point | None, int]:
        # stop at the first ring holding any point and return its nearest: the tempting rule
        centre = self.key(query)
        for k in range(max_rings + 1):
            found = [p for key in self.ring(centre, k) for p in self.buckets.get(key, ())]
            if found:
                return min(found, key=lambda p: math.dist(p, query)), k
        return None, max_rings

    def nearest_safe_stop(
        self, query: Point, max_rings: int = 1000
    ) -> tuple[Point | None, int]:
        # keep searching while the best distance exceeds the reach the rings so far guarantee
        centre = self.key(query)
        best: Point | None = None
        best_d = math.inf
        for k in range(max_rings + 1):
            for key in self.ring(centre, k):
                for p in self.buckets.get(key, ()):
                    d = math.dist(p, query)
                    if d < best_d:
                        best, best_d = p, d
            if best is not None and best_d <= k * self.cell:
                return best, k
        return best, max_rings


def brute_nearest(points: list[Point], query: Point) -> Point:
    return min(points, key=lambda p: math.dist(p, query))


def error_rate(points: list[Point], cell: float, queries: list[Point]) -> dict[str, float]:
    if not queries:
        raise Invalid("at least one query is needed")
    index = GridIndex(points, cell)
    naive_wrong = safe_wrong = 0
    naive_rings = safe_rings = 0
    for q in queries:
        truth = brute_nearest(points, q)
        naive, kn = index.nearest_naive_stop(q)
        safe, ks = index.nearest_safe_stop(q)
        naive_wrong += naive != truth
        safe_wrong += safe != truth
        naive_rings += kn
        safe_rings += ks
    n = len(queries)
    return {
        "naive_wrong": naive_wrong / n,
        "safe_wrong": safe_wrong / n,
        "naive_rings": naive_rings / n,
        "safe_rings": safe_rings / n,
    }


def uniform(n: int, rng: random.Random, side: float = 1000.0) -> list[Point]:
    return [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]


def expected_first_ring(density: float, cell: float) -> float:
    # the mean ring of the first hit for Poisson points
    per_cell = density * cell * cell
    total = 0.0
    survive = 1.0
    for k in range(200):
        cells = 1 if k == 0 else 8 * k
        hit = 1 - math.exp(-per_cell * cells)
        total += k * survive * hit
        survive *= 1 - hit
    return total
