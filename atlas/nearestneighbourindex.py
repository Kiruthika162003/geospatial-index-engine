"""Nearest neighbour index: one number for clustered, random, or regular, and its ceiling.

The Clark-Evans index compresses a point pattern into a single
ratio: the mean distance from each point to its nearest neighbour,
divided by the mean that a random scatter of the same density
would have, which is one over twice the square root of the density.
A ratio of one says random, below one clustered, above one
regular, and the ceiling is set by the most regular arrangement
possible, a hexagonal lattice, where every point sits at the same
distance from six neighbours and the ratio is 2.149. The survey
calibrates the index on known patterns. A uniform random scatter
in the unit square reads above one, since points near the edge
have no neighbours beyond it and their nearest is farther than it
would be in an unbounded scatter; the guess was that this bias
fades as one over the square root of the count, and over ten
seeds each it did not fall so cleanly: 3.1 percent at 100 points,
3.2 at 400, and 0.75 at 1600, while the spread between seeds did
shrink, 0.17, 0.13, 0.047. A square grid of 400 points reads 2.0
exactly when the area is the lattice's own, one cell per point,
and 2.105 when the area is taken as the bounding box, which
crowds the density by the missing half-cell margin; a hexagonal
lattice reads the 2.149 ceiling to six figures. Families of ten
points spread 0.001, 0.005, and 0.02 round fifty random centers
read 0.032, 0.149, and 0.572, the nearest neighbour being a
sibling a small fraction of the random spacing away. And the
index is scale-free: doubling every coordinate and quadrupling
the area leaves it unchanged to the last digit. The finding worth
stating is that the index hits 2.0 on a square grid and 2.149 on
a hexagonal one exactly, drops below one in proportion to how
tightly families cluster, and carries an edge bias of a few
percent on random scatters that the area convention moves more
than the count does, so it is a fair single-number summary whose
weaknesses are the area it is handed and that it cannot say at
what scale the clustering lives. This
module computes the index and its expectations, and a survey
calibrates it on random, grid, hexagonal, and clustered patterns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]
HEXAGONAL_CEILING = 2.0 / math.sqrt(math.sqrt(3.0) / 2.0)  # 2.149, six equidistant neighbours


def nearest_distances(points: Sequence[Point]) -> list[float]:
    n = len(points)
    if n < 2:
        raise Invalid("nearest neighbours need at least two points")
    out = []
    for i, (xi, yi) in enumerate(points):
        best = math.inf
        for j, (xj, yj) in enumerate(points):
            if i != j:
                d = math.hypot(xi - xj, yi - yj)
                best = min(best, d)
        out.append(best)
    return out


def expected_random_distance(n: int, area: float) -> float:
    if n < 1 or area <= 0:
        raise Invalid("need a positive count and area")
    return 0.5 / math.sqrt(n / area)


def index(points: Sequence[Point], area: float) -> float:
    distances = nearest_distances(points)
    return (sum(distances) / len(distances)) / expected_random_distance(len(points), area)


def square_grid(side: int, spacing: float = 1.0) -> list[Point]:
    return [(i * spacing, j * spacing) for i in range(side) for j in range(side)]


def hexagonal_lattice(side: int, spacing: float = 1.0) -> list[Point]:
    # rows offset by half a spacing, rows spaced by spacing times sqrt(3) / 2
    return [
        (i * spacing + (0.5 * spacing if j % 2 else 0.0), j * spacing * math.sqrt(3) / 2)
        for i in range(side)
        for j in range(side)
    ]


def lattice_area(points: Sequence[Point], spacing: float, hexagonal: bool) -> float:
    # the area each point owns in an infinite lattice times the count: the edge-free area
    cell = spacing * spacing * (math.sqrt(3) / 2 if hexagonal else 1.0)
    return cell * len(points)


def families(count: int, size: int, spread: float, rng) -> list[Point]:
    out: list[Point] = []
    for _ in range(count):
        cx, cy = rng.uniform(0, 1), rng.uniform(0, 1)
        out.extend((cx + rng.gauss(0, spread), cy + rng.gauss(0, spread)) for _ in range(size))
    return out
