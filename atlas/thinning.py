"""Point thinning: keeping a representative subset of a dense point set, by grid and distance.

A survey of a million tree positions, a lidar tile, a crowd of
check-ins: the set is too dense to draw or to analyse at every
scale, and thinning keeps a subset that stands for the whole.
Grid thinning keeps one point per cell, the first or the one
nearest the cell's center, and is linear and blind to the
pattern; distance thinning walks the points and keeps each one
that is at least a minimum distance from every point already
kept, a greedy Poisson-disc sample that honours a spacing rather
than a grid. The survey measures what each keeps on 5000 uniform
random points in the unit square, and three of its guesses were
wrong. Grid thinning keeps exactly one point per occupied cell,
2147 of 2500 cells at a cell of 0.02 and all 400 and 100 at 0.05
and 0.1, as guessed; but the guess that its kept set would look
as random as the input was wrong, since one point per cell is
already a regular arrangement: the nearest-neighbour index of the
kept set read 1.25, 1.30, and 1.32 at the three cell sizes
against the input's 1.00, and keeping the point nearest each
cell's center read 1.36, 1.71, and 1.86, nearly as regular as
distance thinning's 1.60, 1.75, and 1.78, which were themselves
above the 1.4 to 1.5 the guess expected. The kept count under
distance thinning is set by the packing: a spacing s on a unit
square holds at most 1.15 over s squared points, the hexagonal
bound, and the greedy walk kept 0.59 of it at a spacing of 0.1,
0.55 at 0.05, and only 0.41 at 0.02, where the 5000 candidates,
two per cell, ran out before the packing filled, so the 0.55 the
guess expected is a property of a well-supplied walk; the walk's
order moved the kept count by 2.5 percent, 245 to 258 over five
shuffles, and no two kept points were ever closer than the
spacing. On thirty clusters of 100 points spread 0.01 the guess
was reversed: distance thinning at a spacing of 0.05 kept 29,
about one per cluster, while grid thinning kept 85, since a
cluster straddles two to four cells, and at 0.02 the counts were
122 and 247. The finding worth stating is that grid thinning
keeps the occupied-cell count in a set already regular at 1.3
and distance thinning keeps a set regular at 1.6 to 1.8 at 41 to
59 percent of the hexagonal bound depending on supply, and that
on clusters the grid keeps the more, so the two methods answer
different questions, how many cells are occupied and how many
points fit at a spacing. This module thins by grid and by
distance, and a survey measures the kept counts and the
regularity of what is kept.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def grid_thin(
    points: Sequence[Point], cell: float, nearest_center: bool = False
) -> list[Point]:
    if cell <= 0:
        raise Invalid("the cell size must be positive")
    kept: dict[tuple[int, int], Point] = {}
    for p in points:
        key = (math.floor(p[0] / cell), math.floor(p[1] / cell))
        if key not in kept:
            kept[key] = p
        elif nearest_center:
            center = ((key[0] + 0.5) * cell, (key[1] + 0.5) * cell)
            if math.dist(p, center) < math.dist(kept[key], center):
                kept[key] = p
    return list(kept.values())


def distance_thin(points: Sequence[Point], spacing: float) -> list[Point]:
    # greedy: keep each point at least the spacing from every kept point, via a cell index
    if spacing <= 0:
        raise Invalid("the spacing must be positive")
    cells: dict[tuple[int, int], list[Point]] = {}
    kept: list[Point] = []
    for p in points:
        cx, cy = math.floor(p[0] / spacing), math.floor(p[1] / spacing)
        blocked = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in cells.get((cx + dx, cy + dy), []):
                    if math.dist(p, q) < spacing:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                break
        if not blocked:
            kept.append(p)
            cells.setdefault((cx, cy), []).append(p)
    return kept


def nearest_neighbour_index(points: Sequence[Point], area: float) -> float:
    if len(points) < 2 or area <= 0:
        raise Invalid("the index needs at least two points and a positive area")
    total = 0.0
    for i, p in enumerate(points):
        total += min(math.dist(p, q) for j, q in enumerate(points) if j != i)
    mean = total / len(points)
    return mean / (0.5 / math.sqrt(len(points) / area))


def hexagonal_bound(spacing: float, area: float = 1.0) -> float:
    # the most points a region can hold with no two closer than the spacing
    if spacing <= 0:
        raise Invalid("the spacing must be positive")
    return area / (spacing * spacing * math.sqrt(3) / 2)


def occupied_cells(points: Sequence[Point], cell: float) -> int:
    if cell <= 0:
        raise Invalid("the cell size must be positive")
    return len({(math.floor(p[0] / cell), math.floor(p[1] / cell)) for p in points})


def min_pair_distance(points: Sequence[Point]) -> float:
    if len(points) < 2:
        raise Invalid("need at least two points")
    return min(
        math.dist(points[i], points[j])
        for i in range(len(points))
        for j in range(i + 1, len(points))
    )
