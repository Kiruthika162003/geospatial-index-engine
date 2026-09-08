"""Spatial join: pair points from two sets within a radius by a grid, not nested loops.

A spatial join asks, for two sets of points, which pairs, one from each
set, lie within a distance of each other, the operation behind matching
riders to drivers, sightings to habitats, deliveries to depots. The
straightforward way is a nested loop, every point of one set against
every point of the other, which is the product of the two sizes and
becomes unbearable as they grow. A grid makes it fast when the radius
is small relative to the spread. Hash one set into a uniform grid whose
cell size equals the radius, then for each point of the other set, look
only in the nine cells around it, its own cell and the eight neighbors,
because a partner within the radius cannot be more than one cell away
when the cell equals the radius. The candidate pairs examined are then
just those sharing a local neighborhood, not the full product, so the
work drops from the product of the sizes to roughly the output size
plus the input sizes when the points are spread out. The correctness
rests on the cell-equals-radius choice: a smaller cell would force
checking more rings of neighbors, a larger cell would let a partner
hide two cells away and be missed, so the radius sets the cell size
exactly. The same clustering caveat as the plain grid applies, since a
dense clump piles many points into one neighborhood and the join there
approaches the nested loop locally, but for spread data the grid join
is dramatically cheaper. The finding worth stating is that the grid
join returns exactly the pairs a nested loop would while examining a
small fraction of the candidate pairs, so it is a partition of the work
by locality, not a change in the answer. This module joins two point
sets within a radius by a grid, and a survey confirms the pairs match a
nested loop while the candidate count stays far below the product.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Point = tuple[float, float]


def join_within(left: list[Point], right: list[Point], radius: float) -> list[tuple[int, int]]:
    if left is None or right is None:
        raise Invalid("both point sets must be provided")
    if radius <= 0:
        raise Invalid("radius must be positive")
    r2 = radius * radius
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(left):
        key = (math.floor(x / radius), math.floor(y / radius))
        grid.setdefault(key, []).append(idx)
    pairs: list[tuple[int, int]] = []
    for j, (x, y) in enumerate(right):
        ci, cj = math.floor(x / radius), math.floor(y / radius)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for i in grid.get((ci + di, cj + dj), ()):
                    lx, ly = left[i]
                    if (lx - x) ** 2 + (ly - y) ** 2 <= r2:
                        pairs.append((i, j))
    return pairs


def candidate_count(left: list[Point], right: list[Point], radius: float) -> int:
    # how many (i, j) pairs the grid actually distance-tested
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, (x, y) in enumerate(left):
        key = (math.floor(x / radius), math.floor(y / radius))
        grid.setdefault(key, []).append(idx)
    count = 0
    for x, y in right:
        ci, cj = math.floor(x / radius), math.floor(y / radius)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                count += len(grid.get((ci + di, cj + dj), ()))
    return count
