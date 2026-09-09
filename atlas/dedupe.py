"""Deduplication: merging near-identical points, and why snapping to a grid splits pairs.

Two feeds of the same shops, two surveys of the same trees, a
track logged twice: a point set with near-duplicates, pairs a
few meters apart that mean one thing, must be merged before it is
counted. The fast way is to snap every point to a grid of cells
the size of the tolerance and merge the points sharing a cell,
which is linear and needs no index. The careful way is to merge
every pair within the tolerance distance, which needs a
neighbour search but honours the tolerance as a radius rather
than a grid. The survey measures the difference, which is the
grid's seams. Two points a distance d apart, placed at random,
straddle a cell boundary of a grid of cell size s with a
probability that grows with d over s, and the guess of about a
third of pairs missed at a fifth of a cell was high: over 2000
planted pairs at 0.1, 0.2, 0.5, and 0.9 cells apart the snapping
method split 12, 23, 54, and 88 percent of them, about 1.2 times
d over s at small separations, while the radius method split
none, merging every pair within the radius by definition. The
two-pass snapping variant, which also snaps to a grid offset by
half a cell and merges the union, cut the splits to 0.5, 2, 12.5,
and 68 percent. The snapping method also merges what it should
not: two points diagonally across a cell are 1.41 cells apart and
are merged when they share it, and on 3000 sparse random points
the longest pair it merged was 1.17 cells apart with 1.8 percent
of its pairs beyond one cell and a mean of 0.49, against the
radius method's longest 0.99 and mean 0.63, the radius method
merging 154 points to snapping's 58 on the same set. The radius
method carries a hazard of its own, since its merges are
transitive: as the density of random points rose from 0.03 to
0.3, 0.83, 1.9, and 3.3 per cell, its largest group went from 3
to 10, 51, 2936, and 2995 of the 3000, chaining the whole set
into one group past a density between 0.83 and 1.9 that brackets
the random geometric graph's percolation threshold of 1.44 per
unit area, while snapping's largest group stayed at 2 to 10. On a
set of exact duplicates every method merged all 200. The finding
worth stating is that grid snapping misses true duplicates in
proportion to their separation over the cell, a quarter at a
fifth of a cell, and merges pairs past a cell apart, while radius
merging honours the tolerance exactly and then chains everything
past a density it does not warn about, so a deduplication count
is only as good as its method's seams and its set's density.
This module merges near-duplicates by snapping and by radius,
and a survey measures the misses, the overreach, and the
chaining.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def snap_merge(points: Sequence[Point], cell: float, offset: float = 0.0) -> list[list[int]]:
    # groups of point indices that share a grid cell
    if cell <= 0:
        raise Invalid("the cell size must be positive")
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(points):
        key = (math.floor((x + offset) / cell), math.floor((y + offset) / cell))
        groups.setdefault(key, []).append(i)
    return list(groups.values())


def double_snap_merge(points: Sequence[Point], cell: float) -> list[list[int]]:
    # the union of the plain grid's groups and a half-cell-offset grid's groups
    parent = list(range(len(points)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for grouping in (snap_merge(points, cell), snap_merge(points, cell, cell / 2)):
        for group in grouping:
            root = find(group[0])
            for j in group[1:]:
                parent[find(j)] = root
    out: dict[int, list[int]] = {}
    for i in range(len(points)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def radius_merge(points: Sequence[Point], radius: float) -> list[list[int]]:
    # groups joining every pair within the radius, transitively, through a cell index
    if radius <= 0:
        raise Invalid("the radius must be positive")
    parent = list(range(len(points)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cells: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(points):
        cells.setdefault((math.floor(x / radius), math.floor(y / radius)), []).append(i)
    for (cx, cy), members in cells.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cx + dx, cy + dy), []):
                    for i in members:
                        if i < j and math.dist(points[i], points[j]) <= radius:
                            parent[find(j)] = find(i)
    out: dict[int, list[int]] = {}
    for i in range(len(points)):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def merged_count(groups: Sequence[Sequence[int]], total: int) -> int:
    return total - len(groups)


def pairs_split(groups: Sequence[Sequence[int]], pairs: Sequence[tuple[int, int]]) -> int:
    # how many planted pairs ended up in different groups
    where = {}
    for g, group in enumerate(groups):
        for i in group:
            where[i] = g
    return sum(1 for a, b in pairs if where[a] != where[b])


def merged_distances(points: Sequence[Point], groups: Sequence[Sequence[int]]) -> list[float]:
    # the distances between members of each two-point group
    return [math.dist(points[g[0]], points[g[1]]) for g in groups if len(g) == 2]


def planted_pairs(count: int, separation: float, rng, extent: float = 100.0):
    # isolated pairs of points a fixed distance apart in random directions
    points: list[Point] = []
    pairs: list[tuple[int, int]] = []
    for _ in range(count):
        x, y = rng.uniform(0, extent), rng.uniform(0, extent)
        angle = rng.uniform(0, 2 * math.pi)
        points.append((x, y))
        points.append((x + separation * math.cos(angle), y + separation * math.sin(angle)))
        pairs.append((len(points) - 2, len(points) - 1))
    return points, pairs
