"""Centroid assignment misplaces 1 - (1 - r/4) squared of a zone's value; a tiling almost none.

Attributes of zones are often given to the grid cell holding each
zone's centroid, and the alternative splits every zone by area.
The share of the total that lands in the wrong cell depends on the
zone's side in cells, r. For a single square zone dropped at random
the share reads 0.0502, 0.1213, 0.2348, 0.4411, 0.75 and 0.9375 at
r of 0.1, 0.25, 0.5, 1, 2 and 4 over 20,000 drops, against the law
1 - (1 - r/4) squared below one cell, where each axis spills a
quarter of the side on average, and 1 - 1/r squared above, 0.0494,
0.1211, 0.2344, 0.4375, 0.75 and 0.9375. Four hundred overlapping
zones with random values misplace less in aggregate, 0.045, 0.123,
0.210, 0.417, 0.718 and 0.898, since one zone's spill into a cell
is offset by another's spill out of it.

The guess that zones the size of a cell must misplace a third of
their value, as a lone zone does, was wrong for a tiling: zones
tiling the extent at the cell's own size misplace 0.0 aligned to
the grid, 0.005 offset by a quarter of a cell and 0.01 offset by
half, because every cell receives from its four neighbours exactly
what by area it would have kept, and only the extent's edge leaks.
Area splitting itself conserves every value: a 10 by 10 zone over
4-unit cells lands 16, 16, 8, 16, 16, 8, 8, 8 and 4 of its 100 on
nine cells, where the centroid puts all 100 on one.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Zone = tuple[float, float, float, float]
Cells = dict[tuple[int, int], float]


def zone_centroid(zone: Zone) -> Point:
    x0, y0, x1, y1 = zone
    if x1 <= x0 or y1 <= y0:
        raise Invalid("a zone needs positive width and height")
    return (x0 + x1) / 2, (y0 + y1) / 2


def cell_of(point: Point, cell: float) -> tuple[int, int]:
    if cell <= 0:
        raise Invalid("the cell must be positive")
    return math.floor(point[0] / cell), math.floor(point[1] / cell)


def by_centroid(zones: list[Zone], values: list[float], cell: float) -> Cells:
    if len(zones) != len(values):
        raise Invalid("one value per zone")
    out: Cells = {}
    for zone, value in zip(zones, values, strict=True):
        key = cell_of(zone_centroid(zone), cell)
        out[key] = out.get(key, 0.0) + value
    return out


def by_area(zones: list[Zone], values: list[float], cell: float) -> Cells:
    if len(zones) != len(values):
        raise Invalid("one value per zone")
    if cell <= 0:
        raise Invalid("the cell must be positive")
    out: Cells = {}
    for (x0, y0, x1, y1), value in zip(zones, values, strict=True):
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            raise Invalid("a zone needs positive area")
        for i in range(math.floor(x0 / cell), math.ceil(x1 / cell)):
            for j in range(math.floor(y0 / cell), math.ceil(y1 / cell)):
                ox = max(0.0, min(x1, (i + 1) * cell) - max(x0, i * cell))
                oy = max(0.0, min(y1, (j + 1) * cell) - max(y0, j * cell))
                if ox > 0 and oy > 0:
                    out[(i, j)] = out.get((i, j), 0.0) + value * ox * oy / area
    return out


def total_error(a: Cells, b: Cells) -> float:
    # half the summed absolute difference: the share of the total that sits in the wrong cell
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / 2


def misplaced_share(zones: list[Zone], values: list[float], cell: float) -> float:
    total = sum(values)
    if total <= 0:
        raise Invalid("the values must sum to something positive")
    return total_error(by_centroid(zones, values, cell), by_area(zones, values, cell)) / total


def square_zones(
    size: float, count: int, rng: random.Random, extent: float = 1000.0
) -> list[Zone]:
    if size <= 0 or count < 1:
        raise Invalid("zones need a positive size and count")
    out = []
    for _ in range(count):
        x, y = rng.uniform(0, extent - size), rng.uniform(0, extent - size)
        out.append((x, y, x + size, y + size))
    return out


def tiled_zones(size: float, extent: float = 1000.0, offset: float = 0.0) -> list[Zone]:
    out = []
    x = offset
    while x < extent:
        y = offset
        while y < extent:
            out.append((x, y, min(x + size, extent), min(y + size, extent)))
            y += size
        x += size
    return out


def misplaced_law(ratio: float) -> float:
    # a square zone of side r cells dropped at random: the mean share of its area outside its
    # centroid's cell; per axis the spill is r/4 for r up to 1, and the two axes overlap
    if ratio <= 0:
        raise Invalid("the ratio must be positive")
    per_axis = min(ratio, 1.0) / 4 if ratio <= 1 else 1 - 1 / ratio
    return 1 - (1 - per_axis) ** 2


def simulated_share(ratio: float, rng: random.Random, trials: int = 20000) -> float:
    if ratio <= 0:
        raise Invalid("the ratio must be positive")
    total = 0.0
    for _ in range(trials):
        zone = square_zones(ratio, 1, rng, extent=100.0)[0]
        total += misplaced_share([zone], [1.0], 1.0)
    return total / trials
