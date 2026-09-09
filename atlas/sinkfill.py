"""Filling sinks without an epsilon leaves more flat cells than the noise had sinks.

Priority flood fills a raster's depressions by flooding inward from
the edges with a heap that never lets the water level fall, so every
cell ends at or above the lowest spill path to the edge. A plane of
slope 1 has no interior sinks and takes no fill. Gaussian noise of
sigma 0.1 on that plane still makes no sink; sigma 0.3 makes one,
filled by 0.1447; sigma 1 makes 211 sinks and sigma 3 makes 374.
The guess that filling removes the sinks was wrong as read by a
steepest-descent flow: the flat lakes the fill leaves count as sinks
too, 244 after filling the sigma 1 noise and 654 after sigma 3, and
only an epsilon of 1e-6 added along the flood, which tilts every
lake toward its spill, brings both to zero. The sigma 1 fill raises
6.56 percent of the cells by 0.535 on average and 2.747 at most,
130.6 in volume; sigma 3 raises 17.6 percent by 1.87 on average and
8.22 at most, 1222.8 in volume. A 61 by 61 fill takes 5
milliseconds.

A paraboloid bowl of depth 5 or 20 fills to the rim with a volume of
7068.13 or 28,272.5, the sum of the paraboloid below the rim over the
disc to the last digit, raising 75.5 percent of the cells and leaving
3481 flat cells. A crater 12 cells in radius and 10 deep on a flat
plane holds 2260.6; on a plane of slope 0.1 the fill stops at the
low rim, 1.8, and holds 1766.0, 78 percent, with a depth of 8.83 at
the centre; on slope 0.5 the low rim sits at 9.0 against the high
rim's 21.0 and the crater holds 543.0, 24 percent, 4.89 deep.
"""

from __future__ import annotations

import heapq
import math
import random

from atlas.errors import Invalid
from atlas.flowdirection import flow_directions, sinks

Grid = list[list[float]]
Cell = tuple[int, int]


def _check(dem: Grid) -> tuple[int, int]:
    if not dem or not dem[0]:
        raise Invalid("the elevation grid is empty")
    return len(dem), len(dem[0])


def priority_flood(dem: Grid, epsilon: float = 0.0) -> Grid:
    # Barnes, Lehman and Mulla: flood inward from the edges, never letting the water level fall
    rows, cols = _check(dem)
    if epsilon < 0:
        raise Invalid("epsilon must not be negative")
    filled = [[math.inf] * cols for _ in range(rows)]
    heap: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if r in (0, rows - 1) or c in (0, cols - 1):
                filled[r][c] = dem[r][c]
                heapq.heappush(heap, (dem[r][c], r, c))
    while heap:
        level, r, c = heapq.heappop(heap)
        if level > filled[r][c]:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if not (dr or dc) or not (0 <= rr < rows and 0 <= cc < cols):
                    continue
                if filled[rr][cc] == math.inf:
                    filled[rr][cc] = max(dem[rr][cc], level + epsilon)
                    heapq.heappush(heap, (filled[rr][cc], rr, cc))
    return filled


def fill_depth(dem: Grid, filled: Grid) -> Grid:
    rows = zip(dem, filled, strict=True)
    return [[f - d for d, f in zip(drow, frow, strict=True)] for drow, frow in rows]


def fill_summary(dem: Grid, filled: Grid) -> dict[str, float]:
    rows, cols = _check(dem)
    depths = [v for row in fill_depth(dem, filled) for v in row]
    raised = [v for v in depths if v > 1e-12]
    return {
        "raised_fraction": len(raised) / (rows * cols),
        "mean_depth": sum(raised) / len(raised) if raised else 0.0,
        "max_depth": max(depths),
        "volume": sum(depths),
    }


def interior_sink_count(dem: Grid) -> int:
    rows, cols = _check(dem)
    directions = flow_directions(dem)
    return sum(1 for r, c in sinks(directions) if 0 < r < rows - 1 and 0 < c < cols - 1)


def tilted_plane(size: int, slope: float) -> Grid:
    return [[slope * c for c in range(size)] for _ in range(size)]


def noisy(dem: Grid, sigma: float, rng: random.Random) -> Grid:
    return [[v + rng.gauss(0, sigma) for v in row] for row in dem]


def bowl(size: int, depth: float) -> Grid:
    centre = (size - 1) / 2
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            d = math.hypot(r - centre, c - centre) / centre
            row.append(depth * min(1.0, d * d))
        out.append(row)
    return out


def bowl_volume_law(size: int, depth: float) -> float:
    # the paraboloid below the rim's level, summed over the disc of radius centre
    centre = (size - 1) / 2
    total = 0.0
    for r in range(size):
        for c in range(size):
            d = math.hypot(r - centre, c - centre) / centre
            if d < 1:
                total += depth * (1 - d * d)
    return total


def crater_on_slope(size: int, slope: float, depth: float, radius: float) -> Grid:
    dem = tilted_plane(size, slope)
    centre = size // 2
    for r in range(size):
        for c in range(size):
            d = math.hypot(r - centre, c - centre)
            if d < radius:
                dem[r][c] -= depth * (1 - (d / radius) ** 2)
    return dem


def spill_height(filled: Grid, r: int, c: int) -> float:
    rows, cols = _check(filled)
    if not (0 <= r < rows and 0 <= c < cols):
        raise Invalid("the cell must lie on the grid")
    return filled[r][c]
