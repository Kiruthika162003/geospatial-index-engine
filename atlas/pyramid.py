"""A pyramid costs a third more memory; a mean pyramid keeps a quarter of a spike per level.

A pyramid stores a raster at every halving until one cell remains,
so that a viewer reads the level that matches its pixels. The
levels of a 64, 256 and 1024 grid number 7, 9 and 11 and hold
5461, 87,381 and 1,398,101 cells, 1.3333 times the base at every
size, the geometric law four thirds. Drawing a 1024 raster at 512,
256, 100 and 10 pixels reads level 1, 2, 4 and 7, which is 262,144,
65,536, 4096 and 64 cells against 1,048,576 from the base. The guess
that a mean pyramid halves a lone peak per level was wrong: a
spike of 100 in one cell reads 100, 25, 6.25, 1.56, 0.39, 0.098 and
0.024 up the levels, a quarter each time, since the spike is one
cell of four in every block, while the mean of every level stays
at 0.0244. A max pyramid keeps the spike at 100 on every level and
inflates the level mean fourfold each time, 0.0244 up to 100, and
a nearest pyramid keeps the spike until the top level, where the
sampled cell misses it and reads 0. A cone of height 100 and radius
40 on a 256 grid reads a mean-pyramid peak of 98.2, 96.3, 92.4,
84.7, 69.4, 38.9, 10.2 and 2.56, the last being the cone's own mean,
while max holds 98.2 throughout and the level means climb from 2.56
to 4.88 by the fifth level.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Grid = list[list[float]]


def _check(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return len(grid), len(grid[0])


def reduce_level(grid: Grid, how: str) -> Grid:
    rows, cols = _check(grid)
    if how not in ("mean", "max", "nearest"):
        raise Invalid("how must be mean, max or nearest")
    if rows < 2 or cols < 2:
        raise Invalid("the level cannot be halved")
    out = []
    for r in range(0, rows - rows % 2, 2):
        row = []
        for c in range(0, cols - cols % 2, 2):
            block = (grid[r][c], grid[r][c + 1], grid[r + 1][c], grid[r + 1][c + 1])
            if how == "mean":
                row.append(sum(block) / 4)
            elif how == "max":
                row.append(max(block))
            else:
                row.append(block[0])
        out.append(row)
    return out


def build(grid: Grid, how: str = "mean") -> list[Grid]:
    levels = [grid]
    while len(levels[-1]) >= 2 and len(levels[-1][0]) >= 2:
        levels.append(reduce_level(levels[-1], how))
    return levels


def cells(levels: list[Grid]) -> int:
    return sum(len(g) * len(g[0]) for g in levels)


def overhead(levels: list[Grid]) -> float:
    base = len(levels[0]) * len(levels[0][0])
    return cells(levels) / base


def overhead_law() -> float:
    return 4 / 3


def peak_at_levels(levels: list[Grid]) -> list[float]:
    return [max(max(row) for row in g) for g in levels]


def mean_at_levels(levels: list[Grid]) -> list[float]:
    return [sum(sum(row) for row in g) / (len(g) * len(g[0])) for g in levels]


def spike(size: int, height: float) -> Grid:
    grid = [[0.0] * size for _ in range(size)]
    grid[size // 2][size // 2] = height
    return grid


def cone(size: int, height: float, radius: float) -> Grid:
    centre = (size - 1) / 2
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            row.append(height * max(0.0, 1 - math.hypot(r - centre, c - centre) / radius))
        out.append(row)
    return out


def level_for_pixels(size: int, pixels: int) -> int:
    # the pyramid level whose side first fits the pixels asked for
    if size < 1 or pixels < 1:
        raise Invalid("size and pixels must be positive")
    level = 0
    side = size
    while side > pixels and side >= 2:
        side //= 2
        level += 1
    return level


def read_cost(size: int, pixels: int) -> tuple[int, int]:
    # cells read to draw the raster at a pixel size, from the base and from the pyramid level
    level = level_for_pixels(size, pixels)
    return size * size, (size >> level) * (size >> level)
