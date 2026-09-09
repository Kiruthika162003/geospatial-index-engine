"""The terrain model sits low on a slope and 1.7 m high under a canopy passing 5 percent.

A lidar survey returns a cloud of points; the highest return in each
cell is the surface model, the lowest is the terrain model, and their
difference is the canopy height model. This module builds a synthetic
scene of conical trees on a plane and reads what each grid gets wrong.
On bare ground sloping 0.2 the terrain model reads 0.0536, 0.175,
0.490 and 0.995 low at cells of 1, 2, 5 and 10 m, against the half
drop across a cell of 0.1, 0.2, 0.5 and 1.0, since the lowest return
sits near the low edge but not on it; the surface model reads the
same amounts high. Vertical noise of 0.3 with 16 pulses in a cell
pulls the terrain 0.5237 low against the expected minimum of 16
normals, 1.766 sigma or 0.5298.

Under a canopy the terrain model rises as the ground stops seeing
pulses. With 16 pulses in a 2 m cell the bias under the crowns reads
0.0 when half the pulses reach the ground, 0.028 at 30 percent, 0.212
at 20, 0.903 at 10, 1.686 at 5 and 3.497 at none, tracking the chance
that a cell holds no ground return: 0.0033, 0.0281, 0.1853, 0.4401 and
1. At a 1 m cell with four pulses the same 30 percent scene reads
1.86 high under the crowns and 0.009 at 2 m.

The guess that a tree top falls short by its slope times half the
pulse spacing was low. At densities 0.25, 1, 4 and 16 per square
meter the mean shortfall of the 40 trees reads 10.15, 3.63, 1.36 and
0.75 against the guess of 4.75, 2.38, 1.19 and 0.59, since the crown's
highest cell is read at its own center. Tree tops found as local
maxima of the canopy model within a radius of 2 cells hit 33 of 40
trees at density 0.25, where 37 percent of cells are empty, and all 40
from density 1 upward. The cell size trades doubles for misses: at 1 m
a radius of 1 cell finds 91 tops on 40 trees and a radius of 2 finds
40; at 4 m radii 1, 2 and 3 find 39, 35 and 26; at 8 m a radius of 1
finds 32 tops with 4 outside any crown and a radius of 3 finds 14.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Return = tuple[float, float, float]
Tree = tuple[float, float, float, float]
Grid = list[list[float | None]]


def ground(x: float, _y: float, slope: float) -> float:
    return slope * x


def plant_trees(count: int, size: float, rng: random.Random) -> list[Tree]:
    trees: list[Tree] = []
    tries = 0
    while len(trees) < count and tries < 100 * count:
        tries += 1
        radius = rng.uniform(3.0, 6.0)
        cx, cy = rng.uniform(radius, size - radius), rng.uniform(radius, size - radius)
        if all(math.hypot(cx - tx, cy - ty) > radius + tr for tx, ty, tr, _ in trees):
            trees.append((cx, cy, radius, rng.uniform(10.0, 30.0)))
    return trees


class TreeIndex:
    def __init__(self, trees: list[Tree], bucket: float = 12.0) -> None:
        if bucket <= 0:
            raise Invalid("the bucket must be positive")
        self.bucket = bucket
        self.cells: dict[tuple[int, int], list[Tree]] = {}
        for tree in trees:
            cx, cy, radius, _ = tree
            low_x, high_x = int((cx - radius) // bucket), int((cx + radius) // bucket)
            low_y, high_y = int((cy - radius) // bucket), int((cy + radius) // bucket)
            for bx in range(low_x, high_x + 1):
                for by in range(low_y, high_y + 1):
                    self.cells.setdefault((bx, by), []).append(tree)

    def nearby(self, x: float, y: float) -> list[Tree]:
        return self.cells.get((int(x // self.bucket), int(y // self.bucket)), [])

    def height(self, x: float, y: float) -> float:
        return canopy_height(self.nearby(x, y), x, y)


def canopy_height(trees: list[Tree], x: float, y: float) -> float:
    best = 0.0
    for cx, cy, radius, height in trees:
        r = math.hypot(x - cx, y - cy)
        if r < radius:
            best = max(best, height * (1 - r / radius))
    return best


def scan(
    trees: list[Tree],
    size: float,
    density: float,
    penetration: float,
    rng: random.Random,
    slope: float = 0.0,
    noise: float = 0.0,
) -> list[Return]:
    if density <= 0 or not 0 <= penetration <= 1:
        raise Invalid("density must be positive and penetration a fraction")
    pulses = int(density * size * size)
    index = TreeIndex(trees)
    returns: list[Return] = []
    for _ in range(pulses):
        x, y = rng.uniform(0, size), rng.uniform(0, size)
        base = ground(x, y, slope)
        top = index.height(x, y)
        z = base
        if top > 0 and rng.random() > penetration:
            z = base + top
        returns.append((x, y, z + rng.gauss(0, noise)))
    return returns


def _grid(returns: list[Return], size: float, cell: float, pick) -> Grid:
    if cell <= 0 or size <= 0:
        raise Invalid("cell and size must be positive")
    n = math.ceil(size / cell)
    grid: Grid = [[None] * n for _ in range(n)]
    for x, y, z in returns:
        c, r = min(int(x / cell), n - 1), min(int(y / cell), n - 1)
        grid[r][c] = z if grid[r][c] is None else pick(grid[r][c], z)
    return grid


def dsm(returns: list[Return], size: float, cell: float) -> Grid:
    return _grid(returns, size, cell, max)


def dtm(returns: list[Return], size: float, cell: float) -> Grid:
    return _grid(returns, size, cell, min)


def chm(surface: Grid, terrain: Grid) -> Grid:
    out: Grid = []
    for srow, trow in zip(surface, terrain, strict=True):
        pairs = zip(srow, trow, strict=True)
        out.append([None if s is None or t is None else s - t for s, t in pairs])
    return out


def empty_fraction(grid: Grid) -> float:
    cells = [v for row in grid for v in row]
    return sum(1 for v in cells if v is None) / len(cells)


def terrain_bias(
    terrain: Grid, cell: float, slope: float, trees: list[Tree]
) -> tuple[float, float]:
    under, clear = [], []
    index = TreeIndex(trees)
    for r, row in enumerate(terrain):
        for c, z in enumerate(row):
            if z is None:
                continue
            x, y = (c + 0.5) * cell, (r + 0.5) * cell
            error = z - ground(x, y, slope)
            (under if index.height(x, y) > 0 else clear).append(error)
    mean_under = sum(under) / len(under) if under else 0.0
    mean_clear = sum(clear) / len(clear) if clear else 0.0
    return mean_under, mean_clear


Top = tuple[int, int, float]


def tree_tops(heights: Grid, min_height: float, radius: int = 1) -> list[Top]:
    if radius < 1:
        raise Invalid("the search radius must be at least one cell")
    n = len(heights)
    tops = []
    for r in range(n):
        for c in range(n):
            z = heights[r][c]
            if z is None or z < min_height:
                continue
            peak = True
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n:
                        other = heights[rr][cc]
                        tie = other == z and (dr, dc) < (0, 0)
                        if other is not None and (other > z or tie):
                            peak = False
            if peak:
                tops.append((r, c, z))
    return tops


def top_shortfall(heights: Grid, cell: float, trees: list[Tree]) -> float:
    shortfalls = []
    for cx, cy, radius, height in trees:
        best = 0.0
        for r, row in enumerate(heights):
            for c, z in enumerate(row):
                if z is None:
                    continue
                if math.hypot((c + 0.5) * cell - cx, (r + 0.5) * cell - cy) <= radius:
                    best = max(best, z)
        shortfalls.append(height - best)
    return sum(shortfalls) / len(shortfalls)


def expected_shortfall(trees: list[Tree], density: float) -> float:
    spacing = 0.5 / math.sqrt(density)
    return sum(height * spacing / radius for _, _, radius, height in trees) / len(trees)


def matched_tops(tops: list[Top], cell: float, trees: list[Tree]) -> tuple[int, int]:
    hit = set()
    stray = 0
    for r, c, _ in tops:
        x, y = (c + 0.5) * cell, (r + 0.5) * cell
        owner = None
        for i, (cx, cy, rad, _) in enumerate(trees):
            if math.hypot(x - cx, y - cy) <= rad:
                owner = i
                break
        if owner is None:
            stray += 1
        else:
            hit.add(owner)
    return len(hit), stray
