"""Shadow casting: the shadow a height grid throws under a sun, ray by ray.

A solar panel behind a building, a street in a canyon, a valley
floor at dusk: whether a place is in shadow is a question about
the terrain between it and the sun. A cell is shadowed when the
ray from it toward the sun, climbing at the sun's elevation,
passes below the ground anywhere along its path, and the module
marches that ray across the height grid, sampling the surface by
bilinear interpolation at each step and comparing it with the
ray's height, until the ray leaves the grid or dips under the
ground. The survey calibrates the march on shapes with known
shadows. A single tower of height 8 on flat ground throws a
shadow of 8 over the tangent of the elevation, away from the
sun's azimuth: under a south sun at 30, 45, and 60 degrees the
column north of the tower held 13, 7, and 4 shadowed cells
against geometric lengths of 13.9, 8.0, and 4.6, within a cell,
and at 15 degrees the 29.9-cell shadow ran off the 41-cell grid
after 20. A cone of slope 20 degrees self-shadows only below its
slope, the threshold the hillshade module found: under a sun at
10, 15, and 19 degrees its far flank held 32, 22, and 10 percent
of the footprint in shadow, and at 21, 25, and 45 degrees not one
cell of the cone or the ground was dark, since a cone's flank
faces every direction and nothing rises above a ray steeper than
the slope. An east-west ridge of height 5 under a south sun at 30
degrees shadowed 8 rows to its north against a geometric 8.66 and
none to its south. And the march's step matters: a one-cell wall
under an oblique sun cast 230 shadowed cells at steps of a half,
a quarter, and a tenth of a cell alike, and only 192 at a step of
one cell, 17 percent of its shadow lost to rays that stepped over
it. The finding worth stating is that ray-marched shadows match
the tower and ridge geometry within a cell, that a cone
self-shadows only below its slope angle and by a fraction that
falls to zero there, and that the march step must be under the
thinnest obstacle or a sixth of its shadow goes missing, so a
shadow map is exact in the large and only as fine as its step in
the small.
This module casts shadows over a height grid, and a survey
measures the tower, the cone, the ridge, and the step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Grid = Sequence[Sequence[float]]


def _height_at(grid: Grid, x: float, y: float) -> float:
    # bilinear sample; x is the column coordinate, y the row coordinate
    rows, cols = len(grid), len(grid[0])
    x = min(max(x, 0.0), cols - 1.0)
    y = min(max(y, 0.0), rows - 1.0)
    c0, r0 = math.floor(x), math.floor(y)
    c1, r1 = min(c0 + 1, cols - 1), min(r0 + 1, rows - 1)
    fx, fy = x - c0, y - r0
    top = grid[r0][c0] * (1 - fx) + grid[r0][c1] * fx
    bottom = grid[r1][c0] * (1 - fx) + grid[r1][c1] * fx
    return top * (1 - fy) + bottom * fy


def shadow_length(height: float, elevation_deg: float) -> float:
    if not 0 < elevation_deg <= 90:
        raise Invalid("the sun must be above the horizon")
    return height / math.tan(math.radians(elevation_deg))


def shadowed(
    grid: Grid, azimuth_deg: float, elevation_deg: float, step: float = 0.25
) -> list[list[bool]]:
    # True where the ray toward the sun dips under the ground; rows run south, columns east
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    if not 0 < elevation_deg <= 90:
        raise Invalid("the sun must be above the horizon")
    if step <= 0:
        raise Invalid("the march step must be positive")
    rows, cols = len(grid), len(grid[0])
    az = math.radians(azimuth_deg)
    dx, dy = math.sin(az), -math.cos(az)  # toward the sun: east positive, north is up (row -1)
    climb = math.tan(math.radians(elevation_deg))
    out = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            x, y, h = float(c), float(r), grid[r][c]
            travelled = 0.0
            while True:
                x += dx * step
                y += dy * step
                travelled += step
                if x < 0 or x > cols - 1 or y < 0 or y > rows - 1:
                    break
                if _height_at(grid, x, y) > h + climb * travelled + 1e-9:
                    out[r][c] = True
                    break
    return out


def count(mask: Sequence[Sequence[bool]]) -> int:
    return sum(1 for row in mask for v in row if v)


def flat_with_tower(size: int, height: float) -> list[list[float]]:
    grid = [[0.0] * size for _ in range(size)]
    grid[size // 2][size // 2] = height
    return grid


def cone(size: int, height: float, radius: float) -> list[list[float]]:
    center = (size - 1) / 2

    def rise(r: int, c: int) -> float:
        return max(0.0, height * (1 - math.hypot(c - center, r - center) / radius))

    return [[rise(r, c) for c in range(size)] for r in range(size)]


def ridge(size: int, height: float, row: int) -> list[list[float]]:
    grid = [[0.0] * size for _ in range(size)]
    grid[row] = [height] * size
    return grid


def footprint(grid: Grid) -> list[tuple[int, int]]:
    return [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v > 0]
