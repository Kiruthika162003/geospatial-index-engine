"""Tile filtering spoils a seam band as wide as the reach; a halo of that reach spoils nothing.

A raster too large for memory is filtered tile by tile, and a
window filter run on a tile alone cannot see across the tile's
edge, so the cells within the filter's reach of an interior seam
come out wrong unless each tile reads a halo of its neighbours. On
a 96 grid of noise cut into 16 or 32 cell tiles, a box filter of
radius 1 with no halo spoils 19.75 or 8.16 percent of the cells, a
radius of 2 spoils 37.3 or 16.0 percent and a radius of 3 spoils
52.7 or 23.4, exactly the cells within radius minus halo of a seam,
1 minus (1 minus the wrong lines over the side) squared; a halo of
one cell brings radius 3 to 37.3 and 16.0 percent, two cells to
19.75 and 8.16, and a halo equal to the radius to 0.0 exactly. The
worst gap at a spoiled cell reads 0.276 for radius 1 and 0.159 for
radius 3, since a wider window averages the missing cells away.

The guess that the halo a repeated filter needs is the radius was
wrong: each pass reaches another radius, so two passes of radius 1
spoil 37.3 percent with no halo and 19.75 with a halo of 1, and
three passes of radius 2 spoil 85.9, 77.0, 66.0, 52.7 and 37.3
percent at halos of 0 to 4 on 16 cell tiles and 0.0 only at 6, the
radius times the passes, which the law with that reach reads to
four places. The halo's price is the cells read per cell written:
1.89 for a halo of 3 on 16 cell tiles, 1.41 on 32 and 1.047 on 256.
"""

from __future__ import annotations

import random

from atlas.errors import Invalid

Grid = list[list[float]]


def _check(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return len(grid), len(grid[0])


def box_filter(grid: Grid, radius: int) -> Grid:
    # a mean over the window, clamped at the raster's edges
    rows, cols = _check(grid)
    if radius < 0:
        raise Invalid("the radius must not be negative")
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            r0, r1 = max(0, r - radius), min(rows - 1, r + radius)
            c0, c1 = max(0, c - radius), min(cols - 1, c + radius)
            block = [grid[rr][cc] for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)]
            row.append(sum(block) / len(block))
        out.append(row)
    return out


def repeated(grid: Grid, radius: int, passes: int) -> Grid:
    if passes < 1:
        raise Invalid("at least one pass is needed")
    out = grid
    for _ in range(passes):
        out = box_filter(out, radius)
    return out


def tiled(grid: Grid, tile: int, halo: int, radius: int, passes: int = 1) -> Grid:
    # filter each tile separately, reading a halo of cells around it, and write back the core
    rows, cols = _check(grid)
    if tile < 1 or halo < 0:
        raise Invalid("the tile must be positive and the halo non-negative")
    out = [[0.0] * cols for _ in range(rows)]
    for r0 in range(0, rows, tile):
        for c0 in range(0, cols, tile):
            r1, c1 = min(rows, r0 + tile), min(cols, c0 + tile)
            hr0, hc0 = max(0, r0 - halo), max(0, c0 - halo)
            hr1, hc1 = min(rows, r1 + halo), min(cols, c1 + halo)
            piece = [row[hc0:hc1] for row in grid[hr0:hr1]]
            done = repeated(piece, radius, passes)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    out[r][c] = done[r - hr0][c - hc0]
    return out


def seam_cells(whole: Grid, pieced: Grid, tolerance: float = 1e-9) -> int:
    rows, cols = _check(whole)
    return sum(
        1
        for r in range(rows)
        for c in range(cols)
        if abs(whole[r][c] - pieced[r][c]) > tolerance
    )


def worst_gap(whole: Grid, pieced: Grid) -> float:
    return max(
        abs(a - b)
        for ra, rb in zip(whole, pieced, strict=True)
        for a, b in zip(ra, rb, strict=True)
    )


def seam_fraction_law(size: int, tile: int, radius: int, halo: int, passes: int = 1) -> float:
    # cells within reach of a seam: a band of (reach - halo) on each side of every interior seam
    band = max(0, radius * passes - halo)
    if band == 0:
        return 0.0
    seams = size // tile - (1 if size % tile == 0 else 0)
    wrong_lines = min(size, seams * 2 * band)
    # rows or columns within the band, counted once for cells in either
    return 1 - (1 - wrong_lines / size) ** 2


def noise_grid(size: int, rng: random.Random) -> Grid:
    return [[rng.random() for _ in range(size)] for _ in range(size)]


def halo_needed(radius: int, passes: int) -> int:
    return radius * passes


def cost_ratio(size: int, tile: int, halo: int) -> float:
    # cells read per cell written when every tile carries a halo
    return ((tile + 2 * halo) / tile) ** 2 if size >= tile else 1.0
