"""Nearest resampling round-trips a step exactly; bilinear blurs a band as wide as the factor.

Regridding a raster by a factor and back is a round trip whose error
depends on the field as much as the method. On a smooth field of one
wave across 64 cells, spread 0.5, the root mean square error of the
round trip at factor 2 reads 0.049 for nearest down and up, 0.035 for
a block mean down and nearest up, 0.035 for nearest down and bilinear
up, and 0.0077 for mean down and bilinear up, six times better than
nearest. At factors 4 and 8 the four read 0.129, 0.077, 0.105 and
0.027, then 0.281, 0.156, 0.238 and 0.087. A field of eight waves,
period 8 cells, regridded by 4 loses everything: nearest reads 0.5,
the whole spread, and mean with bilinear 0.485.

The guess that bilinear always beats nearest was wrong on a sharp
field. A step whose edge sits on a block boundary round-trips with
zero error under nearest at factors 2, 4 and 8, while bilinear
upsampling blurs a band of exactly 2, 4 and 8 columns with root mean
square errors 0.044, 0.070 and 0.101. Stripes of period 4, 8 and 16
cells are exact under nearest at any factor below the period and
blurred by bilinear to 0.246, 0.171 and 0.117 at factor 2. At a
factor equal to the period the block mean reads a constant 0.5 and
the round trip 0.5, while nearest keeps one phase and reads a
constant 0 or 1 with a round trip of 0.7071, so the aliasing of
nearest is worse than the flattening of the mean. Every method keeps
the field's mean on the step, 0.5.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Grid = list[list[float]]


def _check(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return len(grid), len(grid[0])


def downsample_nearest(grid: Grid, factor: int) -> Grid:
    rows, cols = _check(grid)
    if factor < 1 or rows % factor or cols % factor:
        raise Invalid("the factor must divide both sides")
    return [[grid[r][c] for c in range(0, cols, factor)] for r in range(0, rows, factor)]


def downsample_mean(grid: Grid, factor: int) -> Grid:
    rows, cols = _check(grid)
    if factor < 1 or rows % factor or cols % factor:
        raise Invalid("the factor must divide both sides")
    out = []
    for r in range(0, rows, factor):
        row = []
        for c in range(0, cols, factor):
            block = [grid[rr][cc] for rr in range(r, r + factor) for cc in range(c, c + factor)]
            row.append(sum(block) / len(block))
        out.append(row)
    return out


def upsample_nearest(grid: Grid, factor: int) -> Grid:
    rows, cols = _check(grid)
    if factor < 1:
        raise Invalid("the factor must be positive")
    return [
        [grid[r // factor][c // factor] for c in range(cols * factor)]
        for r in range(rows * factor)
    ]


def upsample_bilinear(grid: Grid, factor: int) -> Grid:
    rows, cols = _check(grid)
    if factor < 1:
        raise Invalid("the factor must be positive")
    out = []
    for r in range(rows * factor):
        y = (r + 0.5) / factor - 0.5
        r0 = min(max(math.floor(y), 0), rows - 1)
        r1 = min(r0 + 1, rows - 1)
        ty = min(max(y - r0, 0.0), 1.0)
        row = []
        for c in range(cols * factor):
            x = (c + 0.5) / factor - 0.5
            c0 = min(max(math.floor(x), 0), cols - 1)
            c1 = min(c0 + 1, cols - 1)
            tx = min(max(x - c0, 0.0), 1.0)
            top = grid[r0][c0] * (1 - tx) + grid[r0][c1] * tx
            bottom = grid[r1][c0] * (1 - tx) + grid[r1][c1] * tx
            row.append(top * (1 - ty) + bottom * ty)
        out.append(row)
    return out


def rms(a: Grid, b: Grid) -> float:
    rows, cols = _check(a)
    if (rows, cols) != _check(b):
        raise Invalid("the grids must share a shape")
    total = sum((a[r][c] - b[r][c]) ** 2 for r in range(rows) for c in range(cols))
    return math.sqrt(total / (rows * cols))


def mean(grid: Grid) -> float:
    rows, cols = _check(grid)
    return sum(sum(row) for row in grid) / (rows * cols)


def spread(grid: Grid) -> float:
    m = mean(grid)
    rows, cols = _check(grid)
    return math.sqrt(sum((v - m) ** 2 for row in grid for v in row) / (rows * cols))


def round_trip(grid: Grid, factor: int, down: str, up: str) -> Grid:
    coarse = {"nearest": downsample_nearest, "mean": downsample_mean}[down](grid, factor)
    return {"nearest": upsample_nearest, "bilinear": upsample_bilinear}[up](coarse, factor)


METHODS = (
    ("nearest", "nearest"),
    ("mean", "nearest"),
    ("nearest", "bilinear"),
    ("mean", "bilinear"),
)


def round_trip_errors(grid: Grid, factor: int) -> dict[str, float]:
    return {
        f"{down}-{up}": rms(grid, round_trip(grid, factor, down, up)) for down, up in METHODS
    }


def smooth_field(size: int, waves: float = 1.0) -> Grid:
    out = []
    for r in range(size):
        along = math.cos(2 * math.pi * waves * r / size)
        out.append([math.sin(2 * math.pi * waves * c / size) * along for c in range(size)])
    return out


def step_field(size: int) -> Grid:
    return [[1.0 if c >= size // 2 else 0.0 for c in range(size)] for _ in range(size)]


def stripe_field(size: int, period: int) -> Grid:
    if period < 2:
        raise Invalid("a stripe needs a period of at least two cells")
    row = [1.0 if (c % period) < period // 2 else 0.0 for c in range(size)]
    return [list(row) for _ in range(size)]


def edge_band_width(original: Grid, resampled: Grid, tolerance: float = 1e-9) -> int:
    rows, cols = _check(original)
    columns = set()
    for r in range(rows):
        for c in range(cols):
            if abs(original[r][c] - resampled[r][c]) > tolerance:
                columns.add(c)
    return len(columns)
