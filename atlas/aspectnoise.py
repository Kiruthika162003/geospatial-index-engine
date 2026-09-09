"""Aspect error depends only on slope over the noise gradient: 4 degrees at 11.5, 43 at 1.15.

Horn's kernel reads the gradient from a 3 by 3 window with weights
1, 2, 1 over 8 cells, so white noise of sigma on the heights puts a
noise of root 12 over 8 times sigma over the cell, 0.433 sigma, on
each gradient component. On clean planes the aspect reads its true
value to six places at 0, 90, 200 and 315 degrees and the slope
16.6992 degrees for a gradient of 0.3. With noise the mean absolute
aspect error depends only on the ratio of the slope to that noise
gradient: a slope of 0.5 under sigma 0.1, ratio 11.5, reads 3.94
degrees against the small-angle law atan(noise over slope) times
root of 2 over pi of 3.95; ratio 4.62 reads 9.97 against 9.75; ratio
3.85 reads 12.05 against 11.62; ratio 2.31 reads 21.0 against 18.7;
and ratio 1.15 reads 42.8 against 32.6, where the law fails as the
noise gradient rivals the slope, the same 42.8 whether the pair is
0.5 and 1.0 or 0.05 and 0.1. Ratio 0.77 reads 56.2 and a flat plane
89.6, the uniform 90. The circular standard deviation runs a quarter
above the mean absolute error, 4.96, 15.4 and 54.7 at the first,
third and fifth ratios. A 3 by 3 mean before the gradient cuts the
error about 2.5-fold: 1.62, 4.89 and 17.4 at those ratios.

The guess that noise leaves the slope unbiased was wrong: it can
only add to the gradient's length, so a slope of 5.711 degrees reads
6.27 under sigma 0.1 and 10.33 under 0.3, 2.862 reads 4.03, and a
flat plane reads 3.18, 9.41 and 27.6 degrees under sigma 0.1, 0.3
and 1 against the Rayleigh law of 3.11, 9.25 and 28.5. The cell size
enters through the gradient: a slope of 0.1 under sigma 0.3 reads an
aspect error of 56.7, 12.2 and 1.98 degrees at cells of 1, 5 and 30
against the law's 41.8, 11.6 and 1.98.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Grid = list[list[float]]

HORN_GRADIENT_FACTOR = math.sqrt(12) / 8


def horn(grid: Grid, r: int, c: int, cell: float) -> tuple[float, float]:
    if not (1 <= r < len(grid) - 1 and 1 <= c < len(grid[0]) - 1):
        raise Invalid("Horn's kernel needs an interior cell")
    z = [[grid[r + dr][c + dc] for dc in (-1, 0, 1)] for dr in (-1, 0, 1)]
    dzdx = ((z[0][2] + 2 * z[1][2] + z[2][2]) - (z[0][0] + 2 * z[1][0] + z[2][0])) / (8 * cell)
    dzdy = ((z[2][0] + 2 * z[2][1] + z[2][2]) - (z[0][0] + 2 * z[0][1] + z[0][2])) / (8 * cell)
    return dzdx, dzdy


def slope_degrees(gx: float, gy: float) -> float:
    return math.degrees(math.atan(math.hypot(gx, gy)))


def aspect_degrees(gx: float, gy: float) -> float:
    # downslope direction, clockwise from north (north is decreasing row)
    return math.degrees(math.atan2(-gx, gy)) % 360


def plane(size: int, slope: float, aspect: float, cell: float = 1.0) -> Grid:
    if not 0 <= aspect < 360:
        raise Invalid("aspect runs from 0 to 360")
    a = math.radians(aspect)
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            row.append(-slope * cell * (c * math.sin(a) - r * math.cos(a)))
        out.append(row)
    return out


def noisy(grid: Grid, sigma: float, rng: random.Random) -> Grid:
    return [[v + rng.gauss(0, sigma) for v in row] for row in grid]


def smoothed(grid: Grid) -> Grid:
    rows, cols = len(grid), len(grid[0])
    out = [list(row) for row in grid]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            block = [grid[r + dr][c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            out[r][c] = sum(block) / 9
    return out


def aspect_field(grid: Grid, cell: float = 1.0) -> list[float]:
    rows, cols = len(grid), len(grid[0])
    return [
        aspect_degrees(*horn(grid, r, c, cell))
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
    ]


def slope_field(grid: Grid, cell: float = 1.0) -> list[float]:
    rows, cols = len(grid), len(grid[0])
    return [
        slope_degrees(*horn(grid, r, c, cell))
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
    ]


def circular_error(angles: list[float], truth: float) -> tuple[float, float]:
    # mean absolute angular error and the circular standard deviation, both in degrees
    if not angles:
        raise Invalid("no angles")
    errors = [((a - truth + 180) % 360) - 180 for a in angles]
    mean_abs = sum(abs(e) for e in errors) / len(errors)
    cx = sum(math.cos(math.radians(e)) for e in errors) / len(errors)
    cy = sum(math.sin(math.radians(e)) for e in errors) / len(errors)
    resultant = math.hypot(cx, cy)
    circular = math.degrees(math.sqrt(-2 * math.log(max(resultant, 1e-12))))
    return mean_abs, circular


def gradient_noise(sigma: float, cell: float = 1.0) -> float:
    return HORN_GRADIENT_FACTOR * sigma / cell


def aspect_error_law(slope: float, sigma: float, cell: float = 1.0) -> float:
    # the mean absolute angle a gradient of size slope is turned by noise of size gradient_noise
    g = gradient_noise(sigma, cell)
    if slope <= 0:
        return 90.0
    return math.degrees(math.atan(g / slope)) * math.sqrt(2 / math.pi)


def flat_slope_law(sigma: float, cell: float = 1.0) -> float:
    # the mean slope of pure noise: a Rayleigh mean of the two gradient components
    return math.degrees(math.atan(gradient_noise(sigma, cell) * math.sqrt(math.pi / 2)))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)
