"""A horizontal shift of a DEM shows as a vertical error of shift times gradient over root 2.

Two elevation models of the same ground misregistered by a shift
disagree in height by the shift times the slope in the direction of
the shift, so the vertical error a comparison reads is a horizontal
error in disguise. On a plane of slope 0.1 or 0.5 a shift of 0.25,
0.5, 1 and 2 cells along the slope reads a root mean square height
difference of exactly the shift times the slope, 0.025 to 0.2 and
0.125 to 1.0, a shift at 45 degrees exactly that over root 2, and a
shift across the slope exactly 0.0. The guess that the
direction-averaged law, shift times the root mean square gradient
over root 2, holds on any terrain was right for isotropic terrain:
rolling ground of wavelength 20 and amplitude 5, gradient 1.093,
reads 0.193, 0.386, 0.771 and 1.525 at the four shifts against the
law's 0.193, 0.386, 0.773 and 1.545, within 0.5 percent up to a
cell and 1.3 percent at two, and the same within 3 percent at 45 and
90 degrees. Noise of 0.3 on the heights lifts the measured gradient
to 1.135 and the errors to 0.221, 0.442, 0.885 and 1.587, a tenth
over the law at small shifts, since the noise's own differences add
to the shift's.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Grid = list[list[float]]


def sample(grid: Grid, y: float, x: float) -> float:
    rows, cols = len(grid), len(grid[0])
    y = min(max(y, 0.0), rows - 1.0)
    x = min(max(x, 0.0), cols - 1.0)
    r0, c0 = math.floor(y), math.floor(x)
    r1, c1 = min(r0 + 1, rows - 1), min(c0 + 1, cols - 1)
    ty, tx = y - r0, x - c0
    top = grid[r0][c0] * (1 - tx) + grid[r0][c1] * tx
    bottom = grid[r1][c0] * (1 - tx) + grid[r1][c1] * tx
    return top * (1 - ty) + bottom * ty


def shifted(grid: Grid, dy: float, dx: float) -> Grid:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    rows, cols = len(grid), len(grid[0])
    return [[sample(grid, r - dy, c - dx) for c in range(cols)] for r in range(rows)]


def rms_difference(a: Grid, b: Grid, margin: int) -> float:
    rows, cols = len(a), len(a[0])
    if 2 * margin >= min(rows, cols):
        raise Invalid("the margin eats the whole grid")
    total = count = 0.0
    for r in range(margin, rows - margin):
        for c in range(margin, cols - margin):
            total += (a[r][c] - b[r][c]) ** 2
            count += 1
    return math.sqrt(total / count)


def rms_slope(grid: Grid, margin: int) -> float:
    # the root mean square gradient magnitude by central differences, in height per cell
    rows, cols = len(grid), len(grid[0])
    total = count = 0.0
    for r in range(max(1, margin), min(rows - 1, rows - margin)):
        for c in range(max(1, margin), min(cols - 1, cols - margin)):
            gx = (grid[r][c + 1] - grid[r][c - 1]) / 2
            gy = (grid[r + 1][c] - grid[r - 1][c]) / 2
            total += gx * gx + gy * gy
            count += 1
    return math.sqrt(total / count)


def vertical_law(shift: float, rms_gradient: float) -> float:
    # a shift of s along a random direction against a gradient of size g reads s g over root 2
    return shift * rms_gradient / math.sqrt(2)


def plane(size: int, slope: float) -> Grid:
    return [[slope * c for c in range(size)] for _ in range(size)]


def rolling(size: int, wavelength: float, amplitude: float) -> Grid:
    k = 2 * math.pi / wavelength
    return [
        [amplitude * math.sin(k * c) * math.cos(k * r) for c in range(size)]
        for r in range(size)
    ]


def noisy(grid: Grid, sigma: float, rng: random.Random) -> Grid:
    return [[v + rng.gauss(0, sigma) for v in row] for row in grid]


def error_by_shift(
    grid: Grid, shifts: list[float], angle_deg: float, margin: int
) -> list[tuple[float, float]]:
    a = math.radians(angle_deg)
    out = []
    for s in shifts:
        moved = shifted(grid, s * math.sin(a), s * math.cos(a))
        out.append((s, rms_difference(grid, moved, margin)))
    return out


def mean(values: list[float]) -> float:
    if not values:
        raise Invalid("no values")
    return sum(values) / len(values)
