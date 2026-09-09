"""TRI on a plane is root six times the slope; white noise reads 0.938 of its RMS law.

Three roughness indices on a raster: Riley's terrain ruggedness
index, the root of the summed squared differences to the eight
neighbours; the topographic position index, the cell against the
mean of a ring; and the standard deviation of a window. On a plane
tilted by s along x the differences are s, s, 0, 0 and four
diagonals of s, so TRI reads root six times s, 0.244949, 1.224745
and 2.44949 at slopes 0.1, 0.5 and 1 to six places, TPI reads
exactly 0 and the window deviation s times root two thirds.

The guess that white noise of sigma reads TRI 4 sigma, the root of
eight differences of variance 2 sigma squared, was high by six
percent: the measured mean is 0.9378 of that, 0.3751, 3.751 and
11.25 at sigma 0.1, 1 and 3, because the mean of a root falls under
the root of a mean. The window deviation reads 0.915 sigma and the
absolute TPI 0.844 sigma. Sine hills of amplitude 5 and wavelength
10, 20 and 40 cells read TRI 5.14, 2.62 and 1.31 on average, peaking
at 6.10, 3.66 and 1.90 against the gradient law 2 pi A over lambda
times root six of 7.70, 3.85 and 1.92, since the one-cell difference
undershoots the derivative of a short wave. A one-cell TPI barely
sees the hills, 0.52, 0.15 and 0.038 in absolute mean, where a
five-cell ring reads 1.84, 1.37 and 0.51.

On a paraboloid hilltop of radius 15 and height 30 the summit's TPI
reads 0.2, 1.0889 and 2.6889 at ring radii 1, 3 and 5, exactly the
height times the ring's mean squared distance over the hill radius
squared; the guess of two thirds of the radius squared was low. At
a threshold of 0.5 the one-cell ring classes 0 cells as ridge and
96 as valley, the hill's foot, while the three-cell ring classes
509, 320 and 396 as ridge, valley and flat.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Grid = list[list[float]]


def _check(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return len(grid), len(grid[0])


def _window(grid: Grid, r: int, c: int, radius: int) -> list[float]:
    rows, cols = len(grid), len(grid[0])
    values = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and (dr or dc):
                values.append(grid[rr][cc])
    return values


def tri(grid: Grid, r: int, c: int) -> float:
    # Riley's terrain ruggedness index: the root of the summed squared neighbour differences
    rows, cols = _check(grid)
    if not (0 <= r < rows and 0 <= c < cols):
        raise Invalid("the cell must lie on the grid")
    z = grid[r][c]
    return math.sqrt(sum((v - z) ** 2 for v in _window(grid, r, c, 1)))


def tpi(grid: Grid, r: int, c: int, radius: int = 1) -> float:
    # topographic position index: the cell against the mean of its ring
    rows, cols = _check(grid)
    if radius < 1:
        raise Invalid("the radius must be at least one cell")
    if not (0 <= r < rows and 0 <= c < cols):
        raise Invalid("the cell must lie on the grid")
    values = _window(grid, r, c, radius)
    return grid[r][c] - sum(values) / len(values)


def window_std(grid: Grid, r: int, c: int, radius: int = 1) -> float:
    rows, cols = _check(grid)
    if not (0 <= r < rows and 0 <= c < cols):
        raise Invalid("the cell must lie on the grid")
    values = [*_window(grid, r, c, radius), grid[r][c]]
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def field(grid: Grid, index, margin: int = 1, **kwargs) -> Grid:
    rows, cols = _check(grid)
    if 2 * margin >= min(rows, cols):
        raise Invalid("the margin eats the whole grid")
    return [
        [index(grid, r, c, **kwargs) for c in range(margin, cols - margin)]
        for r in range(margin, rows - margin)
    ]


def mean_of(values: Grid) -> float:
    flat = [v for row in values for v in row]
    return sum(flat) / len(flat)


def plane(size: int, slope: float, cell: float = 1.0) -> Grid:
    return [[slope * c * cell for c in range(size)] for _ in range(size)]


def tri_plane_law(slope: float, cell: float = 1.0) -> float:
    # on a plane tilted along x: two differences of s, two of 0, four diagonals of s
    s = slope * cell
    return math.sqrt(2 * s * s + 4 * s * s)


def sine_hills(size: int, wavelength: float, amplitude: float, cell: float = 1.0) -> Grid:
    k = 2 * math.pi / wavelength
    return [
        [amplitude * math.sin(k * c * cell) * math.sin(k * r * cell) for c in range(size)]
        for r in range(size)
    ]


def noisy_plane(
    size: int, slope: float, sigma: float, rng: random.Random, cell: float = 1.0
) -> Grid:
    return [[slope * c * cell + rng.gauss(0, sigma) for c in range(size)] for _ in range(size)]


def tri_noise_law(sigma: float) -> float:
    # the root-mean-square TRI of white noise: 8 differences of variance 2 sigma squared
    return math.sqrt(8 * 2 * sigma * sigma)


def hilltop(size: int, radius: float, height: float) -> Grid:
    centre = size // 2
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            d = math.hypot(r - centre, c - centre)
            row.append(height * max(0.0, 1 - (d / radius) ** 2))
        out.append(row)
    return out


def tpi_summit_law(radius: int, hill_radius: float, height: float) -> float:
    # the summit sits above its ring by the height times the ring's mean d squared
    squares = [
        dr * dr + dc * dc
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr or dc
    ]
    return height * (sum(squares) / len(squares)) / (hill_radius * hill_radius)


def tpi_classes(values: Grid, threshold: float) -> tuple[int, int, int]:
    ridge = valley = flat = 0
    for row in values:
        for v in row:
            if v > threshold:
                ridge += 1
            elif v < -threshold:
                valley += 1
            else:
                flat += 1
    return ridge, valley, flat
