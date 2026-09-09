"""Two parabolas read a shift 0.3 cells wrong on a diagonal ridge; a 2D quadratic reads 0.02.

The shift between two rasters of the same scene is found where their
normalised cross-correlation peaks, and the peak is refined below a
cell by fitting a curve to the surface around the integer peak. The
guess that two one-dimensional parabolas, one per axis through the
integer peak, refine well enough was wrong on a smooth field: its
correlation surface is a diagonal ridge with a nearly flat top, the
integer peak can land a whole cell off along the ridge, and the
paired parabolas read a true shift of 0.5 and 0.5 cells at an error
of 0.295 cells, 0.25 and 0 at 0.11, and 0.1 and -0.4 at 0.152. A
full quadratic fitted by least squares to the 3 by 3 block reads
the same shifts at 0.021, 0.021 and 0.017 cells, and 0.015 for a
shift of -1.3 and 2.7; on a field with sharper texture it reads
0.009 to 0.027 against the parabolas' 0.066 to 0.182.

Noise on the moved raster grows the error with the field's
smoothness: at sigma 0.05, 0.2, 0.5 and 1 on a field of standard
deviation near 2 the smooth field reads mean errors of 0.018, 0.082,
0.41 and 0.86 cells over five draws and the textured field 0.008,
0.020, 0.044 and 0.090. Sweeping a pure x shift from 0 to 1 in
tenths reads errors between -0.018 and 0.014 cells at every
fraction, so the quadratic shows no pull toward whole cells. A
search of 4 cells each way on a 48 by 48 raster takes 0.024
seconds.
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


def smooth_field(size: int, rng: random.Random, waves: int = 6, top: float = 0.3) -> Grid:
    parts = []
    for _ in range(waves):
        kx, ky = rng.uniform(-top, top), rng.uniform(-top, top)
        parts.append((kx, ky, rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.0)))
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            row.append(sum(a * math.sin(kx * c + ky * r + phase) for kx, ky, phase, a in parts))
        out.append(row)
    return out


def sample(grid: Grid, y: float, x: float) -> float:
    # bilinear sample at a fractional row y and column x, clamped to the edges
    rows, cols = _check(grid)
    y = min(max(y, 0.0), rows - 1.0)
    x = min(max(x, 0.0), cols - 1.0)
    r0, c0 = math.floor(y), math.floor(x)
    r1, c1 = min(r0 + 1, rows - 1), min(c0 + 1, cols - 1)
    ty, tx = y - r0, x - c0
    top = grid[r0][c0] * (1 - tx) + grid[r0][c1] * tx
    bottom = grid[r1][c0] * (1 - tx) + grid[r1][c1] * tx
    return top * (1 - ty) + bottom * ty


def shifted(grid: Grid, dy: float, dx: float) -> Grid:
    # the scene moved by (dy, dx): the new cell (r, c) shows the old cell (r - dy, c - dx)
    rows, cols = _check(grid)
    return [[sample(grid, r - dy, c - dx) for c in range(cols)] for r in range(rows)]


def noisy(grid: Grid, sigma: float, rng: random.Random) -> Grid:
    return [[v + rng.gauss(0, sigma) for v in row] for row in grid]


def _ncc(a: Grid, b: Grid, dy: int, dx: int, margin: int) -> float:
    rows, cols = len(a), len(a[0])
    xs, ys = [], []
    for r in range(margin, rows - margin):
        for c in range(margin, cols - margin):
            xs.append(a[r][c])
            ys.append(b[r + dy][c + dx])
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def correlation_surface(a: Grid, b: Grid, max_shift: int) -> dict[tuple[int, int], float]:
    if _check(a) != _check(b):
        raise Invalid("the rasters must share a shape")
    if max_shift < 1:
        raise Invalid("max_shift must be at least one cell")
    rows, cols = len(a), len(a[0])
    if 2 * max_shift >= min(rows, cols):
        raise Invalid("max_shift must leave an overlap")
    return {
        (dy, dx): _ncc(a, b, dy, dx, max_shift)
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
    }


def integer_peak(surface: dict[tuple[int, int], float]) -> tuple[int, int]:
    return max(surface, key=lambda k: surface[k])


def parabolic_peak(surface: dict[tuple[int, int], float]) -> tuple[float, float]:
    dy, dx = integer_peak(surface)
    refined = []
    for axis in (0, 1):
        before = (dy - 1, dx) if axis == 0 else (dy, dx - 1)
        after = (dy + 1, dx) if axis == 0 else (dy, dx + 1)
        if before not in surface or after not in surface:
            refined.append(float(dy if axis == 0 else dx))
            continue
        left, centre, right = surface[before], surface[(dy, dx)], surface[after]
        denominator = left - 2 * centre + right
        offset = 0.0 if denominator == 0 else 0.5 * (left - right) / denominator
        refined.append((dy if axis == 0 else dx) + max(-0.5, min(0.5, offset)))
    return refined[0], refined[1]


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    m = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-15:
            raise Invalid("the fit is singular")
        for r in range(n):
            if r != col:
                factor = m[r][col] / m[col][col]
                for k in range(col, n + 1):
                    m[r][k] -= factor * m[col][k]
    return [m[i][n] / m[i][i] for i in range(n)]


def quadratic_peak(surface: dict[tuple[int, int], float]) -> tuple[float, float]:
    # a full quadratic fitted by least squares to the 3 by 3 block around the integer peak
    dy, dx = integer_peak(surface)
    rows = []
    values = []
    for y in (-1, 0, 1):
        for x in (-1, 0, 1):
            if (dy + y, dx + x) not in surface:
                return parabolic_peak(surface)
            rows.append([1.0, x, y, x * x, x * y, y * y])
            values.append(surface[(dy + y, dx + x)])
    normal = [[sum(r[i] * r[j] for r in rows) for j in range(6)] for i in range(6)]
    rhs = [sum(r[i] * v for r, v in zip(rows, values, strict=True)) for i in range(6)]
    _, b, c, d, e, f = _solve(normal, rhs)
    det = 4 * d * f - e * e
    if det == 0:
        return parabolic_peak(surface)
    x = (-2 * f * b + e * c) / det
    y = (-2 * d * c + e * b) / det
    x, y = max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))
    return dy + y, dx + x


def find_shift(
    a: Grid, b: Grid, max_shift: int, method: str = "quadratic"
) -> tuple[float, float, tuple[int, int]]:
    if method not in ("quadratic", "parabolic"):
        raise Invalid("method must be quadratic or parabolic")
    surface = correlation_surface(a, b, max_shift)
    coarse = integer_peak(surface)
    fine = quadratic_peak(surface) if method == "quadratic" else parabolic_peak(surface)
    return fine[0], fine[1], coarse


def error(found: tuple[float, float], truth: tuple[float, float]) -> float:
    return math.hypot(found[0] - truth[0], found[1] - truth[1])
