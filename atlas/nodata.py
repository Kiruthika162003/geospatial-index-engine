"""Holes read as zero drag a mean by the true mean times the hole share; ignored, they cost.

A raster with nodata cells can be read four ways: ignoring the
holes, treating them as zero, filling them with the valid mean, or
filling each from its neighbours. On a 60-cell ramp of mean 129.5
and slope 1 with 5, 10 and 30 percent of cells punched at random,
ignoring the holes biases the mean by -0.014, -0.061 and 0.029 and
reads the slope 1.0 exactly, while reading them as zero biases it by
-6.24, -12.68 and -39.05, which is the true mean times the hole
share within 0.06, fakes a slope of 12.9, 23.9 and 54.6 across the
hole edges and inflates the standard deviation from 17.4 to 32.5,
41.8 and 61.2. The guess that filling with the mean is as good as
ignoring was wrong for anything but the mean itself: it reads the
same mean bias but a slope of 2.16, 3.36 and 6.58 and a deviation
shrunk to 16.9, 16.5 and 14.5, since every filled cell is a step to
the middle. Filling from the neighbours reads the mean to 0.003, the
slope to 0.999, 0.998 and 0.993 and the deviation to 17.3.

Ignoring is unbiased only when the holes are random. Hide the
highest 5 and 20 percent of a hill of mean 18.985, as cloud hides
peaks, and ignoring biases the mean by -1.61 and -6.14, zero by
-2.52 and -8.72, and the neighbours by -0.10 and -1.45, since they
climb part way up the missing peak. A 10 by 10 block hole on the
ramp biases the ignored mean by 0.143 because it sits below the
middle, and the neighbours restore the mean exactly, 0.0, by the
symmetry of the fill, though a cell in the block's middle reads
127.4 against the ramp's 125, since each ring of the fill averages
the ring before it. A grid with no valid cell is refused.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Grid = list[list[float | None]]
POLICIES = ("ignore", "zero", "fill_mean", "fill_neighbours")


def _check(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return len(grid), len(grid[0])


def valid_values(grid: Grid) -> list[float]:
    return [v for row in grid for v in row if v is not None]


def hole_fraction(grid: Grid) -> float:
    rows, cols = _check(grid)
    return sum(1 for row in grid for v in row if v is None) / (rows * cols)


def resolve(grid: Grid, policy: str) -> list[list[float]]:
    # replace every hole by the policy's value so that a plain statistic can run
    rows, cols = _check(grid)
    if policy not in POLICIES:
        raise Invalid("policy must be one of " + ", ".join(POLICIES))
    if policy == "zero":
        return [[0.0 if v is None else v for v in row] for row in grid]
    if policy == "fill_mean":
        values = valid_values(grid)
        if not values:
            raise Invalid("no valid cell to take a mean from")
        mean = sum(values) / len(values)
        return [[mean if v is None else v for v in row] for row in grid]
    if policy == "fill_neighbours":
        out = [[v for v in row] for row in grid]
        pending = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] is None]
        for _ in range(rows + cols):
            if not pending:
                break
            still = []
            for r, c in pending:
                near = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (
                            (dr or dc)
                            and 0 <= rr < rows
                            and 0 <= cc < cols
                            and grid[rr][cc] is not None
                        ):
                            near.append(grid[rr][cc])
                if near:
                    out[r][c] = sum(near) / len(near)
                else:
                    still.append((r, c))
            pending = still
            grid = [[v for v in row] for row in out]
        if pending:
            raise Invalid("some holes have no valid neighbour at any distance")
        return out
    return [[v for v in row if v is not None] for row in grid]


def mean_under(grid: Grid, policy: str) -> float:
    if policy == "ignore":
        values = valid_values(grid)
        if not values:
            raise Invalid("no valid cell")
        return sum(values) / len(values)
    filled = resolve(grid, policy)
    return sum(sum(row) for row in filled) / (len(filled) * len(filled[0]))


def std_under(grid: Grid, policy: str) -> float:
    if policy == "ignore":
        values = valid_values(grid)
    else:
        values = [v for row in resolve(grid, policy) for v in row]
    if not values:
        raise Invalid("no valid cell")
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def slope_under(grid: Grid, policy: str) -> float:
    # the mean absolute step between horizontal neighbours, skipping pairs at a hole
    rows, cols = _check(grid)
    filled = grid if policy == "ignore" else resolve(grid, policy)
    total = count = 0.0
    for r in range(rows):
        for c in range(cols - 1):
            a, b = filled[r][c], filled[r][c + 1]
            if a is None or b is None:
                continue
            total += abs(b - a)
            count += 1
    if count == 0:
        raise Invalid("no valid pair")
    return total / count


def zero_bias_law(true_mean: float, hole_share: float) -> float:
    return -true_mean * hole_share


def punch_random(grid: list[list[float]], share: float, rng: random.Random) -> Grid:
    if not 0 <= share <= 1:
        raise Invalid("the share must lie in [0, 1]")
    return [[None if rng.random() < share else v for v in row] for row in grid]


def punch_high(grid: list[list[float]], share: float) -> Grid:
    # the highest cells go missing, as when cloud or saturation hides the peaks
    values = sorted(v for row in grid for v in row)
    cut = values[max(0, math.ceil(len(values) * (1 - share)) - 1)] if share > 0 else math.inf
    return [[None if (share > 0 and v >= cut) else v for v in row] for row in grid]


def punch_block(grid: list[list[float]], r0: int, c0: int, size: int) -> Grid:
    return [
        [None if r0 <= r < r0 + size and c0 <= c < c0 + size else v for c, v in enumerate(row)]
        for r, row in enumerate(grid)
    ]


def ramp(size: int, slope: float, offset: float = 100.0) -> list[list[float]]:
    return [[offset + slope * c for c in range(size)] for _ in range(size)]


def hill(size: int, height: float) -> list[list[float]]:
    centre = (size - 1) / 2
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            d = math.hypot(r - centre, c - centre) / centre
            row.append(height * max(0.0, 1 - d * d))
        out.append(row)
    return out
