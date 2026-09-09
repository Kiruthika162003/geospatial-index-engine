"""Hypsometry: how much of a landscape lies above each height, in one curve and one number.

A catchment's shape in the vertical is summarized by the
hypsometric curve, the fraction of its area lying above each
elevation plotted against the elevation scaled from the lowest
point to the highest, and by the hypsometric integral, the area
under that curve, which is the mean elevation's position between
the floor and the peak. Old, worn-down basins have low integrals,
most of their area near the floor; young, uplifted ones have high
integrals, a plateau with valleys cut into it; and the number is
read from the grid alone. The survey calibrates the curve and the
integral on shapes whose values are known, and one guess was
wrong. A plane tilted across the grid read one half exactly, its
elevations uniform between floor and peak. A cone has an integral
of one third, the mean height of a cone being a third of its
peak, and the survey read 0.3333 on a 321-cell grid masked to the
cone's disc, but 0.2618 over the whole square, since the square's
flat corners are landscape too and pull the mean down, so the
integral belongs to a catchment and not to a grid. A paraboloid
dome, height falling as the square of the distance from the
summit, read 0.4999 over its disc, one half, since the mean of
the squared radius over a disc is one half; and a bowl, its
mirror, was guessed at two thirds and read 0.5001, one half as
well, the two summing to 1.0000 at every size as mirror images
must. The reading converges with the cell: the cone over its
disc read 0.3135, 0.3272, 0.3318, 0.3323, 0.3334, and 0.3333 at
sizes 11 to 321, and the dome 0.4696 to 0.4999, the coarse grids
low because a cell center rarely lands on the apex. The curve on
the cone read the area above a relative height h as 0.5627,
0.2503, and 0.0625 at h of a quarter, a half, and three quarters
against (1 - h) squared, and its quantiles placed a quarter, a
half, and three quarters of the area below relative heights of
0.134, 0.293, and 0.500, which are 1 minus the square root of 1
minus the fraction, so a cone's median height is 29.3 percent of
its peak. The finding worth stating is that the hypsometric
integral reads a plane as one half, a cone as one third over its
own disc and 0.262 over a square, and both a dome and a bowl as
one half, converging within 0.02 by 41 cells, and that a cone's
quantiles follow 1 minus root (1 minus f), so a landscape's stage
is a number that a grid can compute for the catchment it is
asked about and not for the square it comes in. This module computes
hypsometric curves and integrals, and a survey calibrates them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Grid = Sequence[Sequence[float]]


def _values(grid: Grid) -> list[float]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return [h for row in grid for h in row]


def integral(grid: Grid) -> float:
    # the mean elevation's position between the floor and the peak
    values = _values(grid)
    low, high = min(values), max(values)
    if high == low:
        raise Invalid("a flat grid has no hypsometric integral")
    return (sum(values) / len(values) - low) / (high - low)


def curve(grid: Grid, steps: int = 20) -> list[tuple[float, float]]:
    # (relative elevation, fraction of area above it) at evenly spaced relative elevations
    if steps < 1:
        raise Invalid("need at least one step")
    values = _values(grid)
    low, high = min(values), max(values)
    if high == low:
        raise Invalid("a flat grid has no hypsometric curve")
    out = []
    for k in range(steps + 1):
        rel = k / steps
        threshold = low + rel * (high - low)
        above = sum(1 for v in values if v >= threshold)
        out.append((rel, above / len(values)))
    return out


def quantiles(grid: Grid, fractions: Sequence[float]) -> list[float]:
    # the relative elevations below which the given fractions of the area lie
    values = sorted(_values(grid))
    low, high = values[0], values[-1]
    if high == low:
        raise Invalid("a flat grid has no quantiles")
    out = []
    for f in fractions:
        if not 0 <= f <= 1:
            raise Invalid("fractions must lie within 0 and 1")
        index = min(len(values) - 1, int(f * len(values)))
        out.append((values[index] - low) / (high - low))
    return out


def plane(size: int) -> list[list[float]]:
    return [[float(c + r) for c in range(size)] for r in range(size)]


def cone(size: int) -> list[list[float]]:
    center = (size - 1) / 2
    radius = size / 2
    return [
        [max(0.0, 1 - math.hypot(c - center, r - center) / radius) for c in range(size)]
        for r in range(size)
    ]


def dome(size: int) -> list[list[float]]:
    center = (size - 1) / 2
    radius = size / 2
    return [
        [max(0.0, 1 - (math.hypot(c - center, r - center) / radius) ** 2) for c in range(size)]
        for r in range(size)
    ]


def bowl(size: int) -> list[list[float]]:
    return [[1 - h for h in row] for row in dome(size)]


def disc_mask(size: int) -> list[list[bool]]:
    center = (size - 1) / 2
    radius = size / 2
    cells = range(size)
    return [[math.hypot(c - center, r - center) <= radius for c in cells] for r in cells]


def masked(grid: Grid, mask: Sequence[Sequence[bool]]) -> list[list[float]]:
    # keep only the cells inside the mask, as a single row for the integral
    kept = []
    for row, mrow in zip(grid, mask, strict=True):
        kept.extend(h for h, m in zip(row, mrow, strict=True) if m)
    return [kept]
