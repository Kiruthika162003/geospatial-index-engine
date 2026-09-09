"""The cell sum meets a cone's volume within 0.02 percent to cells of 8; corner prisms trail.

Cut and fill are the volumes between two surfaces, read here two
ways: a cell sum of the height change at cell centres times the cell
area, and a prism sum over corner heights with each cell split into
two triangular prisms. The guess that the prisms, using four heights
a cell, would beat the single centre value was wrong from cells of 2
to 8, and at cell 1 the two agree to 0.001 percent. A cone of
radius 50 and height 30, volume 78,539.8, reads by cells 0.001,
0.004, -0.016, 0.016 and 0.452 percent off at cells of 1, 2, 4, 8 and
12, and by prisms -0.001, -0.007, 0.044, 0.187 and -0.315 percent;
a paraboloid of volume 117,809.7 reads 0.001, 0.006, -0.017, -0.095
and 0.703 percent by cells against -0.001, -0.011, 0.044, 0.278 and
-0.817 by prisms. The centre sample is a midpoint rule, whose errors
cancel across a cell, while the corner prisms are a trapezoid rule
that leans one way on a curved surface. Sliding the cone's centre by
half and a quarter of an 8-cell reads 0.016, -0.014 and -0.075
percent, so the grid's phase matters more than its method there. The
cone shows no cut at any cell size, as it should.

The height of a level platform that balances cut against fill is the
mean height of the surface, found by bisection to 1e-6 on a tilted
plane, 6.0 at slope 0.1 and 30.0 at slope 0.5 with cut and fill
equal at 21,600 and 108,000, and on the cone, 5.453298 both ways.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from atlas.errors import Invalid

Grid = list[list[float]]
Field = Callable[[float, float], float]


def _check(before: Grid, after: Grid) -> tuple[int, int]:
    if not before or not before[0]:
        raise Invalid("the surfaces are empty")
    if len(before) != len(after) or len(before[0]) != len(after[0]):
        raise Invalid("the surfaces must share a shape")
    return len(before), len(before[0])


def by_cells(before: Grid, after: Grid, cell: float) -> tuple[float, float]:
    rows, cols = _check(before, after)
    if cell <= 0:
        raise Invalid("the cell size must be positive")
    cut = fill = 0.0
    area = cell * cell
    for r in range(rows):
        for c in range(cols):
            change = (after[r][c] - before[r][c]) * area
            if change < 0:
                cut -= change
            else:
                fill += change
    return cut, fill


def by_prisms(before: Grid, after: Grid, cell: float) -> tuple[float, float]:
    # the grids hold corner heights; each cell is two triangular prisms
    rows, cols = _check(before, after)
    if cell <= 0:
        raise Invalid("the cell size must be positive")
    if rows < 2 or cols < 2:
        raise Invalid("corner grids need at least two rows and columns")
    cut = fill = 0.0
    half = cell * cell / 2
    for r in range(rows - 1):
        for c in range(cols - 1):
            d = [
                after[r][c] - before[r][c],
                after[r][c + 1] - before[r][c + 1],
                after[r + 1][c + 1] - before[r + 1][c + 1],
                after[r + 1][c] - before[r + 1][c],
            ]
            for tri in ((d[0], d[1], d[2]), (d[0], d[2], d[3])):
                change = sum(tri) / 3 * half
                if change < 0:
                    cut -= change
                else:
                    fill += change
    return cut, fill


def sample_centres(field: Field, size: float, cell: float) -> Grid:
    n = round(size / cell)
    if n < 1:
        raise Invalid("the cell must fit the size")
    return [[field((c + 0.5) * cell, (r + 0.5) * cell) for c in range(n)] for r in range(n)]


def sample_corners(field: Field, size: float, cell: float) -> Grid:
    n = round(size / cell)
    if n < 1:
        raise Invalid("the cell must fit the size")
    return [[field(c * cell, r * cell) for c in range(n + 1)] for r in range(n + 1)]


def flat(height: float = 0.0) -> Field:
    return lambda _x, _y: height


def cone(cx: float, cy: float, radius: float, height: float) -> Field:
    def field(x: float, y: float) -> float:
        return max(0.0, height * (1 - math.hypot(x - cx, y - cy) / radius))

    return field


def paraboloid(cx: float, cy: float, radius: float, height: float) -> Field:
    def field(x: float, y: float) -> float:
        d2 = ((x - cx) ** 2 + (y - cy) ** 2) / (radius * radius)
        return max(0.0, height * (1 - d2))

    return field


def cone_volume(radius: float, height: float) -> float:
    return math.pi * radius * radius * height / 3


def paraboloid_volume(radius: float, height: float) -> float:
    return math.pi * radius * radius * height / 2


def tilted(slope: float) -> Field:
    return lambda x, _y: slope * x


def balance_height(
    before: Grid, cell: float, low: float, high: float, steps: int = 60
) -> float:
    rows, cols = _check(before, before)
    if low >= high:
        raise Invalid("the search needs low < high")
    for _ in range(steps):
        mid = (low + high) / 2
        after = [[mid] * cols for _ in range(rows)]
        cut, fill = by_cells(before, after, cell)
        if fill > cut:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def mean_height(grid: Grid) -> float:
    rows, cols = _check(grid, grid)
    return sum(sum(row) for row in grid) / (rows * cols)
