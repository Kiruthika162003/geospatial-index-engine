"""An octile corridor between two ends is a band 1 + t/(root 2 - 1) wide, not an ellipse.

A least-cost corridor holds every cell through which a path from
one end to the other costs no more than the best path plus a
tolerance, read here as the sum of two cost-distance fields. On
uniform cost with the ends 60 cells apart along a row the best path
costs 60.0 and the corridor at tolerances 0, 1, 2, 5, 10 and 20
holds 61, 179, 299, 761, 1451 and 2777 cells. The guess that the
corridor is the ellipse with the ends as foci, area pi a root(a
squared - c squared), was wrong under the eight-move metric: the
ellipse reads 527, 761, 1276, 1982 and 3325 cells, up to three
times the count, while the corridor's width across the axis reads a
constant 3, 5, 13, 25 and 49 cells along the middle, a band, since
a cell k off the axis costs 2k(root 2 - 1) of detour whatever its
position along the path; the band law 1 + t/(root 2 - 1) reads 3.4,
5.8, 13.1, 25.1 and 49.3, and a lattice count under the octile
metric reads 150, 287, 715, 1411 and 2758, within 6 percent from
tolerance 5.

On a rough field of costs uniform on 1 to 5 the best path costs
136.6 against 180 for the straight line at the mean cost, 24
percent less, since it seeks the cheap cells; at tolerance 0 the
path's own cells fall into 3 pieces through floating-point ties,
and tolerances of 5, 10, 20 and 40 make one corridor of 519, 1064,
1864 and 3164 cells with widths that wander from 4 to 12, 14 to 22,
25 to 36 and 42 to 54. A wall with a five-cell gap twenty cells off
the axis lifts the best cost to 74.9 and pinches the corridor at
the gap: at tolerance 2 the band is 16 wide either side and 3 at
the gap, at tolerance 10 29 and 5.
"""

from __future__ import annotations

import math
import random

from atlas.costdistance import cost_distance
from atlas.errors import Invalid

Grid = list[list[float]]
Cell = tuple[int, int]
Mask = list[list[bool]]


def corridor_field(cost: Grid, a: Cell, b: Cell) -> Grid:
    # the least cost of a path from a to b passing through each cell
    da = cost_distance(cost, a)
    db = cost_distance(cost, b)
    return [[x + y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(da, db, strict=True)]


def best_cost(field: Grid) -> float:
    return min(v for row in field for v in row)


def corridor(field: Grid, tolerance: float) -> Mask:
    if tolerance < 0:
        raise Invalid("the tolerance must not be negative")
    floor = best_cost(field)
    return [[v <= floor + tolerance for v in row] for row in field]


def cell_count(mask: Mask) -> int:
    return sum(sum(1 for v in row if v) for row in mask)


def width_profile(mask: Mask, a: Cell, b: Cell, samples: int = 9) -> list[int]:
    # the corridor's width across the straight line from a to b, sampled along it
    if samples < 1:
        raise Invalid("at least one sample is needed")
    rows, cols = len(mask), len(mask[0])
    dr, dc = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dr, dc)
    if length == 0:
        raise Invalid("the ends coincide")
    nr, nc = -dc / length, dr / length
    widths = []
    for k in range(1, samples + 1):
        t = k / (samples + 1)
        pr, pc = a[0] + dr * t, a[1] + dc * t
        count = 0
        for s in range(-max(rows, cols), max(rows, cols) + 1):
            r, c = round(pr + nr * s), round(pc + nc * s)
            if 0 <= r < rows and 0 <= c < cols and mask[r][c]:
                count += 1
        widths.append(count)
    return widths


def ellipse_area(separation: float, tolerance: float) -> float:
    # the Euclidean corridor: the ellipse with foci at the ends and string 2a = d + t
    a = (separation + tolerance) / 2
    c = separation / 2
    if a <= c:
        return 0.0
    return math.pi * a * math.sqrt(a * a - c * c)


def octile_corridor_law(separation: float, tolerance: float, samples: int = 400) -> float:
    # the same set under the octile metric, by counting a fine lattice
    a = (separation + tolerance) / 2 + 1
    count = 0
    step = 2 * a / samples
    for i in range(samples):
        for j in range(samples):
            x = -a + (i + 0.5) * step
            y = -a + (j + 0.5) * step
            d1 = _octile(x + separation / 2, y)
            d2 = _octile(x - separation / 2, y)
            if d1 + d2 <= separation + tolerance:
                count += 1
    return count * step * step


def _octile(dx: float, dy: float) -> float:
    dx, dy = abs(dx), abs(dy)
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def band_width_law(tolerance: float) -> float:
    # a cell k off the axis costs 2k(root 2 - 1) of detour under the octile metric
    return 1 + tolerance / (math.sqrt(2) - 1)


def uniform_cost(size: int) -> Grid:
    return [[1.0] * size for _ in range(size)]


def rough_cost(size: int, rng: random.Random, low: float = 1.0, high: float = 5.0) -> Grid:
    return [[rng.uniform(low, high) for _ in range(size)] for _ in range(size)]


def barrier_cost(size: int, column: int, gap: int, wall: float = 1e6) -> Grid:
    grid = uniform_cost(size)
    for r in range(size):
        if abs(r - gap) > 2:
            grid[r][column] = wall
    return grid


def components(mask: Mask) -> int:
    rows, cols = len(mask), len(mask[0])
    seen = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if not mask[r][c] or seen[r][c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                y, x = stack.pop()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if (
                            0 <= yy < rows
                            and 0 <= xx < cols
                            and mask[yy][xx]
                            and not seen[yy][xx]
                        ):
                            seen[yy][xx] = True
                            stack.append((yy, xx))
    return count
