"""A uniform-cost isochrone is an octagon holding 90 percent of the disc, not a circle.

An isochrone is the set of cells whose cost distance from a source
fits a budget. The cost-distance grid moves in eight directions, so on
uniform cost the reach is the octile ball, max(|x|, |y|) plus
(root 2 - 1) times min(|x|, |y|) at most the budget, an octagon of
area 2 root 2 times the budget squared, 0.9003 of the disc of the same
radius. Measured on a 121 by 121 grid the cell counts at budgets 5,
10, 20, 30 and 50 are 73, 285, 1137, 2549 and 7069 against the octagon
area 70.7, 282.8, 1131.4, 2545.6 and 7071.1, ratios 1.0324, 1.0076,
1.005, 1.0013 and 0.9997, and against the disc 0.9295 falling to
0.9001. The four-connected frontier holds 24, 56, 112, 168 and 280
cells, 5.6 per unit of budget. The effective cost read back from the
count runs 0.9842 at budget 5 and 1.0001 at 50.

The guess that a random cost field behaves like its mean was wrong,
and so was the guess that it behaves like the logarithmic mean. Paths
route around dear cells, so the effective cost read from the reach at
budget 40 sits below both. Costs uniform on 1 to 2, mean 1.5 and
logarithmic mean 1.4427, read 1.3349 over five seeds; 1 to 4, mean 2.5
and 2.164, read 1.9364; 1 to 10, mean 5.5 and 3.9087, read 3.8272; 0.5
to 1.5, mean 1.0 and 0.9102, read 0.8011. The ratio to the mean falls
from 0.89 to 0.70 as the spread grows.

A wall 20 columns from the source with one gap keeps 60.5 percent of
the far side at budget 30 when the gap sits on the source's row, 72.0
percent at 45 and 79.3 percent at 60. A gap 30 rows off the axis keeps
none at 30, 4.9 percent at 45 and 22.9 percent at 60, and a gap in the
corner row keeps none at any of the three, since its octile distance
is 68.3. Taking the minimum of per-source distance grids gives the
same count as the union of the per-source reaches, 207, 451, 794 and
1470 cells for 1, 2, 4 and 8 sources at budget 15 on a random field,
and the eighth source is the first to overlap another: the sum of the
single reaches is 1551, so the union keeps 94.8 percent.
"""

from __future__ import annotations

import math
import random

from atlas.costdistance import cost_distance
from atlas.errors import Invalid

Cell = tuple[int, int]
Grid = list[list[float]]
Mask = list[list[bool]]

OCTILE_BALL = 2 * math.sqrt(2)


def reach_mask(cost: Grid, source: Cell, budget: float) -> Mask:
    if budget < 0:
        raise Invalid("the budget must not be negative")
    dist = cost_distance(cost, source)
    return [[d <= budget for d in row] for row in dist]


def reach(cost: Grid, source: Cell, budget: float) -> list[Cell]:
    mask = reach_mask(cost, source, budget)
    return [(r, c) for r, row in enumerate(mask) for c, inside in enumerate(row) if inside]


def count(mask: Mask) -> int:
    return sum(sum(1 for v in row if v) for row in mask)


def band(cost: Grid, source: Cell, low: float, high: float) -> list[Cell]:
    if low < 0 or high < low:
        raise Invalid("the band must satisfy 0 <= low <= high")
    dist = cost_distance(cost, source)
    return [
        (r, c)
        for r, row in enumerate(dist)
        for c, d in enumerate(row)
        if low < d <= high or (low == 0 and d == 0)
    ]


def frontier(mask: Mask) -> list[Cell]:
    rows, cols = len(mask), len(mask[0])
    edge = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r][c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or not mask[nr][nc]:
                    edge.append((r, c))
                    break
    return edge


def nested_counts(cost: Grid, source: Cell, budgets: list[float]) -> list[int]:
    if not budgets:
        raise Invalid("at least one budget is needed")
    dist = cost_distance(cost, source)
    flat = [d for row in dist for d in row]
    return [sum(1 for d in flat if d <= b) for b in budgets]


def multi_source_distance(cost: Grid, sources: list[Cell]) -> Grid:
    if not sources:
        raise Invalid("at least one source is needed")
    best = cost_distance(cost, sources[0])
    for source in sources[1:]:
        other = cost_distance(cost, source)
        for r, row in enumerate(other):
            for c, d in enumerate(row):
                best[r][c] = min(best[r][c], d)
    return best


def multi_reach_count(cost: Grid, sources: list[Cell], budget: float) -> int:
    dist = multi_source_distance(cost, sources)
    return sum(1 for row in dist for d in row if d <= budget)


def union_reach_count(cost: Grid, sources: list[Cell], budget: float) -> int:
    reached: set[Cell] = set()
    for source in sources:
        reached.update(reach(cost, source, budget))
    return len(reached)


def octile_ball_area(budget: float) -> float:
    return OCTILE_BALL * budget * budget


def disc_area(budget: float) -> float:
    return math.pi * budget * budget


def diamond_area(budget: float) -> float:
    return 2.0 * budget * budget


def uniform_cost(size: int, value: float = 1.0) -> Grid:
    if size <= 0:
        raise Invalid("size must be positive")
    return [[value] * size for _ in range(size)]


def random_cost(size: int, low: float, high: float, rng: random.Random) -> Grid:
    if low <= 0 or high < low:
        raise Invalid("costs must satisfy 0 < low <= high")
    return [[rng.uniform(low, high) for _ in range(size)] for _ in range(size)]


def walled_cost(size: int, column: int, gap: int, wall: float = 1e6) -> Grid:
    grid = uniform_cost(size)
    for r in range(size):
        if r != gap:
            grid[r][column] = wall
    return grid


def effective_cost(cost: Grid, source: Cell, budget: float) -> float:
    cells = count(reach_mask(cost, source, budget))
    return budget / math.sqrt(cells / OCTILE_BALL)


def radius_from_count(cells: int) -> float:
    return math.sqrt(cells / OCTILE_BALL)


def shadow(cost: Grid, source: Cell, budget: float, column: int) -> tuple[int, int]:
    mask = reach_mask(cost, source, budget)
    near = sum(1 for row in mask for c, v in enumerate(row) if v and c < column)
    far = sum(1 for row in mask for c, v in enumerate(row) if v and c > column)
    return near, far
