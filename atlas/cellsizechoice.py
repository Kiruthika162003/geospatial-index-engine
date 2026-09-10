"""The half-root-of-area-per-point cell rule leaves 78 percent of cells empty, not 2 percent.

A raster made from points needs a cell, and a common rule takes a
fraction of the root of the area per point. For 2000 uniform points
over a 1000 square the rule at a half gives a cell of 11.18 with
0.25 points a cell. The guess that such a cell leaves a couple of
percent of cells empty was wrong by a factor of forty: it leaves
78.3 percent empty, the Poisson law e to the minus 0.25 giving
77.9. Factors of 0.25, 1, 2 and 4 give 0.063, 1, 4 and 16 points a
cell and leave 93.9, 37.7, 3.6 and 1.4 percent empty against the law
93.9, 36.8, 1.8 and 0.0, the coarse grids running over the law
because their edge cells are partial; the count's coefficient of
variation falls from 3.99 to 0.44. Five percent empty needs a cell
of 38.7, 1.73 times the rule, and one percent a cell of 48.0.

The rule and the law both fail on clustered points: eight Gaussian
clusters of spread 40 leave 95.4, 89.3, 82.5, 75.4 and 67.4 percent
of the cells empty at the five factors, since the points occupy a
fixed corner of the extent whatever the cell, and the coefficient
of variation reads 5.6 down to 2.7.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


def rule_of_thumb(extent: float, n: int, factor: float = 0.5) -> float:
    # a common rule: the cell is a fraction of the root of the area per point
    if n < 1 or extent <= 0 or factor <= 0:
        raise Invalid("n, extent and factor must be positive")
    return factor * math.sqrt(extent * extent / n)


def counts(points: list[Point], extent: float, cell: float) -> list[list[int]]:
    if cell <= 0 or extent <= 0:
        raise Invalid("cell and extent must be positive")
    size = math.ceil(extent / cell)
    grid = [[0] * size for _ in range(size)]
    for x, y in points:
        c = min(size - 1, int(x / cell))
        r = min(size - 1, int(y / cell))
        grid[r][c] += 1
    return grid


def empty_share(grid: list[list[int]]) -> float:
    cells = [v for row in grid for v in row]
    return sum(1 for v in cells if v == 0) / len(cells)


def poisson_empty_law(points_per_cell: float) -> float:
    return math.exp(-points_per_cell)


def points_per_cell(n: int, extent: float, cell: float) -> float:
    return n * cell * cell / (extent * extent)


def coefficient_of_variation(grid: list[list[int]]) -> float:
    cells = [v for row in grid for v in row]
    m = sum(cells) / len(cells)
    if m == 0:
        raise Invalid("no points")
    return math.sqrt(sum((v - m) ** 2 for v in cells) / len(cells)) / m


def uniform(n: int, rng: random.Random, extent: float) -> list[Point]:
    return [(rng.uniform(0, extent), rng.uniform(0, extent)) for _ in range(n)]


def clustered(
    n: int, rng: random.Random, extent: float, clusters: int = 8, spread: float = 40.0
) -> list[Point]:
    centres = [(rng.uniform(0, extent), rng.uniform(0, extent)) for _ in range(clusters)]
    out = []
    for i in range(n):
        cx, cy = centres[i % clusters]
        out.append(
            (
                min(max(rng.gauss(cx, spread), 0.0), extent - 1e-9),
                min(max(rng.gauss(cy, spread), 0.0), extent - 1e-9),
            )
        )
    return out


def cell_for_empty_share(n: int, extent: float, target: float) -> float:
    # the cell at which a Poisson field leaves the target share of cells empty
    if not 0 < target < 1:
        raise Invalid("the target must lie in (0, 1)")
    per_cell = -math.log(target)
    return math.sqrt(per_cell * extent * extent / n)


def sweep(
    points: list[Point], extent: float, factors: list[float]
) -> dict[float, dict[str, float]]:
    n = len(points)
    out = {}
    for factor in factors:
        cell = rule_of_thumb(extent, n, factor)
        grid = counts(points, extent, cell)
        out[factor] = {
            "cell": cell,
            "per_cell": points_per_cell(n, extent, cell),
            "empty": empty_share(grid),
            "law": poisson_empty_law(points_per_cell(n, extent, cell)),
            "cv": coefficient_of_variation(grid),
        }
    return out
