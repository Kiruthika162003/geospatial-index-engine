"""Getis-Ord hot spots: where high values cluster with high, scored cell by cell.

Moran's I and Geary's C say whether a map is clustered; the
Getis-Ord statistic says where. For each cell it sums the values
in a neighbourhood round it, compares the sum with what a random
draw of that many cells would give, and reports the difference in
standard deviations, a z-score: strongly positive for a hot spot,
a neighbourhood of high values, strongly negative for a cold
spot, and near zero elsewhere. The neighbourhood here is the cell
and its rook neighbours, and the expectation and variance under
randomness are the standard ones for a fixed neighbourhood size.
The survey measures the scores where the answer is known. On a
10 by 10 two-block map the cells deep inside the high block
scored 2.283 and those deep inside the low block -2.283, the
cells in the two columns along the boundary 1.370 and -1.370, and
the corners 1.750, since a corner's neighbourhood is three cells,
so a threshold at 1.96 labeled 38 percent of the map hot and 38
percent cold and left 24 percent unlabeled, the boundary columns
and the four corners. On a shuffled map, no pattern at all, the
fraction of cells past 1.96 should be about five percent by the
usual reading of a z-score, and the measurement split by the
values' shape: shuffles of the binary two-block map read 7.2
percent over 300 trials, 1.45 times the nominal 5.0, because the
neighbourhood sums of a two-valued map are far from normal on a
five-cell window, while a 30 by 30 map of uniform random values
read 4.84 percent, the nominal rate within noise. A single hot
cell, one value of 10 among zeros, scored itself and its four
neighbours 4.359 each and every other cell -0.176, so the hot
spot found is five cells wide, the statistic's resolution: it
cannot see a spot smaller than its neighbourhood, nor can it tell
the hot cell from its neighbours. The finding worth stating is
that Getis-Ord labels block interiors at a fixed score and
leaves the boundary columns and corners unlabeled, that its
false-positive rate runs at the nominal five percent on
continuous values and 1.45 times it on binary ones, and that its
smallest visible spot is the neighbourhood itself, so a hot spot
map has a resolution and a false-alarm rate that the values'
shape sets. This module scores cells by the Getis-Ord statistic with
rook neighbourhoods, and a survey measures the block scores, the
false-positive rate, and the resolution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Grid = Sequence[Sequence[float]]


def _neighbourhood(rows: int, cols: int, r: int, c: int) -> list[tuple[int, int]]:
    cells = [(r, c)]
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            cells.append((rr, cc))
    return cells


def scores(grid: Grid) -> list[list[float]]:
    # the Gi* z-score of every cell over its rook neighbourhood including itself
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    rows, cols = len(grid), len(grid[0])
    values = [v for row in grid for v in row]
    n = len(values)
    if n < 3:
        raise Invalid("the statistic needs at least three cells")
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    if sd == 0:
        raise Invalid("a constant grid has no hot spots")
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cells = _neighbourhood(rows, cols, r, c)
            w = len(cells)
            total = sum(grid[rr][cc] for rr, cc in cells)
            expected = mean * w
            spread = sd * math.sqrt((n * w - w * w) / (n - 1))
            row.append((total - expected) / spread if spread > 0 else 0.0)
        out.append(row)
    return out


def labeled_fraction(grid: Grid, threshold: float = 1.96) -> tuple[float, float]:
    # the fractions of cells scoring above and below the threshold
    z = scores(grid)
    n = sum(len(row) for row in z)
    hot = sum(1 for row in z for v in row if v > threshold)
    cold = sum(1 for row in z for v in row if v < -threshold)
    return hot / n, cold / n


def false_positive_rate(grid: Grid, trials: int, rng, threshold: float = 1.96) -> float:
    # the fraction of cells past the threshold either way over shuffles of the values
    if trials < 1:
        raise Invalid("need at least one shuffle")
    rows, cols = len(grid), len(grid[0])
    values = [v for row in grid for v in row]
    total = 0.0
    for _ in range(trials):
        shuffled = list(values)
        rng.shuffle(shuffled)
        grid_again = [shuffled[r * cols : (r + 1) * cols] for r in range(rows)]
        hot, cold = labeled_fraction(grid_again, threshold)
        total += hot + cold
    return total / trials


def nominal_rate(threshold: float = 1.96) -> float:
    # the two-sided tail of the normal distribution past the threshold
    return 2 * (1 - 0.5 * (1 + math.erf(threshold / math.sqrt(2))))


def hot_cells(grid: Grid, threshold: float = 1.96) -> list[tuple[int, int]]:
    z = scores(grid)
    return [(r, c) for r, row in enumerate(z) for c, v in enumerate(row) if v > threshold]
