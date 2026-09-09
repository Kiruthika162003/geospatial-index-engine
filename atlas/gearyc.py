"""Geary's C: spatial autocorrelation by neighbour differences, and how it relates to Moran's I.

Moran's I asks whether neighbouring cells deviate from the mean
in the same direction; Geary's C asks whether neighbouring cells
differ from each other, summing the squared differences across
every adjacency and dividing by the variance, so that C is one
under no autocorrelation, below one when neighbours match, and
above one when they alternate, the mirror of I's sign. The two
are often described as related by C equals one minus I, and the
survey measures how close that description is, and finds it
looser than the guess. The guess was that the identity holds to
rounding wherever every cell has the same number of neighbours;
on a torus, where every cell has four, C fell short of one minus
I by 0.037 on a 4 by 4 gradient, 0.0027 at 10, and 0.00036 at
20, and by 0.068, 0.0092, and 0.0023 on shuffles of the same
grids, an error near one over the cell count that the equal
degrees do not remove, since C carries a factor of n minus one
over n that I does not. On the ordinary grid with edges the gap
is larger and depends on the pattern: a checkerboard read C of
1.875, 1.98, and 1.995 against one minus I's 2.0 at sizes 4, 10,
and 20, a two-block map 0.3125, 0.11, and 0.0525 against 0.333,
0.111, and 0.0526, and a gradient 0.1875, 0.03, and 0.0075
against the same 0.333, 0.111, and 0.0526, the gradient's C
running far below one minus I because its neighbour differences
are uniformly small while its deviations from the mean are
large, so the two statistics disagree most on smooth ramps. A
shuffle of the 10 by 10 gradient read C of 0.9999 on average over
500 shuffles with a spread of 0.076, the band round one. The
guess that a single outlier moves C several times more than I,
since C squares the differences at the outlier's adjacencies,
was also high: an outlier of 100 in the middle of the 10 by 10
gradient moved C by +0.880 and I by -0.730, a ratio of 1.21, and
an outlier of 50 in the two-block map moved them by +1.002 and
-0.904, a ratio of 1.11. The finding worth stating is that C
equals one minus I only asymptotically, with a gap near one over
n on a torus and a pattern-dependent gap on a bounded grid that
is largest on smooth ramps, that a checkerboard reads C near two
and a shuffle near one within a spread of 0.08, and that C
answers to an outlier only a fifth more than I, so the two
statistics agree on pattern and on outliers and disagree on
ramps. This module computes Geary's C with rook
weights, and a survey measures it against Moran's I.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid
from atlas.morani import morans_i, rook_pairs

Grid = Sequence[Sequence[float]]


def gearys_c(grid: Grid) -> float:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    rows, cols = len(grid), len(grid[0])
    values = [v for row in grid for v in row]
    n = len(values)
    if n < 2:
        raise Invalid("Geary's C needs at least two cells")
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values)
    if variance == 0:
        raise Invalid("a constant grid has no autocorrelation")
    pairs = rook_pairs(rows, cols)
    diffs = sum((values[i] - values[j]) ** 2 for i, j in pairs)
    # the weight sum counts each adjacency twice in the symmetric form
    return (n - 1) * (2 * diffs) / (2 * (2 * len(pairs)) * variance)


def one_minus_i_gap(grid: Grid) -> float:
    return gearys_c(grid) - (1.0 - morans_i(grid))


def shuffle_band(grid: Grid, trials: int, rng) -> tuple[float, float]:
    if trials < 2:
        raise Invalid("need at least two shuffles")
    rows, cols = len(grid), len(grid[0])
    values = [v for row in grid for v in row]
    scores = []
    for _ in range(trials):
        shuffled = list(values)
        rng.shuffle(shuffled)
        scores.append(gearys_c([shuffled[r * cols : (r + 1) * cols] for r in range(rows)]))
    mean = sum(scores) / trials
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (trials - 1))
    return mean, sd


def with_outlier(grid: Grid, row: int, col: int, value: float) -> list[list[float]]:
    out = [list(r) for r in grid]
    out[row][col] = value
    return out


def outlier_shifts(grid: Grid, row: int, col: int, value: float) -> tuple[float, float]:
    # how far C and I move when one cell is replaced by an outlier
    spiked = with_outlier(grid, row, col, value)
    return gearys_c(spiked) - gearys_c(grid), morans_i(spiked) - morans_i(grid)


def torus_pairs(rows: int, cols: int) -> list[tuple[int, int]]:
    # every cell with four neighbours, wrapping at the edges: the equal-degree case
    pairs = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            pairs.append((i, r * cols + (c + 1) % cols))
            pairs.append((i, ((r + 1) % rows) * cols + c))
    return pairs


def torus_c_and_i(grid: Grid) -> tuple[float, float]:
    rows, cols = len(grid), len(grid[0])
    values = [v for row in grid for v in row]
    n = len(values)
    mean = sum(values) / n
    dev = [v - mean for v in values]
    variance = sum(d * d for d in dev)
    pairs = torus_pairs(rows, cols)
    diffs = sum((values[i] - values[j]) ** 2 for i, j in pairs)
    cross = sum(dev[i] * dev[j] for i, j in pairs)
    c = (n - 1) * diffs / (2 * len(pairs) * variance)
    i_value = n * cross / (len(pairs) * variance)
    return c, i_value
