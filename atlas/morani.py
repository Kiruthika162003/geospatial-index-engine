"""Moran's I: whether neighbouring cells hold similar values, and what random looks like.

A grid of values, rainfall by cell, income by district, votes by
precinct, raises the question whether neighbours resemble each
other more than chance would have them. Moran's I answers with a
correlation between each cell's deviation from the mean and its
neighbours' deviations, weighted by adjacency and normalized by
the variance: near +1 when neighbours match, near -1 when they
alternate, and near zero when the map is shuffled. Two things
about the baseline are worth measuring rather than assuming. The
expected value under no autocorrelation is not zero but minus one
over n minus one, a small negative number that matters on small
grids, and the survey confirms it by shuffling a grid's values
500 times and averaging the I of the shuffles, -0.0069 against
the expectation -0.0101 on a 10 by 10 grid, within the standard
error of 0.003 that the shuffles' spread of 0.070 implies. And
the extremes are not exactly plus and minus one: a checkerboard,
the most alternating pattern a rook adjacency allows, reads
exactly -1 on square grids of 4, 5, 7, 10, and 20, since every
neighbour of a high cell is low, while a two-block map, one half
high and one half low, falls short of +1 by twice the fraction of
adjacencies that straddle the boundary, since each straddling
pair swings from +1 to -1: a 10 by 10 grid split in two reads
0.889, not the guessed 0.8, its 10 straddling pairs of 180 costing
2 times 0.056. The guess that a smooth gradient would read higher
than the blocks because its boundary is everywhere and gentle was
wrong in an exact way: the gradient r + c read 0.667, 0.889, and
0.947 on grids of 4, 10, and 20, identical to the blocks at every
size, since along every adjacency the gradient's deviations differ
by exactly one step and the products sum to the same total. The
survey also measures the z-score, the observed I minus its
expectation over the standard deviation of the shuffles, which is
the honest way to call a map clustered: the blocks scored 11.4
standard deviations above random, the gradient 12.8, the
checkerboard 13.4 below, and a shuffled map -0.66. The finding
worth stating is that Moran's I has a negative baseline of minus
one over n minus one, reaches -1 on a checkerboard exactly and
falls short of +1 on blocks by twice their boundary fraction, and
that a z-score from shuffles separates pattern from chance where
the raw value alone cannot. This module
computes I with rook weights and its shuffle z-score, and a
survey measures the baseline, the extremes, and the scores.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Grid = Sequence[Sequence[float]]


def rook_pairs(rows: int, cols: int) -> list[tuple[int, int]]:
    # each adjacency once as a pair of flat indices
    pairs = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                pairs.append((i, i + 1))
            if r + 1 < rows:
                pairs.append((i, i + cols))
    return pairs


def morans_i(grid: Grid) -> float:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    rows, cols = len(grid), len(grid[0])
    values = [v for row in grid for v in row]
    n = len(values)
    if n < 2:
        raise Invalid("Moran's I needs at least two cells")
    mean = sum(values) / n
    dev = [v - mean for v in values]
    variance = sum(d * d for d in dev)
    if variance == 0:
        raise Invalid("a constant grid has no autocorrelation")
    pairs = rook_pairs(rows, cols)
    cross = sum(dev[i] * dev[j] for i, j in pairs)
    # each pair counts twice in the symmetric weight sum, and twice in the cross sum
    return (n / (2 * len(pairs))) * (2 * cross / variance)


def expected_i(n: int) -> float:
    if n < 2:
        raise Invalid("the expectation needs at least two cells")
    return -1.0 / (n - 1)


def shuffle_scores(grid: Grid, trials: int, rng) -> tuple[float, float]:
    # the mean and standard deviation of I over shuffles of the values
    if trials < 2:
        raise Invalid("need at least two shuffles")
    rows, cols = len(grid), len(grid[0])
    values = [v for row in grid for v in row]
    scores = []
    for _ in range(trials):
        shuffled = list(values)
        rng.shuffle(shuffled)
        scores.append(morans_i([shuffled[r * cols : (r + 1) * cols] for r in range(rows)]))
    mean = sum(scores) / trials
    sd = math.sqrt(sum((s - mean) ** 2 for s in scores) / (trials - 1))
    return mean, sd


def z_score(grid: Grid, trials: int, rng) -> float:
    mean, sd = shuffle_scores(grid, trials, rng)
    if sd == 0:
        raise Invalid("the shuffles did not vary; no z-score")
    return (morans_i(grid) - mean) / sd


def checkerboard(size: int) -> list[list[float]]:
    return [[float((r + c) % 2) for c in range(size)] for r in range(size)]


def blocks(size: int) -> list[list[float]]:
    return [[1.0 if c < size // 2 else 0.0 for c in range(size)] for r in range(size)]


def gradient(size: int) -> list[list[float]]:
    return [[float(r + c) for c in range(size)] for r in range(size)]


def boundary_fraction(size: int) -> float:
    # the fraction of rook adjacencies that straddle the two-block boundary
    pairs = rook_pairs(size, size)
    return size / len(pairs)
