"""Quadrat counts: the variance-to-mean ratio of points per cell, and how cell size bends it.

The oldest test for clustering lays a grid of equal cells, the
quadrats, over a point pattern, counts the points in each, and
compares the variance of the counts to their mean. Points
scattered at random fall in each cell as a Poisson count, whose
variance equals its mean, so the ratio is one; clustered points
pile into some cells and leave others empty, inflating the
variance above the mean; evenly spread points give every cell
nearly the same count and a ratio below one. The survey measures
the ratio on a random scatter of 400 points, a clustered pattern
of 20 parents with 20 offspring spread 0.02, and a jittered 20 by
20 grid, across cell counts from 2 to 40 per side, and three
guesses about the curves were wrong. The guess that a random
scatter reads near one at every cell size held only on average:
a single scatter read 0.47 with 4 cells, 1.00 with 16, 1.38 with
64, 1.33 with 100, and 1.05 with 400 and more, wandering by half
either way as the cell count changed, while over 30 scatters the
mean was 1.035 with 100 cells and 1.016 with 400. The guess that
the clustered ratio keeps rising with cell size was wrong at the
top: it rose from 3.2 at cells of 0.025 through 7.6, 9.5, 13.9,
11.5, and 16.0 to a peak of 16.7 at cells of 0.25, about three
cluster diameters, then fell to 7.6 at cells of 0.5, since four
cells each hold several clusters and average them out. And the
guess that the grid reads low only near its spacing and climbs
smoothly away from it was wrong in kind: the ratio is exactly 0.0
whenever the cells per side divide the grid's 20, at 2, 4, 5, 10,
and 20, every cell then holding the same count, and 0.25, 0.33,
and 0.36 at 8, 16, and 25, which do not divide it, and 0.75 at
40, where each half-spacing cell holds 0 or 1 point with
probability a quarter, a Bernoulli count whose variance over mean
is one minus the probability; the grid's reading is a fact of
divisibility, not of distance from the spacing. The survey also
reads the chi-squared statistic, the ratio times the cell count
less one, 132 on 99 degrees of freedom for the scatter at 100
cells, and checks that the counts sum to the point count at every
cell size. The finding worth stating is that the variance-to-mean
ratio reads random as one only on average across scatters, reads
clustering most strongly at cells a few cluster diameters wide,
and reads a grid as zero or not by whether the cells tile it, so
a single quadrat size tells one story and the curve across sizes
tells the truth. This
module counts quadrats and computes the ratio, and a survey
measures it across cell sizes on three patterns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def counts(points: Sequence[Point], cells_per_side: int, size: float = 1.0) -> list[int]:
    if cells_per_side < 1:
        raise Invalid("need at least one cell per side")
    if size <= 0:
        raise Invalid("the window size must be positive")
    out = [0] * (cells_per_side * cells_per_side)
    step = size / cells_per_side
    for x, y in points:
        if not (0 <= x <= size and 0 <= y <= size):
            raise Invalid("a point lies outside the window")
        i = min(cells_per_side - 1, int(x / step))
        j = min(cells_per_side - 1, int(y / step))
        out[i * cells_per_side + j] += 1
    return out


def variance_to_mean(quadrats: Sequence[int]) -> float:
    if len(quadrats) < 2:
        raise Invalid("the ratio needs at least two quadrats")
    mean = sum(quadrats) / len(quadrats)
    if mean == 0:
        raise Invalid("no points fell in any quadrat")
    variance = sum((q - mean) ** 2 for q in quadrats) / (len(quadrats) - 1)
    return variance / mean


def ratio(points: Sequence[Point], cells_per_side: int, size: float = 1.0) -> float:
    return variance_to_mean(counts(points, cells_per_side, size))


def chi_squared(quadrats: Sequence[int]) -> tuple[float, int]:
    # the statistic and its degrees of freedom under randomness
    return variance_to_mean(quadrats) * (len(quadrats) - 1), len(quadrats) - 1


def ratio_curve(points: Sequence[Point], sides: Sequence[int], size: float = 1.0):
    return [(s, ratio(points, s, size)) for s in sides]


def random_scatter(n: int, rng) -> list[Point]:
    return [(rng.random(), rng.random()) for _ in range(n)]


def clustered(parents: int, per_parent: int, spread: float, rng) -> list[Point]:
    out: list[Point] = []
    for _ in range(parents):
        px, py = rng.random(), rng.random()
        for _ in range(per_parent):
            out.append(
                (min(1.0, max(0.0, px + rng.gauss(0, spread))),
                 min(1.0, max(0.0, py + rng.gauss(0, spread))))
            )
    return out


def jittered_grid(side: int, jitter: float, rng) -> list[Point]:
    step = 1.0 / side
    out: list[Point] = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) * step + rng.uniform(-jitter, jitter) * step
            y = (j + 0.5) * step + rng.uniform(-jitter, jitter) * step
            out.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    return out


def poisson_expectation() -> float:
    return 1.0


def cell_side(cells_per_side: int, size: float = 1.0) -> float:
    return size / cells_per_side


def log_spacing(cells: int, side: int) -> float:
    # how far the cell side is from the grid spacing, in doublings
    return math.log2(cell_side(cells) / (1.0 / side))
