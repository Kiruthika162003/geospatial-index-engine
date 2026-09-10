"""Counting points on a grid estimates an area with error falling as n to the -0.73, not -0.50.

An area can be estimated by the share of sample points that fall
inside the shape, and the sampling design sets the error. For a
disc of radius 30 in a 100 square, a regular grid of 25, 100, 400,
1600 and 6400 points dropped at a random offset reads a relative
root mean square error of 0.126, 0.042, 0.0158, 0.0062 and 0.0021
over 200 offsets, while the same counts of random points read
0.307, 0.161, 0.084, 0.040 and 0.020, the binomial law root((1 -
p) over np) of 0.319, 0.159, 0.080, 0.040 and 0.020. The fitted
exponents are -0.731 for the grid and -0.496 for random sampling,
the classical n to the minus three quarters against root n, since a
grid's error comes only from the shape's boundary crossing its
cells while random points miss and double up everywhere; the grid
is 2.4 times better at 25 points and 9.6 times better at 6400.

The guess that a grid fixed at the origin is as good as a shifted
one was wrong in kind: it gives a single number with a bias and no
spread, 27.3 percent high at 5 by 5, 13.2 high at 10 by 10, 1.0 low
at 20 by 20 and 40 by 40 and 0.3 low at 80 by 80, so a survey that
never moves its grid learns nothing about its own error.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


def inside_disc(p: Point, centre: Point, radius: float) -> bool:
    return math.dist(p, centre) <= radius


def grid_estimate(
    n_side: int, side: float, centre: Point, radius: float, offset: Point = (0.0, 0.0)
) -> float:
    if n_side < 1 or side <= 0:
        raise Invalid("the grid needs a positive count and side")
    step = side / n_side
    hits = 0
    for i in range(n_side):
        for j in range(n_side):
            x = ((i + 0.5) * step + offset[0]) % side
            y = ((j + 0.5) * step + offset[1]) % side
            hits += inside_disc((x, y), centre, radius)
    return hits / (n_side * n_side) * side * side


def random_estimate(
    n: int, side: float, centre: Point, radius: float, rng: random.Random
) -> float:
    if n < 1 or side <= 0:
        raise Invalid("the sample needs a positive count and side")
    hits = sum(
        inside_disc((rng.uniform(0, side), rng.uniform(0, side)), centre, radius)
        for _ in range(n)
    )
    return hits / n * side * side


def true_area(radius: float) -> float:
    return math.pi * radius * radius


def rms_error(estimates: list[float], truth: float) -> float:
    if not estimates:
        raise Invalid("no estimates")
    return math.sqrt(sum((e - truth) ** 2 for e in estimates) / len(estimates)) / truth


def grid_errors(
    n_side: int, side: float, centre: Point, radius: float, rng: random.Random, draws: int
) -> float:
    # the grid is shifted by a random offset each draw, as a survey's origin would be
    step = side / n_side
    estimates = [
        grid_estimate(
            n_side, side, centre, radius, (rng.uniform(0, step), rng.uniform(0, step))
        )
        for _ in range(draws)
    ]
    return rms_error(estimates, true_area(radius))


def random_errors(
    n: int, side: float, centre: Point, radius: float, rng: random.Random, draws: int
) -> float:
    estimates = [random_estimate(n, side, centre, radius, rng) for _ in range(draws)]
    return rms_error(estimates, true_area(radius))


def random_law(n: int, share: float) -> float:
    # a binomial count's relative error
    return math.sqrt((1 - share) / (n * share))


def fitted_exponent(counts: list[int], errors: list[float]) -> float:
    # the slope of log error against log count
    if len(counts) != len(errors) or len(counts) < 2:
        raise Invalid("matched counts and errors are needed")
    xs = [math.log(c) for c in counts]
    ys = [math.log(e) for e in errors]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def aligned_grid_bias(n_side: int, side: float, centre: Point, radius: float) -> float:
    # a grid fixed at the origin gives one number, its own bias, not an error distribution
    return grid_estimate(n_side, side, centre, radius) / true_area(radius) - 1
