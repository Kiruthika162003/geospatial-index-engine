"""A ten percent weight wobble holds the ranking at 0.997 and swaps 7 percent of the top decile.

A weighted overlay scores every cell by the weighted mean of
normalised criterion layers, and a planner reads the top cells as
the best sites. On four layers over a 48 grid, two smooth fields,
one of noise and one gradient, weighted 3, 2, 1 and 1, perturbing
every weight by up to 5, 10, 20 and 50 percent of itself keeps
Spearman's rank correlation with the base map at 0.9993, 0.9973,
0.9895 and 0.9413 on average over twenty draws, 0.998, 0.992, 0.969
and 0.846 at worst. The guess that so high a correlation means the
best sites stand still was wrong: the top 5 percent of cells keep
a Jaccard overlap of 0.968, 0.948, 0.896 and 0.719 with the base
set under the same wobbles, the top 10 percent 0.963, 0.928, 0.868
and 0.720 and the top 25 percent 0.981, 0.961, 0.919 and 0.776, so
a ten percent wobble swaps about 7 percent of the top decile while
the whole-map ranking barely moves.

Dropping a layer outright moves far more: without the heaviest
smooth layer the top decile keeps 0.33 of itself and the rank
correlation reads 0.62, without the second smooth layer 0.32 and
0.68, without the noise layer 0.63 and 0.92 and without the
gradient 0.75 and 0.93. Equal weights keep 0.565 of the top decile
at a correlation of 0.90, and the noise layer alone keeps 0.08, the
overlap of two random deciles.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Grid = list[list[float]]


def _check(layers: list[Grid], weights: list[float]) -> tuple[int, int]:
    if not layers or not layers[0] or not layers[0][0]:
        raise Invalid("at least one non-empty layer is needed")
    if len(layers) != len(weights):
        raise Invalid("one weight per layer")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise Invalid("weights must be non-negative and not all zero")
    rows, cols = len(layers[0]), len(layers[0][0])
    if any(len(g) != rows or len(g[0]) != cols for g in layers):
        raise Invalid("the layers must share a shape")
    return rows, cols


def normalise(grid: Grid) -> Grid:
    values = [v for row in grid for v in row]
    low, high = min(values), max(values)
    if high == low:
        return [[0.0 for _ in row] for row in grid]
    return [[(v - low) / (high - low) for v in row] for row in grid]


def overlay(layers: list[Grid], weights: list[float]) -> Grid:
    rows, cols = _check(layers, weights)
    total = sum(weights)
    scaled = [normalise(g) for g in layers]
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(sum(w * g[r][c] for w, g in zip(weights, scaled, strict=True)) / total)
        out.append(row)
    return out


def top_cells(score: Grid, fraction: float) -> set[tuple[int, int]]:
    if not 0 < fraction <= 1:
        raise Invalid("the fraction must lie in (0, 1]")
    cells = [(score[r][c], r, c) for r in range(len(score)) for c in range(len(score[0]))]
    cells.sort(reverse=True)
    keep = max(1, round(fraction * len(cells)))
    return {(r, c) for _, r, c in cells[:keep]}


def overlap(a: set, b: set) -> float:
    if not a or not b:
        raise Invalid("empty sets have no overlap")
    return len(a & b) / len(a | b)


def perturbed_weights(weights: list[float], amount: float, rng: random.Random) -> list[float]:
    if amount < 0:
        raise Invalid("the perturbation must not be negative")
    return [max(0.0, w * (1 + rng.uniform(-amount, amount))) for w in weights]


def stability(
    layers: list[Grid],
    weights: list[float],
    fraction: float,
    amount: float,
    rng: random.Random,
    draws: int = 20,
) -> float:
    base = top_cells(overlay(layers, weights), fraction)
    total = 0.0
    for _ in range(draws):
        other = top_cells(overlay(layers, perturbed_weights(weights, amount, rng)), fraction)
        total += overlap(base, other)
    return total / draws


def rank_correlation(a: Grid, b: Grid) -> float:
    # Spearman's rho over the cells
    xs = [v for row in a for v in row]
    ys = [v for row in b for v in row]
    if len(xs) != len(ys) or len(xs) < 2:
        raise Invalid("the grids must share a shape with at least two cells")

    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, i in enumerate(order):
            out[i] = float(rank)
        return out

    ra, rb = ranks(xs), ranks(ys)
    n = len(xs)
    d2 = sum((p - q) ** 2 for p, q in zip(ra, rb, strict=True))
    return 1 - 6 * d2 / (n * (n * n - 1))


def smooth_layer(size: int, rng: random.Random, waves: int = 3) -> Grid:
    parts = [
        (rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2), rng.uniform(0, 2 * math.pi))
        for _ in range(waves)
    ]
    return [
        [sum(math.sin(kx * c + ky * r + ph) for kx, ky, ph in parts) for c in range(size)]
        for r in range(size)
    ]


def noise_layer(size: int, rng: random.Random) -> Grid:
    return [[rng.random() for _ in range(size)] for _ in range(size)]


def gradient_layer(size: int) -> Grid:
    return [[float(c) for c in range(size)] for _ in range(size)]
