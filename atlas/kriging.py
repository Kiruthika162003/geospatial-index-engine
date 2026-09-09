"""Ordinary kriging: the best linear unbiased estimate between samples, with its own error bar.

Inverse distance weighting guesses the weights; kriging derives
them. Given a semivariogram model, the weights that give an
unbiased estimate with the least variance are the solution of a
linear system built from the model's values between every pair of
samples and between each sample and the target, with a Lagrange
multiplier forcing the weights to sum to one, and the same system
yields the estimate's variance, an error bar that is zero at a
sample and grows with distance from the samples, which no other
interpolator provides. The module solves the system by Gaussian
elimination on the nearest samples and the survey measures what
the theory promises, and two of the promises as guessed were
wrong. At a sample the estimate returned the sample's value to
six places and the kriging variance was exactly zero. Between
samples on a field summed from Gaussian bumps of width 10,
sampled at 300 points and fitted with the exponential model of
range 31.6 and sill 1.41, the kriging error at 200 random probes
was 0.133 against 0.190 for inverse distance weighting at its
best power, better by 1.43, since the weights account for the
samples' arrangement. The guess that the kriging variance is an
honest error bar, the truth lying within one standard deviation
of the estimate about 68 percent of the time, was wrong for this
field: the mean kriging standard deviation was 0.624 against a
root mean square error of 0.133, 4.7 times too wide, so every one
of the 200 probes lay within one standard deviation, 96 percent
within half, and 76 percent within a tenth, because the
exponential model is far rougher at short range than a
Gaussian-bump field, so the error bar is honest only for a field
whose variogram the model fits. The weights summed to one to
twelve places, and the screening effect showed on a constructed
layout: two samples half a unit apart ten units west of the
target took weights of 0.258 and 0.249, together barely more than
the 0.493 of one sample ten units east, each of the pair counting
for half. The guess that a nugget turns the exact interpolator
into a smoother, the estimate at a sample no longer equal to it,
was wrong at the sample and right beside it: with a nugget of
three tenths of the sill the estimate at the sample was still the
sample's value to four places with zero variance, since the
system's diagonal holds the model at zero distance, but a
millionth of a unit away it read -0.1401 against the sample's
-0.1296, a jump of 0.0105 with a variance of 0.668, so a nugget
makes the surface discontinuous at the samples rather than
inexact at them. The finding worth stating is that kriging beats
inverse distance weighting by 1.43 on a field whose structure its
model roughly captures, that its variance is an error bar only as
honest as the model, here 4.7 times too wide, and that a nugget
buys smoothness between samples at the cost of a step beside
each, so the interpolator's honesty is a number it computes from
a model it did not check. This module
kriges from samples and a variogram model, and a survey measures
the error, the error bar, the screening, and the nugget.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from atlas.errors import Invalid

Sample = tuple[float, float, float]
Model = Callable[[float], float]


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    a = [[*row, rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise Invalid("the kriging system is singular; duplicate samples?")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                f = a[r][col] / a[col][col]
                for c in range(col, n + 1):
                    a[r][c] -= f * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def krige(
    samples: Sequence[Sample], model: Model, x: float, y: float, neighbours: int = 16
) -> tuple[float, float]:
    # the estimate and the kriging variance at (x, y) from the nearest samples
    if len(samples) < 2:
        raise Invalid("kriging needs at least two samples")
    if neighbours < 2:
        raise Invalid("need at least two neighbours")
    near = sorted(samples, key=lambda s: math.hypot(s[0] - x, s[1] - y))[:neighbours]
    n = len(near)
    matrix = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = model(math.hypot(near[i][0] - near[j][0], near[i][1] - near[j][1]))
        matrix[i][n] = 1.0
        matrix[n][i] = 1.0
    rhs = [model(math.hypot(s[0] - x, s[1] - y)) for s in near] + [1.0]
    weights = _solve(matrix, rhs)
    estimate = sum(w * s[2] for w, s in zip(weights[:n], near, strict=True))
    variance = sum(w * g for w, g in zip(weights[:n], rhs[:n], strict=True)) + weights[n]
    return estimate, max(0.0, variance)


def weights_at(samples: Sequence[Sample], model: Model, x: float, y: float) -> list[float]:
    n = len(samples)
    matrix = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(n):
            gap = math.hypot(samples[i][0] - samples[j][0], samples[i][1] - samples[j][1])
            matrix[i][j] = model(gap)
        matrix[i][n] = 1.0
        matrix[n][i] = 1.0
    rhs = [model(math.hypot(s[0] - x, s[1] - y)) for s in samples] + [1.0]
    return _solve(matrix, rhs)[:n]


def rms_error(samples, model, field, probes, neighbours: int = 16) -> float:
    if not probes:
        raise Invalid("need at least one probe")
    total = 0.0
    for px, py in probes:
        total += (krige(samples, model, px, py, neighbours)[0] - field(px, py)) ** 2
    return math.sqrt(total / len(probes))


def coverage(samples, model, field, probes, sigmas: float = 1.0, neighbours: int = 16) -> float:
    # the fraction of probes whose true error lies within `sigmas` kriging standard deviations
    if not probes:
        raise Invalid("need at least one probe")
    inside = 0
    for px, py in probes:
        estimate, var = krige(samples, model, px, py, neighbours)
        if abs(estimate - field(px, py)) <= sigmas * math.sqrt(var):
            inside += 1
    return inside / len(probes)
