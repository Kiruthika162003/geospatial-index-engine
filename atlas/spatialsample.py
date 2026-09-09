"""Spatial sampling designs: where to put the sample points, measured by the error they leave.

A soil survey, a forest inventory, or a rainfall network can
afford some hundreds of measurements over a region and must
choose where to make them, and the three classical designs are
simple random, every point placed independently at random;
systematic, points on a regular grid with one random offset; and
stratified random, the region cut into equal cells with one
random point in each. All three estimate the region's mean by the
sample mean, and the survey measures how well, by the root mean
square error of that estimate over 400 repetitions of 100 points
on fields whose true mean is known, and two of its guesses were
wrong. The guess was that the systematic grid would be best by a
wide margin on any smooth field, with stratified between it and
random. On a bowl it was: random erred 0.0105, stratified 0.0025,
and systematic 0.0010, ten times better than random. On a plane
it was not: random erred 0.107 and systematic 0.105, no better,
because the grid's one shared random offset shifts the centroid
of the whole sample and a plane's mean is its value at the
centroid, so the error is the slope times the offset every time;
stratified, whose hundred offsets average out, erred 0.0101, ten
times better than either. The second guess was the textbook
warning that on stripes with the grid's own period every point
lands on the same phase and the estimate is confidently wrong,
biased by the amplitude while repetitions agree; with a random
offset per repetition the phase is random too, so the bias over
repetitions was 0.02 but the scatter was 0.71, ten times the
random and stratified designs' 0.07, a wildly scattered answer
rather than a confident one, and each single survey is wrong by
up to the full amplitude. On stripes of an unrelated period,
0.137, the three designs erred alike at 0.066 to 0.072, and on
white noise all three read 0.0983 against the 0.1 that one over
root n predicts, since no arrangement of points can exploit
structure that is not there. The finding worth stating is that
the systematic grid beats random tenfold on curved fields and not
at all on planes, that stratified sampling is never worse than
fourfold behind the best and tenfold ahead of the worst, and that
a grid at the field's own period scatters by the amplitude, so
the right design is a property of the field. This module generates the
three designs, and a survey measures their estimation error on
smooth, periodic, and noise fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from atlas.errors import Invalid

Point = tuple[float, float]
Field = Callable[[float, float], float]


def simple_random(count: int, rng) -> list[Point]:
    if count < 1:
        raise Invalid("need at least one sample")
    return [(rng.random(), rng.random()) for _ in range(count)]


def systematic(side: int, rng) -> list[Point]:
    # a side by side grid with one random offset shared by every point
    if side < 1:
        raise Invalid("need at least one sample per side")
    step = 1.0 / side
    ox, oy = rng.random() * step, rng.random() * step
    return [(ox + i * step, oy + j * step) for i in range(side) for j in range(side)]


def stratified(side: int, rng) -> list[Point]:
    if side < 1:
        raise Invalid("need at least one sample per side")
    step = 1.0 / side
    return [
        ((i + rng.random()) * step, (j + rng.random()) * step)
        for i in range(side)
        for j in range(side)
    ]


def sample_mean(field: Field, points: Sequence[Point]) -> float:
    if not points:
        raise Invalid("no points to sample")
    return sum(field(x, y) for x, y in points) / len(points)


def estimation_error(
    field: Field, design: Callable[[], list[Point]], truth: float, trials: int
) -> float:
    # the root mean square error of the sample mean over repeated draws of the design
    if trials < 1:
        raise Invalid("need at least one trial")
    total = sum((sample_mean(field, design()) - truth) ** 2 for _ in range(trials))
    return math.sqrt(total / trials)


def bias_and_scatter(
    field: Field, design: Callable[[], list[Point]], truth: float, trials: int
) -> tuple[float, float]:
    # the mean error and its standard deviation over repeated draws
    errors = [sample_mean(field, design()) - truth for _ in range(trials)]
    bias = sum(errors) / trials
    scatter = math.sqrt(sum((e - bias) ** 2 for e in errors) / trials)
    return bias, scatter


def plane(x: float, y: float) -> float:
    return 1.0 + 2.0 * x + 3.0 * y


def bowl(x: float, y: float) -> float:
    return (x - 0.5) ** 2 + (y - 0.5) ** 2


def stripes(period: float, amplitude: float = 1.0) -> Field:
    return lambda x, _y: amplitude * math.sin(2 * math.pi * x / period)


def noise_field(rng) -> Field:
    return lambda _x, _y: rng.gauss(0.0, 1.0)
