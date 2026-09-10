"""A still GPS receiver walks 1.77 sigma a fix; a moving track inflates by (sigma/step) squared.

Position jitter of sigma on each axis makes two fixes of a still
receiver differ by a distance whose mean is sigma root 2 root(pi
over 2), 1.7725 sigma: over 2000 fixes the mean step reads 1.762,
5.286 and 8.810 at sigma 1, 3 and 5, a fraction under the law, and
a receiver that never moved reports 1.77, 5.3 and 8.8 units a second
at one-second fixes and a tenth of that at ten-second fixes. A track
that does move inflates by less as the step outgrows the noise: at
sigma 3 a track stepping 1, 3, 5, 10, 20 and 50 a fix reads 5.39,
1.99, 1.40, 1.096, 1.023 and 1.004 times its true length, matching
a simulation of the step law to three places. The guess that the
inflation is a tenth at four and a half sigma was wrong: the
small-noise expansion puts it at 1 + sigma squared over step
squared, 1.09 at step 10 and 1.0225 at 20 against the 1.096 and
1.0229 measured, so a tenth falls at root ten sigma, 9.5 for sigma
3, and the expansion undershoots below that, 1.36 against 1.40 at
step 5.

Averaging every fix over a window of five brings the inflation to
1.41, 1.04, 1.013, 1.003 and 1.0 at steps 1 to 20, and keeping only
every fifth fix does the same, 1.39, 1.04, 1.014, 1.004 and 1.001,
since both lengthen the step against a noise that does not grow.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

from atlas.errors import Invalid

Point = tuple[float, float]


def jittered(track: list[Point], sigma: float, rng: random.Random) -> list[Point]:
    if sigma < 0:
        raise Invalid("sigma must not be negative")
    return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma)) for x, y in track]


def length(track: list[Point]) -> float:
    return sum(math.dist(a, b) for a, b in pairwise(track))


def still_track(fixes: int) -> list[Point]:
    if fixes < 2:
        raise Invalid("a track needs two fixes")
    return [(0.0, 0.0)] * fixes


def straight_track(fixes: int, step: float) -> list[Point]:
    if fixes < 2:
        raise Invalid("a track needs two fixes")
    return [(k * step, 0.0) for k in range(fixes)]


def still_step_law(sigma: float) -> float:
    # two fixes of a still receiver differ by N(0, 2 sigma^2) on each axis
    return sigma * math.sqrt(2) * math.sqrt(math.pi / 2)


def moving_step_law(step: float, sigma: float, samples: int = 200_000, seed: int = 0) -> float:
    # the mean of hypot(step + N(0, 2 s^2), N(0, 2 s^2)), by simulation
    rng = random.Random(seed)
    s = sigma * math.sqrt(2)
    total = 0.0
    for _ in range(samples):
        total += math.hypot(step + rng.gauss(0, s), rng.gauss(0, s))
    return total / samples


def inflation(track: list[Point], sigma: float, rng: random.Random, draws: int = 20) -> float:
    true = length(track)
    if true == 0:
        raise Invalid("a still track has no length to inflate")
    total = 0.0
    for _ in range(draws):
        total += length(jittered(track, sigma, rng))
    return total / draws / true


def speed_readings(
    track: list[Point], sigma: float, interval_s: float, rng: random.Random
) -> list[float]:
    if interval_s <= 0:
        raise Invalid("the interval must be positive")
    noisy = jittered(track, sigma, rng)
    return [math.dist(a, b) / interval_s for a, b in pairwise(noisy)]


def mean(values: list[float]) -> float:
    if not values:
        raise Invalid("no values")
    return sum(values) / len(values)


def smoothed(track: list[Point], window: int) -> list[Point]:
    if window < 1 or window % 2 == 0:
        raise Invalid("the window must be a positive odd number of fixes")
    half = window // 2
    out = []
    for i in range(len(track)):
        lo, hi = max(0, i - half), min(len(track), i + half + 1)
        xs = [p[0] for p in track[lo:hi]]
        ys = [p[1] for p in track[lo:hi]]
        out.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return out


def thinned(track: list[Point], every: int) -> list[Point]:
    if every < 1:
        raise Invalid("keep at least every fix")
    kept = track[::every]
    if kept[-1] != track[-1]:
        kept.append(track[-1])
    return kept


def inflation_law(step: float, sigma: float) -> float:
    # the small-noise expansion: hypot(step + e1, e2) averages step + sigma^2 / step
    if step <= 0:
        raise Invalid("the step must be positive")
    return 1 + sigma * sigma / (step * step)


def break_even_step(sigma: float) -> float:
    # the step at which the inflation falls to ten percent under the small-noise law
    return sigma * math.sqrt(10)
