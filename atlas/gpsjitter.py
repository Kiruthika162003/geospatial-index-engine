"""A still GPS walks 1.77 sigma a fix; a moving track grows by 1 + (sigma over step) squared.

Position jitter of sigma on each axis puts a random step between
any two fixes. A receiver standing still for 2000 fixes reads a
mean step of 1.762, 5.286 and 8.810 at sigma 1, 3 and 5, within 0.6
percent of the law sigma root 2 root(pi over 2), 1.7725 sigma,
since each axis of the difference has twice the variance; at one
fix a second that is an apparent speed of 1.77, 5.31 and 8.84 units
a second, and at one fix every ten seconds a tenth of it. A track
walked at a step of 1, 3, 5, 10, 20 and 50 units a fix under sigma
3 reads a length 5.39, 1.99, 1.40, 1.096, 1.023 and 1.004 times its
own, matching a simulation of the expected step to three places.
The guess that ten percent of inflation is left by a step of four
and a half sigma was wrong: for a step well over the noise the
inflation is 1 + sigma squared over step squared, 1.09, 1.0225 and
1.0036 at steps 10, 20 and 50, so ten percent is reached at sigma
over root 0.1, 3.16 sigma, 9.49 units here.

Averaging every fix with its two neighbours on each side brings the
inflation to 1.41, 1.039, 1.013, 1.003, 1.000 and 0.999 at the six
steps, and keeping every fifth fix reads 1.39, 1.037, 1.0135, 1.0035,
1.001 and 1.0002, since both cut the noise steps five ways; the
smoothed track shortens a straight track slightly at wide steps.
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


def small_noise_law(step: float, sigma: float) -> float:
    # for a step well over the noise, the inflation is 1 + sigma squared over step squared
    if step <= 0:
        raise Invalid("the step must be positive")
    return 1 + (sigma / step) ** 2


def break_even_step(sigma: float) -> float:
    # the step at which the inflation of a moving track falls to ten percent
    return sigma / math.sqrt(0.1)
