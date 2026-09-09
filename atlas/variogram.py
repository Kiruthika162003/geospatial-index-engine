"""Semivariogram: how a field's values disagree with distance, measured from its samples.

Before a field can be interpolated with any honesty its samples
must be asked how far their values stay related, and the
semivariogram is the question: for every pair of samples, half
the squared difference of their values against their separation,
binned by distance and averaged, rising from near zero at short
range, where neighbours agree, to a plateau, the sill, at the
range beyond which samples are as different as any two drawn at
random. An exponential model with a nugget, a sill, and a range
is fitted to the binned curve by a coarse search over the range
with the sill read from the far bins, and the survey calibrates
the fit on fields whose structure is known, where its first guess
failed. The guess was that on a field summed from forty Gaussian
bumps of a chosen width the fitted range would recover the width
within a fraction; from 400 samples with 4-unit bins to a lag of
80 the exponential range read 37.6 for bumps of width 5, 27.6 for
width 10, and 80.0, the search's cap, for width 20, 7.5, 2.8, and
4.0 widths and not even monotone, because a sum of Gaussian bumps
has a Gaussian-shaped variogram that an exponential model fits
only loosely and the sparser narrow bumps leave a field that is
mostly flat, so the exponential range measures no clean width.
The sill did track the variance, 0.288 against 0.258, 0.599
against 0.628, and 2.10 against 1.55, the last high because a
field correlated over a fifth of its extent has a sample variance
short of its sill, and the first bin was 6, 2.4, and 0.5 percent
of the sill, the smoothness a kriging system depends on, with no
bin holding fewer than 375 pairs. On white noise the curve was
flat from the first bin, 0.94 against a sill of 0.91, the nugget
equal to the sill and the fitted range 0.4, meaningless as it
should be. On a plane the semivariance rose as the square of the
lag, 1, 4.9, 23.8, and 107 times the first bin at lags 2, 6, 14,
and 30, half the square since pairs point in random directions,
and the fit reported the cap, the signature of a trend a
variogram should not be fit to. The finding worth stating is that
the semivariogram reads white noise as pure nugget and a trend as
a range that never arrives, and reads a bump field's variance as
its sill, but that an exponential range on a Gaussian-shaped
field is not a width, so the curve tells a field's structure and
the model's limits together.
This module bins semivariances and fits the exponential model,
and a survey measures the fit on bump, noise, and trend fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from atlas.errors import Invalid

Sample = tuple[float, float, float]


def empirical(samples: Sequence[Sample], bin_width: float, max_lag: float):
    # (lag center, semivariance, pair count) per bin
    if len(samples) < 2:
        raise Invalid("a variogram needs at least two samples")
    if bin_width <= 0 or max_lag <= 0:
        raise Invalid("bin width and maximum lag must be positive")
    bins = math.ceil(max_lag / bin_width)
    total = [0.0] * bins
    count = [0] * bins
    n = len(samples)
    for i in range(n):
        xi, yi, vi = samples[i]
        for j in range(i + 1, n):
            xj, yj, vj = samples[j]
            d = math.hypot(xi - xj, yi - yj)
            if d >= max_lag:
                continue
            b = int(d / bin_width)
            total[b] += 0.5 * (vi - vj) ** 2
            count[b] += 1
    out = []
    for b in range(bins):
        if count[b] > 0:
            out.append(((b + 0.5) * bin_width, total[b] / count[b], count[b]))
    return out


def exponential(nugget: float, sill: float, range_: float) -> Callable[[float], float]:
    def model(h: float) -> float:
        if h <= 0:
            return 0.0
        return nugget + (sill - nugget) * (1 - math.exp(-3 * h / range_))

    return model


def fit(curve: Sequence[tuple[float, float, int]], max_range: float, steps: int = 200):
    # nugget from the first bin, sill from the mean of the far third, range by coarse search
    if len(curve) < 3:
        raise Invalid("the fit needs at least three bins")
    if max_range <= 0 or steps < 2:
        raise Invalid("the search needs a positive range and at least two steps")
    far = curve[2 * len(curve) // 3 :]
    sill = sum(v for _, v, _ in far) / len(far)
    nugget = max(0.0, min(curve[0][1], sill))
    best = (math.inf, max_range)
    for k in range(1, steps + 1):
        candidate = max_range * k / steps
        model = exponential(nugget, sill, candidate)
        error = sum(c * (model(h) - v) ** 2 for h, v, c in curve)
        if error < best[0]:
            best = (error, candidate)
    return nugget, sill, best[1]


def bump_field(width: float, bumps: int, rng, extent: float = 100.0):
    # a smooth field summed from Gaussian bumps of a chosen width
    centers = []
    for _ in range(bumps):
        centers.append((rng.uniform(0, extent), rng.uniform(0, extent), rng.gauss(0, 1)))

    def field(x: float, y: float) -> float:
        total = 0.0
        for cx, cy, a in centers:
            total += a * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width * width))
        return total

    return field


def sample_field(
    field: Callable[[float, float], float], count: int, rng, extent: float = 100.0
) -> list[Sample]:
    out = []
    for _ in range(count):
        x, y = rng.uniform(0, extent), rng.uniform(0, extent)
        out.append((x, y, field(x, y)))
    return out


def variance(samples: Sequence[Sample]) -> float:
    values = [v for _, _, v in samples]
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
