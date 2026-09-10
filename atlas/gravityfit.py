"""The log fit of a decay exponent reads 1.32 for a true 2 at a mean flow of 2; Poisson reads 2.

A gravity model sends flow between places in proportion to their
sizes over their distance to the power beta, and the exponent is
fitted from observed counts. Two fits are tried on 25 towns with
lognormal sizes: a regression of the log of flow over the size
product on minus the log of distance, dropping the pairs with no
flow, and a Poisson maximum likelihood over every pair including
the zeros, found by golden-section search on the deviance. On
noise-free flows both recover beta 2 and the scale 50 exactly. Under
Poisson counts the guess that the log fit recovers the exponent
once counts reach the tens was wrong: for a true exponent of 1 it
reads 0.23, 0.49, 0.81 and 1.02 at mean flows of 0.03, 0.32, 3.2 and
31.6 a pair, where 97, 82, 32 and 0.7 percent of the pairs are
zero, and for a true 2 it reads 0.41, 0.93 and 1.32 at mean flows of
0.02, 0.21 and 2.1, where 98.5, 92.5 and 68 percent are zero, since
the far pairs with small expected flow are the ones that fall to
zero and drop out, which flattens the slope; at a mean flow of 0.004
it reads -9.5 with a spread of 32 across ten draws. The Poisson fit
reads 1.17, 1.01, 1.001 and 0.997 for the true 1 and 2.82, 2.05,
2.02 and 1.996 for the true 2 at the same flows, within 2 percent
from a mean flow of 0.2 a pair.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


def expected_flows(
    origins: list[Point], sizes: list[float], beta: float, scale: float
) -> list[list[float]]:
    if len(origins) != len(sizes):
        raise Invalid("one size per place")
    out = []
    for i, a in enumerate(origins):
        row = []
        for j, b in enumerate(origins):
            if i == j:
                row.append(0.0)
                continue
            d = max(math.dist(a, b), 1e-9)
            row.append(scale * sizes[i] * sizes[j] / d**beta)
        out.append(row)
    return out


def poisson(mean: float, rng: random.Random) -> int:
    if mean <= 0:
        return 0
    if mean > 500:
        return max(0, round(rng.gauss(mean, math.sqrt(mean))))
    limit = math.exp(-mean)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p < limit:
            return k
        k += 1


def observed_flows(expected: list[list[float]], rng: random.Random) -> list[list[int]]:
    return [[poisson(v, rng) for v in row] for row in expected]


def log_fit(
    origins: list[Point], sizes: list[float], flows: list[list[float]], floor: float = 0.5
) -> tuple[float, float]:
    # regress log(flow / (size_i size_j)) on -log(distance) over the pairs above the floor
    xs, ys = [], []
    for i, a in enumerate(origins):
        for j, b in enumerate(origins):
            if i == j or flows[i][j] <= floor:
                continue
            xs.append(-math.log(math.dist(a, b)))
            ys.append(math.log(flows[i][j] / (sizes[i] * sizes[j])))
    if len(xs) < 2:
        raise Invalid("too few pairs with flow to fit")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise Invalid("every pair sits at the same distance")
    beta = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return beta, math.exp(my - beta * mx)


def poisson_fit(
    origins: list[Point],
    sizes: list[float],
    flows: list[list[float]],
    low: float = 0.0,
    high: float = 5.0,
) -> float:
    # the exponent that best explains the counts under Poisson noise, by golden-section search
    def deviance(beta: float) -> float:
        total_expected = total_observed = 0.0
        pairs = []
        for i, a in enumerate(origins):
            for j, b in enumerate(origins):
                if i == j:
                    continue
                d = max(math.dist(a, b), 1e-9)
                mu = sizes[i] * sizes[j] / d**beta
                pairs.append((mu, flows[i][j]))
                total_expected += mu
                total_observed += flows[i][j]
        scale = total_observed / total_expected if total_expected else 0.0
        value = 0.0
        for mu, k in pairs:
            m = scale * mu
            value += m - (k * math.log(m) if k > 0 and m > 0 else 0.0)
        return value

    ratio = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = deviance(c), deviance(d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = deviance(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = deviance(d)
    return (a + b) / 2


def total_flow(flows: list[list[float]]) -> float:
    return sum(sum(row) for row in flows)


def zero_share(flows: list[list[int]]) -> float:
    pairs = [v for i, row in enumerate(flows) for j, v in enumerate(row) if i != j]
    return sum(1 for v in pairs if v == 0) / len(pairs)


def towns(n: int, rng: random.Random, side: float = 100.0) -> tuple[list[Point], list[float]]:
    origins = [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]
    sizes = [math.exp(rng.gauss(0, 0.7)) for _ in range(n)]
    return origins, sizes
