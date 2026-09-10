"""A gravity exponent's log fit reads 0.44 for a true 1 when 90 percent of pairs are empty.

A gravity model sends flows between places in proportion to their
sizes over their distance to the power beta, and the exponent is
fitted from observed counts. On clean flows a log-linear regression
recovers beta 1.5 and its scale exactly, and so does a Poisson fit
that searches the exponent minimising the deviance. Counts are not
clean: they are Poisson, and far or small pairs read zero. The guess
that the log fit only loses precision on sparse counts was wrong;
it loses its centre. With a true beta of 1 over 30 towns, totals of
12, 123, 1268 and 12,612 trips leave 98.7, 89.7, 51.3 and 6.8
percent of the pairs empty, and the log fit, which must drop them,
reads -0.06, 0.44, 0.71 and 0.97 on average over ten draws, since
the pairs it drops are exactly the far ones where the decay shows,
while the Poisson fit reads 1.23, 1.06, 1.003 and 0.998. With a true
beta of 2 the emptiness is worse, 77.4 percent at 851 trips, and the
log fit reads 1.195 against the Poisson fit's 1.995; at 87 trips
0.69 against 2.01, and at 8 trips the Poisson fit reads 2.33 and at
a single trip 4.04, where nothing can be fitted. The log fit's
spread across the ten draws reads 1.02, 0.33, 0.14 and 0.05 at the
four totals for beta 1.
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
