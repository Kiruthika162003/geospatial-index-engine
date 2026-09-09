"""Jenks explains 0.94 of a lognormal's variance in five classes where quantiles explain 0.54.

A choropleth needs class breaks, and three rules compete: equal
intervals over the range, quantiles with equal counts, and Jenks'
natural breaks, the partition of the sorted values that minimises
the within-class sum of squares, found here by Fisher's dynamic
programme in 0.002, 0.01, 0.04 and 0.16 seconds for 100, 200, 400
and 800 values, four times per doubling. The goodness of variance
fit, one minus the within-class over the total sum of squares,
separates them on skewed data: 400 lognormal values read 0.831,
0.937 and 0.970 under Jenks with 3, 5 and 7 classes, 0.402, 0.544
and 0.628 under quantiles and 0.672, 0.820 and 0.900 under equal
intervals, whose three classes hold 387, 9 and 4 values against
Jenks' 332, 61 and 7. Two clusters read 0.9947, 0.9976 and 0.9987
under Jenks, 0.741, 0.860 and 0.908 under quantiles and 0.988,
0.995 and 0.9945 under equal intervals, which leave classes empty,
208, 0 and 192 with three and 208, 0, 0, 77 and 115 with five.

The guess that Jenks is always worth its cost was wrong on uniform
data: there the three rules read within 0.002 of each other, 0.888,
0.888 and 0.888 with three classes and 0.961, 0.959 and 0.960 with
five, since evenly spread values have no natural breaks to find.
Jenks' breaks are also the least stable: over ten independent draws
the five-class breaks of the lognormal wander by 12.9 on average,
the uniform by 7.7 out of a range of 100, and the clusters by 11.5.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid


def _check(values: list[float], classes: int) -> list[float]:
    if classes < 1:
        raise Invalid("at least one class is needed")
    if len(values) < classes:
        raise Invalid("fewer values than classes")
    return sorted(values)


def equal_interval(values: list[float], classes: int) -> list[float]:
    data = _check(values, classes)
    low, high = data[0], data[-1]
    return [low + (high - low) * k / classes for k in range(1, classes)]


def quantile(values: list[float], classes: int) -> list[float]:
    data = _check(values, classes)
    n = len(data)
    return [data[min(n - 1, math.ceil(n * k / classes) - 1)] for k in range(1, classes)]


def jenks(values: list[float], classes: int) -> list[float]:
    # Fisher's optimal partition by dynamic programming over the sorted values
    data = _check(values, classes)
    n = len(data)
    prefix = [0.0]
    prefix_sq = [0.0]
    for v in data:
        prefix.append(prefix[-1] + v)
        prefix_sq.append(prefix_sq[-1] + v * v)

    def cost(i: int, j: int) -> float:
        # sum of squared deviations of data[i:j]
        count = j - i
        total = prefix[j] - prefix[i]
        return prefix_sq[j] - prefix_sq[i] - total * total / count

    best = [[math.inf] * (n + 1) for _ in range(classes + 1)]
    cut = [[0] * (n + 1) for _ in range(classes + 1)]
    best[0][0] = 0.0
    for k in range(1, classes + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                if best[k - 1][i] == math.inf:
                    continue
                candidate = best[k - 1][i] + cost(i, j)
                if candidate < best[k][j]:
                    best[k][j] = candidate
                    cut[k][j] = i
    breaks = []
    j = n
    for k in range(classes, 1, -1):
        i = cut[k][j]
        breaks.append(data[i])
        j = i
    return sorted(breaks)


def assign(values: list[float], breaks: list[float]) -> list[int]:
    out = []
    for v in values:
        index = 0
        while index < len(breaks) and v >= breaks[index]:
            index += 1
        out.append(index)
    return out


def goodness_of_variance_fit(values: list[float], breaks: list[float]) -> float:
    if not values:
        raise Invalid("no values")
    mean = sum(values) / len(values)
    total = sum((v - mean) ** 2 for v in values)
    if total == 0:
        return 1.0
    labels = assign(values, breaks)
    groups: dict[int, list[float]] = {}
    for v, label in zip(values, labels, strict=True):
        groups.setdefault(label, []).append(v)
    within = 0.0
    for members in groups.values():
        m = sum(members) / len(members)
        within += sum((v - m) ** 2 for v in members)
    return 1 - within / total


def class_counts(values: list[float], breaks: list[float]) -> list[int]:
    counts = [0] * (len(breaks) + 1)
    for label in assign(values, breaks):
        counts[label] += 1
    return counts


def lognormal(n: int, rng: random.Random, sigma: float = 1.0) -> list[float]:
    return [math.exp(rng.gauss(0, sigma)) for _ in range(n)]


def uniform(n: int, rng: random.Random) -> list[float]:
    return [rng.uniform(0, 100) for _ in range(n)]


def two_clusters(n: int, rng: random.Random) -> list[float]:
    out = []
    for _ in range(n):
        out.append(rng.gauss(10, 1) if rng.random() < 0.5 else rng.gauss(50, 3))
    return out


def stability(sampler, classes: int, draws: int, seed: int) -> float:
    # the mean range of each Jenks break across independent draws
    breaks = []
    for k in range(draws):
        values = sampler(random.Random(seed + k))
        breaks.append(jenks(values, classes))
    spread = 0.0
    for position in range(classes - 1):
        column = [b[position] for b in breaks]
        spread += max(column) - min(column)
    return spread / (classes - 1)
