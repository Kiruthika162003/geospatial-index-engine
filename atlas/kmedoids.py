"""Medoids shrug off two outliers in 200 points and follow five: their loss is a plain sum.

Partitioning around medoids picks k data points as centres, builds
them greedily and then swaps any medoid for any non-medoid while the
sum of distances falls. On four Gaussian blobs of 50 points at spread
1 and gap 10 the build costs 311.87 and one swap round brings it to
257.84; the medoids sit 0.5711 from the true centres at worst, since
a medoid must be a data point, and 0.3749 from their clusters'
centroids. Lloyd's k-means from a random start sits 0.4115 away.

The guess that medoids are immune to outliers was wrong past a small
count. Add 1 or 2 points at 100 to 200 units away and the medoids do
not move, while random-start k-means spends a centre on the outliers
and merges two blobs, a worst shift of 7.172; k-means seeded with one
point from each blob survives one outlier at a shift of 3.16, the
centroid dragged by 150 over 51, and merges blobs at two. At 5, 10 and
20 outliers the medoids move too: one medoid goes to the outlier
group and the worst shift reads 11.82, because five points at 150
cost more in summed distance than merging two blobs 10 apart.

The swap phase improves the greedy build by 17.2, 17.3, 19.6 and 35.7
percent at 100, 200, 400 and 800 points in one or two rounds, and the
whole run takes 0.07, 0.20, 1.16 and 3.42 seconds, near cubic. The
guess that a medoid sits spread over root n from its centroid, 0.07
for spread 0.5, was low: the gap reads 0.16, 0.32 and 0.64 at spreads
0.5, 1 and 2, a third of the spread, and the worst distance to the
true centre 0.24, 0.47 and 0.94.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def assign(points: list[Point], medoids: list[Point]) -> list[int]:
    return [min(range(len(medoids)), key=lambda m: _dist(p, medoids[m])) for p in points]


def cost(points: list[Point], medoids: list[Point]) -> float:
    return sum(min(_dist(p, m) for m in medoids) for p in points)


def build(points: list[Point], k: int) -> list[int]:
    if k < 1 or k > len(points):
        raise Invalid("k must lie between 1 and the number of points")
    chosen: list[int] = []
    nearest = [math.inf] * len(points)
    for _ in range(k):
        best, best_gain = -1, -math.inf
        for candidate in range(len(points)):
            if candidate in chosen:
                continue
            gain = sum(
                max(0.0, nearest[i] - _dist(points[i], points[candidate]))
                if chosen
                else -_dist(points[i], points[candidate])
                for i in range(len(points))
            )
            if gain > best_gain:
                best, best_gain = candidate, gain
        chosen.append(best)
        for i in range(len(points)):
            nearest[i] = min(nearest[i], _dist(points[i], points[best]))
    return chosen


def swap(
    points: list[Point], medoids: list[int], max_rounds: int = 100
) -> tuple[list[int], int]:
    current = list(medoids)
    current_cost = cost(points, [points[m] for m in current])
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        best_cost, best_swap = current_cost, None
        for slot in range(len(current)):
            for candidate in range(len(points)):
                if candidate in current:
                    continue
                trial = list(current)
                trial[slot] = candidate
                trial_cost = cost(points, [points[m] for m in trial])
                if trial_cost < best_cost - 1e-12:
                    best_cost, best_swap = trial_cost, (slot, candidate)
        if best_swap is None:
            return current, rounds - 1
        current[best_swap[0]] = best_swap[1]
        current_cost = best_cost
    return current, rounds


def pam(points: list[Point], k: int) -> tuple[list[Point], list[int], float, int]:
    if not points:
        raise Invalid("no points to cluster")
    medoid_ids, rounds = swap(points, build(points, k))
    medoids = [points[m] for m in medoid_ids]
    return medoids, assign(points, medoids), cost(points, medoids), rounds


def centroids(points: list[Point], labels: list[int], k: int) -> list[Point]:
    sums = [[0.0, 0.0, 0] for _ in range(k)]
    for p, label in zip(points, labels, strict=True):
        sums[label][0] += p[0]
        sums[label][1] += p[1]
        sums[label][2] += 1
    return [(sx / n, sy / n) if n else (math.nan, math.nan) for sx, sy, n in sums]


def lloyd(points: list[Point], k: int, rng: random.Random, rounds: int = 50) -> list[Point]:
    if k < 1 or k > len(points):
        raise Invalid("k must lie between 1 and the number of points")
    centers = rng.sample(points, k)
    for _ in range(rounds):
        labels = assign(points, centers)
        fresh = centroids(points, labels, k)
        pairs = zip(fresh, centers, strict=True)
        fresh = [c if not math.isnan(c[0]) else old for c, old in pairs]
        if fresh == centers:
            break
        centers = fresh
    return centers


def blobs(
    k: int, per: int, rng: random.Random, spread: float = 1.0, gap: float = 10.0
) -> tuple[list[Point], list[Point]]:
    truth = [(gap * i, gap * (i % 2)) for i in range(k)]
    points = []
    for cx, cy in truth:
        points.extend((rng.gauss(cx, spread), rng.gauss(cy, spread)) for _ in range(per))
    return points, truth


def with_outliers(
    points: list[Point], count: int, rng: random.Random, far: float = 100.0
) -> list[Point]:
    strays = [(rng.uniform(far, 2 * far), rng.uniform(far, 2 * far)) for _ in range(count)]
    return points + strays


def matched_shift(found: list[Point], truth: list[Point]) -> float:
    return max(min(_dist(f, t) for f in found) for t in truth)


def medoid_gap(points: list[Point], labels: list[int], medoids: list[Point]) -> float:
    k = len(medoids)
    means = centroids(points, labels, k)
    return max(_dist(m, c) for m, c in zip(medoids, means, strict=True))
