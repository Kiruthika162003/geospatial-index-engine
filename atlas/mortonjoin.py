"""A window of 8 in Morton order finds 94 percent of the pairs within 5 units and 77 within 20.

A distance join can sweep the points in Morton key order and
compare each with the next few, trusting the curve to keep
neighbours close; what the curve loses is the recall. On 3000
uniform points over a 1000 square the true pairs number 331, 1417
and 5593 within 5, 10 and 20 units, near the law n(n - 1)/2 pi d
squared over the area. Comparing each point with the next 1, 2, 4,
8, 16, 32, 64, 128 and 256 in key order recovers 0.728, 0.834,
0.894, 0.940, 0.955, 0.967, 0.982, 0.988 and 0.997 of the pairs
within 5 units, 0.551, 0.718, 0.817, 0.883, 0.925, 0.950, 0.969,
0.981 and 0.991 within 10, and 0.299, 0.475, 0.645, 0.766, 0.839,
0.896, 0.929, 0.953 and 0.976 within 20, so nine tenths of the
pairs need a window of 8, 16 and 64 and ninety-nine hundredths a
window of 256, 256 and 1024, a third of the points.

The guess that the misses are the pairs straddling the curve's
central seam was wrong by an order: pairs with members on opposite
sides of the top-level quadrant boundary are 0.3, 0.6 and 1.9
percent of the true pairs at the three distances, while a window of
8 misses 6, 12 and 23 percent, since every level of the curve has
its own seams and the lower ones are far more numerous. On 1000
points the pairs within 10 and 30 units number 153 and 1422, a
window of 8 recovers 0.948 and 0.807 and a window of 64 0.987 and
0.950, and nine tenths need windows of 4 and 32.
"""

from __future__ import annotations

import math
import random

from atlas import morton
from atlas.errors import Invalid

Point = tuple[float, float]


def keys(points: list[Point], side: float, order: int = 16) -> list[int]:
    if side <= 0 or order < 1:
        raise Invalid("side and order must be positive")
    steps = (1 << order) - 1
    out = []
    for x, y in points:
        qx = min(max(int(x / side * steps), 0), steps)
        qy = min(max(int(y / side * steps), 0), steps)
        out.append(morton.encode(qx, qy))
    return out


def sweep_join(
    points: list[Point], side: float, distance: float, window: int
) -> set[tuple[int, int]]:
    # pairs within the distance, comparing each point with the next window points in key order
    if distance < 0 or window < 1:
        raise Invalid("distance must not be negative and the window must be positive")
    ks = keys(points, side)
    order = sorted(range(len(points)), key=lambda i: ks[i])
    found: set[tuple[int, int]] = set()
    d2 = distance * distance
    for pos, i in enumerate(order):
        for j in order[pos + 1 : pos + 1 + window]:
            a, b = points[i], points[j]
            if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= d2:
                found.add((min(i, j), max(i, j)))
    return found


def brute_join(points: list[Point], distance: float) -> set[tuple[int, int]]:
    d2 = distance * distance
    found = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a, b = points[i], points[j]
            if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= d2:
                found.add((i, j))
    return found


def recall(
    points: list[Point], side: float, distance: float, window: int
) -> tuple[float, int, int]:
    truth = brute_join(points, distance)
    if not truth:
        raise Invalid("no pair within the distance")
    got = sweep_join(points, side, distance, window)
    return len(got & truth) / len(truth), len(truth), window * len(points)


def window_for_recall(
    points: list[Point], side: float, distance: float, target: float, cap: int = 4096
) -> int:
    if not 0 < target <= 1:
        raise Invalid("the target recall must lie in (0, 1]")
    truth = brute_join(points, distance)
    window = 1
    while window <= cap:
        got = sweep_join(points, side, distance, window)
        if len(got & truth) / len(truth) >= target:
            return window
        window *= 2
    return cap


def uniform(n: int, rng: random.Random, side: float = 1000.0) -> list[Point]:
    return [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]


def expected_pairs(n: int, distance: float, side: float = 1000.0) -> float:
    return n * (n - 1) / 2 * math.pi * distance * distance / (side * side)


def seam_share(points: list[Point], side: float, distance: float) -> float:
    # the share of true pairs whose members sit on opposite sides of the top-level Morton seam
    truth = brute_join(points, distance)
    if not truth:
        raise Invalid("no pair within the distance")
    half = side / 2
    crossing = sum(
        1
        for i, j in truth
        if (points[i][0] < half) != (points[j][0] < half)
        or (points[i][1] < half) != (points[j][1] < half)
    )
    return crossing / len(truth)
