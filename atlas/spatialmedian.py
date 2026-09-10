"""One outlier moves the centroid by D over n + 1 and the geometric median by 0.006 at any.

The centroid minimises summed squared distance and the geometric
median summed distance, found by Weiszfeld's reweighted averaging.
On a Gaussian cloud of 200 points the two sit 0.1 apart, the median
reaching 1e-9 in 27 steps; the summed distance reads 230.98 at the
median against 231.77 at the centroid and the summed squares 341.8
at the centroid against 343.8 at the median. One outlier at 10,
100 and 1000 units moves the centroid by 0.0498, 0.4975 and 4.975,
the law D over n + 1 to four places, and the median by 0.0063 at
all three distances, since a far point pulls with a unit vector
whatever its range; the medoid, the data point of least summed
distance, does not move at all. Ten to 150 outliers at 100 in
random directions move the median by 0.014 to 0.062, their pulls
cancelling, while the centroid moves 1.1 to 3.9.

The guess that the median resists any minority was wrong only at
the edge: 50, 100 and 150 outliers stacked at one point 100 away
move it by 0.32, 0.73 and 1.35, 199 outliers against 200 points
move it by 9.5, and 201, one more than the points, move it by 99.4,
onto the outliers: the breakdown point is exactly one half. Two
hundred outliers in random directions never move it past 5.
Weiszfeld takes 30, 26 and 28 steps to 1e-9 on clouds of 50, 200
and 1000 points, and on 21 points along a line lands on the middle
point in one step, where median and centroid coincide.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


def centroid(points: list[Point]) -> Point:
    if not points:
        raise Invalid("no points")
    n = len(points)
    return sum(p[0] for p in points) / n, sum(p[1] for p in points) / n


def weiszfeld(
    points: list[Point], tolerance: float = 1e-9, max_steps: int = 1000
) -> tuple[Point, int]:
    # the point minimising the summed distance, by iteratively reweighted averaging
    if not points:
        raise Invalid("no points")
    if tolerance <= 0 or max_steps < 1:
        raise Invalid("tolerance and max_steps must be positive")
    x, y = centroid(points)
    for step in range(1, max_steps + 1):
        weights = []
        for px, py in points:
            d = math.hypot(px - x, py - y)
            if d < 1e-12:
                return (px, py), step
            weights.append(1 / d)
        total = sum(weights)
        nx = sum(w * p[0] for w, p in zip(weights, points, strict=True)) / total
        ny = sum(w * p[1] for w, p in zip(weights, points, strict=True)) / total
        moved = math.hypot(nx - x, ny - y)
        x, y = nx, ny
        if moved < tolerance:
            return (x, y), step
    return (x, y), max_steps


def medoid(points: list[Point]) -> Point:
    if not points:
        raise Invalid("no points")
    return min(points, key=lambda p: sum(math.dist(p, q) for q in points))


def summed_distance(points: list[Point], centre: Point) -> float:
    return sum(math.dist(p, centre) for p in points)


def summed_squares(points: list[Point], centre: Point) -> float:
    return sum(math.dist(p, centre) ** 2 for p in points)


def with_outlier(points: list[Point], distance: float, angle_deg: float = 0.0) -> list[Point]:
    cx, cy = centroid(points)
    a = math.radians(angle_deg)
    return [*points, (cx + distance * math.cos(a), cy + distance * math.sin(a))]


def with_outliers(
    points: list[Point], count: int, distance: float, rng: random.Random
) -> list[Point]:
    cx, cy = centroid(points)
    out = list(points)
    for _ in range(count):
        a = rng.uniform(0, 2 * math.pi)
        out.append((cx + distance * math.cos(a), cy + distance * math.sin(a)))
    return out


def shift(before: Point, after: Point) -> float:
    return math.dist(before, after)


def centroid_shift_law(distance: float, n: int) -> float:
    return distance / (n + 1)


def breakdown_count(
    points: list[Point], distance: float, rng: random.Random, limit: float, cap: int = 200
) -> int:
    # the smallest number of far outliers that drags the median beyond the limit
    base, _ = weiszfeld(points)
    for count in range(1, cap + 1):
        moved, _ = weiszfeld(
            with_outliers(points, count, distance, random.Random(rng.random()))
        )
        if shift(base, moved) > limit:
            return count
    return cap


def gaussian_cloud(n: int, rng: random.Random, sigma: float = 1.0) -> list[Point]:
    return [(rng.gauss(0, sigma), rng.gauss(0, sigma)) for _ in range(n)]


def line_cloud(n: int) -> list[Point]:
    return [(float(k), 0.0) for k in range(n)]
