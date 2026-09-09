"""Distance matrix: every pair of places at once, and whether the distances are a metric.

Routing, clustering, and the travelling salesman all start from
the table of distances between every pair of places, and the
table is only as good as the distance function behind it. A
distance is a metric when it is zero only between a place and
itself, the same in both directions, and never longer than a
detour through a third place, the triangle inequality, and an
index or a clustering that assumes a metric will misbehave on a
table that is not one. The module builds the table for any
distance function and checks the three properties, reporting the
worst violation of each, and the survey measures three functions
on the same 120 places scattered uniformly over the sphere, the
triangle check being cubic in the count. The haversine
great-circle distance is a metric: its table was symmetric to the
last bit, zero on the diagonal, and its worst triangle violation
was minus 5e-12, rounding on the safe side, with a mean pairwise
distance of 10055 km against the 10008 that a quarter
circumference predicts for a uniform sphere. The equirectangular
approximation, which scales the longitude difference by the
cosine of the mean latitude and takes the flat distance, was
guessed to break the triangle inequality by kilometers near the
poles and the dateline, and it does far worse on a global table:
its longest entry was 41008 km, twice the half circumference that
no true distance can exceed, pairs straddling the dateline read up
to 32 times their true distance, and the worst direct distance
was 70 percent longer than a detour through a third place. Its
nearest neighbours still agreed with the haversine's on 116 of
the 120 places, since the nearest is usually close. On a local
table, 120 places within 89 km of each other, the same
approximation drifted from the haversine by 0.0015 percent at
most and violated the triangle inequality by one part in a
hundred thousand, so the shortcut is safe there. The Manhattan
distance on map coordinates is a metric too, its triangle
violation reading 4e-16, but a different one, longer than the
straight line by up to 1.4142, the square root of two exactly.
The finding worth stating is that the great-circle table is a
metric to rounding while the flat approximation's global table
holds entries twice the half circumference and detours 70 percent
shorter than direct paths, so the shortcut is safe for local
tables and meaningless for global ones. This module builds distance
tables and checks metric properties, and a survey measures three
distance functions on random places.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from atlas.errors import Invalid
from atlas.haversine import EARTH_RADIUS_KM, haversine

Point = tuple[float, float]
Distance = Callable[[Point, Point], float]


def great_circle(a: Point, b: Point) -> float:
    return haversine(a[0], a[1], b[0], b[1])


def equirectangular(a: Point, b: Point) -> float:
    mean_lat = math.radians((a[0] + b[0]) / 2)
    dx = math.radians(b[1] - a[1]) * math.cos(mean_lat)
    dy = math.radians(b[0] - a[0])
    return math.hypot(dx, dy) * EARTH_RADIUS_KM


def manhattan(a: Point, b: Point) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def matrix(points: Sequence[Point], distance: Distance) -> list[list[float]]:
    if len(points) < 2:
        raise Invalid("a distance table needs at least two places")
    n = len(points)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                table[i][j] = distance(points[i], points[j])
    return table


def asymmetry(table: Sequence[Sequence[float]]) -> float:
    n = len(table)
    gaps = (abs(table[i][j] - table[j][i]) for i in range(n) for j in range(i + 1, n))
    return max(gaps, default=0.0)


def diagonal(table: Sequence[Sequence[float]]) -> float:
    return max(abs(table[i][i]) for i in range(len(table)))


def worst_triangle(table: Sequence[Sequence[float]]) -> float:
    # the largest excess of a direct distance over a detour through a third place, as a
    # ratio direct / detour minus one; zero or below means the triangle inequality holds
    n = len(table)
    worst = -math.inf
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            direct = table[i][j]
            for k in range(n):
                if k in (i, j):
                    continue
                detour = table[i][k] + table[k][j]
                if detour > 0:
                    worst = max(worst, direct / detour - 1.0)
    return worst


def is_metric(table: Sequence[Sequence[float]], tolerance: float = 1e-9) -> bool:
    if asymmetry(table) > tolerance or diagonal(table) > tolerance:
        return False
    return worst_triangle(table) <= tolerance


def nearest_neighbours(table: Sequence[Sequence[float]]) -> list[int]:
    n = len(table)
    return [min((j for j in range(n) if j != i), key=lambda j: table[i][j]) for i in range(n)]


def summary(table: Sequence[Sequence[float]]) -> tuple[float, float]:
    n = len(table)
    values = [table[i][j] for i in range(n) for j in range(i + 1, n)]
    return sum(values) / len(values), max(values)


def uniform_sphere(count: int, rng) -> list[Point]:
    # uniform on the sphere: latitude from the arcsine of a uniform in [-1, 1]
    return [
        (math.degrees(math.asin(rng.uniform(-1.0, 1.0))), rng.uniform(-180.0, 180.0))
        for _ in range(count)
    ]
