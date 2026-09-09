"""A grid scores an orientation order of 1.0, a Delaunay web 0.004, and a 45 degree turn 0.93.

Street orientation entropy bins the bearing of every edge, taken both
ways, into 36 slices of 10 degrees centred on the cardinal
directions, and the orientation order is 1 minus the squared
position of the entropy between a perfect grid's ln 4 and the
uniform ln 36. A 20 by 20 grid reads entropy 1.3863, which is ln 4,
and order 1.0; turned by 3, 4.9, 5.1 or 17 degrees it still reads
1.0, since every bearing stays inside one slice. The guess that any
rotation is harmless was wrong at the slice edges: turned by exactly
5 degrees the entropy reads 1.4118 and turned by 45 it reads 1.9515
and the order 0.9338, because the rotated bearings land on the edge
between two slices and floating point sends them either way.

Adding diagonals to a fraction 0.1, 0.25, 0.5 and 1.0 of the grid's
cells reads order 0.9948, 0.985, 0.9727 and 0.966 by count and
0.9919, 0.9793, 0.9679 and 0.9676 by length. A Delaunay web on 100,
300 and 1000 uniform points reads entropy 3.5571, 3.5727 and 3.5792
against ln 36 of 3.5835, order 0.0239, 0.0098 and 0.0039, and 0.0928,
0.0505 and 0.0363 weighted by length, since the long edges are the
sparser ones. A radial network of 3 rings and 8 spokes reads 0.6019,
of 3 rings and 12 spokes 0.3152, and of 6 rings and 36 spokes 0.008,
a web by another route. Two grids turned 30 degrees apart read
0.9005, 45 apart 0.7802, and 90 apart 1.0, indistinguishable from
one grid.
"""

from __future__ import annotations

import math
import random

from atlas.detourindex import Graph, delaunay_network, grid_network
from atlas.errors import Invalid

Point = tuple[float, float]
BINS = 36


def bearing(a: Point, b: Point) -> float:
    if a == b:
        raise Invalid("a bearing needs two different points")
    angle = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
    return angle % 360


def edge_bearings(points: list[Point], graph: Graph) -> list[tuple[float, float]]:
    out = []
    for i, neighbours in graph.items():
        for j, weight in neighbours.items():
            if i < j:
                theta = bearing(points[i], points[j])
                out.append((theta, weight))
                out.append(((theta + 180) % 360, weight))
    if not out:
        raise Invalid("the network has no edges")
    return out


Bearings = list[tuple[float, float]]


def histogram(bearings: Bearings, bins: int = BINS, weighted: bool = False) -> list[float]:
    if bins <= 0:
        raise Invalid("bins must be positive")
    width = 360 / bins
    counts = [0.0] * bins
    for theta, weight in bearings:
        index = int(((theta + width / 2) % 360) // width)
        counts[index] += weight if weighted else 1.0
    return counts


def entropy(counts: list[float]) -> float:
    total = sum(counts)
    if total <= 0:
        raise Invalid("the histogram is empty")
    return -sum((c / total) * math.log(c / total) for c in counts if c > 0)


def orientation_order(counts: list[float]) -> float:
    bins = len(counts)
    if bins < 4:
        raise Invalid("at least four bins are needed")
    h_max = math.log(bins)
    h_grid = math.log(4)
    h = entropy(counts)
    return 1 - ((h - h_grid) / (h_max - h_grid)) ** 2


def network_order(
    points: list[Point], graph: Graph, weighted: bool = False
) -> tuple[float, float]:
    counts = histogram(edge_bearings(points, graph), BINS, weighted)
    return entropy(counts), orientation_order(counts)


def rotated(points: list[Point], degrees: float) -> list[Point]:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def with_diagonals(
    points: list[Point], graph: Graph, side: int, fraction: float, rng: random.Random
) -> Graph:
    if not 0 <= fraction <= 1:
        raise Invalid("fraction must lie in [0, 1]")
    out: Graph = {i: dict(nb) for i, nb in graph.items()}
    for r in range(side - 1):
        for c in range(side - 1):
            if rng.random() < fraction:
                i, j = r * side + c, (r + 1) * side + c + 1
                out[i][j] = out[j][i] = math.dist(points[i], points[j])
    return out


def radial_network(rings: int, spokes: int) -> tuple[list[Point], Graph]:
    if rings < 1 or spokes < 3:
        raise Invalid("a radial network needs a ring and three spokes")
    points: list[Point] = [(0.0, 0.0)]
    for ring in range(1, rings + 1):
        for k in range(spokes):
            angle = 2 * math.pi * k / spokes
            points.append((ring * math.cos(angle), ring * math.sin(angle)))
    graph: Graph = {i: {} for i in range(len(points))}

    def join(i: int, j: int) -> None:
        graph[i][j] = graph[j][i] = math.dist(points[i], points[j])

    for k in range(spokes):
        join(0, 1 + k)
        for ring in range(1, rings):
            join(1 + (ring - 1) * spokes + k, 1 + ring * spokes + k)
    for ring in range(1, rings + 1):
        for k in range(spokes):
            join(1 + (ring - 1) * spokes + k, 1 + (ring - 1) * spokes + (k + 1) % spokes)
    return points, graph


def two_grids(side: int, angle: float) -> tuple[list[Point], Graph]:
    points, graph = grid_network(side)
    turned = [(x + 2 * side, y) for x, y in rotated(points, angle)]
    offset = len(points)
    merged: Graph = {i: dict(nb) for i, nb in graph.items()}
    for i, nb in graph.items():
        merged[i + offset] = {j + offset: w for j, w in nb.items()}
    return points + turned, merged


def uniform_web(n: int, rng: random.Random) -> tuple[list[Point], Graph]:
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    return points, delaunay_network(points)
