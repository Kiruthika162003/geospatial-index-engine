"""A road network reaches 87 percent of the disc's nodes, as a mean detour of 1.06 predicts.

A Euclidean buffer draws a disc; a network buffer keeps only what
the roads reach within the distance. On a Delaunay road network of
600 uniform nodes over a 100 square, twenty interior sources reach
97.8 percent of the nodes inside the disc at distance 5, 90.5 at
10, 87.1 at 20 and 87.2 at 30, settling where the detour law puts
it: if every trip stretches by the network's mean detour of 1.06
the reachable share of the disc is one over its square, 0.89. The
guess that thinning the network a little costs a little was wrong:
keeping 80 percent of the edges drops the share to 0.77 at every
distance and keeping 60 percent to 0.55 to 0.57, since a missing
edge forces a long way round.

A square grid of spacing 5 reaches all 13 nodes of the disc at
distance 10, 41 of 49 at 20, 85 of 113 at 30 and 181 of 253 at 45,
shares of 1.0, 0.837, 0.752 and 0.715 falling toward the Manhattan
diamond's 2 over pi of 0.637 as the lattice's coarseness fades. The
network buffer's own area estimate reads 316.7 against the disc's
314.2 at distance 10 and 1166.7 against 1256.6 at 20.
"""

from __future__ import annotations

import math
import random

from atlas.detourindex import Graph, delaunay_network, dijkstra, grid_network, sparsified
from atlas.errors import Invalid

Point = tuple[float, float]


def reachable(_points: list[Point], graph: Graph, source: int, distance: float) -> set[int]:
    if distance < 0:
        raise Invalid("the distance must not be negative")
    dist = dijkstra(graph, source)
    return {node for node, d in dist.items() if d <= distance}


def within_disc(points: list[Point], source: int, distance: float) -> set[int]:
    sx, sy = points[source]
    return {i for i, (x, y) in enumerate(points) if math.hypot(x - sx, y - sy) <= distance}


def coverage_ratio(points: list[Point], graph: Graph, source: int, distance: float) -> float:
    disc = within_disc(points, source, distance)
    if not disc:
        raise Invalid("the disc holds no node")
    return len(reachable(points, graph, source, distance) & disc) / len(disc)


def mean_coverage(
    points: list[Point], graph: Graph, distance: float, sources: list[int]
) -> float:
    if not sources:
        raise Invalid("at least one source is needed")
    return sum(coverage_ratio(points, graph, s, distance) for s in sources) / len(sources)


def network_area_estimate(
    points: list[Point], graph: Graph, source: int, distance: float, side: float
) -> float:
    # the reachable share of the nodes times the extent, as the area a network buffer covers
    return len(reachable(points, graph, source, distance)) / len(points) * side * side


def disc_area(distance: float) -> float:
    return math.pi * distance * distance


def interior_sources(
    points: list[Point], margin: float, count: int, rng: random.Random, side: float = 100.0
) -> list[int]:
    inside = [
        i
        for i, (x, y) in enumerate(points)
        if margin <= x <= side - margin and margin <= y <= side - margin
    ]
    if len(inside) < count:
        raise Invalid("not enough interior nodes")
    return rng.sample(inside, count)


def detour_law(mean_detour: float) -> float:
    # trips stretched by the mean detour reach 1 over its square of the disc
    return 1 / (mean_detour * mean_detour)


def town(n: int, seed: int) -> tuple[list[Point], Graph]:
    rng = random.Random(seed)
    points = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
    return points, delaunay_network(points)


def thinned_town(n: int, seed: int, keep: float) -> tuple[list[Point], Graph]:
    points, graph = town(n, seed)
    return points, sparsified(graph, keep, random.Random(seed + 1))


def grid(side: int, spacing: float) -> tuple[list[Point], Graph]:
    return grid_network(side, spacing)


def manhattan_law() -> float:
    # a Manhattan diamond of radius d over the Euclidean disc of radius d
    return 2 / math.pi
