"""A Delaunay road network detours 6 percent on average, a grid 27, and one bridge 41 to 85.

The detour index of a pair of places is the network distance over the
straight line between them. On a Delaunay network of 300 uniform
points, 881 edges and 5.87 edges a node, 500 random pairs read a mean
of 1.0609, a median of 1.057 and a worst of 1.2595, far under the
proven stretch bound near 2 for Delaunay graphs. The mean hardly
moves with size: 1.0698, 1.062, 1.0555 and 1.0584 at 50, 100, 200 and
400 points, with the worst pair at 1.32 to 1.37. A square grid of 20
by 20 reads a mean of 1.2708 against the Manhattan law 4 over pi of
1.2732, a median of 1.3077 and a worst of root 2.

Thinning the Delaunay network at random raises the detour slowly and
then breaks it: keeping 90, 80, 70, 60 and 50 percent of the edges
reads means of 1.1008, 1.1491, 1.2149, 1.3552 and 1.5355, worst pairs
of 2.25, 2.25, 3.67, 4.51 and 4.51, and 0, 0, 1.8, 7 and 8.6 percent
of pairs disconnected. The guess that the long edges are the ones to
drop was wrong at the median: cutting every edge above the median
length keeps 441 edges and disconnects 93 percent of pairs; cutting
above 1.5 times the median keeps 696 and disconnects 3.2 percent with
a worst detour of 7.42; cutting above twice the median keeps 815,
disconnects none and reads a mean of 1.0758 with a worst of 1.3255;
above three times the median nothing changes.

A river down the middle with one bridge leaves the same-bank pairs at
1.0642 and puts the cross-river pairs at a mean of 1.4099, a median of
1.2437 and a worst of 5.07 when the bridge sits mid-river, and at
1.8454, 1.5335 and 10.58 when the bridge sits ten units from the
corner.
"""

from __future__ import annotations

import heapq
import math
import random

from atlas.delaunay import triangulate
from atlas.errors import Invalid

Point = tuple[float, float]
Graph = dict[int, dict[int, float]]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def delaunay_network(points: list[Point]) -> Graph:
    if len(points) < 3:
        raise Invalid("a network needs at least three points")
    index = {p: i for i, p in enumerate(points)}
    graph: Graph = {i: {} for i in range(len(points))}
    for tri in triangulate(points):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            i, j = index[a], index[b]
            graph[i][j] = graph[j][i] = _dist(a, b)
    return graph


def grid_network(side: int, spacing: float = 1.0) -> tuple[list[Point], Graph]:
    if side < 2:
        raise Invalid("a grid needs at least two points a side")
    points = [(c * spacing, r * spacing) for r in range(side) for c in range(side)]
    graph: Graph = {i: {} for i in range(len(points))}
    for r in range(side):
        for c in range(side):
            i = r * side + c
            if c + 1 < side:
                graph[i][i + 1] = graph[i + 1][i] = spacing
            if r + 1 < side:
                graph[i][i + side] = graph[i + side][i] = spacing
    return points, graph


def edge_count(graph: Graph) -> int:
    return sum(len(v) for v in graph.values()) // 2


def dijkstra(graph: Graph, source: int) -> dict[int, float]:
    if source not in graph:
        raise Invalid("the source must be a node of the graph")
    dist = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, math.inf):
            continue
        for other, weight in graph[node].items():
            nd = d + weight
            if nd < dist.get(other, math.inf):
                dist[other] = nd
                heapq.heappush(heap, (nd, other))
    return dist


Pair = tuple[int, int]


def detour_ratios(points: list[Point], graph: Graph, pairs: list[Pair]) -> list[float]:
    if not pairs:
        raise Invalid("at least one pair is needed")
    cache: dict[int, dict[int, float]] = {}
    ratios = []
    for i, j in pairs:
        if i == j:
            raise Invalid("a pair must join two different nodes")
        if i not in cache:
            cache[i] = dijkstra(graph, i)
        straight = _dist(points[i], points[j])
        ratios.append(cache[i].get(j, math.inf) / straight if straight > 0 else 1.0)
    return ratios


def random_pairs(n: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    pairs = []
    while len(pairs) < count:
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            pairs.append((i, j))
    return pairs


def summary(ratios: list[float]) -> dict[str, float]:
    finite = [r for r in ratios if math.isfinite(r)]
    if not finite:
        raise Invalid("no pair is connected")
    finite.sort()
    return {
        "mean": sum(finite) / len(finite),
        "median": finite[len(finite) // 2],
        "max": finite[-1],
        "disconnected": (len(ratios) - len(finite)) / len(ratios),
    }


def sparsified(graph: Graph, keep: float, rng: random.Random) -> Graph:
    if not 0 < keep <= 1:
        raise Invalid("keep must be a fraction in (0, 1]")
    out: Graph = {i: {} for i in graph}
    for i, neighbours in graph.items():
        for j, weight in neighbours.items():
            if i < j and rng.random() < keep:
                out[i][j] = out[j][i] = weight
    return out


def without_long_edges(graph: Graph, factor: float) -> Graph:
    lengths = sorted(w for i, nb in graph.items() for j, w in nb.items() if i < j)
    if not lengths:
        raise Invalid("the graph has no edges")
    median = lengths[len(lengths) // 2]
    out: Graph = {i: {} for i in graph}
    for i, neighbours in graph.items():
        for j, weight in neighbours.items():
            if weight <= factor * median:
                out[i][j] = weight
    return out


def bridged(points: list[Point], graph: Graph, river_x: float, bridge_y: float) -> Graph:
    out: Graph = {i: {} for i in graph}
    left = [i for i, p in enumerate(points) if p[0] < river_x]
    right = [i for i, p in enumerate(points) if p[0] >= river_x]
    if not left or not right:
        raise Invalid("the river must have points on both banks")
    for i, neighbours in graph.items():
        for j, weight in neighbours.items():
            if (points[i][0] < river_x) == (points[j][0] < river_x):
                out[i][j] = weight
    a = min(left, key=lambda i: abs(points[i][1] - bridge_y) + (river_x - points[i][0]))
    b = min(right, key=lambda i: abs(points[i][1] - bridge_y) + (points[i][0] - river_x))
    out[a][b] = out[b][a] = _dist(points[a], points[b])
    return out


def manhattan_law() -> float:
    return 4 / math.pi


def uniform_points(n: int, rng: random.Random, side: float = 100.0) -> list[Point]:
    return [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]
