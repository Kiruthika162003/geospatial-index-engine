"""A bridge's ends carry the crossing share; a grid's centre carries 1.2 to 1.35 over its side.

Betweenness counts, for each node, the share of shortest paths
between other pairs that pass through it, read here by Brandes'
algorithm with Dijkstra on the weighted road network and normalised
by the pairs. On a Delaunay town of 100 uniform points the busiest
node carries 0.167 of the pairs, the mean node 0.0433, and 7 percent
of nodes carry none; at 300 points the busiest carries 0.1059, the
mean 0.026, and 3.7 percent carry none, in 0.1 seconds. Cut the
town by a river with one bridge and the bridge's two ends each
carry 0.5027 of the pairs, which is the share of pairs with one
point on each bank, 0.5048, less the pairs the ends belong to
themselves; the next busiest nodes, the bridge's approaches, carry
0.2226 and 0.2064. A bridge near the corner reads 0.5044 and 0.5014
at its ends and 0.3993 at the first approach.

The guess that a grid's centre carries a fifth of the pairs held
only at the smallest grids: the centre of a 5, 7, 11 and 15 grid
carries 0.2382, 0.1777, 0.1192 and 0.09, the largest share in each,
which is 1.19, 1.24, 1.31 and 1.35 over the side, while a corner
carries 0.0097, 0.0032, 0.0007 and 0.0002 and the mean node 0.1014,
0.078, 0.0532 and 0.0404; no grid node carries nothing, since every
node sits on some shortest path between its neighbours' pairs.
"""

from __future__ import annotations

import heapq
import math
import random

from atlas.detourindex import Graph, bridged, delaunay_network, grid_network, uniform_points
from atlas.errors import Invalid

Point = tuple[float, float]


def brandes(graph: Graph) -> dict[int, float]:
    # Brandes' algorithm with Dijkstra, counting every ordered pair once each way, then halved
    if not graph:
        raise Invalid("the graph is empty")
    score = dict.fromkeys(graph, 0.0)
    for source in graph:
        stack: list[int] = []
        preds: dict[int, list[int]] = {v: [] for v in graph}
        sigma = dict.fromkeys(graph, 0.0)
        sigma[source] = 1.0
        dist = dict.fromkeys(graph, math.inf)
        dist[source] = 0.0
        heap = [(0.0, source)]
        seen: set[int] = set()
        while heap:
            d, v = heapq.heappop(heap)
            if v in seen:
                continue
            seen.add(v)
            stack.append(v)
            for w, weight in graph[v].items():
                nd = d + weight
                if nd < dist[w] - 1e-12:
                    dist[w] = nd
                    sigma[w] = sigma[v]
                    preds[w] = [v]
                    heapq.heappush(heap, (nd, w))
                elif abs(nd - dist[w]) <= 1e-12:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = dict.fromkeys(graph, 0.0)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
            if w != source:
                score[w] += delta[w]
    return {v: s / 2 for v, s in score.items()}


def normalised(score: dict[int, float]) -> dict[int, float]:
    n = len(score)
    if n < 3:
        raise Invalid("normalisation needs three nodes")
    pairs = (n - 1) * (n - 2) / 2
    return {v: s / pairs for v, s in score.items()}


def top(score: dict[int, float], count: int = 5) -> list[tuple[int, float]]:
    return sorted(score.items(), key=lambda item: -item[1])[:count]


def summary(score: dict[int, float]) -> dict[str, float]:
    values = list(score.values())
    return {
        "max": max(values),
        "mean": sum(values) / len(values),
        "zero_fraction": sum(1 for v in values if v == 0) / len(values),
    }


def bridge_ends(points: list[Point], graph: Graph) -> tuple[int, int]:
    # the two nodes joined by the only edge across x = 50
    for i, neighbours in graph.items():
        for j in neighbours:
            if (points[i][0] < 50) != (points[j][0] < 50):
                return (i, j) if points[i][0] < 50 else (j, i)
    raise Invalid("no bridge found")


def crossing_pairs(points: list[Point]) -> int:
    left = sum(1 for p in points if p[0] < 50)
    return left * (len(points) - left)


def town(n: int, seed: int) -> tuple[list[Point], Graph]:
    points = uniform_points(n, random.Random(seed))
    return points, delaunay_network(points)


def bridged_town(n: int, seed: int, bridge_y: float) -> tuple[list[Point], Graph]:
    points, graph = town(n, seed)
    return points, bridged(points, graph, 50.0, bridge_y)


def grid(side: int) -> tuple[list[Point], Graph]:
    return grid_network(side)
