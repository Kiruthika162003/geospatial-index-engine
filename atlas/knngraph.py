"""k-nearest-neighbor graph: a directed graph whose edges only sometimes point both ways.

Linking every point to its k nearest neighbors builds the graph that
spatial clustering, manifold learning, and proximity routing all start
from, and the first thing to understand about it is that it is
directed. If point a is among b's k nearest, b need not be among a's:
a lonely outlier reaches out to k points in a dense cluster, but none
of those cluster points count the outlier among their own k nearest,
since they have closer company. So the graph has one-way edges, and
the fraction of edges that are mutual, present in both directions, is
a measurable property of how evenly the points are spread. Uniformly
scattered points give a high mutual fraction, because neighbors tend
to be neighbors of each other; a clustered set with stragglers gives a
lower one, since every straggler adds k one-way edges. Two more
counts pin the structure. Every vertex has out-degree exactly k, by
construction, while in-degrees vary, and a hub, a point that many
others count among their nearest, has a high in-degree that the
out-degree never reveals. And the symmetrized graph, keeping an edge
if it exists in either direction, has between n k over two and n k
edges, the lower bound when every edge is mutual and the upper when
none is. The graph is built with the k-d tree so each query is a log
walk, and the survey checks the neighbor lists against brute force.
The finding worth stating is that the k-nearest-neighbor graph is
directed with out-degree exactly k and a mutual-edge fraction that
falls as the points cluster, so its asymmetry is a direct reading of
the point set's unevenness. This module builds the graph and reports
its mutual fraction, in-degree spread, and symmetrized edge count, and
a survey measures those on uniform and clustered inputs.
"""

from __future__ import annotations

from atlas.errors import Invalid
from atlas.kdtree import KDTree

Point = tuple[float, float]


class KnnGraph:
    def __init__(self, points: list[Point], k: int) -> None:
        if points is None:
            raise Invalid("points must not be None")
        if k <= 0:
            raise Invalid("k must be positive")
        if len(set(points)) <= k:
            raise Invalid("need more than k distinct points")
        self.points = list(dict.fromkeys(points))
        self.k = k
        tree = KDTree(self.points)
        index = {p: i for i, p in enumerate(self.points)}
        self.out: list[list[int]] = []
        for p in self.points:
            neighbors = tree.k_nearest(p, k + 1)  # the point itself comes back first
            self.out.append([index[q] for q in neighbors if q != p][:k])

    def out_degree(self, i: int) -> int:
        return len(self.out[i])

    def in_degrees(self) -> list[int]:
        degrees = [0] * len(self.points)
        for targets in self.out:
            for j in targets:
                degrees[j] += 1
        return degrees

    def mutual_fraction(self) -> float:
        total = mutual = 0
        for i, targets in enumerate(self.out):
            for j in targets:
                total += 1
                if i in self.out[j]:
                    mutual += 1
        return mutual / total if total else 0.0

    def symmetrized_edges(self) -> int:
        edges = set()
        for i, targets in enumerate(self.out):
            for j in targets:
                edges.add((min(i, j), max(i, j)))
        return len(edges)
