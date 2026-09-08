"""Visibility graph: the shortest path around obstacles bends only at obstacle corners.

Routing a path from a start to a goal around polygonal obstacles is
a search over an infinite continuum of paths, and the visibility graph
reduces it to a finite one by a single geometric fact: the shortest
obstacle-avoiding path is a chain of straight segments that turns only
at obstacle vertices, because any bend in open space could be
straightened to shorten the path and any segment cutting an obstacle
is not allowed. So the candidate turning points are the obstacle
corners plus the start and goal, an edge joins two of them when the
segment between them crosses no obstacle edge, and Dijkstra over that
graph with Euclidean edge weights returns the exact shortest path. The
graph is dense in the worst case, every pair of the n corners tested
for mutual visibility, quadratic edges each costing a crossing test
against every obstacle edge, which is why the construction is the
expensive part and the search the cheap part, and why it suits a
fixed map queried many times. Three properties pin the result and are
worth measuring. Every path edge is unobstructed, by construction of
the graph, so the path never cuts a wall. The path is never shorter
than the straight-line distance, since nothing beats a straight line,
and equals it exactly when the straight line is clear, so the detour
cost is a measurable excess over the crow's flight that grows as
obstacles intervene. And every interior turn of the path lands on an
obstacle vertex, the theorem stated as a check. When the goal is
walled off entirely no path exists and the honest answer is to say so
rather than return a path through a wall. The finding worth stating
is that the visibility graph's shortest path is exact, clear of every
obstacle, equal to the straight line when unobstructed, and turns only
at corners, so the continuum of routes collapses to a finite graph
without loss. This module builds the visibility graph over obstacle
corners and runs Dijkstra, and a survey confirms clearance, the
straight-line bound, and the corner-turn property.
"""

from __future__ import annotations

import heapq
import math

from atlas.errors import Invalid
from atlas.lineofsight import Visibility

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _edges_of(polygon: list[Point]) -> list[Segment]:
    n = len(polygon)
    return [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]


def _centroid(polygon: list[Point]) -> Point:
    n = len(polygon)
    return (sum(p[0] for p in polygon) / n, sum(p[1] for p in polygon) / n)


class VisibilityGraph:
    def __init__(self, obstacles: list[list[Point]]) -> None:
        if obstacles is None:
            raise Invalid("obstacles must not be None")
        for poly in obstacles:
            if len(poly) < 3:
                raise Invalid("each obstacle needs at least three vertices")
        self.obstacles = [list(p) for p in obstacles]
        # shrink each obstacle edge a hair toward its centroid so corner-to-corner
        # sight lines along a polygon's own edges are not counted as crossings
        walls: list[Segment] = []
        for poly in self.obstacles:
            cx, cy = _centroid(poly)
            for a, b in _edges_of(poly):
                a2 = (a[0] + (cx - a[0]) * 1e-9, a[1] + (cy - a[1]) * 1e-9)
                b2 = (b[0] + (cx - b[0]) * 1e-9, b[1] + (cy - b[1]) * 1e-9)
                walls.append((a2, b2))
        self._sight = Visibility(walls)
        self.corners: list[Point] = [v for poly in self.obstacles for v in poly]

    def visible(self, a: Point, b: Point) -> bool:
        return self._sight.visible(a, b)

    def shortest_path(self, start: Point, goal: Point) -> tuple[list[Point], float]:
        nodes = [start, goal, *self.corners]
        n = len(nodes)
        adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self.visible(nodes[i], nodes[j]):
                    d = math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1])
                    adjacency[i].append((j, d))
                    adjacency[j].append((i, d))
        dist = [math.inf] * n
        prev = [-1] * n
        dist[0] = 0.0
        heap = [(0.0, 0)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if u == 1:
                break
            for v, w in adjacency[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if dist[1] == math.inf:
            raise Invalid("the goal is walled off; no path exists")
        path = []
        u = 1
        while u != -1:
            path.append(nodes[u])
            u = prev[u]
        path.reverse()
        return path, dist[1]
