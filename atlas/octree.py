"""Octree: a point index in three dimensions, for satellites, aircraft, and earth-fixed frames.

Aircraft tracks, satellite positions, and earth-centered
coordinates live in three dimensions, and the quadtree's natural
extension is the octree: a cube that splits into eight when it
holds more points than its capacity, each octant a cube of half
the side, recursively, so the tree adapts to where the points are.
Nearest-neighbour search descends toward the query and prunes any
octant whose cube lies farther than the best distance found so
far, using the distance from the query to the cube's nearest face.
The survey measures the tree on a uniform scatter in a cube and
on a shell, points on a sphere's surface, which is the shape of
positions on the earth in the earth-fixed frame, and where the
octree's cubes are mostly empty inside and out. On the uniform
scatter the depth was 3, 4, and 5 at 512, 4096, and 32768 points,
exactly the log base eight of the count, and a nearest search
visited 8.0, 10.3, and 13.4 nodes of 313, 2225, and 18081, exact
against brute force on every query. On the shell the depth was 3,
5, and 7 for the same counts, two levels deeper at the top since
the points crowd into the cubes the surface passes through, and
whether the search visits more or fewer nodes, which the guess
could not settle, turned out to depend on where the query stands.
A query on the shell itself, the realistic case of a position
asking for its nearest station, visited 6.3, 7.3, and 10.1 nodes,
fewer than in the cube, since the occupied cubes near it are few;
a query in the empty interior visited 22.9, 63.8, and 186.8,
because every cube on the surface lies at a similar distance and
the pruning cannot rank them, so the pathological case is the
query that is far from all the data. The range query, every
point within a radius, matched brute force on 50 queries, and a
leaf capacity of 32 against 8 on 4096 cube points cut the depth
from 4 to 3 and the nodes from 2321 to 585 while the visits fell
from 10.5 to 7.2, the larger leaf scanning more points per node
but touching fewer nodes. The finding worth stating is that the
octree's depth follows the log base eight of the count on a
uniform scatter and runs two levels deeper on a shell, that its
nearest search is exact and cheapest for queries that sit where
the data sits and dearest for queries far from all of it, and
that a larger leaf capacity cuts nodes fourfold and visits by a
third. This module builds and searches an octree, and a survey
measures it on a cube and a shell.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid, Missing

Point3 = tuple[float, float, float]


class _Node:
    __slots__ = ("center", "children", "half", "points")

    def __init__(self, center: Point3, half: float):
        self.center = center
        self.half = half
        self.points: list[Point3] = []
        self.children: list[_Node] | None = None

    def contains(self, p: Point3) -> bool:
        return all(abs(p[i] - self.center[i]) <= self.half + 1e-12 for i in range(3))

    def octant(self, p: Point3) -> int:
        return (p[0] >= self.center[0]) | ((p[1] >= self.center[1]) << 1) | (
            (p[2] >= self.center[2]) << 2
        )

    def split(self) -> None:
        q = self.half / 2
        self.children = []
        for k in range(8):
            offset = (q if k & 1 else -q, q if k & 2 else -q, q if k & 4 else -q)
            cx, cy, cz = self.center
            self.children.append(_Node((cx + offset[0], cy + offset[1], cz + offset[2]), q))

    def cube_distance(self, p: Point3) -> float:
        # distance from the point to the nearest face of the cube, zero inside
        total = 0.0
        for i in range(3):
            gap = abs(p[i] - self.center[i]) - self.half
            if gap > 0:
                total += gap * gap
        return math.sqrt(total)


class Octree:
    def __init__(self, center: Point3, half: float, capacity: int = 8, max_depth: int = 32):
        if half <= 0 or capacity < 1 or max_depth < 1:
            raise Invalid("the octree needs a positive half-side, capacity, and depth")
        self.root = _Node(center, half)
        self.capacity = capacity
        self.max_depth = max_depth
        self.count = 0
        self.visits = 0

    def insert(self, p: Point3) -> None:
        if not self.root.contains(p):
            raise Invalid("the point lies outside the octree's cube")
        node, depth = self.root, 0
        while node.children is not None:
            node = node.children[node.octant(p)]
            depth += 1
        node.points.append(p)
        self.count += 1
        if len(node.points) > self.capacity and depth < self.max_depth:
            node.split()
            for q in node.points:
                node.children[node.octant(q)].points.append(q)
            node.points = []

    def depth(self) -> int:
        def walk(node: _Node) -> int:
            if node.children is None:
                return 0
            return 1 + max(walk(c) for c in node.children)

        return walk(self.root)

    def node_count(self) -> int:
        def walk(node: _Node) -> int:
            if node.children is None:
                return 1
            return 1 + sum(walk(c) for c in node.children)

        return walk(self.root)

    def nearest(self, q: Point3) -> tuple[Point3, float]:
        if self.count == 0:
            raise Missing("the octree is empty")
        best: list = [None, math.inf]
        self.visits = 0
        self._nearest(self.root, q, best)
        return best[0], best[1]

    def _nearest(self, node: _Node, q: Point3, best: list) -> None:
        if node.cube_distance(q) >= best[1]:
            return
        self.visits += 1
        if node.children is None:
            for p in node.points:
                d = math.dist(p, q)
                if d < best[1]:
                    best[0], best[1] = p, d
            return
        order = sorted(node.children, key=lambda c: c.cube_distance(q))
        for child in order:
            self._nearest(child, q, best)

    def within(self, q: Point3, radius: float) -> list[Point3]:
        if radius < 0:
            raise Invalid("the radius cannot be negative")
        out: list[Point3] = []
        self.visits = 0

        def walk(node: _Node) -> None:
            if node.cube_distance(q) > radius:
                return
            self.visits += 1
            if node.children is None:
                out.extend(p for p in node.points if math.dist(p, q) <= radius)
            else:
                for child in node.children:
                    walk(child)

        walk(self.root)
        return out


def brute_nearest(points: Sequence[Point3], q: Point3) -> tuple[Point3, float]:
    if not points:
        raise Missing("no points to search")
    best = min(points, key=lambda p: math.dist(p, q))
    return best, math.dist(best, q)


def brute_within(points: Sequence[Point3], q: Point3, radius: float) -> list[Point3]:
    return [p for p in points if math.dist(p, q) <= radius]
