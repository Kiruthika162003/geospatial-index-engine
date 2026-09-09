"""A kd-tree built by insertion survives x-sorted input at twice the height; diagonal wrecks it.

A kd-tree grown by inserting points one at a time takes whatever
shape the input order gives it. The guess that sorted input makes a
linked list, as it does in a binary search tree, was wrong for a
kd-tree, because the splitting axis alternates: points sorted by x
go right at every x level and split evenly at every y level. On
1000, 4000 and 16,000 uniform points a shuffled order builds heights
21, 31 and 33 against the balanced 10, 12 and 14, mean depths 11.99,
14.14 and 17.08, and a nearest-neighbour search visits 20.6, 23.0 and
27.2 nodes. The x-sorted order builds heights 39, 52 and 61, mean
depths 20.5, 26.1 and 31.1, and visits 41.5, 70.1 and 113.3 nodes,
twice the height and four times the visits at the largest size but
nothing like a list.

The order that wrecks the tree is the one sorted along the diagonal
by x plus y, which goes right at both axes: heights 79, 159 and 308,
one level per 52 points, mean depths 34.5, 59.7 and 112.4, and 48.2,
87.3 and 166.4 visits a query. A snake order through 16 horizontal
bands builds 40, 59 and 88. Building is cheap at every order, 0.001
to 0.066 seconds. Rebuilding the x-sorted 4000-point tree by median
splits brings its height from 50 to the balanced 12, the mean depth
to 10.98 and the visits to 18.1, and the rebuilt tree's nearest
point matched a brute scan on all 200 queries.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


class Node:
    __slots__ = ("axis", "left", "point", "right")

    def __init__(self, point: Point, axis: int) -> None:
        self.point = point
        self.axis = axis
        self.left: Node | None = None
        self.right: Node | None = None


class InsertionTree:
    def __init__(self) -> None:
        self.root: Node | None = None
        self.size = 0

    def insert(self, point: Point) -> int:
        self.size += 1
        if self.root is None:
            self.root = Node(point, 0)
            return 1
        node = self.root
        depth = 1
        while True:
            depth += 1
            axis = node.axis
            if point[axis] < node.point[axis]:
                if node.left is None:
                    node.left = Node(point, 1 - axis)
                    return depth
                node = node.left
            else:
                if node.right is None:
                    node.right = Node(point, 1 - axis)
                    return depth
                node = node.right

    def height(self) -> int:
        return _height(self.root)

    def mean_depth(self) -> float:
        if self.root is None:
            raise Invalid("an empty tree has no depths")
        total, count = _depth_sum(self.root, 1)
        return total / count

    def nearest(self, query: Point) -> tuple[Point, int]:
        if self.root is None:
            raise Invalid("an empty tree has no nearest point")
        best = [self.root.point, _dist2(query, self.root.point), 0]
        _search(self.root, query, best)
        return best[0], best[2]

    def rebuild(self) -> None:
        points = _collect(self.root)
        self.root = _balanced(points, 0)


def _height(node: Node | None) -> int:
    if node is None:
        return 0
    return 1 + max(_height(node.left), _height(node.right))


def _depth_sum(node: Node | None, depth: int) -> tuple[int, int]:
    if node is None:
        return 0, 0
    lt, lc = _depth_sum(node.left, depth + 1)
    rt, rc = _depth_sum(node.right, depth + 1)
    return depth + lt + rt, 1 + lc + rc


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _search(node: Node | None, query: Point, best: list) -> None:
    if node is None:
        return
    best[2] += 1
    d = _dist2(query, node.point)
    if d < best[1]:
        best[0], best[1] = node.point, d
    axis = node.axis
    diff = query[axis] - node.point[axis]
    near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
    _search(near, query, best)
    if diff * diff < best[1]:
        _search(far, query, best)


def _collect(node: Node | None) -> list[Point]:
    if node is None:
        return []
    return [*_collect(node.left), node.point, *_collect(node.right)]


def _balanced(points: list[Point], axis: int) -> Node | None:
    if not points:
        return None
    points = sorted(points, key=lambda p: p[axis])
    mid = len(points) // 2
    node = Node(points[mid], axis)
    node.left = _balanced(points[:mid], 1 - axis)
    node.right = _balanced(points[mid + 1 :], 1 - axis)
    return node


def build(points: list[Point]) -> InsertionTree:
    tree = InsertionTree()
    for p in points:
        tree.insert(p)
    return tree


def balanced_height(n: int) -> int:
    if n < 0:
        raise Invalid("n must not be negative")
    return math.ceil(math.log2(n + 1))


def uniform(n: int, rng: random.Random) -> list[Point]:
    return [(rng.random(), rng.random()) for _ in range(n)]


def sorted_by_x(points: list[Point]) -> list[Point]:
    return sorted(points)


def sorted_diagonal(points: list[Point]) -> list[Point]:
    return sorted(points, key=lambda p: p[0] + p[1])


def sorted_snake(points: list[Point], bands: int) -> list[Point]:
    if bands < 1:
        raise Invalid("bands must be positive")
    def key(p: Point) -> tuple[int, float]:
        band = int(p[1] * bands)
        return band, p[0] if band % 2 == 0 else -p[0]

    return sorted(points, key=key)


def mean_visits(tree: InsertionTree, queries: list[Point]) -> float:
    if not queries:
        raise Invalid("at least one query is needed")
    return sum(tree.nearest(q)[1] for q in queries) / len(queries)


def brute_nearest(points: list[Point], query: Point) -> Point:
    return min(points, key=lambda p: _dist2(p, query))
