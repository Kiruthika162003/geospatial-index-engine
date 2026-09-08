"""k-d tree: split the plane by alternating axes so nearest-neighbor prunes to a log walk.

A k-d tree indexes points by cutting space with axis-aligned planes
that alternate direction down the levels: the root splits on x, its
children on y, their children on x again, and so on. Each node holds a
point and the cut through it, so the left subtree holds everything on
the low side of the cut and the right subtree everything on the high
side. Building from the median at each level keeps the tree balanced,
depth logarithmic. The reason to build it is nearest-neighbor search,
and the reason that is fast is pruning. To find the closest stored
point to a query, descend to the leaf the query would occupy, take
that point as the current best, then unwind, and at each node ask one
question: could anything on the other side of this cut be closer than
the best found so far? The distance from the query to the cutting
plane answers it exactly, since every point beyond the plane is at
least that far away on that axis; if the plane is farther than the
current best, the whole far subtree is skipped without looking at a
single point in it. That geometric test is what turns a linear scan
into an average logarithmic one on well-spread data. The honest limit,
worth stating, is that pruning weakens as dimensions rise, the curse
of dimensionality, because in high dimensions the query is close to
many cutting planes at once and few subtrees can be skipped; in the
two dimensions of a map it works well. The finding worth stating is
that the plane-distance prune visits a small fraction of the nodes a
brute scan would while returning the identical nearest neighbor. This
module builds a 2D k-d tree with nearest, k-nearest, and range
queries, and a survey measures the fraction of nodes visited against
the answer's exactness.
"""

from __future__ import annotations

import heapq

from atlas.bbox import BBox
from atlas.errors import Invalid, Missing

Point = tuple[float, float]


class _Node:
    __slots__ = ("axis", "left", "point", "right")

    def __init__(self, point: Point, axis: int) -> None:
        self.point = point
        self.axis = axis
        self.left: _Node | None = None
        self.right: _Node | None = None


class KDTree:
    def __init__(self, points: list[Point]) -> None:
        if points is None:
            raise Invalid("points must not be None")
        self._size = len(points)
        self._root = self._build(list(points), 0)
        self.visits = 0  # nodes touched by the most recent query

    def __len__(self) -> int:
        return self._size

    def _build(self, points: list[Point], depth: int) -> _Node | None:
        if not points:
            return None
        axis = depth % 2
        points.sort(key=lambda p: p[axis])
        mid = len(points) // 2
        node = _Node(points[mid], axis)
        node.left = self._build(points[:mid], depth + 1)
        node.right = self._build(points[mid + 1 :], depth + 1)
        return node

    def nearest(self, query: Point) -> Point:
        if self._root is None:
            raise Missing("the tree is empty")
        self.visits = 0
        best = self._search(self._root, query, None, float("inf"))
        return best[0]

    def _search(
        self, node: _Node | None, query: Point, best: Point | None, best_d2: float
    ) -> tuple[Point | None, float]:
        if node is None:
            return best, best_d2
        self.visits += 1
        d2 = _dist2(node.point, query)
        if d2 < best_d2:
            best, best_d2 = node.point, d2
        axis = node.axis
        diff = query[axis] - node.point[axis]
        near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
        best, best_d2 = self._search(near, query, best, best_d2)
        if diff * diff < best_d2:  # the cutting plane is closer than the best; check far
            best, best_d2 = self._search(far, query, best, best_d2)
        return best, best_d2

    def k_nearest(self, query: Point, k: int) -> list[Point]:
        if k <= 0:
            raise Invalid("k must be positive")
        heap: list[tuple[float, int, Point]] = []
        counter = 0
        self.visits = 0
        stack = [self._root]
        # simple correct traversal with plane pruning against the k-th best
        def visit(node: _Node | None) -> None:
            nonlocal counter
            if node is None:
                return
            self.visits += 1
            d2 = _dist2(node.point, query)
            if len(heap) < k:
                heapq.heappush(heap, (-d2, counter, node.point))
                counter += 1
            elif d2 < -heap[0][0]:
                heapq.heapreplace(heap, (-d2, counter, node.point))
                counter += 1
            diff = query[node.axis] - node.point[node.axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(near)
            if len(heap) < k or diff * diff < -heap[0][0]:
                visit(far)

        stack.clear()
        visit(self._root)
        return [p for _, _, p in sorted(heap, key=lambda t: -t[0])]

    def range_query(self, box: BBox) -> list[Point]:
        found: list[Point] = []
        self._range(self._root, box, found)
        return found

    def _range(self, node: _Node | None, box: BBox, found: list[Point]) -> None:
        if node is None:
            return
        x, y = node.point
        if box.contains_point(x, y):
            found.append(node.point)
        axis = node.axis
        lo = box.min_x if axis == 0 else box.min_y
        hi = box.max_x if axis == 0 else box.max_y
        coord = node.point[axis]
        if lo <= coord:
            self._range(node.left, box, found)
        if coord <= hi:
            self._range(node.right, box, found)


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
