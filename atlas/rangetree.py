"""Range tree: a tree over x whose every node carries a sorted y list, for rectangle counts.

A range tree answers how many points, or which points, lie in an
axis-aligned rectangle, and it does so with a guaranteed bound rather
than the average-case bound the k-d tree offers. It is a balanced
binary search tree over the points' x coordinates, and the trick is
that every node also stores, in a secondary structure, all the points
of its subtree sorted by y. A rectangle query first descends the x tree
to find the canonical nodes, the handful of subtrees whose x ranges
together exactly tile the query's x span, of which there are at most
about two per level, so order log n of them. Each canonical node's
subtree lies wholly inside the x span, so within it only y matters,
and its sorted y list answers the y part by two binary searches,
counting or listing the points between the bounds. Summing over the
canonical nodes gives the answer in order log squared n plus the output
size, every step a binary search, no data-dependent pruning that could
fail on a bad layout. The cost is space: every point appears in the y
list of each of its log n ancestors, so the structure holds n log n
point copies, which is the price of the guaranteed query time and the
reason range trees are used when query latency must be predictable
rather than when memory is tight. The finding worth stating is that the
range tree's canonical decomposition touches order log n subtrees for
any rectangle, each settled by binary search, so the query bound holds
for every input and the count matches a brute scan exactly. This module
builds a static range tree with rectangle count and report queries, and
a survey confirms the results against brute force and measures the
canonical node count.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

from atlas.bbox import BBox
from atlas.errors import Invalid

Point = tuple[float, float]


class _RNode:
    __slots__ = ("hi", "left", "lo", "points", "right", "x", "ys")

    def __init__(self, points: list[Point]) -> None:
        self.points = points  # subtree points, sorted by y
        self.ys = [p[1] for p in points]
        self.lo = min(p[0] for p in points)
        self.hi = max(p[0] for p in points)
        self.x = 0.0
        self.left: _RNode | None = None
        self.right: _RNode | None = None


class RangeTree:
    def __init__(self, points: list[Point]) -> None:
        if points is None:
            raise Invalid("points must not be None")
        self._size = len(points)
        self.canonical = 0
        self._root = self._build(sorted(points)) if points else None

    def __len__(self) -> int:
        return self._size

    def _build(self, by_x: list[Point]) -> _RNode:
        node = _RNode(sorted(by_x, key=lambda p: p[1]))
        if len(by_x) > 1:
            mid = len(by_x) // 2
            node.x = by_x[mid][0]
            node.left = self._build(by_x[:mid])
            node.right = self._build(by_x[mid:])
        return node

    def count(self, box: BBox) -> int:
        self.canonical = 0
        return self._count(self._root, box)

    def _count(self, node: _RNode | None, box: BBox) -> int:
        if node is None or node.hi < box.min_x or node.lo > box.max_x:
            return 0
        if box.min_x <= node.lo and node.hi <= box.max_x:
            self.canonical += 1
            return bisect_right(node.ys, box.max_y) - bisect_left(node.ys, box.min_y)
        if node.left is None:
            return 0
        return self._count(node.left, box) + self._count(node.right, box)

    def report(self, box: BBox) -> list[Point]:
        found: list[Point] = []
        self._report(self._root, box, found)
        return found

    def _report(self, node: _RNode | None, box: BBox, found: list[Point]) -> None:
        if node is None or node.hi < box.min_x or node.lo > box.max_x:
            return
        if box.min_x <= node.lo and node.hi <= box.max_x:
            lo = bisect_left(node.ys, box.min_y)
            hi = bisect_right(node.ys, box.max_y)
            found.extend(node.points[lo:hi])
            return
        if node.left is None:
            return
        self._report(node.left, box, found)
        self._report(node.right, box, found)
