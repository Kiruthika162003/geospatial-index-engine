"""Point quadtree: subdivide space into four until each cell is sparse enough.

A quadtree indexes points by recursively cutting a square region into
four equal quadrants. A node holds points up to a small capacity, and
the moment it would overflow it splits into four children covering its
four sub-squares and hands each point down to the child that contains
it. This is the defining contrast with the k-d tree. The k-d tree is
data-driven: it cuts at the median of the actual points, so the split
position depends on the data and the tree stays balanced regardless of
how the points cluster. The quadtree is space-driven: it always cuts a
region exactly in half on each axis, at a position fixed by the region,
not the data, so the tree's shape is dictated by where the points fall,
not how many there are. That difference has a sharp consequence worth
measuring rather than assuming. Uniformly spread points make a shallow,
even quadtree, because every quadrant gets a similar share and the
recursion stops quickly. But points packed into a tiny area force the
quadtree to keep halving that area, subdividing empty space around them
again and again, until the cells are finally small enough to separate
the cluster, so a tight cluster produces a deep, spindly tree even when
the point count is modest. The k-d tree does not suffer this, because
its median cut lands among the clustered points immediately. The
finding worth stating is that quadtree depth is governed by the spatial
spread of the data, not its size, so clustering deepens it while the
k-d tree stays balanced. This module builds a point quadtree with
insertion and range query, and a survey measures the depth uniform
versus clustered inputs produce, so the space-driven cost is a number.
"""

from __future__ import annotations

from atlas.bbox import BBox
from atlas.errors import Invalid

Point = tuple[float, float]


class _QNode:
    __slots__ = ("box", "children", "points")

    def __init__(self, box: BBox) -> None:
        self.box = box
        self.points: list[Point] = []
        self.children: list[_QNode] | None = None


class QuadTree:
    def __init__(self, box: BBox, capacity: int = 4) -> None:
        if capacity <= 0:
            raise Invalid("capacity must be positive")
        self._root = _QNode(box)
        self._capacity = capacity
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, point: Point) -> None:
        x, y = point
        if not self._root.box.contains_point(x, y):
            raise Invalid("point lies outside the quadtree's region")
        self._insert(self._root, point)
        self._size += 1

    def _insert(self, node: _QNode, point: Point) -> None:
        if node.children is not None:
            self._insert(self._child_for(node, point), point)
            return
        node.points.append(point)
        if len(node.points) > self._capacity:
            self._split(node)

    def _split(self, node: _QNode) -> None:
        b = node.box
        mx, my = b.center()
        node.children = [
            _QNode(BBox(b.min_x, b.min_y, mx, my)),
            _QNode(BBox(mx, b.min_y, b.max_x, my)),
            _QNode(BBox(b.min_x, my, mx, b.max_y)),
            _QNode(BBox(mx, my, b.max_x, b.max_y)),
        ]
        stranded = node.points
        node.points = []
        for p in stranded:
            self._insert(self._child_for(node, p), p)

    def _child_for(self, node: _QNode, point: Point) -> _QNode:
        x, y = point
        mx, my = node.box.center()
        east = x >= mx
        north = y >= my
        index = (2 if north else 0) + (1 if east else 0)
        return node.children[index]

    def range_query(self, box: BBox) -> list[Point]:
        found: list[Point] = []
        self._range(self._root, box, found)
        return found

    def _range(self, node: _QNode, box: BBox, found: list[Point]) -> None:
        if not node.box.intersects(box):
            return
        if node.children is None:
            found.extend(p for p in node.points if box.contains_point(*p))
            return
        for child in node.children:
            self._range(child, box, found)

    def depth(self) -> int:
        return self._depth(self._root)

    def _depth(self, node: _QNode) -> int:
        if node.children is None:
            return 1
        return 1 + max(self._depth(child) for child in node.children)
