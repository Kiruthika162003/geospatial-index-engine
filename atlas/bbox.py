"""Bounding box: the minimum axis-aligned rectangle, the atom of every spatial tree.

A bounding box, or minimum bounding rectangle, is the smallest
axis-aligned rectangle that contains a set of points or shapes, named
by its low and high corner. It is the workhorse approximation of
spatial indexing: testing whether two boxes overlap is four
comparisons, far cheaper than testing the shapes inside them, so a
tree stores boxes and only descends into the exact geometry when the
boxes cannot rule a query out. The operations that matter are
containment, whether a box holds a point or another box; intersection,
whether two boxes share any area; union, the smallest box covering
both; and the two costs an R-tree balances when it decides where to
put a new entry, the area of a box and the enlargement, how much a
box's area must grow to swallow a new one. The enlargement is the
quantity worth understanding, because it drives the whole family of
rectangle trees. Merging two disjoint boxes produces a box whose area
is at least the sum of their areas and usually more; the excess, the
covered area belonging to neither original box, is dead space, and
every rectangle tree is a strategy for keeping dead space and box
overlap small so that queries prune more and descend less. The
finding worth stating is that box union is superadditive in area,
enlargement is never negative, and that gap is exactly the waste the
index tries to minimize, which is why this humble rectangle carries so
much weight. This module implements the box and its area, containment,
intersection, union, and enlargement, and a survey measures the dead
space merging random boxes creates, so the superadditivity is a number.
"""

from __future__ import annotations

from atlas.errors import Invalid


class BBox:
    __slots__ = ("max_x", "max_y", "min_x", "min_y")

    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float) -> None:
        if min_x > max_x or min_y > max_y:
            raise Invalid("a box's low corner must not exceed its high corner")
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    @classmethod
    def from_points(cls, points: list[tuple[float, float]]) -> BBox:
        if not points:
            raise Invalid("cannot bound an empty set of points")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    def area(self) -> float:
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)

    def margin(self) -> float:
        # half-perimeter, the quantity R*-trees minimize alongside area
        return (self.max_x - self.min_x) + (self.max_y - self.min_y)

    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    def contains_point(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def contains_box(self, other: BBox) -> bool:
        return (
            self.min_x <= other.min_x
            and self.min_y <= other.min_y
            and self.max_x >= other.max_x
            and self.max_y >= other.max_y
        )

    def intersects(self, other: BBox) -> bool:
        return not (
            other.min_x > self.max_x
            or other.max_x < self.min_x
            or other.min_y > self.max_y
            or other.max_y < self.min_y
        )

    def intersection_area(self, other: BBox) -> float:
        if not self.intersects(other):
            return 0.0
        dx = min(self.max_x, other.max_x) - max(self.min_x, other.min_x)
        dy = min(self.max_y, other.max_y) - max(self.min_y, other.min_y)
        return dx * dy

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    def enlargement(self, other: BBox) -> float:
        return self.union(other).area() - self.area()

    def dead_space(self, other: BBox) -> float:
        # area of the union belonging to neither box (minus their overlap credit)
        return (
            self.union(other).area()
            - self.area()
            - other.area()
            + self.intersection_area(other)
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BBox):
            return NotImplemented
        return self.as_tuple() == other.as_tuple()

    def __hash__(self) -> int:
        return hash(self.as_tuple())

    def __repr__(self) -> str:
        return f"BBox({self.min_x}, {self.min_y}, {self.max_x}, {self.max_y})"
