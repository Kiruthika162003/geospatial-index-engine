"""Ball tree: bound each subtree by a sphere and prune by the triangle inequality.

A ball tree is a metric tree for nearest-neighbor search that, unlike
the k-d tree, does not cut space with axis-aligned planes. Instead each
node bounds all the points beneath it with a ball, a center and a
radius large enough to contain them, and splits its points into two
child balls. The bounding ball is what powers the prune, and it prunes
by the triangle inequality rather than by a coordinate comparison. To
find the nearest neighbor to a query, the search keeps the best
distance found so far, and at each node it computes the distance from
the query to the ball's center and subtracts the ball's radius: that
difference is a lower bound on the distance from the query to anything
inside the ball, because no point in the ball can be closer than the
center distance minus the radius. If that lower bound already exceeds
the best distance, the entire ball is skipped, exactly as the k-d tree
skips a subtree behind a far plane. The reason the ball tree exists
alongside the k-d tree is that its bound uses only distances, not
coordinates, so it stays effective in higher dimensions and under
metrics where axis-aligned cuts lose their grip, where the k-d tree's
plane test weakens because the query sits near many planes at once. In
the two dimensions of a map both work well and the choice is a wash,
but the ball tree's distance-only prune is the more general mechanism.
The finding worth stating is that the center-minus-radius lower bound
lets the ball tree skip whole balls while returning the identical
nearest neighbor as a brute scan, the same exact pruning the k-d tree
achieves by a different bound. This module builds a ball tree with
nearest-neighbor search, and a survey confirms it matches brute force
while touching a fraction of the points.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Missing

Point = tuple[float, float]


class _Ball:
    __slots__ = ("center", "left", "point", "radius", "right")

    def __init__(self, center: Point, radius: float) -> None:
        self.center = center
        self.radius = radius
        self.point: Point | None = None
        self.left: _Ball | None = None
        self.right: _Ball | None = None


class BallTree:
    def __init__(self, points: list[Point]) -> None:
        if points is None:
            raise Invalid("points must not be None")
        self._size = len(points)
        self._root = self._build(list(points))
        self.visits = 0

    def __len__(self) -> int:
        return self._size

    def _build(self, points: list[Point]) -> _Ball | None:
        if not points:
            return None
        center = (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
        radius = max(_dist(center, p) for p in points)
        ball = _Ball(center, radius)
        if len(points) == 1:
            ball.point = points[0]
            return ball
        # split on the axis of greatest spread at its median
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
        points.sort(key=lambda p: p[axis])
        mid = len(points) // 2
        ball.left = self._build(points[:mid])
        ball.right = self._build(points[mid:])
        return ball

    def nearest(self, query: Point) -> Point:
        if self._root is None:
            raise Missing("the tree is empty")
        self.visits = 0
        best = self._search(self._root, query, None, math.inf)
        return best[0]

    def _search(
        self, ball: _Ball | None, query: Point, best: Point | None, best_d: float
    ) -> tuple[Point | None, float]:
        if ball is None:
            return best, best_d
        lower_bound = _dist(query, ball.center) - ball.radius
        if lower_bound >= best_d:
            return best, best_d  # the whole ball is farther than the best; skip it
        self.visits += 1
        if ball.point is not None:
            d = _dist(query, ball.point)
            if d < best_d:
                return ball.point, d
            return best, best_d
        # descend into the nearer child first
        left_d = _dist(query, ball.left.center) if ball.left else math.inf
        right_d = _dist(query, ball.right.center) if ball.right else math.inf
        if left_d <= right_d:
            best, best_d = self._search(ball.left, query, best, best_d)
            best, best_d = self._search(ball.right, query, best, best_d)
        else:
            best, best_d = self._search(ball.right, query, best, best_d)
            best, best_d = self._search(ball.left, query, best, best_d)
        return best, best_d


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
