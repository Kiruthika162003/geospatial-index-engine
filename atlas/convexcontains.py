"""Convex containment: point-in-convex-polygon by binary search over the fan, log n not n.

Ray casting answers point-in-polygon for any polygon in time linear
in its edges, because it must consider every edge as a possible
crossing. A convex polygon allows a much faster test, and the reason
is that its vertices, taken counterclockwise from any one of them,
fan out in angular order around that vertex with no reversals. Fix
vertex zero as the fan's apex. The polygon is the union of the
triangles apex, vertex i, vertex i plus one, and these triangles tile
it without overlap in angular order, so a query point falls into at
most one of them, and which one can be found by binary search on the
sign of the cross product with the spokes from the apex. Two cross
products decide whether the query lies inside the fan's angular range
at all, between the first and last spokes; a binary search over the
spokes then finds the wedge the query lies in; and one final
orientation test against that wedge's outer edge says whether the
query is inside the polygon or beyond the edge. That is order log n
orientation tests against ray casting's order n, and the survey
measures the count. The verdict is identical to ray casting for every
point, which the survey confirms, because both are exact predicates on
the same polygon; the difference is only in how many edges are
consulted. Points exactly on the boundary are counted inside by the
non-strict orientation test, a convention worth naming. The test
assumes convexity and counterclockwise order and gives no meaningful
answer otherwise, which is the price of the speed. The finding worth
stating is that convexity turns point-in-polygon into a binary search,
matching ray casting's verdict on every query while testing a
logarithmic number of edges, so for a convex region the fan search is
the right tool. This module implements the fan binary search and
counts the orientation tests, and a survey confirms the verdict and
the logarithmic count.
"""

from __future__ import annotations

from atlas.errors import Invalid
from atlas.pointinpolygon import ray_casting

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


class ConvexPolygon:
    def __init__(self, vertices: list[Point]) -> None:
        if vertices is None or len(vertices) < 3:
            raise Invalid("a convex polygon needs at least three vertices")
        n = len(vertices)
        for i in range(n):
            if _cross(vertices[i], vertices[(i + 1) % n], vertices[(i + 2) % n]) < 0:
                raise Invalid("vertices must be convex and counterclockwise")
        self.vertices = list(vertices)
        self.tests = 0

    def contains(self, p: Point) -> bool:
        v = self.vertices
        n = len(v)
        self.tests = 2
        # outside the fan's angular range between the first and last spokes
        if _cross(v[0], v[1], p) < 0 or _cross(v[0], v[n - 1], p) > 0:
            return False
        lo, hi = 1, n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            self.tests += 1
            if _cross(v[0], v[mid], p) >= 0:
                lo = mid
            else:
                hi = mid
        self.tests += 1
        return _cross(v[lo], v[hi], p) >= 0

    def contains_by_ray(self, p: Point) -> bool:
        return ray_casting(p, self.vertices)
