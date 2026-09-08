"""Visvalingam-Whyatt: simplify a line by removing the least-area points first.

Visvalingam's method simplifies a polyline by a different criterion
than Douglas-Peucker, and the difference in criterion gives a different
character of result. Where Douglas-Peucker keeps the points farthest
from a chord, Visvalingam scores each interior point by its effective
area, the area of the triangle it forms with its two neighbors, and
removes the point with the smallest such triangle, the one whose
removal changes the line's enclosed area the least. Removing a point
changes its neighbors' triangles, so their areas are recomputed, and
the process repeats, always deleting the current least-area point,
until the desired number of points remains or every surviving triangle
exceeds an area threshold. The two methods disagree about what detail
matters. Douglas-Peucker is driven by distance, so it clings to a
single far spike even if that spike is a thin sliver enclosing almost
no area; Visvalingam is driven by area, so it removes that thin sliver
early and instead preserves the points that bound real area, which
tends to keep the overall silhouette and shrink the spikes. Neither is
simply better; they answer different questions, keep-the-outliers
versus keep-the-mass, and on the same line simplified to the same
number of points they generally keep different subsets of vertices.
The finding worth stating is that Visvalingam and Douglas-Peucker,
given the same point budget, retain measurably different vertices
because one ranks by distance and the other by area, so the choice of
simplifier is a choice of which features to defend. This module
simplifies by effective area to a target size or an area threshold, and
a survey measures how much the retained vertex set differs from
Douglas-Peucker's on the same line and budget.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]


def _triangle_area(a: Point, b: Point, c: Point) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2


def _effective_areas_removal_order(points: list[Point]) -> list[int]:
    # returns interior indices in the order Visvalingam would remove them
    n = len(points)
    prev = {i: i - 1 for i in range(n)}
    nxt = {i: i + 1 for i in range(n)}
    removed_order: list[int] = []
    # repeatedly find the least-area interior point among the living ones
    interior = set(range(1, n - 1))
    while interior:
        best = min(
            interior,
            key=lambda i: _triangle_area(points[prev[i]], points[i], points[nxt[i]]),
        )
        removed_order.append(best)
        interior.discard(best)
        p, q = prev[best], nxt[best]
        nxt[p] = q
        prev[q] = p
    return removed_order


def simplify_to(points: list[Point], keep: int) -> list[Point]:
    if points is None:
        raise Invalid("points must not be None")
    if keep < 2:
        raise Invalid("must keep at least the two endpoints")
    if len(points) <= keep:
        return list(points)
    order = _effective_areas_removal_order(points)
    to_remove = set(order[: len(points) - keep])
    return [p for i, p in enumerate(points) if i not in to_remove]


def simplify_by_area(points: list[Point], min_area: float) -> list[Point]:
    if points is None:
        raise Invalid("points must not be None")
    if min_area < 0:
        raise Invalid("min_area must not be negative")
    if len(points) <= 2:
        return list(points)
    # remove points whose effective area at removal time is below the threshold
    n = len(points)
    prev = {i: i - 1 for i in range(n)}
    nxt = {i: i + 1 for i in range(n)}
    interior = set(range(1, n - 1))
    removed: set[int] = set()
    while interior:
        best = min(
            interior,
            key=lambda i: _triangle_area(points[prev[i]], points[i], points[nxt[i]]),
        )
        if _triangle_area(points[prev[best]], points[best], points[nxt[best]]) >= min_area:
            break
        interior.discard(best)
        removed.add(best)
        p, q = prev[best], nxt[best]
        nxt[p] = q
        prev[q] = p
    return [p for i, p in enumerate(points) if i not in removed]
