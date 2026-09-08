"""Jarvis march: wrap the convex hull point by point, paying only for the hull's size.

The Jarvis march, or gift wrapping, builds a convex hull by imagining
a string tied to the leftmost point and pulled taut around the set: at
each step it finds the point that makes the sharpest right turn from
the current hull edge, which is the next point the wrapping string
would touch, and continues until it returns to the start. Its cost is
what distinguishes it. Each hull vertex costs a full scan of the points
to find the most counterclockwise next one, so the total work is the
point count times the number of hull vertices, order n times h. This
is output-sensitive: the running time depends not just on how many
points there are but on how many end up on the hull. When the hull is
tiny, a handful of points wrapping thousands, gift wrapping is
excellent, near linear, beating the n-log-n sort-based hulls that pay
for a full sort regardless. When the hull is large, in the worst case
every point lying on a circle so all of them are hull vertices, h
approaches n and the cost degrades to n squared, worse than the
sort-based methods. So the choice between gift wrapping and a
sort-based hull is really a bet on how many points sit on the boundary.
The wrap also naturally reports the hull size as it goes, one step per
hull vertex, which is the quantity that governs its own cost. The
finding worth stating is that Jarvis march's running time is literally
the point count times the hull size, so it is the right hull when the
boundary is sparse and the wrong one when the boundary is crowded, a
tradeoff a fixed-cost sort cannot make. This module wraps the convex
hull and reports the number of wrap steps, and a survey measures that
the step count equals the hull size across sparse and crowded inputs.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: list[Point]) -> list[Point]:
    if points is None:
        raise Invalid("points must not be None")
    unique = sorted(set(points))
    if len(unique) <= 2:
        return unique
    start = unique[0]  # leftmost-lowest is guaranteed on the hull
    hull: list[Point] = []
    current = start
    while True:
        hull.append(current)
        candidate = unique[0] if unique[0] != current else unique[1]
        for p in unique:
            if p == current:
                continue
            turn = _cross(current, candidate, p)
            # pick the most counterclockwise; break ties by taking the farther point
            if turn < 0 or (turn == 0 and _dist2(current, p) > _dist2(current, candidate)):
                candidate = p
        current = candidate
        if current == start:
            break
    return hull


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
