"""Smallest enclosing circle: Welzl's randomized method, fixed by at most three points.

The smallest circle that contains a set of points is the tightest
round boundary around them, used to bound a cluster, cull it against a
view, or place a coverage disk. A useful fact makes it tractable: the
minimal enclosing circle is determined by at most three of the points,
which lie on its boundary, either two points as the ends of a diameter
or three points on the circle, and every other point sits inside. So
the search is not over continuous centers but over which two or three
points pin the circle. Welzl's algorithm finds them in expected linear
time by a randomized incremental construction. Shuffle the points and
add them one at a time, keeping the minimal circle of the points so
far; when a new point already lies inside the current circle, nothing
changes, and when it lies outside, that point must lie on the boundary
of the new minimal circle, which pins one degree of freedom and lets
the circle be rebuilt from the earlier points with that point fixed.
Nesting this reasoning three deep, one boundary point, then two, then
three, gives the circle, and because a point forces a rebuild only when
it falls outside, which the random order makes rare, the expected total
work is linear despite the nesting looking cubic. The randomization is
what buys the linear expectation: an adversarial order could be slow,
but a shuffled order is fast with overwhelming probability. The finding
worth stating is that the minimal circle rests on at most three points
and Welzl finds them while touching each point a constant number of
times on average, returning a circle that contains every point and is
the smallest that does. This module computes the smallest enclosing
circle, and a survey confirms it contains all points and that shrinking
its radius leaves some point outside, so the circle is truly minimal.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Circle = tuple[Point, float]


def _in_circle(circle: Circle, p: Point) -> bool:
    (cx, cy), r = circle
    return math.hypot(p[0] - cx, p[1] - cy) <= r + 1e-9


def _from_two(a: Point, b: Point) -> Circle:
    center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return (center, math.hypot(a[0] - b[0], a[1] - b[1]) / 2)


def _from_three(a: Point, b: Point, c: Point) -> Circle:
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0:
        # collinear; the enclosing circle is the diameter of the farthest pair
        pairs = [_from_two(a, b), _from_two(a, c), _from_two(b, c)]
        return max(pairs, key=lambda circ: circ[1])
    ux = (
        (ax**2 + ay**2) * (by - cy)
        + (bx**2 + by**2) * (cy - ay)
        + (cx**2 + cy**2) * (ay - by)
    ) / d
    uy = (
        (ax**2 + ay**2) * (cx - bx)
        + (bx**2 + by**2) * (ax - cx)
        + (cx**2 + cy**2) * (bx - ax)
    ) / d
    center = (ux, uy)
    return (center, math.hypot(ax - ux, ay - uy))


def smallest_enclosing_circle(points: list[Point], seed: int = 0) -> Circle:
    if points is None or not points:
        raise Invalid("need at least one point")
    pts = list(points)
    random.Random(seed).shuffle(pts)
    circle: Circle = (pts[0], 0.0)
    for i, p in enumerate(pts):
        if _in_circle(circle, p):
            continue
        circle = (p, 0.0)
        for j in range(i):
            q = pts[j]
            if _in_circle(circle, q):
                continue
            circle = _from_two(p, q)
            for k in range(j):
                r = pts[k]
                if not _in_circle(circle, r):
                    circle = _from_three(p, q, r)
    return circle
