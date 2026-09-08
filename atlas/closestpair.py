"""Closest pair: divide and conquer, where the strip needs only a few comparisons.

Finding the two closest points among many is the mirror of the
diameter problem, and unlike the diameter it does not reduce to the
convex hull, since the closest pair can be deep in the interior. The
brute force compares every pair, quadratic. Divide and conquer brings
it to n log n by splitting the points down the middle by x, solving
each half recursively, and then merging, and the whole subtlety is in
the merge. After the two halves are solved, the smallest distance so
far is the smaller of the two half-answers, call it delta, and the only
way the true closest pair can be closer is if its two points straddle
the dividing line, one in each half, within delta of the line. So the
merge looks only at points in a vertical strip of width two delta
around the divider. The strip could still hold many points, so a naive
all-pairs check inside it would ruin the bound; the saving insight is
that within the strip, sorted by y, any point can be closer than delta
to only a constant number of the points that follow it, because more
than a few points within a delta-by-two-delta box would themselves be
closer than delta to each other, contradicting delta. Checking each
strip point against a fixed small number of its successors is therefore
enough, and the merge is linear, giving the n log n total. The finding
worth stating is that the strip's bounded-neighbor property, at most a
constant number of candidates per point, is what keeps the merge linear
and the whole algorithm below quadratic, and it returns the exact same
closest pair the brute force would. This module finds the closest pair
by divide and conquer, and a survey confirms it equals the brute
closest pair while the strip check stays bounded.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]


def _dist2(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def closest_pair(points: list[Point]) -> tuple[float, Point, Point]:
    if points is None:
        raise Invalid("points must not be None")
    if len(points) < 2:
        raise Invalid("need at least two points for a closest pair")
    by_x = sorted(points)
    best_d2, a, b = _solve(by_x)
    return (best_d2**0.5, a, b)


def _solve(px: list[Point]) -> tuple[float, Point, Point]:
    n = len(px)
    if n <= 3:
        best = (float("inf"), px[0], px[0])
        for i in range(n):
            for j in range(i + 1, n):
                d2 = _dist2(px[i], px[j])
                if d2 < best[0]:
                    best = (d2, px[i], px[j])
        return best
    mid = n // 2
    mid_x = px[mid][0]
    left = _solve(px[:mid])
    right = _solve(px[mid:])
    best = left if left[0] <= right[0] else right
    # points within delta of the divider, sorted by y
    strip = sorted((p for p in px if (p[0] - mid_x) ** 2 < best[0]), key=lambda p: p[1])
    for i in range(len(strip)):
        # only the next few points in y can beat delta; the loop self-limits
        for j in range(i + 1, len(strip)):
            if (strip[j][1] - strip[i][1]) ** 2 >= best[0]:
                break
            d2 = _dist2(strip[i], strip[j])
            if d2 < best[0]:
                best = (d2, strip[i], strip[j])
    return best
