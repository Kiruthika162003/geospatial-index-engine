"""Discrete Frechet distance: the dog-leash measure that respects the order along a path.

Two GPS traces of the same road can be near each other point for point
and still be different routes, if one doubles back or runs the road in
reverse, and the Hausdorff distance cannot tell, because it looks at
the sets of points and ignores the order in which they were visited.
The Frechet distance is the path measure that honors order. Its image
is a person walking one curve and a dog walking the other, each free
to vary its speed but never to go backward, joined by a leash; the
Frechet distance is the shortest leash that lets both reach the end.
The discrete version, over the vertex sequences, is a dynamic program:
the cost of pairing vertex i of one path with vertex j of the other is
the larger of their distance and the best cost of arriving there from
the three predecessor pairings, i minus one, j minus one, or both, and
the answer sits at the final pair. Because the walk can only advance,
the pairing is monotone, and that monotonicity is what makes Frechet
see structure Hausdorff misses. The survey exhibits it with two paths
that share exactly the same vertex set so their Hausdorff distance is
zero, one traversed forward and one traversed backward: the Frechet
distance between them is the full length of the path, since the leash
must stretch from one end to the other as the walkers pass. On paths
that are genuinely similar the two measures agree closely, and the
Frechet distance is never less than the Hausdorff distance, since any
monotone pairing is also a pairing the max-of-mins could have used.
The finding worth stating is that Frechet is bounded below by Hausdorff,
equals it on well-aligned paths, and exceeds it, up to the whole path
length, exactly when the order of traversal differs, so the gap
between them is a measurement of how much a route's direction matters.
This module computes the discrete Frechet distance, and a survey
measures it against Hausdorff on aligned and reversed paths.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Point = tuple[float, float]


def frechet(p: list[Point], q: list[Point]) -> float:
    if p is None or q is None or not p or not q:
        raise Invalid("both paths must be non-empty")
    n, m = len(p), len(q)
    # ca[i][j] is the leash needed to reach vertex i of p paired with vertex j of q
    ca = [[-1.0] * m for _ in range(n)]

    def dist(i: int, j: int) -> float:
        return math.hypot(p[i][0] - q[j][0], p[i][1] - q[j][1])

    ca[0][0] = dist(0, 0)
    for i in range(1, n):
        ca[i][0] = max(ca[i - 1][0], dist(i, 0))
    for j in range(1, m):
        ca[0][j] = max(ca[0][j - 1], dist(0, j))
    for i in range(1, n):
        for j in range(1, m):
            arrive = min(ca[i - 1][j], ca[i][j - 1], ca[i - 1][j - 1])
            ca[i][j] = max(arrive, dist(i, j))
    return ca[n - 1][m - 1]
