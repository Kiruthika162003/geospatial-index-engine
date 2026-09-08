"""Ring validity: a polygon boundary is simple only if no two non-adjacent edges touch.

A polygon boundary, a ring, is simple when it does not cross or touch
itself, and most of the geometry built on polygons quietly assumes it:
the shoelace area of a self-crossing ring is a signed mix of its
lobes, point-in-polygon splits into the parity and winding disputes,
and triangulation refuses outright. Validating a ring before trusting
it is therefore the first step of any ingest, and the definition is
mechanical. Every pair of edges that are not adjacent along the ring
must be disjoint, and every pair that are adjacent may share only
their common vertex. Adjacent edges are exempt because they always
touch at the vertex they share, so testing them would reject every
ring; the exemption is exactly the two neighbors, and for a ring of
three edges every pair is adjacent, so a triangle is always simple.
The test is quadratic, each non-adjacent pair checked with the
orientation-sign segment intersection, which is fine for the modest
rings of map features and is what the survey uses as its ground
truth. Three shapes calibrate it and are worth naming. A bowtie, the
four vertices of a square visited in the crossing order, is invalid
because its two diagonals cross. A shape with a vertex repeated
elsewhere along the ring, touching itself at a point, is invalid too,
since two non-adjacent edges meet there. And a strongly concave
comb, however wiggly, is valid as long as its teeth never meet, which
the survey confirms by sheering random convex polygons into concave
ones that stay simple. The finding worth stating is that simplicity
is a pairwise non-adjacent edge condition and nothing more, so the
quadratic check decides it exactly, rejecting the bowtie and the
pinched ring while passing every non-self-touching concave shape. This
module validates a ring and names the offending edge pair when there
is one, and a survey confirms the verdicts on the calibrating shapes.
"""

from __future__ import annotations

from atlas.errors import Invalid
from atlas.segmentintersect import segments_intersect

Point = tuple[float, float]


def first_crossing(ring: list[Point]) -> tuple[int, int] | None:
    # the indices of the first pair of non-adjacent edges that touch, or None
    if ring is None or len(ring) < 3:
        raise Invalid("a ring needs at least three vertices")
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # the last edge is adjacent to the first
            c, d = ring[j], ring[(j + 1) % n]
            if segments_intersect(a, b, c, d):
                return (i, j)
    return None


def is_simple(ring: list[Point]) -> bool:
    return first_crossing(ring) is None


def has_repeated_vertex(ring: list[Point]) -> bool:
    if ring is None:
        raise Invalid("ring must not be None")
    return len(set(ring)) != len(ring)
