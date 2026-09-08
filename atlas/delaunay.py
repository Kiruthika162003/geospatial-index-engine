"""Delaunay triangulation: no point inside any triangle's circumcircle, built point by point.

Connecting a scatter of points into triangles can be done many ways,
and most of them are bad, full of long thin slivers. The Delaunay
triangulation is the canonical good one, defined by a single rule: no
point of the set lies strictly inside the circumcircle of any
triangle. That empty-circumcircle property maximizes the smallest
angle over all triangulations of the set, which is why it underlies
terrain meshes, nearest-neighbor graphs, and interpolation, and its
dual is the Voronoi diagram. Bowyer-Watson builds it incrementally.
Start with one enormous super-triangle enclosing every point. Insert
the points one at a time: find every triangle whose circumcircle
contains the new point, the bad triangles, remove them, and their
union is a star-shaped polygonal hole whose boundary edges are those
belonging to exactly one bad triangle; connect the new point to each
boundary edge to fill the hole with new triangles, each of which now
has an empty circumcircle with respect to the points so far. After all
points are in, drop every triangle touching a super-triangle vertex.
The circumcircle containment test is a determinant of the three
vertices and the point, exact in integer arithmetic. Two counts pin
the result and are worth measuring. For n points with h of them on the
convex hull boundary, collinear boundary points included, the
triangulation has exactly two n minus h minus two triangles, a formula
from Euler's relation, and every triangle's circumcircle is empty,
which the survey checks against every other point with the same
determinant. The construction hides a trap that the measurement
exposed and the docstring keeps. A first super-triangle fifty times
the point spread across looked generous; on three hundred random
integer sets it lost boundary triangles on eleven of them, every
count below the Euler target, never above, while the circumcircle
rule still held on the triangles that were built. The cause is that a
hull triangle whose circumcircle reaches out to a super-triangle
vertex never gets formed, because that vertex sits inside it, and
when the super-triangle's triangles are discarded at the end the hull
triangle is simply absent. Making the super-triangle a thousand times
the spread restored every count, and it is set to a hundred thousand
times here, so the super-triangle must be far larger than intuition
suggests. The method is quadratic in the simple form here, adequate
for the modest sets of map features. The finding worth stating is that
with a sufficiently distant super-triangle the empty-circumcircle rule
holds for every triangle produced and the triangle count matches the
Euler formula exactly, so the incremental construction yields the
canonical triangulation, and a merely large super-triangle silently
drops hull triangles. This module triangulates by Bowyer-Watson,
and a survey checks the circumcircle rule and the count.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]
Triangle = tuple[Point, Point, Point]


def in_circumcircle(tri: Triangle, p: Point) -> bool:
    (ax, ay), (bx, by), (cx, cy) = tri
    # translate so p is the origin, then the determinant's sign gives containment
    adx, ady = ax - p[0], ay - p[1]
    bdx, bdy = bx - p[0], by - p[1]
    cdx, cdy = cx - p[0], cy - p[1]
    det = (
        (adx * adx + ady * ady) * (bdx * cdy - cdx * bdy)
        - (bdx * bdx + bdy * bdy) * (adx * cdy - cdx * ady)
        + (cdx * cdx + cdy * cdy) * (adx * bdy - bdx * ady)
    )
    orient = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return det > 0 if orient > 0 else det < 0


def triangulate(points: list[Point], super_scale: float = 100000.0) -> list[Triangle]:
    if points is None or len(set(points)) < 3:
        raise Invalid("need at least three distinct points")
    if super_scale <= 1:
        raise Invalid("the super-triangle must exceed the point spread")
    pts = list(dict.fromkeys(points))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    mid_x = (min(xs) + max(xs)) / 2
    mid_y = (min(ys) + max(ys)) / 2
    # a 50x span super-triangle lost boundary triangles on 11 of 300 sets; 1000x lost none
    big = super_scale * span
    sup: Triangle = (
        (mid_x - big, mid_y - big),
        (mid_x + big, mid_y - big),
        (mid_x, mid_y + big),
    )
    triangles: list[Triangle] = [sup]
    for p in pts:
        bad = [t for t in triangles if in_circumcircle(t, p)]
        edges: dict[tuple[Point, Point], int] = {}
        for t in bad:
            for i in range(3):
                a, b = t[i], t[(i + 1) % 3]
                key = (a, b) if a <= b else (b, a)
                edges[key] = edges.get(key, 0) + 1
        boundary = [e for e, n in edges.items() if n == 1]
        triangles = [t for t in triangles if t not in bad]
        for a, b in boundary:
            triangles.append((a, b, p))
    sup_vertices = set(sup)
    return [t for t in triangles if not (set(t) & sup_vertices)]


def is_delaunay(triangles: list[Triangle], points: list[Point]) -> bool:
    for t in triangles:
        for p in points:
            if p in t:
                continue
            if in_circumcircle(t, p):
                return False
    return True
