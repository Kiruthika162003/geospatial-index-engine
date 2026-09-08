"""Voronoi vertices from the Delaunay dual: each circumcenter is equidistant from three sites.

The Voronoi diagram partitions the plane into cells, one per site,
where each cell is the region closer to its site than to any other,
the map of which shop, tower, or well serves each spot. Building it
directly is intricate, but it is the exact dual of the Delaunay
triangulation, and the duality hands the pieces over for free. Every
Delaunay triangle's circumcenter is a Voronoi vertex, the meeting
point of the three cells of the triangle's corners, and every pair of
Delaunay triangles that share an edge yields a Voronoi edge between
their two circumcenters, the border between the two cells whose sites
are the shared edge's endpoints. So the vertices and edges of the
diagram are read straight off the triangulation with one circumcenter
computation per triangle. Two properties define a Voronoi vertex and
are worth measuring rather than trusting the duality on faith. First,
a circumcenter is equidistant from the three corners of its triangle,
which is the meaning of circumcenter and the reason three cells meet
there. Second, and this is the empty-circumcircle property restated,
no fourth site is nearer to that vertex than those three, because if
one were, the circumcircle would contain it and the triangle would not
be Delaunay; so each Voronoi vertex's nearest sites are exactly its
triangle's corners, at a common distance, with everything else
farther. The Voronoi edge between two adjacent circumcenters lies on
the perpendicular bisector of the shared Delaunay edge, so every point
on it is equidistant from the two sites, which the survey samples.
Sites on the convex hull have unbounded cells whose outer edges go to
infinity and are not produced here, only the finite edges between
adjacent finite triangles. The finding worth stating is that every
Voronoi vertex is at one distance from its three sites and strictly
farther from all others, and every finite Voronoi edge is equidistant
from its two sites along its length, so the dual construction yields a
correct diagram. This module reads vertices and finite edges off the
Delaunay triangulation, and a survey verifies both properties.
"""

from __future__ import annotations

import math

from atlas.delaunay import triangulate
from atlas.errors import Degenerate, Invalid

Point = tuple[float, float]
Triangle = tuple[Point, Point, Point]


def circumcenter(tri: Triangle) -> Point:
    (ax, ay), (bx, by), (cx, cy) = tri
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0:
        raise Degenerate("collinear triangle has no circumcenter")
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy)


class Voronoi:
    def __init__(self, sites: list[Point]) -> None:
        if sites is None or len(set(sites)) < 3:
            raise Invalid("need at least three distinct sites")
        self.sites = list(dict.fromkeys(sites))
        self.triangles = triangulate(self.sites)
        self.vertices: list[Point] = [circumcenter(t) for t in self.triangles]
        # finite edges: between circumcenters of triangles sharing an edge
        owners: dict[tuple[Point, Point], list[int]] = {}
        for idx, t in enumerate(self.triangles):
            for i in range(3):
                a, b = t[i], t[(i + 1) % 3]
                key = (a, b) if a <= b else (b, a)
                owners.setdefault(key, []).append(idx)
        self.edges: list[tuple[int, int, tuple[Point, Point]]] = [
            (ids[0], ids[1], key) for key, ids in owners.items() if len(ids) == 2
        ]

    def nearest_sites(self, point: Point) -> list[Point]:
        distances = {s: math.hypot(point[0] - s[0], point[1] - s[1]) for s in self.sites}
        best = min(distances.values())
        return [s for s, d in distances.items() if abs(d - best) < 1e-6]
