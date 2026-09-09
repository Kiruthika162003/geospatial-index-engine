"""Geodesic grid: the sphere tiled from an icosahedron, with the cells' evenness measured.

Climate models and global games tile the sphere by starting from
an icosahedron, whose twenty triangular faces are the most even
way to cover a sphere with flat pieces, subdividing each face
into smaller triangles, and pushing the new vertices out onto the
sphere. The result at frequency f has 20 f squared triangles and
10 f squared plus 2 vertices, of which exactly twelve, the
icosahedron's own, have five neighbours and every other has six,
which is why every such grid has twelve pentagons however fine
it is. The survey measures the evenness that is the grid's whole
point, and the guess about it was too kind. The guess was that
the triangles' spherical areas would vary by about 1.2 to one at
moderate frequency and settle near 1.3 as the frequency grew;
with the plain subdivision used here, new vertices placed at
equal barycentric steps across each face and pushed to the
sphere, the ratio of the largest to the smallest triangle read
1.00, 1.20, 1.52, 1.74, and 1.86 at frequencies 1, 2, 4, 8, and
16, still climbing, since the projection stretches the triangles
near the face centers more than those near the icosahedron's
vertices; the edge lengths varied by 1.00, 1.13, 1.29, 1.38, and
1.43. That is a third of the cube-face scheme's fivefold spread
and the sum of the areas was the sphere's to eight places at
every frequency, but it is not 1.3, and an evener grid needs the
vertices spaced by arc rather than by chord. The vertex count and
the triangle count followed 10 f squared plus 2 and 20 f squared
exactly at every frequency, with twelve five-neighbour vertices
and the rest six, 30, 150, 630, and 2550 of them. And the grid
partitions the sphere: decoding the centroid of the triangle
holding a random point and locating it again returned the same
triangle for all 300 random points at frequency 6. The finding
worth stating is that a plain icosahedral grid's triangles differ
in area by 1.86 to one at frequency 16 and still climbing, a
third of the cube's spread but not the near-uniformity the guess
expected, that twelve vertices have five neighbours and the rest
six however fine the grid, and that the grid locates points
consistently, so it buys evener cells than a cube at the cost of
twelve places where the neighbour count changes and a spread its
subdivision rule sets. This module
builds geodesic grids and locates points in them, and a survey
measures the area spread, the counts, and the containment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid, Outside
from atlas.sphericalarea import triangle_area_km2

Vector = tuple[float, float, float]
Point = tuple[float, float]


def _normalize(v: Vector) -> Vector:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / n, v[1] / n, v[2] / n)


def _to_latlon(v: Vector) -> Point:
    lat = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
    return (lat, math.degrees(math.atan2(v[1], v[0])))


def _to_vector(lat: float, lon: float) -> Vector:
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise Outside("the point lies off the globe")
    phi, lam = math.radians(lat), math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))


def icosahedron() -> tuple[list[Vector], list[tuple[int, int, int]]]:
    t = (1 + math.sqrt(5)) / 2
    raw = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    vertices = [_normalize((float(x), float(y), float(z))) for x, y, z in raw]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    return vertices, faces


class GeodesicGrid:
    def __init__(self, frequency: int):
        if frequency < 1:
            raise Invalid("the frequency must be at least one")
        self.frequency = frequency
        base_vertices, base_faces = icosahedron()
        self.vertices: list[Vector] = []
        self.triangles: list[tuple[int, int, int]] = []
        index: dict[tuple[float, float, float], int] = {}

        def vertex(v: Vector) -> int:
            key = (round(v[0], 10), round(v[1], 10), round(v[2], 10))
            if key not in index:
                index[key] = len(self.vertices)
                self.vertices.append(v)
            return index[key]

        f = frequency
        for a, b, c in base_faces:
            va, vb, vc = base_vertices[a], base_vertices[b], base_vertices[c]
            grid: dict[tuple[int, int], int] = {}
            for i in range(f + 1):
                for j in range(f + 1 - i):
                    k = f - i - j
                    point = _normalize(
                        (
                            (i * va[0] + j * vb[0] + k * vc[0]) / f,
                            (i * va[1] + j * vb[1] + k * vc[1]) / f,
                            (i * va[2] + j * vb[2] + k * vc[2]) / f,
                        )
                    )
                    grid[(i, j)] = vertex(point)
            for i in range(f):
                for j in range(f - i):
                    self.triangles.append((grid[(i, j)], grid[(i + 1, j)], grid[(i, j + 1)]))
                    if i + j < f - 1:
                        upper = (grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)])
                        self.triangles.append(upper)

    def neighbour_counts(self) -> dict[int, int]:
        edges: set[tuple[int, int]] = set()
        for a, b, c in self.triangles:
            for u, v in ((a, b), (b, c), (c, a)):
                edges.add((min(u, v), max(u, v)))
        degree = [0] * len(self.vertices)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        counts: dict[int, int] = {}
        for d in degree:
            counts[d] = counts.get(d, 0) + 1
        return counts

    def areas_km2(self) -> list[float]:
        corners = [
            tuple(_to_latlon(self.vertices[k]) for k in tri) for tri in self.triangles
        ]
        return [triangle_area_km2(a, b, c) for a, b, c in corners]

    def edge_lengths(self) -> list[float]:
        out = []
        for a, b, c in self.triangles:
            for u, v in ((a, b), (b, c), (c, a)):
                pu, pv = self.vertices[u], self.vertices[v]
                dot = pu[0] * pv[0] + pu[1] * pv[1] + pu[2] * pv[2]
                out.append(math.acos(max(-1.0, min(1.0, dot))))
        return out

    def locate(self, lat: float, lon: float) -> int:
        # the triangle whose spherical interior holds the point: the one where the point
        # lies on the inner side of all three edge planes
        p = _to_vector(lat, lon)
        for t, (a, b, c) in enumerate(self.triangles):
            va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
            sides = (_inside(p, va, vb), _inside(p, vb, vc), _inside(p, vc, va))
            if all(s >= -1e-12 for s in sides):
                return t
        raise Outside("no triangle holds the point")

    def centroid(self, t: int) -> Point:
        a, b, c = self.triangles[t]
        va, vb, vc = self.vertices[a], self.vertices[b], self.vertices[c]
        total = (va[0] + vb[0] + vc[0], va[1] + vb[1] + vc[1], va[2] + vb[2] + vc[2])
        return _to_latlon(_normalize(total))


def _inside(p: Vector, a: Vector, b: Vector) -> float:
    # the signed volume of (a, b, p): positive when p lies on the left of the edge a-b
    return (
        a[0] * (b[1] * p[2] - b[2] * p[1])
        - a[1] * (b[0] * p[2] - b[2] * p[0])
        + a[2] * (b[0] * p[1] - b[1] * p[0])
    )


def expected_counts(frequency: int) -> tuple[int, int]:
    return 10 * frequency * frequency + 2, 20 * frequency * frequency


def spread(values: Sequence[float]) -> float:
    return max(values) / min(values)
