"""Cube-face cells: the globe projected onto a cube and each face cut as a quadtree.

Geohash and plus codes cut the globe by latitude and longitude,
which makes their cells shrink toward the poles and break at the
dateline. The cube-face scheme avoids both: the sphere is
projected from its center onto the six faces of an enclosing
cube, each face is a square with no poles and no seam inside it,
and each face is cut as a quadtree, so a cell at any level is a
face, a level, and a pair of quadtree coordinates, with neighbours
on the same face found by arithmetic and the poles sitting in the
middle of two faces where nothing is special. The projection is
the gnomonic one, so the cells are not equal in area: a face's
center cell, where the sphere touches the cube, is smaller on the
ground than a corner cell by a factor the survey measures, and
the classic remedy is a quadratic transform of the face
coordinates that squeezes the corners and spreads the center,
which the survey measures too. The ratio of the largest to the
smallest cell area on a face, read from the spherical
quadrilaterals the cells cover, was 3.78, 4.86, and 5.11 at
levels 3, 5, and 7 for the plain projection, climbing toward the
5.2 the gnomonic's corner stretch implies, and 1.56, 1.94, and
2.06 after the transform, so the transform halves the spread. The
six faces' level-3 cells summed to the sphere's area to nine
places. Every point on the sphere lands in exactly one cell at
each level, which the survey checks by encoding 20000 random
points at random levels, decoding each cell's center, and
re-encoding it: all 20000 centers came back to their own cells,
with the six faces holding 3229 to 3423 points each. The first
version of the inverse for the bottom face swapped its two
coordinates, so 16 percent of centers, one face's worth, came
back in the wrong cell, until the inverse was made to mirror the
forward map. The poles encode into the top and bottom faces'
central cells, the dateline runs through the middle of a cell on
the minus-x face at latitude -2.5 rather than along a boundary,
and the cell edge is 10008 km at level 0, 2.44 km at level 12,
9.5 meters at level 20, and 9.3 millimeters at level 30. The
finding worth stating is that cube-face cells have no poles and
no seam at the cost of a fivefold area spread that a quadratic
transform halves to two, that every point lands in one cell
whose center lands back in it, and that the cell schedule halves
per level like any quadtree, so the scheme buys uniformity of
topology with a measured non-uniformity of area. This module
encodes, decodes, and measures cube-face cells, and a survey
measures the area spread and the containment.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside
from atlas.sphericalarea import triangle_area_km2

Point = tuple[float, float]
Vector = tuple[float, float, float]


def _to_vector(lat: float, lon: float) -> Vector:
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise Outside("the point lies off the globe")
    phi, lam = math.radians(lat), math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))


def _to_latlon(v: Vector) -> Point:
    x, y, z = v
    return (math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x)))


def face_of(v: Vector) -> int:
    # 0 +x, 1 +y, 2 +z, 3 -x, 4 -y, 5 -z: the axis with the largest magnitude
    axis = max(range(3), key=lambda i: abs(v[i]))
    return axis if v[axis] > 0 else axis + 3


def face_uv(v: Vector, face: int) -> tuple[float, float]:
    # gnomonic coordinates on the face in [-1, 1]
    x, y, z = v
    if face == 0:
        return y / x, z / x
    if face == 1:
        return -x / y, z / y
    if face == 2:
        return -x / z, -y / z
    if face == 3:
        return z / x, y / x
    if face == 4:
        return z / y, -x / y
    return y / z, x / z


def uv_to_vector(face: int, u: float, v: float) -> Vector:
    if face == 0:
        raw = (1.0, u, v)
    elif face == 1:
        raw = (-u, 1.0, v)
    elif face == 2:
        raw = (-u, -v, 1.0)
    elif face == 3:
        raw = (-1.0, -v, -u)
    elif face == 4:
        raw = (v, -1.0, -u)
    else:
        raw = (-v, -u, -1.0)
    n = math.sqrt(sum(c * c for c in raw))
    return (raw[0] / n, raw[1] / n, raw[2] / n)


def quadratic(u: float) -> float:
    # the S2 quadratic transform from a face coordinate to a cell coordinate, both in
    # [-1, 1]: it squeezes the corners and spreads the center so cell areas even out
    if u >= 0:
        return math.sqrt(1 + 3 * u) - 1
    return 1 - math.sqrt(1 - 3 * u)


def inverse_quadratic(s: float) -> float:
    if s >= 0:
        return ((s + 1) ** 2 - 1) / 3
    return (1 - (1 - s) ** 2) / 3


def encode(
    lat: float, lon: float, level: int, transform: bool = True
) -> tuple[int, int, int, int]:
    if not 0 <= level <= 30:
        raise Invalid("the level must lie within 0 and 30")
    vec = _to_vector(lat, lon)
    face = face_of(vec)
    u, v = face_uv(vec, face)
    if transform:
        u, v = quadratic(u), quadratic(v)
    size = 2**level
    i = min(size - 1, int((u + 1) / 2 * size))
    j = min(size - 1, int((v + 1) / 2 * size))
    return face, level, i, j


def cell_center(face: int, level: int, i: int, j: int, transform: bool = True) -> Point:
    size = 2**level
    u = (i + 0.5) / size * 2 - 1
    v = (j + 0.5) / size * 2 - 1
    if transform:
        u, v = inverse_quadratic(u), inverse_quadratic(v)
    return _to_latlon(uv_to_vector(face, u, v))


def cell_corners(face: int, level: int, i: int, j: int, transform: bool = True) -> list[Point]:
    size = 2**level
    out = []
    for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
        u = (i + du) / size * 2 - 1
        v = (j + dv) / size * 2 - 1
        if transform:
            u, v = inverse_quadratic(u), inverse_quadratic(v)
        out.append(_to_latlon(uv_to_vector(face, u, v)))
    return out


def cell_area_km2(face: int, level: int, i: int, j: int, transform: bool = True) -> float:
    a, b, c, d = cell_corners(face, level, i, j, transform)
    return triangle_area_km2(a, b, c) + triangle_area_km2(a, c, d)


def area_spread(level: int, transform: bool = True) -> float:
    # largest over smallest cell area on face 0 at the level
    size = 2**level
    areas = [cell_area_km2(0, level, i, j, transform) for i in range(size) for j in range(size)]
    return max(areas) / min(areas)


def edge_km(level: int) -> float:
    # the face's quarter-circumference divided among the cells along one side
    return 10007.5 / 2**level
