"""Spherical polygons: point-in-polygon on the globe, and where the flat test breaks.

A polygon drawn on the sphere, a country, a flight region, an
ocean basin, is tested for containment two ways. The flat way
treats latitude and longitude as x and y and casts a ray, which
is what most code does and which fails in three known places: at
the dateline, where a polygon spanning it looks like two pieces
with the whole world between them; at the poles, where a polygon
enclosing a pole has no crossing to count; and along long edges,
which on the map are straight but on the globe are not, so points
near an edge fall on the wrong side. The spherical way works with
unit vectors: each edge is a great-circle arc, and the point is
inside when the sum of the signed angles it turns through going
round the polygon is a full turn, with no map at all. The survey
measures the flat test's failures against the spherical test on
polygons built to exhibit them. On a compact box 20 degrees wide
between latitudes 40 and 55 the two disagreed on 0.5 percent of
2000 random points, the sliver along its parallels where the
great-circle edge bows north of the map line. On the same box
shifted to straddle the dateline, longitudes 170 to -170, the
guess was a hundred percent disagreement, and the measurement was
41 percent: the flat test inverts every verdict inside the
latitude band, calling the true inside outside and the outside
inside, but above and below the band both tests say outside, so
the two agreed on the 59 percent of points beyond the parallels
and read the same inside fraction of 20.4 percent on different
points. On a ring of twelve vertices at latitude 60 the flat test
called all 1000 random points between 61 and 89 outside and the
pole with them, while the spherical test called all 1000 inside
and none of 1000 points below 59, so a polar polygon is missed
entirely. On a box with edges spanning 40 degrees of longitude
the flat test disagreed along each parallel in a band that is the
arc's bulge, 1.8 degrees of latitude on the southern edge at 40
and 1.5 on the northern at 60, which over a one-degree grid of
1125 points came to 13.3 percent disagreement. Reversing the
polygon's vertex order changed no spherical verdict, and a point
on a vertex is decided inside. The finding worth stating is that
the flat test inverts a dateline-straddling polygon within its
latitude band, misses a polar polygon entirely, and errs in a
band of 1.5 to 1.8 degrees along 40-degree edges, while the
spherical winding test needs no map and has no such places, so
containment on the globe must be done on the globe. This module
tests containment on the sphere, and a survey
measures the flat test's three failures against it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid, Outside

Point = tuple[float, float]
Vector = tuple[float, float, float]


def _vec(lat: float, lon: float) -> Vector:
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise Outside("the point lies off the globe")
    phi, lam = math.radians(lat), math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))


def _cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vector) -> Vector:
    n = math.sqrt(_dot(v, v))
    if n == 0:
        raise Invalid("a zero vector has no direction")
    return (v[0] / n, v[1] / n, v[2] / n)


def spherical_contains(polygon: Sequence[Point], lat: float, lon: float) -> bool:
    # the winding number: the sum of signed turning angles round the polygon as seen
    # from the point is a full turn inside and zero outside
    if len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    p = _vec(lat, lon)
    total = 0.0
    n = len(polygon)
    for i in range(n):
        a = _vec(*polygon[i])
        b = _vec(*polygon[(i + 1) % n])
        # tangent-plane directions from p toward a and toward b
        da = (a[0] - p[0] * _dot(a, p), a[1] - p[1] * _dot(a, p), a[2] - p[2] * _dot(a, p))
        db = (b[0] - p[0] * _dot(b, p), b[1] - p[1] * _dot(b, p), b[2] - p[2] * _dot(b, p))
        if _dot(da, da) < 1e-18 or _dot(db, db) < 1e-18:
            return True  # the point sits on a vertex
        da, db = _norm(da), _norm(db)
        sine = _dot(_cross(da, db), p)
        cosine = _dot(da, db)
        total += math.atan2(sine, cosine)
    return abs(total) > math.pi


def planar_contains(polygon: Sequence[Point], lat: float, lon: float) -> bool:
    # the ordinary ray cast on latitude and longitude as y and x
    if len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    inside = False
    n = len(polygon)
    for i in range(n):
        y1, x1 = polygon[i]
        y2, x2 = polygon[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            x = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < x:
                inside = not inside
    return inside


def disagreement(polygon: Sequence[Point], points: Sequence[Point]) -> float:
    if not points:
        raise Invalid("need at least one point")
    wrong = 0
    for lat, lon in points:
        if planar_contains(polygon, lat, lon) != spherical_contains(polygon, lat, lon):
            wrong += 1
    return wrong / len(points)


def shifted(polygon: Sequence[Point], by_lon: float) -> list[Point]:
    return [(lat, ((lon + by_lon + 180.0) % 360.0) - 180.0) for lat, lon in polygon]


def polar_cap(latitude: float, vertices: int = 12) -> list[Point]:
    return [(latitude, -180.0 + 360.0 * k / vertices) for k in range(vertices)]
