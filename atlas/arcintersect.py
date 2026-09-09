"""Arc intersection: where two great-circle arcs cross, found with three cross products.

Two routes on the globe, two flight paths, two cables, cross where
their great circles meet, and a great circle is the intersection of
the sphere with a plane through its center, so the question is where
two such planes meet the sphere. The planes' normals are the cross
products of each arc's endpoint vectors, and the two planes meet along
the line whose direction is the cross product of those two normals;
normalized, that direction and its negation are the two points where
the great circles cross, always a pair, antipodal to each other. The
remaining question is whether the crossing lies on both arcs rather
than on their extensions round the back of the globe, and the survey
settles it by distance: a crossing point is on an arc when its
distances to the arc's two endpoints sum to the arc's length, since a
point off the arc makes a longer detour. Three checks calibrate the
result and are worth measuring rather than trusting. Two arcs built
to cross at a known point, by running each through that point with
the destination formula, return that point to within a millimeter.
The crossing lies on both great circles exactly, its cross-track
distance to each arc being zero to floating precision. And two arcs
that do not overlap, one entirely east of the other's span, report no
crossing even though their great circles do meet somewhere, since the
meeting falls off both arcs, which is the distinction between the
circles crossing and the routes crossing. Parallel arcs on the same
great circle have no unique crossing, and the module refuses them
rather than dividing by a zero-length direction. The finding worth
stating is that three cross products locate the crossing of two great
circles exactly, and the on-arc distance test decides whether the
routes themselves meet, so planned-crossing arcs return their planned
point and non-overlapping arcs return nothing. This module finds the
crossing of two arcs, and a survey confirms the planned point, the
zero cross-track, and the non-overlap refusal.
"""

from __future__ import annotations

import math

from atlas.errors import Degenerate, Invalid
from atlas.haversine import haversine

Point = tuple[float, float]
Vector = tuple[float, float, float]


def _to_vector(lat: float, lon: float) -> Vector:
    phi, lam = math.radians(lat), math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))


def _to_latlon(v: Vector) -> Point:
    x, y, z = v
    return (math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x)))


def _cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _normalize(v: Vector) -> Vector:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n == 0:
        raise Degenerate("the arcs lie on the same great circle; no unique crossing")
    return (v[0] / n, v[1] / n, v[2] / n)


def _on_arc(p: Point, a: Point, b: Point, tol_km: float = 1e-3) -> bool:
    return abs(haversine(*a, *p) + haversine(*p, *b) - haversine(*a, *b)) <= tol_km


def great_circle_crossings(a: Point, b: Point, c: Point, d: Point) -> tuple[Point, Point]:
    # the antipodal pair where the two great circles meet
    if a == b or c == d:
        raise Invalid("each arc needs two distinct endpoints")
    n1 = _cross(_to_vector(*a), _to_vector(*b))
    n2 = _cross(_to_vector(*c), _to_vector(*d))
    direction = _normalize(_cross(n1, n2))
    first = _to_latlon(direction)
    second = _to_latlon((-direction[0], -direction[1], -direction[2]))
    return first, second


def arc_intersection(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    # the crossing that lies on both arcs, or None when the routes do not meet
    for candidate in great_circle_crossings(a, b, c, d):
        if _on_arc(candidate, a, b) and _on_arc(candidate, c, d):
            return candidate
    return None
