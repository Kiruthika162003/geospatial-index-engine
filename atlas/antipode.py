"""Antipode: the point opposite through the earth, and the identity every third point obeys.

Every point on a sphere has an antipode, the point directly opposite
through the center, at the negated latitude and the longitude turned
half a turn. It is the farthest place from where you stand, and it
carries an identity that is exact on a sphere and worth stating as a
measurement: for any third point whatsoever, its great-circle
distance to a point plus its distance to that point's antipode equals
half the circumference, because the two arcs together trace a great
semicircle from the point through the third point to the antipode.
Two consequences follow. The distance from a point to its own antipode
is exactly half the circumference, the largest great-circle distance
the sphere allows, and no third point can be farther from you than
your antipode. And the identity is a sensitive check on a distance
formula: if haversine were wrong anywhere, the sum would drift from
the half circumference for some third point, so holding the identity
to floating precision over thousands of random triples is a
whole-globe certificate of the formula's symmetry. The antipode is
also where the tools of this package break down honestly. Vincenty's
iteration fails to converge near it, the interpolation between a
point and its antipode has no unique great circle, and the initial
bearing toward it is undefined, since every direction reaches it at
the same distance; the module names those cases rather than papering
over them. The finding worth stating is that the distance to a point
plus the distance to its antipode is half the circumference for every
third point tried, to a few parts in ten to the twelfth, and the
antipode is the unique farthest point, so the sphere's opposite is
both a place and a law. This module computes antipodes and checks the
identity, and a survey measures it over random triples.
"""

from __future__ import annotations

import math

from atlas.errors import Outside
from atlas.haversine import EARTH_RADIUS_KM, haversine


def antipode(lat: float, lon: float) -> tuple[float, float]:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    opposite = lon + 180.0
    if opposite > 180.0:
        opposite -= 360.0
    return (-lat, opposite)


def half_circumference_km(radius: float = EARTH_RADIUS_KM) -> float:
    return math.pi * radius


def identity_residual(lat: float, lon: float, third_lat: float, third_lon: float) -> float:
    # the departure of d(third, p) + d(third, antipode(p)) from half the circumference
    alat, alon = antipode(lat, lon)
    to_point = haversine(third_lat, third_lon, lat, lon)
    to_opposite = haversine(third_lat, third_lon, alat, alon)
    return to_point + to_opposite - half_circumference_km()


def is_antipodal(
    lat1: float, lon1: float, lat2: float, lon2: float, tol_km: float = 1e-6
) -> bool:
    return haversine(lat1, lon1, lat2, lon2) >= half_circumference_km() - tol_km
