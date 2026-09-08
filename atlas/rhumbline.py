"""Rhumb line: the constant-heading route, longer than the great circle as latitude climbs.

A rhumb line, or loxodrome, is the path a ship sails holding one compass
heading the whole way. On a Mercator chart it is a straight line, which
is exactly why Mercator charts exist: a navigator draws a straight line
between two ports, reads the angle, and holds it. The price of that
convenience is distance. The shortest path between two points on a
sphere is the great circle, whose heading changes continuously, and the
rhumb line, by refusing to change heading, spirals gently toward the
pole and takes longer. The rhumb distance comes from the projected
latitude, the log of the tangent of half the co-latitude that Mercator
uses, which stretches latitude so a constant bearing becomes a
straight line; the bearing is the arctangent of the longitude change
over the change in that projected latitude, and the distance is the
latitude change over the cosine of that bearing, with the east-west
special case handled by the cosine of the mean latitude when the
latitudes are equal. Two measurements define the relationship and are
worth having as numbers. Along the equator or along any meridian the
rhumb line and the great circle coincide, since those are both
constant-heading and shortest, so their distances match exactly. Away
from them the rhumb line is always longer, and the excess grows with
latitude and with east-west extent: an east-west leg at latitude sixty
spanning ninety degrees of longitude runs about nine percent longer on
the rhumb line than the great circle, a gap that a ship crossing an
ocean would feel as a day. The finding worth stating is that the rhumb
line is never shorter than the great circle, equal only along the
equator and meridians, and the excess rises with latitude, which is the
navigator's trade of distance for a heading that never changes. This
module computes rhumb distance and bearing, and a survey measures the
rhumb-to-great-circle excess across latitudes.
"""

from __future__ import annotations

import math

from atlas.errors import Outside

EARTH_RADIUS_KM = 6371.0088


def _check(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")


def _projected(phi: float) -> float:
    # Mercator's stretched latitude, where a constant bearing becomes a straight line
    return math.log(math.tan(math.pi / 4 + phi / 2))


def rhumb_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _check(lat1, lon1)
    _check(lat2, lon2)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    if abs(dlam) > math.pi:
        dlam = dlam - math.copysign(2 * math.pi, dlam)  # take the short way round
    dpsi = _projected(phi2) - _projected(phi1)
    # q is dphi/dpsi, which tends to cos(phi) on an east-west leg
    q = dphi / dpsi if abs(dpsi) > 1e-12 else math.cos(phi1)
    return math.sqrt(dphi * dphi + q * q * dlam * dlam) * EARTH_RADIUS_KM


def rhumb_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _check(lat1, lon1)
    _check(lat2, lon2)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    if abs(dlam) > math.pi:
        dlam = dlam - math.copysign(2 * math.pi, dlam)
    dpsi = _projected(phi2) - _projected(phi1)
    return (math.degrees(math.atan2(dlam, dpsi)) + 360.0) % 360.0
