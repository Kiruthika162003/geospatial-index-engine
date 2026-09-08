"""Haversine: great-circle distance on a sphere, stable where the cosine law is not.

The straight-line distance between two points given by latitude and
longitude is not the distance you would walk, because the earth is
curved and the shortest path between two points on a sphere follows a
great circle, an arc of the largest circle that passes through both.
The haversine formula computes that arc length from the two latitudes
and the longitude difference. Its name is the half-versed-sine, the
function that keeps it numerically well behaved: the alternative, the
spherical law of cosines, involves the cosine of the central angle,
and for two nearby points that angle is tiny and its cosine is very
close to one, so subtracting it from one loses almost all the
significant digits, and the distance comes out grainy or zero. The
haversine formula is built from sines of half the differences
instead, which stay far from the cancellation cliff for small angles,
so it is accurate from meters to the antipode. A tempting shortcut is
the equirectangular approximation, treating a small patch of the
globe as a flat rectangle and using Pythagoras on the scaled degree
differences; it is much cheaper and fine for short hops, but its
error grows with distance and with latitude, since meridians converge
toward the poles. The finding worth stating is that haversine is the
right default because it is both globally correct and locally stable,
where the cosine law is globally correct but locally unstable and the
flat approximation is locally cheap but globally wrong. This module
computes haversine distance and the flat approximation beside it, and
a survey measures how far the flat approximation drifts from
haversine as the separation grows, so the tradeoff is a number.
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


def haversine(
    lat1: float, lon1: float, lat2: float, lon2: float, radius: float = EARTH_RADIUS_KM
) -> float:
    _check(lat1, lon1)
    _check(lat2, lon2)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


def equirectangular(
    lat1: float, lon1: float, lat2: float, lon2: float, radius: float = EARTH_RADIUS_KM
) -> float:
    _check(lat1, lon1)
    _check(lat2, lon2)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = dlambda * math.cos((phi1 + phi2) / 2)
    y = math.radians(lat2 - lat1)
    return radius * math.hypot(x, y)
