"""Bearing and destination: great-circle navigation, where the heading changes en route.

Navigating on a sphere is not like navigating on a flat map. On a
great-circle route, the shortest path between two points, the compass
heading is not constant: it changes continuously along the way, so the
bearing you set off on, the initial bearing, differs from the bearing
you arrive on, the final bearing. This is why a great-circle route
drawn on a flat Mercator map looks curved, bowing toward the nearer
pole, while the constant-heading route, the rhumb line, is a straight
diagonal but a longer path. The initial bearing from one point to
another comes from a spherical-trigonometry formula in the two
latitudes and the longitude difference, giving the angle clockwise
from north to aim at first. The inverse operation, the destination
point, takes a start, an initial bearing, and a distance along the
great circle, and returns where you land, which is what you need to
draw the curved route or to place a point a fixed distance away in a
given direction. The two operations are consistent: travel a distance
on a bearing to reach a destination, and the great-circle distance
back from that destination to the start equals the distance you
travelled, and the bearing back is the reverse of the arrival bearing,
not of the departure bearing, precisely because the heading turned
along the way. The finding worth stating is that the initial and final
bearings of a great-circle route genuinely differ, by an amount that
grows with the route's length and latitude, which is the measurable
signature of the sphere's curvature. This module computes initial
bearing, final bearing, and destination, and a survey measures the
initial-to-final bearing gap and confirms the distance round-trips.
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


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _check(lat1, lon1)
    _check(lat2, lon2)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def final_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # the final bearing is the reverse of the initial bearing of the reverse route
    back = initial_bearing(lat2, lon2, lat1, lon1)
    return (back + 180.0) % 360.0


def destination(
    lat: float, lon: float, bearing: float, distance: float, radius: float = EARTH_RADIUS_KM
) -> tuple[float, float]:
    _check(lat, lon)
    if distance < 0:
        raise Outside("distance must not be negative")
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    theta = math.radians(bearing)
    delta = distance / radius
    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    lat2 = math.degrees(phi2)
    lon2 = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return (lat2, lon2)
