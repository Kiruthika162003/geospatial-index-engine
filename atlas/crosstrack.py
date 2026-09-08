"""Cross-track and along-track distance: how far a point strays from a great-circle path.

A vessel or a vehicle following a great-circle route between two
waypoints wants to know, at any moment, how far it has drifted off the
route and how far along it has come. Those are the cross-track and
along-track distances. The cross-track distance is the shortest
distance from the point to the great circle through the two waypoints,
measured along the perpendicular arc, and it carries a sign: negative
for a point to the left of the route as travelled and positive to the
right, so it says not just how far off but which side, which is what a
steering correction needs. The along-track distance is how far along
the route, from the start, the foot of that perpendicular lies, the
point on the route nearest the wanderer. Both come from the same
spherical triangle: the angular distance from the start to the point
and the difference between the route's initial bearing and the bearing
from the start to the point. The cross-track angle is the arcsine of
the sine of the angular distance times the sine of the bearing
difference; the along-track angle is the arccosine of the cosine of the
angular distance over the cosine of the cross-track angle. Two facts
are worth confirming as measurements rather than trusting. A point that
lies on the route has zero cross-track distance and an along-track
distance equal to its distance from the start, and mirroring a point
across the route flips the sign of its cross-track distance without
changing its magnitude, which is the property the sign convention
promises. The finding worth stating is that cross-track distance is a
signed perpendicular offset from a great circle, zero on the route and
antisymmetric across it, so it is the exact quantity a navigator
corrects against. This module computes signed cross-track and
along-track distances, and a survey confirms the on-route zero and the
mirror antisymmetry.
"""

from __future__ import annotations

import math

from atlas.bearing import initial_bearing
from atlas.errors import Outside
from atlas.haversine import EARTH_RADIUS_KM, haversine


def cross_track_km(
    lat1: float, lon1: float, lat2: float, lon2: float, lat_p: float, lon_p: float
) -> float:
    if (lat1, lon1) == (lat2, lon2):
        raise Outside("the route needs two distinct waypoints")
    d13 = haversine(lat1, lon1, lat_p, lon_p) / EARTH_RADIUS_KM
    theta13 = math.radians(initial_bearing(lat1, lon1, lat_p, lon_p))
    theta12 = math.radians(initial_bearing(lat1, lon1, lat2, lon2))
    return math.asin(math.sin(d13) * math.sin(theta13 - theta12)) * EARTH_RADIUS_KM


def along_track_km(
    lat1: float, lon1: float, lat2: float, lon2: float, lat_p: float, lon_p: float
) -> float:
    if (lat1, lon1) == (lat2, lon2):
        raise Outside("the route needs two distinct waypoints")
    d13 = haversine(lat1, lon1, lat_p, lon_p) / EARTH_RADIUS_KM
    dxt = cross_track_km(lat1, lon1, lat2, lon2, lat_p, lon_p) / EARTH_RADIUS_KM
    cos_dxt = math.cos(dxt)
    if cos_dxt == 0:
        return 0.0
    ratio = max(-1.0, min(1.0, math.cos(d13) / cos_dxt))
    return math.acos(ratio) * EARTH_RADIUS_KM
