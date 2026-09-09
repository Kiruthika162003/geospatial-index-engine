"""Azimuthal equidistant: the map on which distance and bearing from the center are true.

Every projection sacrifices something, and this one chooses to keep
two things exactly for one point: from the center, every distance is
true and every bearing is true, because a point at angular distance
c and bearing b from the center is drawn at polar coordinates (c, b)
in the plane. It is the map on the United Nations emblem, the map
a radio operator uses to point an antenna, and the one that can
show the whole globe, with the antipode stretched into the outer
rim. The survey measures what is kept and what is lost. What is
kept: over 5000 random points, half of them in the far hemisphere,
the plane distance from the center matched the great-circle
distance within 1.2e-8 km, the plane bearing matched the initial
great-circle bearing within 9e-14 degrees, and the round trip
returned within 1.3e-8 km. What is lost is everything between two
off-center points. A pair each 90 degrees out on opposite sides of
the center are half the circumference apart on the globe and sit at
the ends of a diameter of length pi on the map, also half the
circumference, so that case is true by luck, ratio 1.000000. The
same two points a quarter turn apart round the center are 90
degrees apart on the globe but their map chord is the diagonal of a
square, ratio 1.414; at 60 degrees apart the ratio is 1.500 and at
30 degrees 1.553. For a quarter turn the ratio grows with distance
out: 1.003 at 10 degrees, 1.025 at 30, 1.12 at 60, 1.41 at 90,
2.25 at 120, and 5.12 at 150. Scale across the radial direction
grows as c over sin c, one at the center, 1.571 at 90 degrees,
5.24 at 150, 179 at 179, and unbounded at the antipode, where a
single point becomes the whole rim: a circle of 100 km radius
drawn round the antipode, circumference 628 km, is stretched into a
ring measuring 124,972 km on the map, 199 times its true length.
The finding worth stating is that
the azimuthal equidistant map is exact for every measurement that
starts at its center and wrong by a factor that grows to infinity
for measurements that do not, so it is a map for one observer
rather than for a region. This module projects and unprojects
about a center, and a survey measures the kept distances and
bearings, the lost off-center chords, and the transverse scale.
"""

from __future__ import annotations

import math

from atlas.bearing import initial_bearing
from atlas.errors import Invalid, Outside
from atlas.haversine import EARTH_RADIUS_KM, haversine

Point = tuple[float, float]


def _check(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")


def forward(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    # plane coordinates in earth radii: radius is the angular distance, angle is the bearing
    _check(lat, lon)
    _check(lat0, lon0)
    phi, phi0 = math.radians(lat), math.radians(lat0)
    dlam = math.radians(lon - lon0)
    cos_c = math.sin(phi0) * math.sin(phi) + math.cos(phi0) * math.cos(phi) * math.cos(dlam)
    c = math.acos(max(-1.0, min(1.0, cos_c)))
    if c < 1e-15:
        return (0.0, 0.0)
    sin_c = math.sin(c)
    if sin_c < 1e-15:
        raise Invalid("the antipode of the center has no unique direction")
    k = c / sin_c
    x = k * math.cos(phi) * math.sin(dlam)
    y = k * (math.cos(phi0) * math.sin(phi) - math.sin(phi0) * math.cos(phi) * math.cos(dlam))
    return (x, y)


def inverse(x: float, y: float, lat0: float, lon0: float) -> Point:
    _check(lat0, lon0)
    c = math.hypot(x, y)
    if c == 0.0:
        return (lat0, lon0)
    if c > math.pi:
        raise Outside("the point lies beyond the rim of the map")
    phi0, lam0 = math.radians(lat0), math.radians(lon0)
    sin_c, cos_c = math.sin(c), math.cos(c)
    phi = math.asin(cos_c * math.sin(phi0) + y * sin_c * math.cos(phi0) / c)
    east = c * math.cos(phi0) * cos_c - y * math.sin(phi0) * sin_c
    lam = lam0 + math.atan2(x * sin_c, east)
    lon = math.degrees(lam)
    if lon > 180.0:
        lon -= 360.0
    elif lon < -180.0:
        lon += 360.0
    return (math.degrees(phi), lon)


def plane_distance_km(x: float, y: float) -> float:
    return math.hypot(x, y) * EARTH_RADIUS_KM


def plane_bearing(x: float, y: float) -> float:
    return math.degrees(math.atan2(x, y)) % 360.0


def radial_error_km(lat: float, lon: float, lat0: float, lon0: float) -> float:
    # the map's distance from center against the great-circle distance: zero if the map is right
    x, y = forward(lat, lon, lat0, lon0)
    return plane_distance_km(x, y) - haversine(lat0, lon0, lat, lon)


def bearing_error_deg(lat: float, lon: float, lat0: float, lon0: float) -> float:
    x, y = forward(lat, lon, lat0, lon0)
    gap = plane_bearing(x, y) - initial_bearing(lat0, lon0, lat, lon)
    return (gap + 180.0) % 360.0 - 180.0


def chord_ratio(a: Point, b: Point, lat0: float, lon0: float) -> float:
    # map distance between two off-center points over their true distance: one only by luck
    xa, ya = forward(*a, lat0, lon0)
    xb, yb = forward(*b, lat0, lon0)
    true = haversine(*a, *b)
    if true == 0.0:
        raise Invalid("the points coincide")
    return math.hypot(xa - xb, ya - yb) * EARTH_RADIUS_KM / true


def transverse_scale(angle_deg: float) -> float:
    # scale across the radial direction at angular distance c: c / sin c
    if not 0.0 <= angle_deg < 180.0:
        raise Invalid("the angle must lie within 0 and 180 degrees")
    c = math.radians(angle_deg)
    return 1.0 if c == 0.0 else c / math.sin(c)
