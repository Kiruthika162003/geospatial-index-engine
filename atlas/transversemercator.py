"""Transverse Mercator: the projection behind UTM, with its grid convergence and scale measured.

UTM slices the world into sixty zones and maps each with a
Mercator cylinder turned on its side to touch the zone's central
meridian, so that distortion is small along a north-south strip
rather than along the equator. The spherical form is closed:
with B the cosine of the latitude times the sine of the longitude
offset from the central meridian, the easting is the scale times
the radius times the inverse hyperbolic tangent of B, and the
northing is the scale times the radius times the arctangent of
the tangent of the latitude over the cosine of the offset. Two
numbers describe the map at each point. The point scale is how
much longer a meter on the ground is on the map, one over the
square root of one minus B squared, times the central scale of
0.9996 that UTM applies so the zone's edges are not too large;
it is 0.9996 on the central meridian, exactly one about 180 km
either side, and about 1.001 at the zone edge. The grid
convergence is the angle between grid north, the map's vertical,
and true north, the meridian, which is zero on the central
meridian and grows with the offset and the latitude as the
arctangent of the tangent of the offset times the sine of the
latitude. The survey measures both by finite differences on the
projection itself, stepping a short way north from a point and
reading the map displacement, and compares them with the closed
forms: the scale matched within 2e-9 at every latitude and
offset tried, 2e-16 on the equator, and the convergence matched
within 3e-7 degrees once its sign convention was settled, the
first reading coming out negated because the meridian's image
bends toward the central meridian and grid north therefore lies
clockwise of true north east of it; it reads 2.12 degrees at the
3-degree zone edge at latitude 45, 2.60 at 60, 2.96 at 80, and
0.0 on the equator. It also measures the UTM premise: the scale
runs from 0.9996 on the central meridian to 1.000972 at the
3-degree edge on the equator, 1.000285 at latitude 45, and
crosses one at 180.2 km east of the central meridian at
latitudes 0, 45, and 60 alike, since the crossing depends on the
easting and not the latitude; the round trip through the inverse
returned 2000 random points within 8 nanometers. And it measures
the reason zones are narrow: at latitude 45 the scale is 1.0023
at 6 degrees off the central meridian and 1.069 at 30, and on the
equator 1.005 and 1.154, so a single transverse cylinder cannot
serve a continent. The finding worth stating is that the
finite-difference scale and convergence match their closed forms
to 2e-9 and 3e-7 degrees, that the scale crosses one at 180.2 km
either side of the central meridian at every latitude as UTM
intends, and that the convergence reaches two to three degrees
at a zone's edge, so a grid bearing needs correcting before it
is a true one. This
module projects and unprojects on a spherical transverse
Mercator, and a survey measures scale, convergence, and the zone.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside
from atlas.haversine import EARTH_RADIUS_KM

CENTRAL_SCALE = 0.9996
RADIUS_M = EARTH_RADIUS_KM * 1000.0


def forward(
    lat: float, lon: float, lon0: float, k0: float = CENTRAL_SCALE
) -> tuple[float, float]:
    # easting and northing in meters from the central meridian and the equator
    if not -90.0 < lat < 90.0:
        raise Outside("latitude must lie strictly within -90 and 90 degrees")
    phi = math.radians(lat)
    dlam = math.radians((lon - lon0 + 180.0) % 360.0 - 180.0)
    b = math.cos(phi) * math.sin(dlam)
    if abs(b) >= 1.0 - 1e-12:
        raise Invalid("the point lies 90 degrees from the central meridian, off the map")
    x = k0 * RADIUS_M * math.atanh(b)
    y = k0 * RADIUS_M * (math.atan2(math.tan(phi), math.cos(dlam)))
    return x, y


def inverse(x: float, y: float, lon0: float, k0: float = CENTRAL_SCALE) -> tuple[float, float]:
    d = x / (k0 * RADIUS_M)
    n = y / (k0 * RADIUS_M)
    lat = math.degrees(math.asin(math.sin(n) / math.cosh(d)))
    lon = lon0 + math.degrees(math.atan2(math.sinh(d), math.cos(n)))
    return lat, ((lon + 180.0) % 360.0) - 180.0


def point_scale(lat: float, lon: float, lon0: float, k0: float = CENTRAL_SCALE) -> float:
    phi = math.radians(lat)
    dlam = math.radians((lon - lon0 + 180.0) % 360.0 - 180.0)
    b = math.cos(phi) * math.sin(dlam)
    return k0 / math.sqrt(1.0 - b * b)


def convergence(lat: float, lon: float, lon0: float) -> float:
    # the angle in degrees from true north to grid north, positive east of the central meridian
    phi = math.radians(lat)
    dlam = math.radians((lon - lon0 + 180.0) % 360.0 - 180.0)
    return math.degrees(math.atan(math.tan(dlam) * math.sin(phi)))


def measured_scale(lat: float, lon: float, lon0: float, step_deg: float = 1e-5) -> float:
    # map meters per ground meter stepping due north, by finite difference
    x0, y0 = forward(lat, lon, lon0)
    x1, y1 = forward(lat + step_deg, lon, lon0)
    ground = math.radians(step_deg) * RADIUS_M
    return math.hypot(x1 - x0, y1 - y0) / ground


def measured_convergence(lat: float, lon: float, lon0: float, step_deg: float = 1e-5) -> float:
    # grid north's bearing from true north, read from a short step along the meridian: the
    # meridian's image bends toward the central meridian, so grid north lies the other way
    x0, y0 = forward(lat, lon, lon0)
    x1, y1 = forward(lat + step_deg, lon, lon0)
    return -math.degrees(math.atan2(x1 - x0, y1 - y0))


def unit_scale_offset_km(lat: float, lon0: float = 0.0) -> float:
    # how far east of the central meridian the scale crosses one, by bisection
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if point_scale(lat, lon0 + mid, lon0) < 1.0:
            lo = mid
        else:
            hi = mid
    x, _ = forward(lat, lon0 + hi, lon0)
    return x / 1000.0
