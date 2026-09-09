"""Gnomonic projection: the map on which every great circle is a straight line.

Project the sphere from its center onto a plane tangent at a chosen
point and every great circle becomes a straight line, because a
great circle is the sphere cut by a plane through the center, and
that plane meets the tangent plane in a line. No other projection
has the property, and it is why the gnomonic is the navigator's map
for plotting a geodesic: draw the straight line, read off the
waypoints, and transfer them to the working chart. The price is
distortion that grows without bound toward the horizon. A point at
angular distance c from the tangent point lands at radius tan c, so
scale along the radius grows as one over cos squared c, four times at
sixty degrees, and the far hemisphere cannot be shown at all since
its rays never reach the plane. The survey measures the straightness
directly: sampling a long geodesic and projecting the samples, the
largest perpendicular distance of any sample from the straight line
through the projected endpoints is at floating precision, while the
same test on a line straight in latitude and longitude, the map line
that is not a geodesic, shows a departure of kilometers, so the
projection distinguishes routes from map lines. It also measures the
round trip, forward then inverse landing on the starting point to
floating precision across the visible hemisphere, and the radial
scale at sixty and eighty degrees, four and thirty-three. The finding
worth stating is that gnomonic straightness holds for every geodesic
tried to a millimeter, which makes the projection a test of
geodesic-ness as much as a map, while its scale doubles by forty-five
degrees and explodes past eighty, so it is a map for one region at a
time. This module projects and unprojects about a tangent point, and
a survey measures straightness, round trip, and scale.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside
from atlas.haversine import EARTH_RADIUS_KM
from atlas.interpolate import sample

Point = tuple[float, float]


def _check(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")


def forward(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    # plane coordinates in earth radii; the far hemisphere and the horizon are refused
    _check(lat, lon)
    _check(lat0, lon0)
    phi, phi0 = math.radians(lat), math.radians(lat0)
    dlam = math.radians(lon - lon0)
    cos_c = math.sin(phi0) * math.sin(phi) + math.cos(phi0) * math.cos(phi) * math.cos(dlam)
    if cos_c <= 1e-12:
        raise Outside("the point lies on or beyond the horizon of the tangent point")
    x = math.cos(phi) * math.sin(dlam) / cos_c
    north = math.cos(phi0) * math.sin(phi) - math.sin(phi0) * math.cos(phi) * math.cos(dlam)
    return (x, north / cos_c)


def inverse(x: float, y: float, lat0: float, lon0: float) -> Point:
    _check(lat0, lon0)
    rho = math.hypot(x, y)
    if rho == 0.0:
        return (lat0, lon0)
    c = math.atan(rho)
    phi0, lam0 = math.radians(lat0), math.radians(lon0)
    sin_c, cos_c = math.sin(c), math.cos(c)
    phi = math.asin(cos_c * math.sin(phi0) + y * sin_c * math.cos(phi0) / rho)
    east = rho * math.cos(phi0) * cos_c - y * math.sin(phi0) * sin_c
    lam = lam0 + math.atan2(x * sin_c, east)
    lon = math.degrees(lam)
    if lon > 180.0:
        lon -= 360.0
    elif lon < -180.0:
        lon += 360.0
    return (math.degrees(phi), lon)


def angular_distance_deg(lat: float, lon: float, lat0: float, lon0: float) -> float:
    phi, phi0 = math.radians(lat), math.radians(lat0)
    dlam = math.radians(lon - lon0)
    cos_c = math.sin(phi0) * math.sin(phi) + math.cos(phi0) * math.cos(phi) * math.cos(dlam)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_c))))


def radial_scale(angle_deg: float) -> float:
    # scale along the radius at angular distance c from the tangent point: 1 / cos^2 c
    if not 0.0 <= angle_deg < 90.0:
        raise Invalid("the angle must lie within 0 and 90 degrees")
    return 1.0 / math.cos(math.radians(angle_deg)) ** 2


def _deviation_km(points: list[tuple[float, float]]) -> float:
    # the largest perpendicular distance of the points from the line through the first
    # and last, in kilometers of plane distance at one earth radius per unit
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        raise Invalid("the endpoints coincide")
    worst = 0.0
    for x, y in points[1:-1]:
        worst = max(worst, abs(dx * (y - y1) - dy * (x - x1)) / norm)
    return worst * EARTH_RADIUS_KM


def geodesic_deviation_km(a: Point, b: Point, lat0: float, lon0: float, samples: int = 64):
    # how far the projected geodesic strays from straight: at precision if the map is right
    pts = [forward(*p, lat0, lon0) for p in sample(*a, *b, samples)]
    return _deviation_km(pts)


def map_line_deviation_km(a: Point, b: Point, lat0: float, lon0: float, samples: int = 64):
    # the same test on the line straight in latitude and longitude, which is not a geodesic
    pts = []
    for i in range(samples):
        t = i / (samples - 1)
        lat = a[0] + t * (b[0] - a[0])
        lon = a[1] + t * (b[1] - a[1])
        pts.append(forward(lat, lon, lat0, lon0))
    return _deviation_km(pts)
