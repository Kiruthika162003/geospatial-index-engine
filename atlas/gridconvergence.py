"""Grid north leans off true north by atan(tan dlon sin lat), 2.12 degrees at a zone's edge.

A transverse Mercator grid's north is the central meridian's, and
away from it true north and grid north part by the convergence,
atan(tan of the longitude offset times the sine of the latitude).
Measured as the grid bearing of a step due north on a spherical
projection, the convergence reads 0.25000, 0.50004 and 1.50103
degrees at latitude 30 for offsets of 0.5, 1 and 3 degrees,
0.35356, 0.70714 and 2.12229 at 45, 0.43302, 0.86605 and 2.59867 at
60 and 0.4924, 0.98481 and 2.9545 at 80, the law to five places,
and zero at the equator; west of the meridian the sign flips. The
guess that the small-angle rule, offset times sine of latitude,
serves across a zone was right to 0.05 percent at 3 degrees, 2.1213
against 2.1223 at latitude 45, and 0.5 percent at 10 degrees,
7.0711 against 7.1071. Walking 10 km along grid north instead of
true north at a zone's 3 degree edge misses by 0 m at the equator,
262 m at 30, 370 m at 45, 453 m at 60 and 515 m at 80.

The scale along the same step reads 0.9996 on the meridian and at
3 degrees off 1.000972 at the equator, 1.000628 at 30, 1.000285 at
45, 0.999942 at 60 and 0.999641 at 80, the law k0 over root(1 - (cos
lat sin dlon) squared) to six places, so the zone edge's scale
error crosses zero near latitude 60 and the zone is nowhere more
than a thousandth off.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside


def forward(
    lat: float, lon: float, lon0: float, k0: float = 0.9996, radius: float = 6371008.8
) -> tuple[float, float]:
    # spherical transverse Mercator about the central meridian lon0
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise Outside("latitude within 90 and longitude within 180")
    phi = math.radians(lat)
    dlam = math.radians(((lon - lon0 + 180) % 360) - 180)
    b = math.cos(phi) * math.sin(dlam)
    if abs(b) >= 1 - 1e-12:
        raise Invalid("the point sits 90 degrees from the central meridian")
    x = 0.5 * k0 * radius * math.log((1 + b) / (1 - b))
    y = k0 * radius * math.atan2(math.tan(phi), math.cos(dlam))
    return x, y


def convergence_law(lat: float, lon: float, lon0: float) -> float:
    # the angle from true north to grid north, in degrees, positive east of the meridian
    phi = math.radians(lat)
    dlam = math.radians(((lon - lon0 + 180) % 360) - 180)
    return math.degrees(math.atan(math.tan(dlam) * math.sin(phi)))


def small_angle_law(lat: float, lon: float, lon0: float) -> float:
    return (((lon - lon0 + 180) % 360) - 180) * math.sin(math.radians(lat))


def convergence_measured(lat: float, lon: float, lon0: float, step: float = 1e-5) -> float:
    # true north's image on the grid leans west of grid north east of the meridian, so the
    # convergence, grid north measured from true north, is the negative of that bearing
    x0, y0 = forward(lat, lon, lon0)
    x1, y1 = forward(lat + step, lon, lon0)
    return -math.degrees(math.atan2(x1 - x0, y1 - y0))


def scale_measured(
    lat: float, lon: float, lon0: float, step: float = 1e-5, radius: float = 6371008.8
) -> float:
    x0, y0 = forward(lat, lon, lon0, radius=radius)
    x1, y1 = forward(lat + step, lon, lon0, radius=radius)
    ground = math.radians(step) * radius
    return math.hypot(x1 - x0, y1 - y0) / ground


def scale_law(lat: float, lon: float, lon0: float, k0: float = 0.9996) -> float:
    phi = math.radians(lat)
    dlam = math.radians(((lon - lon0 + 180) % 360) - 180)
    b = math.cos(phi) * math.sin(dlam)
    return k0 / math.sqrt(1 - b * b)


def grid_bearing_error(lat: float, lon: float, lon0: float) -> float:
    # walking a true bearing by the grid's north misses by the convergence, in degrees
    return convergence_law(lat, lon, lon0)


def zone_extremes(lat: float, half_width: float = 3.0) -> tuple[float, float]:
    # the convergence at the zone's edge and the scale there, for a 6 degree UTM-like zone
    if half_width <= 0:
        raise Invalid("the half width must be positive")
    return convergence_law(lat, half_width, 0.0), scale_law(lat, half_width, 0.0)


def offset_at_distance(convergence_deg: float, distance: float) -> float:
    # the sideways miss after walking a distance along grid north instead of true north
    return distance * math.sin(math.radians(convergence_deg))
