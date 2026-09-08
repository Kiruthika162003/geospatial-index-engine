"""Web Mercator: the projection that squares the map and inflates the poles.

Web Mercator is the projection nearly every online map uses, because it
turns the round earth into a square that tiles cleanly and preserves
angles, so a small shape keeps its true form and north stays up
everywhere. It projects longitude linearly to the horizontal axis and
latitude through a logarithmic function of the tangent to the vertical
axis, mapping the world into a unit square. Preserving angles comes at
a famous cost: it does not preserve area, and the distortion grows
without bound toward the poles. The vertical stretch at a given
latitude is the secant of that latitude, and because the horizontal
scale stretches by the same secant to keep angles true, area is
inflated by the secant squared. At forty-five degrees that is a
doubling of area; at sixty degrees a factor of four; near the poles it
runs to infinity, which is why Greenland looks the size of Africa on a
Mercator map though it is a fourteenth the area. The infinity at the
poles is not merely large but undefined, so Web Mercator cannot
represent them at all and is clamped to about eighty-five degrees of
latitude, the value that makes the projected world exactly square. The
finding worth stating, and the one the survey measures, is that the
area inflation is precisely the secant squared of the latitude, a
predictable law rather than a vague distortion, so a region's Mercator
area can be corrected back to its true area by dividing by that factor.
This module projects to and from Web Mercator in the unit square, and a
survey measures the area inflation against the secant-squared law and
confirms the projection round-trips.
"""

from __future__ import annotations

import math

from atlas.errors import Outside

# the latitude where the projected world becomes exactly square
MAX_LATITUDE = 85.05112877980659


def forward(lat: float, lon: float) -> tuple[float, float]:
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    if not -MAX_LATITUDE <= lat <= MAX_LATITUDE:
        raise Outside("latitude must lie within the Web Mercator limit of ~85.051 degrees")
    x = (lon + 180.0) / 360.0
    phi = math.radians(lat)
    y = (1 - math.log(math.tan(phi) + 1 / math.cos(phi)) / math.pi) / 2
    return (x, y)


def inverse(x: float, y: float) -> tuple[float, float]:
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        raise Outside("projected coordinates must lie within the unit square")
    lon = x * 360.0 - 180.0
    n = math.pi * (1 - 2 * y)
    lat = math.degrees(math.atan(math.sinh(n)))
    return (lat, lon)


def area_inflation(lat: float) -> float:
    # the factor by which Web Mercator magnifies area at a latitude: sec^2(lat)
    if not -MAX_LATITUDE <= lat <= MAX_LATITUDE:
        raise Outside("latitude must lie within the Web Mercator limit")
    return 1.0 / math.cos(math.radians(lat)) ** 2
