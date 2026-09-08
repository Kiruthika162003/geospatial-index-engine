"""Spherical polygon area: two formulas, and the octant triangle that tells them apart.

A polygon drawn in latitude and longitude lives on a sphere, and
treating its degrees as flat coordinates gets the area wrong twice:
longitude degrees shrink toward the poles, and a big polygon bends with
the curvature. The common spherical shoelace fixes the first. For each
edge it accumulates the longitude difference times two plus the sines
of the two latitudes, and half the total times the squared radius is
the signed area. The sine of latitude is height along the sphere's
axis, so each term is a band-area increment, the cylindrical equal-area
projection's version of a trapezoid under an edge. A first guess about
its reach was recorded and then refuted by measurement, and the
correction is the most useful sentence here. The guess was that this
formula is exact on the sphere, so the octant triangle with vertices at
two equator points ninety degrees apart and the pole, whose three right
angles give a spherical excess of exactly half pi, should return one
eighth of the sphere's surface. It returns exactly one sixteenth. The
formula is exact only for edges that run along meridians and
parallels, because those are straight lines in the equal-area
projection it silently assumes; the octant's hypotenuse is a great
circle, which bows in that projection, and the formula measures the
straight chord instead, losing half the triangle. A spherical rectangle
bounded by two meridians and two parallels comes out exact to four
figures, and a tenth-of-a-degree patch agrees with the flat shoelace to
a ten-thousandth of a percent, so the band formula is right whenever
its edges are graticule-aligned or short. For a true great-circle
triangle the exact answer needs the spherical excess, and L'Huilier's
theorem gives it from the three side lengths alone, returning the
octant's one eighth exactly. The finding worth stating is that the band
shoelace is exact for graticule-aligned edges and small patches, and
halves the octant triangle, while spherical excess is exact for any
great-circle triangle, so the choice of formula depends on what the
edges are. This module offers both, and a survey pins the octant at one
sixteenth by the bands and one eighth by the excess.
"""

from __future__ import annotations

import math

from atlas.errors import Degenerate, Invalid, Outside

EARTH_RADIUS_KM = 6371.0088


def _check(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")


def _wrap(dlon: float) -> float:
    # take the short way round the antimeridian
    return (dlon + 180.0) % 360.0 - 180.0


def band_signed_area_km2(
    polygon: list[tuple[float, float]], radius: float = EARTH_RADIUS_KM
) -> float:
    if polygon is None or len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    for lat, lon in polygon:
        _check(lat, lon)
    total = 0.0
    n = len(polygon)
    for i in range(n):
        lat1, lon1 = polygon[i]
        lat2, lon2 = polygon[(i + 1) % n]
        dlon = math.radians(_wrap(lon2 - lon1))
        total += dlon * (2 + math.sin(math.radians(lat1)) + math.sin(math.radians(lat2)))
    return total * radius * radius / 2


def band_area_km2(polygon: list[tuple[float, float]], radius: float = EARTH_RADIUS_KM) -> float:
    return abs(band_signed_area_km2(polygon, radius))


def _central_angle(a: tuple[float, float], b: tuple[float, float]) -> float:
    phi1, phi2 = math.radians(a[0]), math.radians(b[0])
    dlam = math.radians(_wrap(b[1] - a[1]))
    h = math.sin((phi2 - phi1) / 2) ** 2
    h += math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * math.asin(math.sqrt(h))


def triangle_area_km2(
    a: tuple[float, float], b: tuple[float, float], c: tuple[float, float],
    radius: float = EARTH_RADIUS_KM,
) -> float:
    # L'Huilier's theorem: spherical excess from the three great-circle sides
    for lat, lon in (a, b, c):
        _check(lat, lon)
    sa, sb, sc = _central_angle(b, c), _central_angle(a, c), _central_angle(a, b)
    s = (sa + sb + sc) / 2
    product = (
        math.tan(s / 2)
        * math.tan((s - sa) / 2)
        * math.tan((s - sb) / 2)
        * math.tan((s - sc) / 2)
    )
    if product <= 0:
        raise Degenerate("the three points do not form a spherical triangle")
    excess = 4 * math.atan(math.sqrt(product))
    return excess * radius * radius


def sphere_surface_km2(radius: float = EARTH_RADIUS_KM) -> float:
    return 4 * math.pi * radius * radius
