"""A straight Mercator line bows 110 km off the great circle over a 2345 km leg at 45 north.

A straight line between two points on a Mercator map is a rhumb
line, and the great circle bows away from it toward the pole. The
bow is read as the largest cross-track distance of any point of the
map chord from the great circle. An east-west leg spanning 10, 30
and 60 degrees of longitude reads 10.5, 96.5 and 410 km at latitude
30 over legs of 963, 2881 and 5706 km, 12.1, 110.4 and 456.6 km at
latitude 45 over 786, 2345 and 4605 km, and 10.5, 94.8 and 382 km at
latitude 60 over 555, 1654 and 3220 km; at the equator the chord is
the circle and the bow is 0.0, as it is along any meridian, where an
earlier reading that compared points at equal fractions of the two
curves showed 588 km of pure parametrisation. From New York to
London, 5628 km, the bow is 723 km. The guess that a short leg bows
by d squared sin(lat) over 8R was low by one over the cosine of the
latitude at every leg: the measured bow is 1.16 to 1.29 times it at
latitude 30, 1.42 to 1.55 at 45 and 2.0 to 2.17 at 60, so the law
carries the tangent, d squared tan(lat) over 8R, which the shortest
legs meet within 0.3 percent.

Splitting the great circle into pieces whose map chords stay
within a tolerance needs 4, 16 and 32 pieces for 1 km at spans of
10, 30 and 60 degrees at every latitude from 30 to 60, and 2, 4 and
8 pieces for 10 km; New York to London needs 32 for 1 km and 16 for
5 km.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

EARTH_RADIUS_KM = 6371.0088
Point = tuple[float, float]


def _vec(lat: float, lon: float) -> tuple[float, float, float]:
    phi, lam = math.radians(lat), math.radians(lon)
    return math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)


def _latlon(v: tuple[float, float, float]) -> Point:
    x, y, z = v
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


def great_circle_distance_km(a: Point, b: Point) -> float:
    va, vb = _vec(*a), _vec(*b)
    dot = max(-1.0, min(1.0, sum(p * q for p, q in zip(va, vb, strict=True))))
    return EARTH_RADIUS_KM * math.acos(dot)


def great_circle_point(a: Point, b: Point, fraction: float) -> Point:
    if not 0 <= fraction <= 1:
        raise Invalid("the fraction must lie in [0, 1]")
    va, vb = _vec(*a), _vec(*b)
    dot = max(-1.0, min(1.0, sum(p * q for p, q in zip(va, vb, strict=True))))
    omega = math.acos(dot)
    if omega < 1e-12:
        return a
    wa = math.sin((1 - fraction) * omega) / math.sin(omega)
    wb = math.sin(fraction * omega) / math.sin(omega)
    return _latlon(tuple(wa * p + wb * q for p, q in zip(va, vb, strict=True)))


def mercator(lat: float, lon: float) -> Point:
    if not -85.05 <= lat <= 85.05:
        raise Invalid("latitude within the Mercator limit of 85.05")
    return math.radians(lon), math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def inverse_mercator(x: float, y: float) -> Point:
    return math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2), math.degrees(x)


def chord_point(a: Point, b: Point, fraction: float) -> Point:
    # the point a fraction along the straight map line between a and b
    xa, ya = mercator(*a)
    xb, yb = mercator(*b)
    return inverse_mercator(xa + (xb - xa) * fraction, ya + (yb - ya) * fraction)


def cross_track_km(a: Point, b: Point, p: Point) -> float:
    # the distance from p to the great circle through a and b
    va, vb, vp = _vec(*a), _vec(*b), _vec(*p)
    nx = va[1] * vb[2] - va[2] * vb[1]
    ny = va[2] * vb[0] - va[0] * vb[2]
    nz = va[0] * vb[1] - va[1] * vb[0]
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-15:
        raise Invalid("the ends coincide or are antipodal")
    dot = (nx * vp[0] + ny * vp[1] + nz * vp[2]) / norm
    return EARTH_RADIUS_KM * abs(math.asin(max(-1.0, min(1.0, dot))))


def sagitta_km(a: Point, b: Point, samples: int = 200) -> float:
    # the largest distance of any point of the map chord from the great circle
    if samples < 2:
        raise Invalid("at least two samples are needed")
    worst = 0.0
    for k in range(1, samples):
        worst = max(worst, cross_track_km(a, b, chord_point(a, b, k / samples)))
    return worst


def segments_for_tolerance(a: Point, b: Point, tolerance_km: float, cap: int = 4096) -> int:
    # the number of great-circle pieces whose map chords stay within the tolerance
    if tolerance_km <= 0:
        raise Invalid("the tolerance must be positive")
    n = 1
    while n <= cap:
        worst = 0.0
        for k in range(n):
            p, q = great_circle_point(a, b, k / n), great_circle_point(a, b, (k + 1) / n)
            worst = max(worst, sagitta_km(p, q, 20))
        if worst <= tolerance_km:
            return n
        n *= 2
    return cap


def sine_guess(distance_km: float, lat: float) -> float:
    # the guess before measuring, d squared sin(lat) over 8R; it reads low by 1 over cos(lat)
    return distance_km**2 * abs(math.sin(math.radians(lat))) / (8 * EARTH_RADIUS_KM)


def tangent_law(distance_km: float, lat: float) -> float:
    # the bow of a short east-west map chord off its great circle: d squared tan(lat) over 8R
    return distance_km**2 * abs(math.tan(math.radians(lat))) / (8 * EARTH_RADIUS_KM)


def east_west_leg(lat: float, span_deg: float) -> tuple[Point, Point]:
    return (lat, -span_deg / 2), (lat, span_deg / 2)
