"""Great-circle interpolation: points along the shortest path, none of them off it.

Drawing a route on a map, animating a flight, or sampling a path for a
corridor search all need intermediate points along the great circle
between two endpoints, not along the straight line in latitude and
longitude, which is a different and longer curve that bows away from
the true route. The intermediate point at a fraction f of the way is
found by slerp, spherical linear interpolation: treat the endpoints as
unit vectors from the earth's center, weight them by the sines of f and
one minus f times the total angle, over the sine of the total angle,
and the weighted sum, renormalized, points at the interpolated
position. The weights are what keep the point on the sphere and on the
arc; a plain linear blend of the vectors would cut through the earth
and, projected back out, would bunch the points toward the middle. Two
properties are worth confirming as measurements. Every interpolated
point lies exactly on the great circle through the endpoints, which is
to say its cross-track distance from the route is zero, and the
fraction is honored as arc length: the point at fraction f is f times
the total distance from the start, so the midpoint is equidistant from
both ends and the points at even fractions are evenly spaced along the
route. The contrast with naive latitude-longitude averaging is the
whole reason to slerp: the naive midpoint of two points at the same
high latitude sits at that latitude, while the true midpoint sits
poleward of it, because the great circle bows toward the pole. The
finding worth stating is that slerp's points have zero cross-track
distance and arc-proportional spacing, while the naive average drifts
off the route by an amount that grows with latitude and span. This
module interpolates along a great circle and exposes the naive average
beside it, and a survey measures the cross-track error of both.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside


def _check(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")


def _to_vector(lat: float, lon: float) -> tuple[float, float, float]:
    phi, lam = math.radians(lat), math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))


def _to_latlon(v: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = v
    return (math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x)))


def intermediate(
    lat1: float, lon1: float, lat2: float, lon2: float, fraction: float
) -> tuple[float, float]:
    _check(lat1, lon1)
    _check(lat2, lon2)
    if not 0.0 <= fraction <= 1.0:
        raise Invalid("fraction must lie within 0 and 1")
    a = _to_vector(lat1, lon1)
    b = _to_vector(lat2, lon2)
    dot = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    angle = math.acos(dot)
    if angle < 1e-12:
        return (lat1, lon1)
    if abs(angle - math.pi) < 1e-12:
        raise Invalid("antipodal endpoints have no unique great circle")
    wa = math.sin((1 - fraction) * angle) / math.sin(angle)
    wb = math.sin(fraction * angle) / math.sin(angle)
    return _to_latlon((wa * a[0] + wb * b[0], wa * a[1] + wb * b[1], wa * a[2] + wb * b[2]))


def naive_midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    # the tempting average of the coordinates, which is not on the great circle
    return ((lat1 + lat2) / 2, (lon1 + lon2) / 2)


def sample(
    lat1: float, lon1: float, lat2: float, lon2: float, count: int
) -> list[tuple[float, float]]:
    if count < 2:
        raise Invalid("need at least the two endpoints")
    return [intermediate(lat1, lon1, lat2, lon2, i / (count - 1)) for i in range(count)]
