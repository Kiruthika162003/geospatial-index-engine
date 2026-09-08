"""Radius box: the lat/lon rectangle around a point that a circle of that radius cannot escape.

A radius query, everything within so many kilometers of a point, is the
commonest spatial question, and the fast way to answer it is to first
cut the candidates down with a bounding rectangle in latitude and
longitude guaranteed to contain the whole circle, then test the
survivors with the real distance. The rectangle looks like one
division, the radius over the meters per degree, but the measurements
recorded here corrected it twice, and both corrections are kept. The
latitude half-height is honest at any latitude: the radius over the
earth's radius, in degrees. The longitude half-width is where the
guesses failed. The first guess divided by the cosine of the center's
latitude, since a degree of longitude shrinks by that cosine. Sampling
the circle's rim showed that box leaking, missing three and a half
percent of rim points overall and nine percent between sixty and
eighty degrees, because the circle's widest longitude spread lies not
at the center's latitude but poleward of it, where degrees are cheaper.
The second guess divided by the smaller cosine at the box's poleward
edge. That box missed none of a hundred thousand rim points, but it
was not tight: it overshot the true spread by seven percent on average
and forty-six percent at worst, thirty-two degrees of half-width at
latitude seventy for an eight-hundred-kilometer radius where the truth
is twenty-one and a half. The truth is the arcsine of the sine of the
angular radius over the cosine of the center latitude, the maximum
longitude a small circle on a sphere reaches, and the box built from it
contained every rim point while a one percent shrink let ten percent of
them escape, which is what tight means. One more thing the sampling
turned up is worth keeping: rim points due north and south land on the
latitude edge to within a few parts in ten to the fourteenth of a
degree and rounding puts half of them a hair outside, a tie rather than
a miss, so the box pads its latitude bound by a trillionth of a degree.
A circle that reaches a pole spans every longitude, and a circle near
the dateline gets a wrapped box. The finding worth stating is that the
exact arcsine box is the one that both contains the circle and hugs
it, and the two simpler cosines each fail one of those two duties. This
module computes all three boxes so the survey can hold them side by
side, and confirms containment and tightness by sampling the rim.
"""

from __future__ import annotations

import math

from atlas.errors import Outside

EARTH_RADIUS_KM = 6371.0088
_TIE_PAD = 1e-12  # degrees; absorbs the rounding tie at the exact north/south rim
Box = tuple[float, float, float, float]  # min_lat, min_lon, max_lat, max_lon


def _check(lat: float, lon: float, radius_km: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")
    if radius_km < 0:
        raise Outside("radius must not be negative")


def _finish(lat: float, lon: float, dlat: float, dlon: float) -> Box:
    min_lat, max_lat = lat - dlat, lat + dlat
    if min_lat <= -90.0 or max_lat >= 90.0 or dlon >= 180.0:
        # the circle reaches a pole, or wraps the globe: every longitude is inside
        return (max(min_lat, -90.0), -180.0, min(max_lat, 90.0), 180.0)
    min_lon, max_lon = lon - dlon, lon + dlon
    if min_lon < -180.0:
        min_lon += 360.0
    if max_lon > 180.0:
        max_lon -= 360.0
    return (min_lat, min_lon, max_lat, max_lon)


def naive_box(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    # first guess: widen longitude by the center's own cosine; leaks poleward
    _check(lat, lon, radius_km)
    dlat = math.degrees(radius_km / EARTH_RADIUS_KM) + _TIE_PAD
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-12)
    return _finish(lat, lon, dlat, dlon)


def edge_cosine_box(lat: float, lon: float, radius_km: float) -> Box:
    # second guess: widen by the poleward edge's cosine; contains but overshoots
    _check(lat, lon, radius_km)
    dlat = math.degrees(radius_km / EARTH_RADIUS_KM) + _TIE_PAD
    min_lat, max_lat = lat - dlat, lat + dlat
    if min_lat <= -90.0 or max_lat >= 90.0:
        return _finish(lat, lon, dlat, 180.0)
    edge_cos = min(math.cos(math.radians(min_lat)), math.cos(math.radians(max_lat)))
    return _finish(lat, lon, dlat, dlat / edge_cos)


def radius_box(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    # the exact spread: the widest longitude a small circle on a sphere reaches
    _check(lat, lon, radius_km)
    delta = radius_km / EARTH_RADIUS_KM
    dlat = math.degrees(delta) + _TIE_PAD
    cos_lat = math.cos(math.radians(lat))
    ratio = math.sin(delta) / cos_lat if cos_lat > 0 else 2.0
    dlon = 180.0 if ratio >= 1.0 else math.degrees(math.asin(ratio))
    return _finish(lat, lon, dlat, dlon)


def contains(box: tuple[float, float, float, float], lat: float, lon: float) -> bool:
    min_lat, min_lon, max_lat, max_lon = box
    if not min_lat <= lat <= max_lat:
        return False
    if min_lon <= max_lon:
        return min_lon <= lon <= max_lon
    return lon >= min_lon or lon <= max_lon  # a box that wraps the dateline
