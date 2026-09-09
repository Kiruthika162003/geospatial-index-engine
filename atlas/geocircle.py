"""Geodesic circle: a lat/lon circle as a polygon, short of pi r squared as it grows.

A circle of a given radius around a point on the earth is not a
circle in latitude and longitude, since degrees of longitude shrink
toward the poles, and it is not a flat disk either, since it lies on
a sphere. The honest construction places its rim by the destination
formula, one point per bearing at the given great-circle distance, so
every rim point is exactly the radius from the center along the
surface, and the polygon those points form is the geodesic circle.
Its area is the quantity that reveals the sphere. On a flat plane a
circle of radius r encloses pi r squared; on a sphere the surface
inside a great-circle radius r is the spherical cap, whose area is two
pi R squared times one minus the cosine of r over R, which is less
than pi r squared and falls further below it as r grows, because the
surface curves away under the circle. For a small circle the two are
indistinguishable, a kilometer radius giving a ratio to pi r squared
within a hundred-millionth of one. The larger radii were guessed
before they were measured, and the guesses were wrong in a way worth
keeping: the docstring first said a thousand kilometers holds about
ninety-eight percent of the flat disk and a hemisphere about sixty-one
percent. Measured, a thousand kilometers holds 99.79 percent, three
thousand 98.2, five thousand 95.0, and the hemisphere, at the quarter
circumference of ten thousand kilometers, holds 81.06 percent, which
is exactly eight over pi squared, the cap's two pi R squared over the
flat pi times the square of pi R over two. The sixty-one was bad
arithmetic and the measurement caught it. The polygon's area, measured
on the sphere, tracks the exact cap formula from below as the rim
resolution rises, 82.7 percent of the cap at six rim points, 95.5 at
twelve, 99.87 at seventy-two, and 99.995 at three hundred and sixty,
since an inscribed polygon lies inside its circle. The finding worth
stating is that a geodesic circle's area is the cap formula, equal to
pi r squared only in the small-radius limit and falling to eight over
pi squared of it at a hemisphere, so treating a large radius query as
a flat disk overstates its ground by up to nineteen percent. This
module builds the rim and computes cap and polygon areas, and a survey
measures the ratio to pi r squared across radii.
"""

from __future__ import annotations

import math

from atlas.bearing import destination
from atlas.errors import Invalid, Outside
from atlas.haversine import EARTH_RADIUS_KM, haversine
from atlas.sphericalarea import band_area_km2


def rim(
    lat: float, lon: float, radius_km: float, points: int = 72
) -> list[tuple[float, float]]:
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise Outside("center must lie on the globe")
    if radius_km <= 0:
        raise Invalid("radius must be positive")
    if points < 3:
        raise Invalid("a rim needs at least three points")
    return [destination(lat, lon, 360.0 * i / points, radius_km) for i in range(points)]


def cap_area_km2(radius_km: float, sphere_radius: float = EARTH_RADIUS_KM) -> float:
    if radius_km <= 0:
        raise Invalid("radius must be positive")
    if radius_km > math.pi * sphere_radius:
        raise Invalid("radius exceeds half the circumference")
    return 2 * math.pi * sphere_radius**2 * (1 - math.cos(radius_km / sphere_radius))


def flat_area_km2(radius_km: float) -> float:
    return math.pi * radius_km * radius_km


def polygon_area_km2(lat: float, lon: float, radius_km: float, points: int = 72) -> float:
    return band_area_km2(rim(lat, lon, radius_km, points))


def rim_is_exact(lat: float, lon: float, radius_km: float, points: int = 72) -> float:
    # the worst deviation of any rim point from the intended radius
    points_on_rim = rim(lat, lon, radius_km, points)
    return max(abs(haversine(lat, lon, *p) - radius_km) for p in points_on_rim)
