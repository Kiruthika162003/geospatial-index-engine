"""Geographic centroid: the mean of points on a sphere, taken through unit vectors.

Averaging latitudes and longitudes is the obvious way to find the
middle of a set of places, and it is wrong in two ways that a survey
can measure. Across the dateline it is wrong catastrophically: two
points at longitude 179 and -179 are a few hundred kilometers apart
in the Pacific, and their averaged longitude is 0, in the Atlantic,
on the far side of the globe. Away from the dateline it is wrong
quietly, because degrees of longitude shrink toward the poles, so a
flat average of coordinates weights high-latitude points as if their
longitude spans were as wide as at the equator. The proper mean
converts each point to a unit vector on the sphere, averages the
vectors, and converts the resulting vector back to a latitude and
longitude. That mean is the point on the sphere that minimizes the
summed squared chord distance to the inputs, it is unaffected by
where the dateline happens to sit, and it degrades honestly: when the
points are spread evenly round the globe the mean vector shrinks
toward the center, and its length reports how concentrated the set
is, one for a single spot, near zero for a balanced spread. The
survey measures both wrongnesses. Two points straddling the dateline
average to longitude 0 by coordinates and 180 by vectors, a
difference of half the globe; the same pair shifted to straddle the
prime meridian agree, which shows the coordinate failure is an
artifact of where the seam is drawn, not of the points. And a ring
of nineteen points at latitude 60 spread over 180 degrees of
longitude has a vector mean at latitude 70.85, poleward of every
point in the ring, because the ring bends round the pole and its
middle on the ground lies inside the bend; the coordinate mean sits
on the ring at latitude 60, 1206 km away, and costs 3.78 in summed
squared chord against the vector mean's 3.16. The minimization was
checked directly: over 300 random sets and 6000 perturbations of
the vector mean, no perturbation ever lowered the chord cost. The
finding worth stating is that the vector mean is the same function
of the points wherever the seam sits, while coordinate averaging is
a different function on each side of it, so the vector mean is the
only one of the two that is a mean of places. This module computes
the vector centroid and its concentration, and a survey measures
the dateline gap and the ring offset against coordinate averaging.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from atlas.errors import Degenerate, Invalid

Point = tuple[float, float]


def _to_vector(lat: float, lon: float) -> tuple[float, float, float]:
    phi, lam = math.radians(lat), math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))


def mean_vector(points: Iterable[Point]) -> tuple[float, float, float]:
    xs = ys = zs = 0.0
    count = 0
    for lat, lon in points:
        x, y, z = _to_vector(lat, lon)
        xs += x
        ys += y
        zs += z
        count += 1
    if count == 0:
        raise Invalid("the centroid of no points is undefined")
    return (xs / count, ys / count, zs / count)


def concentration(points: Iterable[Point]) -> float:
    # the length of the mean vector: one for a single spot, near zero for a balanced spread
    x, y, z = mean_vector(points)
    return math.sqrt(x * x + y * y + z * z)


def centroid(points: Iterable[Point], min_concentration: float = 1e-9) -> Point:
    x, y, z = mean_vector(points)
    length = math.sqrt(x * x + y * y + z * z)
    if length < min_concentration:
        raise Degenerate("the points balance round the globe; no centroid direction")
    return (math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x)))


def coordinate_mean(points: Iterable[Point]) -> Point:
    # the naive average of latitudes and longitudes, kept for measuring its failures
    lats, lons = [], []
    for lat, lon in points:
        lats.append(lat)
        lons.append(lon)
    if not lats:
        raise Invalid("the centroid of no points is undefined")
    return (sum(lats) / len(lats), sum(lons) / len(lons))


def chord_cost(points: Iterable[Point], center: Point) -> float:
    # the summed squared chord distance from a candidate center, which the centroid minimizes
    cx, cy, cz = _to_vector(*center)
    total = 0.0
    for lat, lon in points:
        x, y, z = _to_vector(lat, lon)
        total += (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
    return total
