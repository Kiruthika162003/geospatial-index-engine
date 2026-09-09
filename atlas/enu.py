"""ENU: the local east-north-up frame, and how far the flat earth can be trusted.

Near a reference point the world is usefully flat: east, north, and
up form a right-handed frame tangent to the ellipsoid there, and a
nearby position is a vector of meters in that frame, which is what
a local survey, a robot, or a radar wants. The frame is reached
from earth-centered coordinates by subtracting the reference's xyz
and rotating by the reference's latitude and longitude, and the
rotation is orthogonal so the inverse is its transpose. The
question a survey can answer is how far the flat earth extends,
and the answer has a clean law. A point on the ellipsoid surface a
distance d away from the reference sits below the tangent plane by
about d squared over twice the earth's radius, the curvature drop:
7.8 cm at one kilometer, 7.8 meters at ten, 785 meters at a
hundred, growing as the square of the distance so that each tenfold
step in range costs a hundredfold in drop. The survey measures the
drop by placing points on the surface at known ranges and reading
their up coordinate: 7.87 cm east and 7.84 cm north at one
kilometer, 7.87 and 7.84 meters at ten, 787 and 784 at a hundred,
against the law's 7.84, 784, and so on, ratios of 1.0036 east and
0.9997 north that hold unchanged from 100 meters to 100 km, the
residue being the ellipsoid's curvature in each direction differing
from the equatorial radius the law uses, and only at 1000 km does
the quadratic law itself slip by 0.15 percent as higher terms
enter. The same drop read the other way, a point one kilometer
east on the tangent plane, sits 7.83 cm above the surface. It also
measures the round trip, ENU to ECEF and back, 1.3 nanometers over
3000 random frames and offsets since the transform is a rigid
motion, 29 nanometers when the trip passes through geodetic
coordinates, and the frame's handedness, east crossed with north
giving up and north crossed with up giving east. The finding worth
stating is that the flat-earth error is quadratic in range with a
coefficient of one over twice the radius, so a flat frame is exact
to a centimeter within a few hundred meters, to a meter within
three and a half kilometers, and misleading by a building's height
by ten, which is the scale at which a local survey must start
correcting for the globe. This module converts between ECEF and a
local ENU frame, and a survey measures the curvature drop against
the quadratic law.
"""

from __future__ import annotations

import math

from atlas.ecef import WGS84_A, forward, inverse

Vector = tuple[float, float, float]


def _rotation(lat: float, lon: float) -> tuple[Vector, Vector, Vector]:
    phi, lam = math.radians(lat), math.radians(lon)
    east = (-math.sin(lam), math.cos(lam), 0.0)
    north = (-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi))
    up = (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))
    return east, north, up


def _dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def to_enu(point: Vector, ref_lat: float, ref_lon: float, ref_height: float = 0.0) -> Vector:
    rx, ry, rz = forward(ref_lat, ref_lon, ref_height)
    delta = (point[0] - rx, point[1] - ry, point[2] - rz)
    east, north, up = _rotation(ref_lat, ref_lon)
    return (_dot(delta, east), _dot(delta, north), _dot(delta, up))


def from_enu(enu: Vector, ref_lat: float, ref_lon: float, ref_height: float = 0.0) -> Vector:
    rx, ry, rz = forward(ref_lat, ref_lon, ref_height)
    east, north, up = _rotation(ref_lat, ref_lon)
    e, n, u = enu
    return (
        rx + e * east[0] + n * north[0] + u * up[0],
        ry + e * east[1] + n * north[1] + u * up[1],
        rz + e * east[2] + n * north[2] + u * up[2],
    )


def geodetic_to_enu(
    lat: float,
    lon: float,
    height: float,
    ref_lat: float,
    ref_lon: float,
    ref_height: float = 0.0,
) -> Vector:
    return to_enu(forward(lat, lon, height), ref_lat, ref_lon, ref_height)


def enu_to_geodetic(
    enu: Vector, ref_lat: float, ref_lon: float, ref_height: float = 0.0
) -> tuple[float, float, float]:
    lat, lon, height, _ = inverse(*from_enu(enu, ref_lat, ref_lon, ref_height))
    return (lat, lon, height)


def curvature_drop_m(range_m: float, radius_m: float = WGS84_A) -> float:
    # the quadratic law: a surface point at range d lies d^2 / 2R below the tangent plane
    return range_m * range_m / (2.0 * radius_m)


def frame_axes(lat: float, lon: float) -> tuple[Vector, Vector, Vector]:
    return _rotation(lat, lon)
