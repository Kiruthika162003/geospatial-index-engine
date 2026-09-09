"""Equal-area maps keep area and pay in angle: 74 degrees at latitude 60 on the cylinder.

Four projections here have a Jacobian of exactly one: Lambert's
cylindrical, the sinusoidal, Lambert's azimuthal in the polar aspect,
and Mollweide's, whose auxiliary angle comes from Newton's iteration
in 1 step at the equator, 5 at latitude 30, 7 at 80, 11 at 89 and 14
at 89.9, where the derivative vanishes. The finite-difference
indicatrix reads the area factor within 1e-5 of one for the cylinder,
2e-10 for the sinusoidal, 4e-7 for the azimuthal and 3e-6 for
Mollweide, the residue of the 1e-4 degree step and not of the maps.
The exact test is a 10 by 10 degree cell: on the cylinder the
projected polygon's area meets the spherical cell to 3e-15 with no
densification, since parallels and meridians are straight lines
there. On the other three a cell's straight-edged image falls short
by 2.54e-3 on the sinusoidal and 5.07e-3 on the azimuthal at every
latitude tried, twice the one on the other, and by 1.58e-3, 2.87e-3
and 1.87e-2 on Mollweide at latitudes 0, 40 and 70; each shrinks
sixteenfold per fourfold densification, 6.2e-7 for the sinusoidal at
64 segments an edge.

The price is angle. The cylinder's maximum angular distortion reads
0, 16.43, 38.94, 73.74, 121.96 and 160.08 degrees at latitudes 0, 30,
45, 60, 75 and 85, and its indicatrix axis ratio 1, 1.333, 2, 4, 14.93
and 131.6 is the Mercator area inflation of the same latitude, since
both are the secant squared. The sinusoidal reads 0 along its central
meridian at every latitude and 42.88, 58.09, 68.45, 74.37 and 76.08 at
longitude 90. The polar azimuthal reads 38.94 at the equator falling
to 0.109 at latitude 85. Mollweide reads 12.01 at the equator, a
minimum of 2.95 at latitude 45 and 79.68 at 85 on the central
meridian, and 108.4 at latitude 85 and longitude 90. Mercator reads
under 2e-4 everywhere, its conformal due. A hemisphere ring of six
vertices projects to 1.0 of its area on the cylinder at any
densification, and to pi over 4, 3 over pi and 2 over pi on the
sinusoidal, azimuthal and Mollweide with straight edges, rising to
0.99984, 0.99873 and 0.99946 at 36 segments an edge.
"""

from __future__ import annotations

import math

from atlas import mercator
from atlas.errors import Invalid, Outside
from atlas.tissot import Forward, indicatrix

ROOT_TWO = math.sqrt(2.0)
LatLon = tuple[float, float]


def _check(lat: float, lon: float) -> tuple[float, float]:
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise Outside("latitude within 90 and longitude within 180")
    return math.radians(lat), math.radians(lon)


def lambert_cylindrical(lat: float, lon: float) -> tuple[float, float]:
    phi, lam = _check(lat, lon)
    return lam, math.sin(phi)


def sinusoidal(lat: float, lon: float) -> tuple[float, float]:
    phi, lam = _check(lat, lon)
    return lam * math.cos(phi), phi


def lambert_azimuthal(lat: float, lon: float) -> tuple[float, float]:
    phi, lam = _check(lat, lon)
    rho = 2 * math.sin((math.pi / 2 - phi) / 2)
    return rho * math.sin(lam), -rho * math.cos(lam)


def mollweide_theta(lat: float, tolerance: float = 1e-12) -> tuple[float, int]:
    phi = math.radians(lat)
    if abs(abs(phi) - math.pi / 2) < 1e-12:
        return math.copysign(math.pi / 2, phi), 0
    theta = phi
    target = math.pi * math.sin(phi)
    steps = 0
    while True:
        steps += 1
        delta = (2 * theta + math.sin(2 * theta) - target) / (2 + 2 * math.cos(2 * theta))
        theta -= delta
        if abs(delta) < tolerance or steps >= 50:
            return theta, steps


def mollweide(lat: float, lon: float) -> tuple[float, float]:
    _, lam = _check(lat, lon)
    theta, _ = mollweide_theta(lat)
    return 2 * ROOT_TWO / math.pi * lam * math.cos(theta), ROOT_TWO * math.sin(theta)


PROJECTIONS: dict[str, Forward] = {
    "cylindrical": lambert_cylindrical,
    "sinusoidal": sinusoidal,
    "azimuthal": lambert_azimuthal,
    "mollweide": mollweide,
}


def area_factor(forward: Forward, lat: float, lon: float) -> float:
    return indicatrix(forward, lat, lon)["area"]


def omega(forward: Forward, lat: float, lon: float) -> float:
    return indicatrix(forward, lat, lon)["omega"]


def axis_ratio(forward: Forward, lat: float, lon: float) -> float:
    ellipse = indicatrix(forward, lat, lon)
    if ellipse["b"] == 0:
        raise Invalid("the indicatrix has collapsed")
    return ellipse["a"] / ellipse["b"]


def cylindrical_omega(lat: float) -> float:
    ratio = 1 / math.cos(math.radians(lat)) ** 2
    return 2 * math.degrees(math.asin((ratio - 1) / (ratio + 1)))


def mercator_inflation(lat: float) -> float:
    return mercator.area_inflation(lat)


def worst_omega(
    forward: Forward, lats: list[float], lons: list[float]
) -> tuple[float, float, float]:
    worst = (-1.0, 0.0, 0.0)
    for lat in lats:
        for lon in lons:
            value = omega(forward, lat, lon)
            if value > worst[0]:
                worst = (value, lat, lon)
    return worst


def densify_edge(a: LatLon, b: LatLon, n: int) -> list[LatLon]:
    return [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(n)]


def projected_area(forward: Forward, polygon: list[LatLon], per_edge: int = 1) -> float:
    if len(polygon) < 3:
        raise Invalid("a polygon needs three vertices")
    if per_edge < 1:
        raise Invalid("each edge needs at least one segment")
    ring = []
    for i, vertex in enumerate(polygon):
        ring.extend(densify_edge(vertex, polygon[(i + 1) % len(polygon)], per_edge))
    points = [forward(lat, lon) for lat, lon in ring]
    total = 0.0
    for i, (x1, y1) in enumerate(points):
        x2, y2 = points[(i + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def cell_true_area(south: float, north: float, west: float, east: float) -> float:
    dlon = math.radians(east - west)
    return dlon * (math.sin(math.radians(north)) - math.sin(math.radians(south)))


def cell_polygon(south: float, north: float, west: float, east: float) -> list[LatLon]:
    return [(south, west), (south, east), (north, east), (north, west)]


def hemisphere_check(forward: Forward, per_edge: int = 36) -> float:
    top = 89.999
    ring = [(0.0, -180.0), (0.0, 0.0), (0.0, 180.0), (top, 180.0), (top, 0.0), (top, -180.0)]
    return projected_area(forward, ring, per_edge) / (2 * math.pi)
