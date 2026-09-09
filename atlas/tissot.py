"""Tissot's indicatrix: how much each projection stretches and shears, read from finite steps.

Every flat map distorts the globe, and Tissot's indicatrix says
how at each point: an infinitesimal circle on the sphere lands on
the map as an ellipse whose semi-axes are the scale along the
meridian, h, and along the parallel, k, whose area is h times k
times the sine of the angle between them, and whose maximum
angular distortion omega satisfies sin(omega / 2) equals (a - b)
over (a + b) for the ellipse's axes a and b. The module reads h
and k from the projections already in the package by finite
differences, stepping a small distance north and east on the
sphere and measuring the map displacement per unit of ground,
and the survey calibrates the readings against the known laws of
each projection. Mercator is conformal, so h equals k everywhere
and omega is zero: the survey read h and k as equal within 5
parts in a million at latitudes 0, 30, 60, and 80, the residue
being the finite step, both as the secant of the latitude, 1,
1.155, 2.000, and 5.759, the area factor as the secant squared,
33.16 at latitude 80, and omega below 3e-4 degrees. The
equirectangular map, latitude and longitude as y and x, has h of
1 and k of the secant, and the guess of its angular distortion,
29 degrees at latitude 60 and 55 at 80, was well short: omega
read 8.23 at 30, 38.94 at 60, and 89.51 at 80, matching twice the
arcsine of (1 - cos) over (1 + cos) exactly, so an
equirectangular map near latitude 80 turns a right angle into
one of a half degree. The azimuthal equidistant map read 1.000
along every radius from its center and c over sin c across it,
1.0051, 1.1107, 1.5708, and 3.3322 at 10, 45, 90, and 135 degrees
out, so omega is 25.66 degrees at the quarter globe and 65.1 at
135, with the area factor equal to the transverse scale. And the
gnomonic map read one over cos squared c along the radius, 2.000
at 45 and 4.000 at 60 degrees out, and one over cos c across,
1.414 and 2.000, so its omega at 60 degrees is 38.94, the same
as the equirectangular's at latitude 60 since both have a
two-to-one axis ratio there, and its area factor one over cos
cubed, 8.000. The finding worth stating is that
finite-difference indicatrices reproduce the closed-form scales
of four projections to a few parts in a million, reading
Mercator as angle-true and area-false by the secant squared and
the equidistant map as distance-true along its radii and
angle-false by 25.7 degrees at the quarter globe, so a
projection's honesty is a number per point and not a slogan.
This module computes indicatrix parameters for any projection
function, and a survey calibrates them on four.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from atlas.errors import Invalid

Forward = Callable[[float, float], tuple[float, float]]


def scales(
    forward: Forward, lat: float, lon: float, step_deg: float = 1e-4
) -> tuple[float, float]:
    # h along the meridian and k along the parallel, per unit of ground at the sphere's scale
    if step_deg <= 0:
        raise Invalid("the step must be positive")
    if abs(lat) + step_deg > 90.0:
        raise Invalid("the point is too close to a pole to step")
    x0, y0 = forward(lat, lon)
    xn, yn = forward(lat + step_deg, lon)
    xe, ye = forward(lat, lon + step_deg)
    ground_north = math.radians(step_deg)
    ground_east = math.radians(step_deg) * math.cos(math.radians(lat))
    h = math.hypot(xn - x0, yn - y0) / ground_north
    k = math.hypot(xe - x0, ye - y0) / ground_east
    return h, k


def axes_angle(forward: Forward, lat: float, lon: float, step_deg: float = 1e-4) -> float:
    # the angle in degrees between the map's meridian and parallel directions
    x0, y0 = forward(lat, lon)
    xn, yn = forward(lat + step_deg, lon)
    xe, ye = forward(lat, lon + step_deg)
    dot = (xn - x0) * (xe - x0) + (yn - y0) * (ye - y0)
    norms = math.hypot(xn - x0, yn - y0) * math.hypot(xe - x0, ye - y0)
    if norms == 0:
        raise Invalid("a step produced no displacement")
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / norms))))


def indicatrix(forward: Forward, lat: float, lon: float, step_deg: float = 1e-4) -> dict:
    h, k = scales(forward, lat, lon, step_deg)
    theta = math.radians(axes_angle(forward, lat, lon, step_deg))
    area = h * k * math.sin(theta)
    # the ellipse axes from h, k, and the angle between them
    a_plus_b = math.sqrt(h * h + k * k + 2 * h * k * math.sin(theta))
    a_minus_b = math.sqrt(max(0.0, h * h + k * k - 2 * h * k * math.sin(theta)))
    a, b = (a_plus_b + a_minus_b) / 2, (a_plus_b - a_minus_b) / 2
    omega = 2 * math.degrees(math.asin(min(1.0, (a - b) / (a + b)))) if a + b > 0 else 0.0
    return {"h": h, "k": k, "area": area, "a": a, "b": b, "omega": omega}


def equirectangular(lat: float, lon: float) -> tuple[float, float]:
    return math.radians(lon), math.radians(lat)


def mercator_sphere(lat: float, lon: float) -> tuple[float, float]:
    # the spherical Mercator on the unit sphere, so scales read directly
    if not -89.999 < lat < 89.999:
        raise Invalid("Mercator has no finite image of the poles")
    return math.radians(lon), math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
