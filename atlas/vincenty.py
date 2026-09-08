"""Vincenty: distance on the flattened ellipsoid, more exact than the spherical haversine.

The earth is not a sphere but an oblate ellipsoid, flattened at the
poles by about a third of a percent, so a distance computed on a sphere
carries a small systematic error. Vincenty's formula computes the
geodesic distance on the reference ellipsoid, the WGS84 shape that GPS
uses, and is accurate to within a fraction of a millimeter. It is
iterative: it solves for the angular separation on an auxiliary sphere
and refines a term that accounts for the ellipsoid's flattening,
repeating until the angle stops changing. Because the sphere is only an
approximation of the ellipsoid, haversine and Vincenty disagree, and
the disagreement has a sign that flips, which is more interesting than
a uniform gap and worth stating as the measured truth. A first guess
was that the gap would be smallest along the equator, since the
equator is a circle on both shapes; the measurement refutes that. With
haversine using the mean earth radius, Vincenty comes out shorter than
haversine on a north-south route, by about ten kilometers over eighty
degrees of latitude, because the ellipsoid is squashed at the poles,
but longer than haversine on an equatorial east-west route of the same
span, because the ellipsoid bulges at the equator to a radius larger
than the mean. So the two differ by a few tenths of a percent
everywhere, and the direction of the difference reverses between polar
and equatorial travel. Vincenty's accuracy has a cost worth stating
honestly: the iteration converges quickly for ordinary pairs but slowly
or not at all for nearly antipodal points, where the geometry is
ill-conditioned, so a correct implementation caps the iterations and
reports non-convergence rather than returning a wrong number or looping
forever. The finding worth stating is that Vincenty and haversine
differ by the earth's shape, a small but real gap whose sign reverses
between north-south and east-west travel, so the choice between them is
a choice of how much accuracy the task needs against the sphere's
simplicity. This module computes the Vincenty inverse distance and
refuses to converge past a cap, and a survey measures the
Vincenty-to-haversine gap and confirms it is negative north-south and
positive along the equator.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid, Outside

# WGS84 ellipsoid parameters
_A = 6378137.0  # semi-major axis, meters
_F = 1 / 298.257223563  # flattening
_B = (1 - _F) * _A  # semi-minor axis


def _check(lat: float, lon: float) -> None:
    if not -90.0 <= lat <= 90.0:
        raise Outside("latitude must lie within -90 and 90 degrees")
    if not -180.0 <= lon <= 180.0:
        raise Outside("longitude must lie within -180 and 180 degrees")


def distance_m(
    lat1: float, lon1: float, lat2: float, lon2: float, max_iters: int = 200
) -> float:
    _check(lat1, lon1)
    _check(lat2, lon2)
    if (lat1, lon1) == (lat2, lon2):
        return 0.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    ll = math.radians(lon2 - lon1)
    u1 = math.atan((1 - _F) * math.tan(phi1))
    u2 = math.atan((1 - _F) * math.tan(phi2))
    sin_u1, cos_u1 = math.sin(u1), math.cos(u1)
    sin_u2, cos_u2 = math.sin(u2), math.cos(u2)
    lam = ll
    for _ in range(max_iters):
        sin_lam, cos_lam = math.sin(lam), math.cos(lam)
        sin_sigma = math.sqrt(
            (cos_u2 * sin_lam) ** 2
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam) ** 2
        )
        if sin_sigma == 0:
            return 0.0
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2
        cos_2sigma_m = (
            cos_sigma - 2 * sin_u1 * sin_u2 / cos_sq_alpha if cos_sq_alpha != 0 else 0.0
        )
        c = _F / 16 * cos_sq_alpha * (4 + _F * (4 - 3 * cos_sq_alpha))
        lam_prev = lam
        lam = ll + (1 - c) * _F * sin_alpha * (
            sigma
            + c
            * sin_sigma
            * (cos_2sigma_m + c * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
        )
        if abs(lam - lam_prev) < 1e-12:
            u_sq = cos_sq_alpha * (_A**2 - _B**2) / _B**2
            big_a = 1 + u_sq / 16384 * (
                4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq))
            )
            big_b = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
            delta_sigma = (
                big_b
                * sin_sigma
                * (
                    cos_2sigma_m
                    + big_b
                    / 4
                    * (
                        cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                        - big_b
                        / 6
                        * cos_2sigma_m
                        * (-3 + 4 * sin_sigma**2)
                        * (-3 + 4 * cos_2sigma_m**2)
                    )
                )
            )
            return _B * big_a * (sigma - delta_sigma)
    raise Invalid("Vincenty failed to converge; the points are near-antipodal")
