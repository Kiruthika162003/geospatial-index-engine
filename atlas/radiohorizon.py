"""The radio horizon is 3.57 root h km, 4.12 with refraction, and a clear ray can still fade.

The distance to the horizon from a height h over a sphere is root of
2Rh plus h squared, which the rule of thumb 3.5696 root h matches to
a metre up to 300 m and to 4 m at 1000 m, where the exact arc reads
112.873 km against the chord's 112.885. Standard refraction bends a
radio ray toward the ground, which the four-thirds Earth models by
scaling the radius, and every horizon grows by root of four thirds,
1.1547: 4.372 becomes 5.048 km at 1.5 m, 11.288 becomes 13.034 at 10
m, 35.696 becomes 41.218 at 100 m. Two antennas see each other at
the sum of their horizons, 22.576 km for 10 m each and 26.069 with
refraction, 40.068 and 46.266 for 100 m and 1.5 m, and a search over
a flat profile at 0.1 km steps finds the same ranges within a step.

The Earth's bulge under a path is d1 d2 over 2kR, 7.848 m at the
middle of 20 km and 5.886 with refraction, 49.05 and 36.79 at the
middle of 50 km. The guess that a ray clearing every ridge is a
working link was wrong: with 30 m antennas 20 km apart the ray sits
22.15 m over the middle of the path, 24.11 with refraction, so a 20 m
ridge there leaves 2.15 m of clearance, while the first Fresnel zone
at that point is 25.0 m across at 2.4 GHz and 40.8 m at 0.9 GHz, so
the ridge fills most of the zone and the link fades. A 30 m ridge
blocks the ray outright by 7.85 m, 5.89 with refraction, and the
same 20 m ridge 5 km from one end leaves 4.11 m against a Fresnel
radius of 21.65 m.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

EARTH_RADIUS_KM = 6371.0088
Profile = list[tuple[float, float]]


def horizon_km(
    height_m: float, k_factor: float = 1.0, radius_km: float = EARTH_RADIUS_KM
) -> float:
    if height_m < 0:
        raise Invalid("the height must not be negative")
    if k_factor <= 0:
        raise Invalid("the k factor must be positive")
    effective = radius_km * k_factor
    return math.sqrt(2 * effective * height_m / 1000 + (height_m / 1000) ** 2)


def rule_of_thumb_km(height_m: float, k_factor: float = 1.0) -> float:
    return math.sqrt(2 * EARTH_RADIUS_KM * k_factor / 1000) * math.sqrt(height_m)


def link_range_km(tx_m: float, rx_m: float, k_factor: float = 1.0) -> float:
    return horizon_km(tx_m, k_factor) + horizon_km(rx_m, k_factor)


def earth_bulge_m(d1_km: float, d2_km: float, k_factor: float = 1.0) -> float:
    if d1_km < 0 or d2_km < 0:
        raise Invalid("distances must not be negative")
    return 1000 * d1_km * d2_km / (2 * EARTH_RADIUS_KM * k_factor)


def ray_height_m(
    tx_m: float, rx_m: float, total_km: float, at_km: float, k_factor: float = 1.0
) -> float:
    if total_km <= 0 or at_km < -1e-9 or at_km > total_km + 1e-9:
        raise Invalid("the sample must lie on the path")
    at_km = min(max(at_km, 0.0), total_km)
    straight = tx_m + (rx_m - tx_m) * at_km / total_km
    return straight - earth_bulge_m(at_km, total_km - at_km, k_factor)


def clearance_m(
    tx_m: float, rx_m: float, total_km: float, profile: Profile, k_factor: float = 1.0
) -> float:
    if not profile:
        raise Invalid("the profile needs at least one point")
    worst = math.inf
    for at_km, ground_m in profile:
        worst = min(worst, ray_height_m(tx_m, rx_m, total_km, at_km, k_factor) - ground_m)
    return worst


def fresnel_radius_m(freq_ghz: float, d1_km: float, d2_km: float) -> float:
    if freq_ghz <= 0:
        raise Invalid("the frequency must be positive")
    total = d1_km + d2_km
    if total <= 0:
        raise Invalid("the path needs length")
    return 17.32 * math.sqrt(d1_km * d2_km / (freq_ghz * total))


def flat_profile(total_km: float, samples: int, height_m: float = 0.0) -> Profile:
    if samples < 2:
        raise Invalid("a profile needs two samples")
    return [(total_km * k / (samples - 1), height_m) for k in range(samples)]


def ridge_profile(total_km: float, at_km: float, ridge_m: float, samples: int) -> Profile:
    profile = flat_profile(total_km, samples)
    half_step = total_km / (2 * (samples - 1))
    return [(d, ridge_m if abs(d - at_km) <= half_step else h) for d, h in profile]


def max_range_over_flat(
    tx_m: float, rx_m: float, k_factor: float = 1.0, step_km: float = 0.1
) -> float:
    total = step_km
    while clearance_m(tx_m, rx_m, total, flat_profile(total, 201), k_factor) >= 0:
        total += step_km
    return total - step_km


def exact_geometric_horizon_km(height_m: float, radius_km: float = EARTH_RADIUS_KM) -> float:
    h = height_m / 1000
    return radius_km * math.acos(radius_km / (radius_km + h))
