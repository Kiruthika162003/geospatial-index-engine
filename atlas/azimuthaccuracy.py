"""A two-station bearing fix misses by root 2 d sigma over sin theta, sharpest at a right angle.

A bearing taken to a point whose position is uncertain by sigma at
a distance d is off by about sigma over d radians: with sigma 5 the
bearing error reads 2.855, 0.2852 and 0.0285 degrees at distances
of 100, 1000 and 10,000 over 4000 draws against the law's 2.865,
0.2865 and 0.0286. A fix from two stations 1000 units from the
target, each bearing jittered by 1 degree, misses by a root mean
square of 157.6, 49.9, 28.5, 24.6, 28.4, 49.2 and 144.9 units when
the rays cross at 10, 30, 60, 90, 120, 150 and 170 degrees, against
the law root 2 d sigma over the sine of the crossing angle, 142.1,
49.4, 28.5, 24.7, 28.5, 49.4 and 142.1: within 1 percent from 30 to
150 degrees and 11 percent over at 10, where the near-parallel rays
make the intersection nonlinear in the jitter. The guess that the
fix is symmetric about the right angle held to 1 percent, 28.5
against 28.4 at 60 and 120 and 49.9 against 49.2 at 30 and 150, and
the miss grows linearly with the bearing sigma, 2.47, 12.34, 24.69
and 49.44 at 0.1, 0.5, 1 and 2 degrees at a right angle. The
dilution one over sin theta reads 5.76 at 10 degrees and 2.0 at 30.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


def bearing(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360


def bearing_error_law(position_sigma: float, distance: float) -> float:
    # a position error of sigma at a distance d turns the bearing by about sigma over d radians
    if distance <= 0:
        raise Invalid("the distance must be positive")
    return math.degrees(position_sigma / distance)


def measured_bearing_error(
    a: Point, b: Point, sigma: float, rng: random.Random, draws: int = 4000
) -> float:
    true = bearing(a, b)
    total = 0.0
    for _ in range(draws):
        jittered = (b[0] + rng.gauss(0, sigma), b[1] + rng.gauss(0, sigma))
        diff = ((bearing(a, jittered) - true + 180) % 360) - 180
        total += diff * diff
    return math.sqrt(total / draws)


def intersect(a: Point, bearing_a: float, b: Point, bearing_b: float) -> Point:
    # the point where two rays from a and b at the given bearings meet
    ax, ay = a
    bx, by = b
    ux, uy = math.sin(math.radians(bearing_a)), math.cos(math.radians(bearing_a))
    vx, vy = math.sin(math.radians(bearing_b)), math.cos(math.radians(bearing_b))
    det = ux * (-vy) - uy * (-vx)
    if abs(det) < 1e-12:
        raise Invalid("the rays are parallel")
    t = ((bx - ax) * (-vy) - (by - ay) * (-vx)) / det
    return ax + t * ux, ay + t * uy


def fix_error(
    target: Point,
    stations: tuple[Point, Point],
    bearing_sigma: float,
    rng: random.Random,
    draws: int = 4000,
) -> float:
    # the root mean square miss of a two-bearing fix with bearings jittered by sigma degrees
    a, b = stations
    total = 0.0
    for _ in range(draws):
        try:
            fix = intersect(
                a,
                bearing(a, target) + rng.gauss(0, bearing_sigma),
                b,
                bearing(b, target) + rng.gauss(0, bearing_sigma),
            )
        except Invalid:
            continue
        total += (fix[0] - target[0]) ** 2 + (fix[1] - target[1]) ** 2
    return math.sqrt(total / draws)


def fix_error_law(distance: float, bearing_sigma: float, angle_deg: float) -> float:
    # two rays of length d crossing at an angle theta: the miss is root 2 d sigma over sin theta
    if not 0 < angle_deg < 180:
        raise Invalid("the crossing angle must lie between 0 and 180")
    return (
        math.sqrt(2)
        * distance
        * math.radians(bearing_sigma)
        / math.sin(math.radians(angle_deg))
    )


def stations_at_angle(target: Point, distance: float, angle_deg: float) -> tuple[Point, Point]:
    # two stations at the distance from the target whose rays cross at the angle
    half = math.radians(angle_deg) / 2
    a = (target[0] - distance * math.sin(half), target[1] - distance * math.cos(half))
    b = (target[0] + distance * math.sin(half), target[1] - distance * math.cos(half))
    return a, b


def dilution(angle_deg: float) -> float:
    return 1 / math.sin(math.radians(angle_deg))
