"""Five decimals hold a coordinate within 0.79 m, float32 loses 0.42 m at longitude 100.

Rounding a coordinate to d decimals moves it by at most half a unit
in the last place along each axis, 111,319 m a degree of latitude
and cos(lat) times that of longitude. Measured over 2000 random
points at each latitude the worst shift approaches the formula from
below, within 2 percent from two decimals on: 784.8 against 787.1 m
at two decimals on the equator, 78.5 against 78.7 at three, 7.77
against 7.87 at four, 0.773 against 0.787 at five, 0.078 at six and
0.008 at seven, and the mean shift is 0.54 of the worst. At latitude 80
the worst falls to 0.72 of the equator's, since the east axis
shrinks to cos 80. One decimal reads under the formula only because
the probe latitudes were held within 0.01 degree of the band. A one
metre tolerance needs five decimals at any latitude, ten metres
four, and a millimetre eight.

Storing a coordinate as a 32 bit float is a rounding of its own,
and the guess that it is harmless was wrong away from the prime
meridian. The unit in the last place of a float32 longitude is 0.85
m at longitude 100 and 1.70 m at 179.9, so the worst shift reads
0.423 and 0.848 m there, 0.053 m at longitude 10 and 0.0036 m at
0.5; a latitude between 80 and 89 shifts up to 0.423 m.

The guess that rounding makes small rings cross themselves was
wrong: rounding repeats vertices and collapses rings instead. With
eight vertices at a radius of twice the grid cell 27.8 percent of
rings gain a repeated vertex and none cross or collapse; at one cell
99 percent repeat; at half a cell 50.8 percent collapse to a line or
point, at a fifth 93.2 percent and at a tenth 98.4 percent. Twenty
vertex rings at five and two cells cross in 0.4 and 0.2 percent of
trials. A ring of radius a cell rounded to that cell changes its
area by 22.7 percent on average and 97.4 at worst; ten cells across
the change is 2.2 percent on average and 7.7 at worst, the same at
every decimal place.
"""

from __future__ import annotations

import math
import random
import struct

from atlas.errors import Invalid

METRES_PER_DEGREE = 111_319.49
Point = tuple[float, float]


def rounded(lat: float, lon: float, decimals: int) -> Point:
    if decimals < 0:
        raise Invalid("decimals must not be negative")
    return round(lat, decimals), round(lon, decimals)


def worst_shift_m(lat: float, decimals: int) -> float:
    half = 0.5 * 10**-decimals
    north = half * METRES_PER_DEGREE
    east = half * METRES_PER_DEGREE * math.cos(math.radians(lat))
    return math.hypot(north, east)


def shift_m(a: Point, b: Point) -> float:
    mid = math.radians((a[0] + b[0]) / 2)
    north = (b[0] - a[0]) * METRES_PER_DEGREE
    east = (b[1] - a[1]) * METRES_PER_DEGREE * math.cos(mid)
    return math.hypot(north, east)


def measured_shift_m(
    lat: float, decimals: int, rng: random.Random, samples: int = 2000
) -> tuple[float, float]:
    worst = 0.0
    total = 0.0
    for _ in range(samples):
        p = (lat + rng.uniform(-0.01, 0.01), rng.uniform(-180, 180))
        d = shift_m(p, rounded(*p, decimals))
        worst = max(worst, d)
        total += d
    return worst, total / samples


def to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def float32_shift_m(lat: float, lon: float) -> float:
    return shift_m((lat, lon), (to_float32(lat), to_float32(lon)))


def float32_ulp_m(lon: float, lat: float = 0.0) -> float:
    exponent = math.floor(math.log2(abs(lon))) if lon else -149
    ulp = 2.0 ** (exponent - 23)
    return ulp * METRES_PER_DEGREE * math.cos(math.radians(lat))


def polygon_area_deg2(ring: list[Point]) -> float:
    total = 0.0
    for i, (lat1, lon1) in enumerate(ring):
        lat2, lon2 = ring[(i + 1) % len(ring)]
        total += lon1 * lat2 - lon2 * lat1
    return abs(total) / 2


def area_change(ring: list[Point], decimals: int) -> float:
    before = polygon_area_deg2(ring)
    if before == 0:
        raise Invalid("the ring has no area")
    after = polygon_area_deg2([rounded(*p, decimals) for p in ring])
    return after / before - 1


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0) and d1 * d2 < 0 and d3 * d4 < 0


def self_intersects(ring: list[Point]) -> bool:
    n = len(ring)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(ring[i], ring[(i + 1) % n], ring[j], ring[(j + 1) % n]):
                return True
    return False


def collapsed(ring: list[Point]) -> bool:
    return len(set(ring)) < 3 or polygon_area_deg2(ring) == 0


def small_ring(
    centre: Point, radius_deg: float, vertices: int, rng: random.Random
) -> list[Point]:
    if vertices < 3:
        raise Invalid("a ring needs three vertices")
    ring = []
    for k in range(vertices):
        angle = 2 * math.pi * k / vertices + rng.uniform(-0.2, 0.2)
        r = radius_deg * rng.uniform(0.5, 1.0)
        ring.append((centre[0] + r * math.sin(angle), centre[1] + r * math.cos(angle)))
    return ring


def damage_rates(
    radius_deg: float, decimals: int, rng: random.Random, trials: int = 500, vertices: int = 8
) -> dict[str, float]:
    crossed = collapsed_count = repeated = 0
    for _ in range(trials):
        centre = (rng.uniform(-60, 60), rng.uniform(-180, 180))
        ring = small_ring(centre, radius_deg, vertices, rng)
        coarse = [rounded(*p, decimals) for p in ring]
        if len(set(coarse)) < len(coarse):
            repeated += 1
        if collapsed(coarse):
            collapsed_count += 1
        elif self_intersects(coarse):
            crossed += 1
    return {
        "crossed": crossed / trials,
        "collapsed": collapsed_count / trials,
        "repeated": repeated / trials,
    }


def decimals_for(metres: float, lat: float = 0.0) -> int:
    if metres <= 0:
        raise Invalid("the tolerance must be positive")
    decimals = 0
    while worst_shift_m(lat, decimals) > metres:
        decimals += 1
    return decimals
