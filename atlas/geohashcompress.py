"""Sorted geohashes front-code a city to 2 bytes a point; unsorted ones cost more than raw.

A sorted list of geohashes can be front-coded, each entry keeping
only the length of the prefix it shares with its predecessor and
the suffix, at two bytes of overhead a point. Twenty thousand
points over the whole world share 2.21 characters between sorted
neighbours at any precision, so precision 6, 8, 10 and 12 hashes
code to 5.79, 7.79, 9.79 and 11.79 bytes a point against 7, 9, 11
and 13 raw, a saving of a character and a fraction. The guess that
sorted uniform neighbours share log base 32 of n characters, 2.86
for 20,000, was high by two thirds of a character: they share 1.35,
1.86, 2.21 and 2.65 at 1000, 5000, 20,000 and 80,000 points against
the law's 1.99, 2.46, 2.86 and 3.26. Coding the hashes in their
arrival order is worse than raw, 7.97 bytes a point at precision 6,
since the overhead is paid and nothing is shared.

A city of 20,000 points spread 0.05 degrees about its centre shares
5.9 to 6.3 characters and codes to 2.07, 3.67, 5.67 and 7.67 bytes
a point at the four precisions, 0.30 to 0.59 of raw; a tight city of
spread 0.005 shares 7.6 to 7.7 and codes to 2.0, 2.37, 4.32 and
6.32. At precision 6 the city's 20,000 points fall into only 1272
distinct hashes, since a cell is 611 m across; the worst decoding
error reads 611 m at precision 6, 19.1 m at 8, 0.60 m at 10 and
0.019 m at 12. Decoding the front code returns every hash.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

from atlas.errors import Invalid
from atlas.geohash import decode, encode

Point = tuple[float, float]


def hashes(points: list[Point], precision: int) -> list[str]:
    if precision < 1:
        raise Invalid("precision must be at least one character")
    return [encode(lat, lon, precision) for lat, lon in points]


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def front_code(sorted_hashes: list[str]) -> list[tuple[int, str]]:
    # each entry keeps the shared prefix length with its predecessor and the suffix
    out = []
    previous = ""
    for h in sorted_hashes:
        shared = common_prefix(previous, h)
        out.append((shared, h[shared:]))
        previous = h
    return out


def front_decode(coded: list[tuple[int, str]]) -> list[str]:
    out = []
    previous = ""
    for shared, suffix in coded:
        if shared > len(previous):
            raise Invalid("a shared prefix longer than the previous hash")
        current = previous[:shared] + suffix
        out.append(current)
        previous = current
    return out


def coded_bytes(coded: list[tuple[int, str]]) -> int:
    # one byte for the shared length, one per suffix character, one terminator
    return sum(2 + len(suffix) for _, suffix in coded)


def raw_bytes(sorted_hashes: list[str]) -> int:
    return sum(len(h) + 1 for h in sorted_hashes)


def ratio(points: list[Point], precision: int, presorted: bool = True) -> tuple[float, float]:
    hs = hashes(points, precision)
    if presorted:
        hs = sorted(hs)
    coded = front_code(hs)
    return coded_bytes(coded) / len(hs), raw_bytes(hs) / len(hs)


def mean_shared(sorted_hashes: list[str]) -> float:
    if len(sorted_hashes) < 2:
        raise Invalid("at least two hashes are needed")
    total = sum(common_prefix(a, b) for a, b in pairwise(sorted_hashes))
    return total / (len(sorted_hashes) - 1)


def decode_error_m(points: list[Point], precision: int) -> float:
    worst = 0.0
    for lat, lon in points:
        d_lat, d_lon = decode(encode(lat, lon, precision))
        worst = max(worst, abs(d_lat - lat) * 111_320, abs(d_lon - lon) * 111_320)
    return worst


def uniform_world(n: int, rng: random.Random) -> list[Point]:
    return [(rng.uniform(-90, 90), rng.uniform(-180, 180)) for _ in range(n)]


def city(
    n: int, rng: random.Random, centre: Point = (51.5, -0.12), spread: float = 0.05
) -> list[Point]:
    return [(rng.gauss(centre[0], spread), rng.gauss(centre[1], spread)) for _ in range(n)]


def log32_law(n: int) -> float:
    # n uniform hashes sorted: neighbours share about log base 32 of n characters
    if n < 2:
        raise Invalid("the law needs at least two hashes")
    return math.log(n, 32)
