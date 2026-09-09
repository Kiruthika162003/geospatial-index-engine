"""Bearing statistics: the mean of directions, which is not the mean of their numbers.

Bearings live on a circle, and the arithmetic mean does not know
that. Two headings of 350 and 10 degrees are twenty degrees apart
and point roughly north together, but their numbers average to 180,
due south, the one direction they least agree on. The right mean
treats each bearing as a unit vector, sums the vectors, and reads the
angle of the sum: 350 and 10 sum to a vector pointing at 0, and the
length of the summed vector divided by the count, the resultant
length, says how much the bearings agree, one when they coincide and
zero when they cancel. From the resultant length come the circular
variance, one minus that length, and the circular standard deviation,
the square root of minus two times its logarithm, which for
clustered bearings matches the ordinary standard deviation and for
cancelling ones grows without bound. The survey measures the
arithmetic failure and the vector success on the same inputs. The
350-and-10 pair is the loud case, 0 against 180. The quiet case is
200 bearings clustered round 5 degrees with a spread of 8, where 58
members wrap past 360 and pull the arithmetic mean to 109 degrees,
104 off the cluster, while the vector mean sits at 4.9. Rotating
every bearing by the same angle and rotating the answer back moves
the vector mean by 3e-14 degrees and the arithmetic mean by up to
148, since each rotation pushes a different set of members across
the seam. The guess that the circular standard deviation would only
match the ordinary one on tight clusters was too cautious: on 5000
Gaussian bearings it agreed to 0.004 percent at a spread of 5
degrees and still to 0.13 percent at a spread of 60, and only the
cancelling case, 0 and 180 together, sends it to 495 degrees while
the mean direction is refused. The finding worth stating is that the
vector mean is the same function of the directions wherever the seam
sits, while the arithmetic mean changes by up to 180 degrees
depending on which side of north the numbers fall, so only the
vector mean is a mean of directions. This module computes circular
means and spreads, and a survey measures them against the arithmetic
versions.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from atlas.errors import Degenerate, Invalid


def _resultant(bearings: Iterable[float]) -> tuple[float, float, int]:
    x = y = 0.0
    count = 0
    for b in bearings:
        r = math.radians(b)
        x += math.cos(r)
        y += math.sin(r)
        count += 1
    if count == 0:
        raise Invalid("the mean of no bearings is undefined")
    return (x / count, y / count, count)


def resultant_length(bearings: Iterable[float]) -> float:
    x, y, _ = _resultant(bearings)
    return math.hypot(x, y)


def circular_mean(bearings: Iterable[float], min_length: float = 1e-12) -> float:
    x, y, _ = _resultant(bearings)
    if math.hypot(x, y) < min_length:
        raise Degenerate("the bearings cancel; no mean direction")
    mean = math.degrees(math.atan2(y, x)) % 360.0
    # a hair below zero wraps to 360.0 under the modulus; north is 0
    return 0.0 if mean >= 360.0 else mean


def circular_variance(bearings: Iterable[float]) -> float:
    return 1.0 - resultant_length(bearings)


def circular_std_deg(bearings: Iterable[float]) -> float:
    length = resultant_length(bearings)
    if length <= 0.0:
        return math.inf
    return math.degrees(math.sqrt(-2.0 * math.log(length)))


def arithmetic_mean(bearings: Iterable[float]) -> float:
    # the mean of the numbers, kept for measuring how far it strays from the direction
    values = list(bearings)
    if not values:
        raise Invalid("the mean of no bearings is undefined")
    return sum(values) / len(values)


def angular_gap(a: float, b: float) -> float:
    # the unsigned separation of two bearings the short way round, in [0, 180]
    return abs((a - b + 180.0) % 360.0 - 180.0)


def rotate(bearings: Iterable[float], by: float) -> list[float]:
    return [(b + by) % 360.0 for b in bearings]
