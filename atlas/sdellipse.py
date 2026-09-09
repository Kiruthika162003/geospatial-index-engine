"""Standard deviational ellipse: the direction and spread of a point pattern in one shape.

A set of crime locations, of tree falls after a storm, of shops
along a corridor, has a center, a spread, and often a direction,
and the standard deviational ellipse draws all three: its center
is the mean of the points, its orientation is the axis along
which the points spread most, found from the covariance of the
coordinates, and its semi-axes are the standard deviations along
and across that axis. The module computes the ellipse from the
covariance's eigenvectors and the survey measures what it reads
on known patterns. On an isotropic scatter the two axes should be
equal and the orientation meaningless, and over 200 random
scatters the axis ratio averaged 1.133 on a hundred points,
ranging from 1.009 to 1.311, and 1.043 on a thousand, ranging to
1.123, so a ratio of 1.1 on a hundred points is noise and not
direction, and even 1.3 can be. On scatters of 500 points
stretched three to one along bearings of 30, 170, and 95 degrees
the ellipse recovered the bearing within 0.71 to 0.86 degrees on
average over 100 trials each but by as much as 3.3 degrees on a
single trial, and its axis ratio averaged 3.006, 3.003, and 2.997
while single trials ranged from 2.65 to 3.43, so the guess that
the ratio comes within a few percent of three holds for the mean
and not for any one survey, which can be off by 14 percent. The
ellipse's coverage is a fixed fraction of the points: on 20000
Gaussian points the one-standard-deviation ellipse held 39.45
percent against the 39.35 the two-dimensional Gaussian predicts,
not the 68 of one dimension, the two-sigma ellipse 86.56 against
86.47, and the three-sigma 98.97 against 98.89, and a stretched
scatter read 39.06 and 86.45 alike. And the orientation has a
symmetry the survey checks, a bearing and its opposite being the
same axis, so the angle is reported in a half turn and a stretch
along 170 degrees read as 171.1 and not 351. The finding worth
stating is that the ellipse recovers a stretch direction within a
degree on average and three at worst, that its ratio on an
isotropic hundred wanders to 1.3, and that the one-sigma ellipse
holds 39 percent of a Gaussian scatter and the two-sigma 86, so
the shape is a summary whose numbers mean what the survey says
and not what one-dimensional habit expects. This module computes the
ellipse, and a survey measures it on isotropic, stretched, and
Gaussian scatters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def ellipse(points: Sequence[Point]) -> tuple[Point, float, float, float]:
    # center, semi-axis along the major direction, semi-axis across, bearing in [0, 180)
    n = len(points)
    if n < 3:
        raise Invalid("the ellipse needs at least three points")
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    sxx = sum((p[0] - cx) ** 2 for p in points) / n
    syy = sum((p[1] - cy) ** 2 for p in points) / n
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in points) / n
    if sxx == 0 and syy == 0:
        raise Invalid("all points coincide; no ellipse")
    # eigenvalues of the covariance
    trace, det = sxx + syy, sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, trace * trace / 4 - det))
    major, minor = trace / 2 + disc, max(0.0, trace / 2 - disc)
    # the major eigenvector's direction as a compass bearing from the y axis
    if sxy != 0:
        vx, vy = major - syy, sxy
    elif sxx >= syy:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    bearing = math.degrees(math.atan2(vx, vy)) % 180.0
    return (cx, cy), math.sqrt(major), math.sqrt(minor), bearing


def axis_ratio(points: Sequence[Point]) -> float:
    _, a, b, _ = ellipse(points)
    if b == 0:
        raise Invalid("the points are collinear; the ratio is unbounded")
    return a / b


def bearing_gap(a: float, b: float) -> float:
    # the unsigned gap between two axis bearings, in [0, 90]
    gap = abs(a - b) % 180.0
    return min(gap, 180.0 - gap)


def coverage(points: Sequence[Point], sigmas: float) -> float:
    # the fraction of points inside the ellipse scaled by `sigmas`
    (cx, cy), a, b, bearing = ellipse(points)
    if a == 0 or b == 0:
        raise Invalid("a flat ellipse has no interior")
    t = math.radians(bearing)
    inside = 0
    for x, y in points:
        dx, dy = x - cx, y - cy
        along = dx * math.sin(t) + dy * math.cos(t)
        across = dx * math.cos(t) - dy * math.sin(t)
        if (along / (sigmas * a)) ** 2 + (across / (sigmas * b)) ** 2 <= 1.0:
            inside += 1
    return inside / len(points)


def stretched(count: int, ratio: float, bearing_deg: float, rng) -> list[Point]:
    # a Gaussian scatter with unit spread across and `ratio` along the bearing
    t = math.radians(bearing_deg)
    out = []
    for _ in range(count):
        along, across = rng.gauss(0, ratio), rng.gauss(0, 1)
        x = along * math.sin(t) + across * math.cos(t)
        y = along * math.cos(t) - across * math.sin(t)
        out.append((x, y))
    return out


def isotropic(count: int, rng) -> list[Point]:
    return [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(count)]


def gaussian_coverage(sigmas: float) -> float:
    # the fraction of a 2-D Gaussian inside the `sigmas` ellipse: 1 - e^(-s^2 / 2)
    return 1.0 - math.exp(-sigmas * sigmas / 2)
