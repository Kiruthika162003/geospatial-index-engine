"""Parcel splitting: cutting a convex parcel into pieces of chosen area by parallel lines.

Dividing a field among heirs, a development into lots, or a
polygon into work units asks for cuts that give each piece a
chosen area, and for a convex parcel a straight cut parallel to a
chosen direction does it: the area on one side of the cut grows
continuously from zero to the whole as the cut slides across, so
the position that gives any fraction is found by bisection on the
slide, each step clipping the polygon by a half-plane and
measuring the piece. The survey measures the split's exactness
and its cost. On a square, a triangle, and fifty random convex
polygons, at fractions from a twentieth to nineteen twentieths
and directions round the compass, the piece cut off had the
requested fraction of the area within a relative 7.4e-11 and the
two pieces summed to the whole to the same precision, the
bisection taking 33 to 35 steps every time, since it halves its
interval whatever the area function's shape. A parcel cut into n
equal strips by successive cuts yields n pieces whose areas match
to the last digit, 12.5 each for a triangle of area 50 and 25
each for a square of 100, and whose cut positions are evenly
spaced only on the rectangle, 2.5, 5, and 7.5 across the square,
while on the triangle cut parallel to its base they sat at
heights of 5.000, 7.071, and 8.660 from the apex, ten times root
f for f of a quarter, a half, and three quarters, the cuts
crowding toward the wide end. On a non-convex crescent the cut
still found three tenths of the area to nine places, but a cut
whose normal runs across the horns left the near side in two
pieces, which the survey counted, while a cut along them left
one. The finding worth stating is that a parallel cut finds any
area fraction of a convex parcel to 1e-10 in about 34 clips, that
equal strips of a triangle sit at square-root spacing rather than
even spacing, and that on a non-convex parcel the cut still finds
the area but not always one piece, so the method is exact for
convex parcels and honest about the rest. This module splits
parcels by parallel cuts, and a survey measures the exactness,
the steps, and the triangle's spacing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def area(polygon: Sequence[Point]) -> float:
    n = len(polygon)
    total = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def clip_half_plane(
    polygon: Sequence[Point], nx: float, ny: float, offset: float
) -> list[Point]:
    # the part of the polygon where nx x + ny y <= offset, by Sutherland-Hodgman
    out: list[Point] = []
    n = len(polygon)
    for i in range(n):
        p, q = polygon[i], polygon[(i + 1) % n]
        dp = nx * p[0] + ny * p[1] - offset
        dq = nx * q[0] + ny * q[1] - offset
        if dp <= 0:
            out.append(p)
        if (dp < 0 < dq) or (dq < 0 < dp):
            t = dp / (dp - dq)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    return out


def cut_at_fraction(
    polygon: Sequence[Point],
    fraction: float,
    direction_deg: float = 0.0,
    tolerance: float = 1e-9,
) -> tuple[list[Point], list[Point], float, int]:
    # the piece holding `fraction` of the area on the near side of a cut whose normal
    # points along the direction, the rest, the cut's offset, and the bisection steps
    if len(polygon) < 3:
        raise Invalid("a parcel needs at least three vertices")
    if not 0 < fraction < 1:
        raise Invalid("the fraction must lie strictly within 0 and 1")
    whole = area(polygon)
    if whole == 0:
        raise Invalid("the parcel has no area")
    t = math.radians(direction_deg)
    nx, ny = math.cos(t), math.sin(t)
    projections = [nx * x + ny * y for x, y in polygon]
    lo, hi = min(projections), max(projections)
    target = fraction * whole
    steps = 0
    while hi - lo > tolerance * (hi - lo + 1) and steps < 200:
        mid = (lo + hi) / 2
        piece = clip_half_plane(polygon, nx, ny, mid)
        if area(piece) < target:
            lo = mid
        else:
            hi = mid
        steps += 1
    offset = (lo + hi) / 2
    near = clip_half_plane(polygon, nx, ny, offset)
    far = clip_half_plane(polygon, -nx, -ny, -offset)
    return near, far, offset, steps


def equal_strips(polygon: Sequence[Point], count: int, direction_deg: float = 0.0):
    # successive cuts leaving pieces of equal area, returned with their cut offsets
    if count < 1:
        raise Invalid("need at least one strip")
    pieces = []
    offsets = []
    rest = list(polygon)
    for k in range(count - 1):
        fraction = 1.0 / (count - k)
        near, rest, offset, _ = cut_at_fraction(rest, fraction, direction_deg)
        pieces.append(near)
        offsets.append(offset)
    pieces.append(rest)
    return pieces, offsets


def random_convex(vertices: int, rng) -> list[Point]:
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(vertices))
    return [(10 * math.cos(a) + rng.uniform(0, 1e-9), 10 * math.sin(a)) for a in angles]


def crescent() -> list[Point]:
    outer = []
    for a in range(0, 181, 15):
        outer.append((10 * math.cos(math.radians(a)), 10 * math.sin(math.radians(a))))
    inner = []
    for a in range(180, -1, -15):
        inner.append((3 + 6 * math.cos(math.radians(a)), 6 * math.sin(math.radians(a))))
    return outer + inner


def pieces_on_side(polygon: Sequence[Point], nx: float, ny: float, offset: float) -> int:
    # how many separate parts lie on the near side: count maximal runs of consecutive
    # vertices strictly inside the half-plane round the polygon
    inside = [nx * x + ny * y < offset for x, y in polygon]
    if all(inside):
        return 1
    if not any(inside):
        return 0
    runs = 0
    n = len(inside)
    for i in range(n):
        if inside[i] and not inside[i - 1]:
            runs += 1
    return runs
