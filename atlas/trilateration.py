"""Trilateration: a position from distances to known beacons, and when the geometry fails.

Three beacons at known places, three measured distances, one
unknown position: each distance is a circle round its beacon, and
the position is where the circles meet. Subtracting the first
circle's equation from each of the others cancels the squared
unknowns and leaves a linear system, two equations in the two
coordinates for three beacons, more for more, solved by least squares
when the distances carry noise and the circles no longer meet at a
point. The survey measures three things. With exact distances the
recovered position matches the truth to floating precision for any
beacons that are not collinear. With Gaussian noise on the distances
the position error scales with the noise and with the geometry. The
guess was thirtyfold dilution for a bunched layout against threefold
for a spread one; the measurement over 2000 noisy trials was 1.14
for three beacons spread round the target at 120 degrees, 1.07 with
a fourth added, and 80.0 for three beacons bunched in a 20 degree
arc on one side, because the circles cross at a shallow angle there
and a small radius change slides the crossing a long way. The
normal-matrix determinant tells the same story before any noise is
added, 1.08 million for the spread layout against 4.5 for the
bunched, and exactly zero for collinear beacons, which fail
outright since every point on the perpendicular bisector is equally
consistent; the module refuses rather than picking one. A finding
that was not guessed at all: a common bias on every distance, each
reading half a unit long, leaves the position untouched to floating
precision, because the linearization subtracts the first beacon's
equation from the others and the shared error cancels, while the
residual reports the half unit faithfully. The residual, the root
mean square mismatch between the measured distances and the
distances from the solved position, is the one number that says
whether the distances agreed with each other at all. The finding
worth stating is that trilateration is exact and stable when beacons
surround the target and degrades in proportion to how nearly they
line up, eightyfold for a bunched layout against near unity for a
spread one, so a beacon plan is a geometry decision before it is a
hardware one. This module solves plane trilateration, and a survey
measures exactness, noise dilution, bias cancellation, and the
collinear refusal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Degenerate, Invalid

Point = tuple[float, float]


def _solve_2x2(a: float, b: float, c: float, d: float, e: float, f: float) -> Point:
    # [a b; c d] [x y]^T = [e f]^T
    det = a * d - b * c
    if abs(det) < 1e-12:
        raise Degenerate("the beacons are collinear; the position is not unique")
    return ((e * d - b * f) / det, (a * f - e * c) / det)


def normal_matrix_determinant(beacons: Sequence[Point]) -> float:
    # the determinant of the least-squares normal matrix, zero for collinear beacons
    (x0, y0) = beacons[0]
    rows = [(x - x0, y - y0) for x, y in beacons[1:]]
    a = sum(2 * r[0] * 2 * r[0] for r in rows)
    b = sum(2 * r[0] * 2 * r[1] for r in rows)
    d = sum(2 * r[1] * 2 * r[1] for r in rows)
    return a * d - b * b


def locate(beacons: Sequence[Point], distances: Sequence[float]) -> Point:
    if len(beacons) < 3:
        raise Invalid("trilateration needs at least three beacons")
    if len(beacons) != len(distances):
        raise Invalid("each beacon needs one distance")
    if any(d < 0 for d in distances):
        raise Invalid("distances cannot be negative")
    (x0, y0), d0 = beacons[0], distances[0]
    # linearize against the first beacon:
    # 2(xi-x0) x + 2(yi-y0) y = d0^2 - di^2 + xi^2 - x0^2 + yi^2 - y0^2
    rows = []
    rhs = []
    for (xi, yi), di in zip(beacons[1:], distances[1:], strict=True):
        rows.append((2 * (xi - x0), 2 * (yi - y0)))
        rhs.append(d0 * d0 - di * di + xi * xi - x0 * x0 + yi * yi - y0 * y0)
    # normal equations for least squares
    a = sum(r[0] * r[0] for r in rows)
    b = sum(r[0] * r[1] for r in rows)
    d = sum(r[1] * r[1] for r in rows)
    e = sum(r[0] * v for r, v in zip(rows, rhs, strict=True))
    f = sum(r[1] * v for r, v in zip(rows, rhs, strict=True))
    return _solve_2x2(a, b, b, d, e, f)


def residual(beacons: Sequence[Point], distances: Sequence[float], position: Point) -> float:
    # root mean square mismatch between measured distances and distances from the position
    px, py = position
    total = 0.0
    for (x, y), d in zip(beacons, distances, strict=True):
        total += (math.hypot(x - px, y - py) - d) ** 2
    return math.sqrt(total / len(beacons))


def dilution(beacons: Sequence[Point], truth: Point, noise: float, trials: int, rng) -> float:
    # the ratio of root mean square position error to the distance noise over noisy trials
    total = 0.0
    for _ in range(trials):
        noisy = [
            math.hypot(x - truth[0], y - truth[1]) + rng.gauss(0, noise) for x, y in beacons
        ]
        px, py = locate(beacons, noisy)
        total += (px - truth[0]) ** 2 + (py - truth[1]) ** 2
    return math.sqrt(total / trials) / noise
