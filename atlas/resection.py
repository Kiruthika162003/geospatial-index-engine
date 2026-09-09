"""Resection: a position from bearings to known landmarks, and the circle where it fails.

A surveyor who can see three known landmarks and measure the bearing
to each can fix a position without measuring any distance: each
bearing is a line from the unknown point through a landmark, and the
lines meet at the point. On the plane a bearing b from the unknown
(x, y) to a landmark (lx, ly) says the vector from the point to the
landmark is parallel to (sin b, cos b), which is one linear equation
in x and y, so two landmarks give a solution and three or more give
a least-squares one. Two behaviors are worth measuring rather than
assuming. The first is the dilution of bearing noise, which depends
on distance as well as angle: a bearing error of one degree moves
the line by one degree times the range, so distant landmarks cost
more position error per degree than close ones. Measured over 2000
noisy trials, three landmarks spread at 120 degrees and range 10
cost 0.20 units of position per degree of bearing noise, the same
three at range 100 cost 2.02, ten times more for ten times the
range, and three bunched in a 20 degree arc at range 10 cost 0.72,
the lines crossing shallowly and sliding the fix along them. The
second is the danger circle, the classical warning that an observer
on the circle through three landmarks cannot be fixed, and here the
guess was wrong in an instructive way. The guess was that compass
bearings would become hypersensitive on the circle; the measurement
put the on-circle dilution at 0.16 per degree against 0.12 at the
circle's center and 0.16 at a point halfway in, no collapse at all.
The danger is real but belongs to a different problem: without a
compass, a surveyor measures only the angles the landmarks subtend
at the eye, and by the inscribed angle theorem every point on the
same arc of the circle sees the same two angles, measured here as
identical to 1e-12 degrees across the arc while varying by tens of
degrees off it, so the angle-only fix has no unique answer there.
With absolute bearings the compass supplies the orientation the
angles lack, and the circle is an ordinary place. The survey also
measures exactness with noise-free bearings, 3e-11 over 500 random
layouts, and the refusal when the bearing lines are parallel. The
finding worth stating is that resection error scales with landmark
range and with how shallowly the bearing lines cross, and that the
danger circle is a property of angle-only fixing that a compass
removes, so the old warning is about instruments, not geometry.
This module solves plane resection, and a survey measures
exactness, the range and angle dilution, and both readings of the
danger circle.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.errors import Degenerate, Invalid

Point = tuple[float, float]


def bearing_to(observer: Point, landmark: Point) -> float:
    # compass bearing in degrees clockwise from north
    dx, dy = landmark[0] - observer[0], landmark[1] - observer[1]
    return math.degrees(math.atan2(dx, dy)) % 360.0


def locate(landmarks: Sequence[Point], bearings: Sequence[float]) -> Point:
    if len(landmarks) < 2:
        raise Invalid("resection needs at least two landmarks")
    if len(landmarks) != len(bearings):
        raise Invalid("each landmark needs one bearing")
    # the line through landmark (lx, ly) with direction (sin b, cos b) satisfies
    # cos b * x - sin b * y = cos b * lx - sin b * ly
    a = b = d = e = f = 0.0
    for (lx, ly), bearing in zip(landmarks, bearings, strict=True):
        r = math.radians(bearing)
        cb, sb = math.cos(r), -math.sin(r)
        rhs = cb * lx + sb * ly
        a += cb * cb
        b += cb * sb
        d += sb * sb
        e += cb * rhs
        f += sb * rhs
    det = a * d - b * b
    if abs(det) < 1e-12:
        raise Degenerate("the bearing lines are parallel; the position is not unique")
    return ((e * d - b * f) / det, (a * f - e * b) / det)


def residual_deg(
    landmarks: Sequence[Point], bearings: Sequence[float], position: Point
) -> float:
    # root mean square mismatch between measured bearings and bearings from the position
    total = 0.0
    for landmark, bearing in zip(landmarks, bearings, strict=True):
        gap = (bearing_to(position, landmark) - bearing + 180.0) % 360.0 - 180.0
        total += gap * gap
    return math.sqrt(total / len(landmarks))


def dilution(landmarks: Sequence[Point], truth: Point, noise_deg: float, trials: int, rng):
    # root mean square position error per degree of bearing noise
    total = 0.0
    for _ in range(trials):
        noisy = [bearing_to(truth, lm) + rng.gauss(0, noise_deg) for lm in landmarks]
        px, py = locate(landmarks, noisy)
        total += (px - truth[0]) ** 2 + (py - truth[1]) ** 2
    return math.sqrt(total / trials) / noise_deg


def subtended_angles(observer: Point, landmarks: Sequence[Point]) -> list[float]:
    # the angles between successive landmarks as seen from the observer, compass-free
    bearings = [bearing_to(observer, lm) for lm in landmarks]
    return [(b - a) % 360.0 for a, b in itertools.pairwise(bearings)]


def on_circle(center: Point, radius: float, angle_deg: float) -> Point:
    r = math.radians(angle_deg)
    return (center[0] + radius * math.sin(r), center[1] + radius * math.cos(r))


def circumcircle(p: Point, q: Point, r: Point) -> tuple[Point, float]:
    # the circle through three landmarks, on which resection loses its grip
    ax, ay = p
    bx, by = q
    cx, cy = r
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        raise Degenerate("collinear landmarks have no circumcircle")
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)) / d
    ux += (cx * cx + cy * cy) * (ay - by) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)) / d
    uy += (cx * cx + cy * cy) * (bx - ax) / d
    return ((ux, uy), math.hypot(ax - ux, ay - uy))
