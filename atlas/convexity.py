"""A star's convexity is its inner ratio over cos(pi over n); noise costs a circle 0.026 a unit.

Convexity here is a ring's area over the area of its convex hull,
one for any convex ring. A circle of 360 vertices and radius 100
reads 1.0, and Gaussian noise of 0.5, 1, 2 and 5 on the radius
brings the mean over twenty draws to 0.9888, 0.9745, 0.9441 and
0.8533, about 0.026 lost per unit of noise, since every inward
excursion is area the hull keeps and the ring does not. A star of n
points with inner radius a ratio of the outer reads 0.309, 0.618 and
0.927 at ratios of 0.25, 0.5 and 0.75 with five points, 0.2706,
0.5412 and 0.8118 with eight and 0.2588, 0.5176 and 0.7765 with
twelve, and 1.0 when the ratio is 1 and the hull takes every
vertex. The guess before measuring, a formula with two cosines that
read the five-point star at 0.2236, was wrong: the measured
convexity is exactly the inner ratio over cos(pi over n), 1.236,
1.0824 and 1.0353 times the ratio for five, eight and twelve points,
because the star and its hull share the same n triangles from the
centre and differ only in the triangles' heights. The hull of a
star holds n vertices below a ratio of 1 and 2n at it, and the hull
of a square is the square.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Ring = list[Point]


def area(ring: Ring) -> float:
    if len(ring) < 3:
        raise Invalid("a ring needs three vertices")
    total = 0.0
    for i, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(i + 1) % len(ring)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2


def hull(points: list[Point]) -> Ring:
    # Andrew's monotone chain
    pts = sorted(set(points))
    if len(pts) < 3:
        raise Invalid("a hull needs three distinct points")

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: Ring = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: Ring = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def convexity(ring: Ring) -> float:
    # the ring's area over its hull's area, one for a convex ring
    hull_area = area(hull(ring))
    if hull_area == 0:
        raise Invalid("the ring has no hull area")
    return area(ring) / hull_area


def noisy_circle(n: int, radius: float, noise: float, rng: random.Random) -> Ring:
    if n < 3:
        raise Invalid("a ring needs three vertices")
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        r = radius + rng.gauss(0, noise)
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def star(n: int, outer: float, inner: float) -> Ring:
    if n < 3 or not 0 < inner <= outer:
        raise Invalid("a star needs three points and an inner radius within the outer")
    out = []
    for k in range(2 * n):
        r = outer if k % 2 == 0 else inner
        a = math.pi * k / n
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def star_guess(n: int, ratio: float) -> float:
    # the guess before measuring, which read a five-point star at 0.22 where it measures 0.31
    return (
        ratio * math.cos(math.pi / n) / math.cos(math.pi / (2 * n)) ** 2 if ratio < 1 else 1.0
    )


def star_law(n: int, ratio: float) -> float:
    # the measured law: a star's area over its hull's is the inner ratio over cos(pi over n)
    if n < 3 or not 0 < ratio <= 1:
        raise Invalid("a star needs three points and a ratio in (0, 1]")
    return min(1.0, ratio / math.cos(math.pi / n))


def mean_convexity(n: int, radius: float, noise: float, seed: int, draws: int = 20) -> float:
    total = 0.0
    for k in range(draws):
        total += convexity(noisy_circle(n, radius, noise, random.Random(seed + k)))
    return total / draws
