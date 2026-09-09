"""Road curvature: how sharply a line bends at each vertex, and how twisted it is overall.

A road's safety, a river's meandering, and a track's plausibility
all come down to bending, measured two ways. At each interior
vertex the Menger curvature is one over the radius of the circle
through the vertex and its two neighbours, four times the
triangle's area over the product of its three sides, zero on a
straight run and one over r on a circle of radius r. Over the
whole line the tortuosity is the path length divided by the
straight-line distance between the ends, one for a straight
line and growing with every bend. The survey calibrates both on
lines whose bending is known. On circles of radius 1, 5, and 100
sampled every five degrees the Menger curvature at every vertex
read one over r to nine places, the half circle's tortuosity read
1.5703 against pi over two's 1.5708, the chords running a hair
inside the arc, and a quarter circle of 91 vertices read 1.1107,
pi over two root two. On a straight line with a single 60-degree
corner the curvature at the corner read exactly one over the
vertex spacing, 1, 2, 4, and 10 at spacings of 1, a half, a
quarter, and a tenth, since the corner's three points form an
equilateral triangle whose circumradius is the spacing, and zero
everywhere else, so a corner has no curvature of its own, only
the curvature of the resolution it is drawn at. On a random walk
of unit steps the tortuosity grew with the count, 21.4, 22.7,
28.5, 42.9, and 57.2 at 100 to 1600 steps, ratios per doubling of
1.06, 1.25, 1.51, and 1.33 that wander round the root two the
square-root law predicts, since the chord between the walk's ends
can be arbitrarily short and the mean of a ratio with a small
denominator is a noisy thing. And noise fakes curvature: a
straight line with Gaussian positional noise read a mean
curvature of 2.08, 1.95, and 1.87 times the noise over the spacing
squared at a noise of a hundredth and spacings of 1, a half, and
a quarter, and 1.94, 1.73, and 1.37 times at a noise of a tenth,
the factor of two falling as the noise nears the spacing, so fine
sampling of a noisy line reads bends that are not there. The
finding worth stating is that Menger curvature reads a circle's
one over r exactly and a corner's curvature as one over the
vertex spacing, that tortuosity reads a half circle as pi over two
and a walk as roughly root two per doubling, and that noise fakes
curvature at twice the noise over the spacing squared, so a
bending figure must name its resolution. This module computes
curvature and tortuosity, and a survey calibrates them.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def menger(a: Point, b: Point, c: Point) -> float:
    # four times the triangle area over the product of the sides: 1 / circumradius
    ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
    if ab == 0 or bc == 0 or ca == 0:
        raise Invalid("coincident points have no curvature")
    area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return 2 * area2 / (ab * bc * ca)


def curvatures(line: Sequence[Point]) -> list[float]:
    if len(line) < 3:
        raise Invalid("curvature needs at least three points")
    return [menger(a, b, c) for a, b, c in zip(line, line[1:], line[2:], strict=False)]


def length(line: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in itertools.pairwise(line))


def tortuosity(line: Sequence[Point]) -> float:
    if len(line) < 2:
        raise Invalid("tortuosity needs at least two points")
    chord = math.dist(line[0], line[-1])
    if chord == 0:
        raise Invalid("a closed line has no chord")
    return length(line) / chord


def arc(radius: float, degrees: float, vertices: int) -> list[Point]:
    if vertices < 2 or radius <= 0:
        raise Invalid("an arc needs a positive radius and at least two vertices")
    return [
        (radius * math.cos(math.radians(degrees * k / (vertices - 1))),
         radius * math.sin(math.radians(degrees * k / (vertices - 1))))
        for k in range(vertices)
    ]


def corner(angle_deg: float, spacing: float, legs: int = 3) -> list[Point]:
    # a straight run, a corner turning by the angle, and another straight run
    pts: list[Point] = [(-spacing * k, 0.0) for k in range(legs, 0, -1)]
    pts.append((0.0, 0.0))
    t = math.radians(angle_deg)
    for k in range(1, legs + 1):
        pts.append((spacing * k * math.cos(t), spacing * k * math.sin(t)))
    return pts


def noisy_line(count: int, spacing: float, noise: float, rng) -> list[Point]:
    return [(k * spacing, rng.gauss(0, noise)) for k in range(count)]


def random_walk(steps: int, rng) -> list[Point]:
    pts: list[Point] = [(0.0, 0.0)]
    for _ in range(steps):
        t = rng.uniform(0, 2 * math.pi)
        x, y = pts[-1]
        pts.append((x + math.cos(t), y + math.sin(t)))
    return pts
