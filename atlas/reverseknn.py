"""Reverse nearest neighbours: who counts this place as their closest, measured against the six.

A shop wants to know not which customers are nearest to it but
which customers have it as their nearest shop, which is a
different set: the reverse nearest neighbours of a point are the
points whose own nearest neighbour is that point, and a point can
have none, when every neighbour has someone closer, or several,
when it stands alone in a crowd. In the plane the count is
bounded: a point can be the nearest neighbour of at most six
others, since seven points round a center would put two of them
within 60 degrees of each other and closer to one another than to
the center. The survey measures the distribution of the count on
12000 random points in forty sets of 300: it averaged exactly
one, since every point is somebody's nearest exactly once, and
read 28.3 percent of points with none, 46.5 with one, 22.2 with
two, 2.9 with three, and 0.12 with four, the largest count in any
set being four, nowhere near six. It tested the bound of six by
construction and found the guess needed a word changed: a center
with k points equally spaced on a circle round it is the nearest
neighbour of all k for k of 3, 4, and 5, whose ring sides of
1.73, 1.41, and 1.18 radii exceed the radius, but at six the side
equals the radius exactly, since a regular hexagon's side is its
circumradius, so each ring point is as close to its neighbours as
to the center and the count of six exists only as a tie, which
the index-order tie-break here resolves to one; at seven and
eight the sides of 0.87 and 0.77 radii make the ring points
prefer each other and the center's count is zero. Six is the
bound with ties allowed and five without, and random points never
reached either. The survey also measured the asymmetry of
nearness: 62.2 percent of points had their nearest neighbour
return the favour over the forty sets, between 58.7 and 67.3 in
any one, matching the known two-dimensional value of about 62,
so nearly four in ten nearest-neighbour relations are one-sided.
The finding worth stating is that reverse nearest neighbour
counts average one, top out at four on random points against a
bound of five in general position and six only with ties, and
leave 28 percent of points as nobody's nearest, so a catchment
built on nearest-shop logic is uneven in a way these numbers
describe. This module computes reverse nearest neighbours by
exhaustive search, and a survey measures the count distribution,
the bound, and the mutual fraction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def nearest_of_each(points: Sequence[Point]) -> list[int]:
    if len(points) < 2:
        raise Invalid("need at least two points")
    out = []
    for i, p in enumerate(points):
        best, best_d = -1, math.inf
        for j, q in enumerate(points):
            if i != j:
                d = math.dist(p, q)
                if d < best_d:
                    best, best_d = j, d
        out.append(best)
    return out


def reverse_neighbours(points: Sequence[Point]) -> list[list[int]]:
    nearest = nearest_of_each(points)
    out: list[list[int]] = [[] for _ in points]
    for i, j in enumerate(nearest):
        out[j].append(i)
    return out


def counts(points: Sequence[Point]) -> list[int]:
    return [len(r) for r in reverse_neighbours(points)]


def count_distribution(points: Sequence[Point]) -> dict[int, int]:
    dist: dict[int, int] = {}
    for c in counts(points):
        dist[c] = dist.get(c, 0) + 1
    return dict(sorted(dist.items()))


def mutual_fraction(points: Sequence[Point]) -> float:
    # the fraction of points whose nearest neighbour has them as its nearest
    nearest = nearest_of_each(points)
    return sum(1 for i, j in enumerate(nearest) if nearest[j] == i) / len(points)


def ring_with_center(count: int, radius: float = 1.0) -> list[Point]:
    # a center with `count` points equally spaced on a circle round it, the center first
    if count < 1 or radius <= 0:
        raise Invalid("the ring needs at least one point and a positive radius")
    pts: list[Point] = [(0.0, 0.0)]
    step = 2 * math.pi / count
    pts.extend((radius * math.cos(k * step), radius * math.sin(k * step)) for k in range(count))
    return pts


def ring_side(count: int, radius: float = 1.0) -> float:
    # the distance between neighbours on the ring: equals the radius exactly at six
    return 2 * radius * math.sin(math.pi / count)
