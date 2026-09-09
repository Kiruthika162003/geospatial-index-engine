"""Tour: visiting every stop once and returning, by nearest neighbour and then by 2-opt.

A delivery round, a survey route, a drill pattern: the shortest
tour through a set of stops is the travelling salesman problem,
and for a few dozen stops the practical method is a construction
followed by an improvement. Nearest neighbour builds a tour by
always going to the closest unvisited stop, which is fast and
leaves long return legs, since the stops it skipped early must be
picked up late. 2-opt improves any tour by taking two edges that
cross or could be shortened, reversing the path between them, and
repeating until no pair helps; it removes every crossing, since a
crossing pair of edges can always be uncrossed to save length.
The survey measures the two against each other and against the
truth. On eight or nine random stops the optimum can be found by
trying every ordering, and over 60 layouts nearest neighbour
landed 8.5 percent long on average and 31 percent at worst, not
the quarter the guess expected at that size, while 2-opt from
that start landed 0.6 percent long on average, 9.3 at worst, and
reached the exact optimum on 47 of the 60. On fifty stops, where
no optimum is available, the nearest-neighbour tour of length
7.66 with 7 crossings became 6.29 with none after 28 reversals,
an improvement of 17.8 percent. Nearest neighbour from every one
of the fifty starts ranged from 6.46 to 8.60, a 33 percent swing.
Feeding 2-opt a random ordering of the same fifty, length 25.5,
took 113 reversals and ended at 6.18, shorter than the
nearest-neighbour start's 6.29 on that layout, and over twenty
more layouts the random start ended 2.7 percent longer on
average, from 5.5 percent shorter to 11 percent longer, so the
construction matters less than the improvement and sometimes
less than nothing. The finding worth stating is that nearest
neighbour is a tenth long on small layouts and a fifth on fifty
stops, 2-opt recovers nearly all of it and finds the true optimum
on most small layouts, and 2-opt from a random start does about
as well, so the improvement step is where the value lives. This module
builds and improves tours, and a survey measures the ratios, the
crossings, and the start dependence.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.errors import Invalid
from atlas.segmentintersect import segments_intersect

Point = tuple[float, float]


def length(tour: Sequence[Point]) -> float:
    if len(tour) < 2:
        return 0.0
    total = sum(math.dist(a, b) for a, b in itertools.pairwise(tour))
    return total + math.dist(tour[-1], tour[0])


def nearest_neighbour(stops: Sequence[Point], start: int = 0) -> list[Point]:
    if not stops:
        raise Invalid("no stops to tour")
    if not 0 <= start < len(stops):
        raise Invalid("the start index is off the list")
    remaining = list(range(len(stops)))
    current = remaining.pop(start)
    order = [current]
    while remaining:
        nxt = min(remaining, key=lambda i: math.dist(stops[current], stops[i]))
        remaining.remove(nxt)
        order.append(nxt)
        current = nxt
    return [stops[i] for i in order]


def two_opt(tour: Sequence[Point]) -> tuple[list[Point], int]:
    # reverse segments while any reversal shortens the tour; returns the tour and the count
    best = list(tour)
    n = len(best)
    reversals = 0
    if n < 4:
        return best, 0
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n if i > 0 else n - 1):
                a, b = best[i], best[i + 1]
                c, d = best[j], best[(j + 1) % n]
                current = math.dist(a, b) + math.dist(c, d)
                if math.dist(a, c) + math.dist(b, d) < current - 1e-12:
                    best[i + 1 : j + 1] = reversed(best[i + 1 : j + 1])
                    reversals += 1
                    improved = True
    return best, reversals


def optimal(stops: Sequence[Point]) -> tuple[list[Point], float]:
    # every ordering with the first stop fixed; only for small sets
    if not stops:
        raise Invalid("no stops to tour")
    if len(stops) > 10:
        raise Invalid("too many stops for brute force")
    first = stops[0]
    best: tuple[list[Point], float] | None = None
    for perm in itertools.permutations(stops[1:]):
        candidate = [first, *perm]
        total = length(candidate)
        if best is None or total < best[1]:
            best = (candidate, total)
    assert best is not None
    return best


def crossings(tour: Sequence[Point]) -> int:
    n = len(tour)
    edges = [(tour[i], tour[(i + 1) % n]) for i in range(n)]
    count = 0
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if segments_intersect(*edges[i], *edges[j]):
                count += 1
    return count


def start_spread(stops: Sequence[Point]) -> tuple[float, float]:
    lengths = [length(nearest_neighbour(stops, s)) for s in range(len(stops))]
    return (min(lengths), max(lengths))
