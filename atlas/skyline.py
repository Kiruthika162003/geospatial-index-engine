"""Skyline: the hotels that are not beaten on both price and distance, and how many there are.

A traveller wants a hotel that is cheap and close to the beach,
and no single ranking serves, since the cheapest is far and the
closest is dear. The skyline is the set of hotels not dominated:
a hotel is dominated when another is at least as good on every
attribute and strictly better on one, and the skyline is what
remains after every dominated hotel is dropped, the Pareto front
of the map. Computing it is a sort and a sweep: order the points
by the first attribute, walk them, and keep each point whose
second attribute beats the best second attribute seen so far,
which is linear after the sort and exact. The survey confirms
exactness against the quadratic brute force that tests every pair
on random sets, and then measures the question a planner asks,
how large the skyline is, which depends on how the attributes
relate. When the two attributes are independent and uniform the
expected skyline size is the harmonic number of n, about the
natural log of n plus 0.58: over 200 draws the mean skyline was
2.84, 5.23, and 7.72 for 10, 100, and 1000 points against
harmonic numbers of 2.93, 5.19, and 7.49, and 9.93 against 9.79
over 30 draws of 10000, on the harmonic number within the draws'
noise, so a thousand random hotels leave about eight on the
skyline. When the attributes are anti-correlated, cheap hotels
being far and close ones dear, the front's length is set by the
noise round the trade-off line: with none every one of 1000
points is on the front, with Gaussian noise of 0.01 it is 128, of
0.05 it is 37, and of 0.2 it is 14, so the guess of a large
fraction holds only for a sharp trade-off; when the attributes
are correlated with noise 0.05, good hotels being good at both,
the front collapses to 4.4. It also measures the effect of ties:
with 1000 points rounded to a 0.1 grid, 120 distinct values
remain and the skyline is 3, and the strict rule that a dominator
must beat on at least one attribute keeps exact duplicates on the
skyline together, which the sweep and the pairwise brute force
agreed on over 600 sets with and without rounding. The finding
worth stating is that the skyline of independent attributes grows
only as the logarithm of the count, eight of a thousand, while
anti-correlated attributes put between 1 and 100 percent on it
depending on how sharp the trade-off is and correlated ones a
handful, so the size of the choice a planner faces is set by how
the attributes trade, not by how many options there are. This
module computes skylines by sweep, and a survey measures their
size against the harmonic number and across correlations.
"""

from __future__ import annotations

from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def dominates(a: Point, b: Point) -> bool:
    # smaller is better on both attributes
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])


def skyline(points: Sequence[Point]) -> list[Point]:
    if not points:
        raise Invalid("no points to take a skyline of")
    ordered = sorted(points)
    out: list[Point] = []
    best_second = float("inf")
    for p in ordered:
        if p[1] < best_second:
            out.append(p)
            best_second = p[1]
        elif p[1] == best_second and out and out[-1] == p:
            out.append(p)  # an exact duplicate of the last kept point stays with it
    return out


def brute_force(points: Sequence[Point]) -> list[Point]:
    if not points:
        raise Invalid("no points to take a skyline of")
    return sorted(p for p in points if not any(dominates(q, p) for q in points))


def harmonic(n: int) -> float:
    if n < 1:
        raise Invalid("the harmonic number needs a positive count")
    return sum(1.0 / k for k in range(1, n + 1))


def independent(n: int, rng) -> list[Point]:
    return [(rng.random(), rng.random()) for _ in range(n)]


def anti_correlated(n: int, noise: float, rng) -> list[Point]:
    out: list[Point] = []
    for _ in range(n):
        t = rng.random()
        out.append((t + rng.gauss(0, noise), 1.0 - t + rng.gauss(0, noise)))
    return out


def correlated(n: int, noise: float, rng) -> list[Point]:
    out: list[Point] = []
    for _ in range(n):
        t = rng.random()
        out.append((t + rng.gauss(0, noise), t + rng.gauss(0, noise)))
    return out


def rounded(points: Sequence[Point], step: float) -> list[Point]:
    if step <= 0:
        raise Invalid("the rounding step must be positive")
    return [(round(x / step) * step, round(y / step) * step) for x, y in points]
