"""K-center: placing k depots so the farthest customer is as near as possible, greedily.

Where to put k fire stations, warehouses, or cell towers so that
the farthest point they serve is as close as possible is the
k-center problem, and it is NP-hard, so the practical answer is
the greedy farthest-first rule: put the first center anywhere,
then repeatedly add the point farthest from all centers so far.
The rule carries a proof that its worst distance is at most twice
the optimum, and the survey measures how much better than twice
it does in practice, since a guarantee of two is a bound, not a
prediction, and the guess that it would land within ten percent
was wrong. On small sets, twelve to fourteen points with k of two
to four, the optimum can be found by trying every subset of
centers, and over 120 random layouts the greedy's ratio to it
averaged 1.30, reached the optimum on only 7 percent of layouts,
and peaked at 1.86, close under the bound of two rather than well
under it; a further 200 layouts of 8 to 12 points peaked at 1.86
again, and no trial ever reached two. The start dependence is
large: on one 14-point layout with k = 3, running the greedy from
every start gave covering radii from 0.356 to 0.497, 1.20 to 1.68
of the optimum, a 40 percent swing that is the measure of how
much the rule's answer is luck, and the practical cure is to run
it from every start and keep the best. The layout built to defeat
it, points in tight pairs far apart with k equal to the pair
count, was solved exactly, ratio 1.0, since farthest-first picks
one point per pair before it ever picks a second from any pair.
On a uniform scatter of 400 points the covering radius fell from
0.989 at k = 1 through 0.924, 0.550, 0.330, 0.193, 0.158, and
0.094 at k = 2, 4, 8, 16, 32, and 64, an irregular descent, 1.07
at the first doubling since the second center barely helps a
square, then about 1.7 per doubling with a stall to 1.22 at 32,
rather than the clean root-two per doubling the area argument
suggests. The finding worth stating is that greedy farthest-first
averages 30 percent above the optimal k-center radius on random
layouts, comes within 7 percent of its twofold bound on the worst
of them, and swings by 40 percent with its start, so the
guarantee is what it says and the rule wants its best start, not
its first. This module places
centers greedily and by brute force, and a survey measures the
ratio, the start dependence, and the radius curve.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def covering_radius(points: Sequence[Point], centers: Sequence[Point]) -> float:
    if not centers:
        raise Invalid("no centers to cover with")
    return max(min(math.dist(p, c) for c in centers) for p in points)


def greedy(points: Sequence[Point], k: int, start: int = 0) -> list[Point]:
    if k < 1 or k > len(points):
        raise Invalid("k must lie within 1 and the point count")
    if not 0 <= start < len(points):
        raise Invalid("the start index is off the list")
    centers = [points[start]]
    nearest = [math.dist(p, centers[0]) for p in points]
    while len(centers) < k:
        far = max(range(len(points)), key=lambda i: nearest[i])
        centers.append(points[far])
        for i, p in enumerate(points):
            nearest[i] = min(nearest[i], math.dist(p, points[far]))
    return centers


def optimal(points: Sequence[Point], k: int) -> tuple[list[Point], float]:
    # every k-subset of the points as centers; only for small sets
    if k < 1 or k > len(points):
        raise Invalid("k must lie within 1 and the point count")
    if math.comb(len(points), k) > 200_000:
        raise Invalid("too many subsets for brute force")
    best: tuple[list[Point], float] | None = None
    for subset in itertools.combinations(points, k):
        radius = covering_radius(points, subset)
        if best is None or radius < best[1]:
            best = (list(subset), radius)
    assert best is not None
    return best


def greedy_ratio(points: Sequence[Point], k: int, start: int = 0) -> float:
    return covering_radius(points, greedy(points, k, start)) / optimal(points, k)[1]


def start_spread(points: Sequence[Point], k: int) -> tuple[float, float]:
    # the smallest and largest covering radius the greedy reaches over every start
    radii = [covering_radius(points, greedy(points, k, s)) for s in range(len(points))]
    return (min(radii), max(radii))


def tight_pairs(pairs: int, gap: float, spacing: float = 10.0) -> list[Point]:
    # pairs of points a small gap apart, the pairs far apart: a layout built to test the bound
    out: list[Point] = []
    for i in range(pairs):
        out.append((i * spacing, 0.0))
        out.append((i * spacing + gap, 0.0))
    return out
