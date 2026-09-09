"""Fair capacities at no slack raise the mean trip 61 percent uniform, 94 percent clustered.

Nearest-facility allocation ignores capacity: eight random
facilities over a 100 square take 531, 529, 191, 123, 161, 41, 36
and 388 of 2000 uniform points at a mean trip of 22.55, and 872 and
703 at two of them for clustered demand at 21.07. Give every
facility a fair share, 250 with no slack, and a greedy allocation
that serves each point at its nearest facility with room reads a
mean trip of 36.21 on the uniform demand, 61 percent more, sending
946 points past their nearest, and 40.88 on the clustered demand,
94 percent more, sending 1233. Slack of 1.1, 1.25, 1.5 and 2 times
the fair share brings the uniform mean to 32.78, 29.79, 25.52 and
22.60 and the clustered mean to 38.27, 35.48, 32.08 and 26.69, so
even double the fair share leaves the clustered demand 27 percent
over its free mean.

The guess that serving the points with the largest regret first,
the gap between their first and second choice, is always best held
on uniform demand, 36.21 against 37.12 for nearest-first and 37.88
for input order at no slack, and failed on clustered demand from
slack 1.25 on, where nearest-first reads 35.31, 30.73 and 26.07
against regret's 35.48, 32.08 and 26.69. Swapping pairs of points
between facilities while the summed trip falls improves the greedy
uniform result to 31.97, 28.96, 26.34, 23.51 and 22.58 at the five
slacks with 1585 down to 45 swaps, keeping every load, and sends
more points past their nearest while travelling less. On 500 points
the same swaps bring 34.90 to 30.58 at no slack in 422 swaps and
24.85 to 23.24 at slack 1.5 in 156, in a quarter of a second.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ranked(points: list[Point], facilities: list[Point]) -> list[list[int]]:
    if not facilities:
        raise Invalid("at least one facility is needed")
    return [
        sorted(range(len(facilities)), key=lambda f: _dist(p, facilities[f])) for p in points
    ]


def nearest(points: list[Point], facilities: list[Point]) -> list[int]:
    return [order[0] for order in ranked(points, facilities)]


def greedy(
    points: list[Point],
    facilities: list[Point],
    capacities: list[int],
    priority: str = "regret",
) -> list[int]:
    if len(capacities) != len(facilities):
        raise Invalid("one capacity per facility")
    if sum(capacities) < len(points):
        raise Invalid("the capacities must hold every point")
    if priority not in ("regret", "nearest", "input"):
        raise Invalid("priority must be regret, nearest or input")
    orders = ranked(points, facilities)
    if priority == "regret":
        keys = []
        for i, order in enumerate(orders):
            first = _dist(points[i], facilities[order[0]])
            second = _dist(points[i], facilities[order[1]]) if len(order) > 1 else first
            keys.append(-(second - first))
        sequence = sorted(range(len(points)), key=lambda i: keys[i])
    elif priority == "nearest":
        sequence = sorted(
            range(len(points)), key=lambda i: _dist(points[i], facilities[orders[i][0]])
        )
    else:
        sequence = list(range(len(points)))
    room = list(capacities)
    assignment = [-1] * len(points)
    for i in sequence:
        for f in orders[i]:
            if room[f] > 0:
                room[f] -= 1
                assignment[i] = f
                break
    return assignment


def mean_distance(points: list[Point], facilities: list[Point], assignment: list[int]) -> float:
    if len(assignment) != len(points):
        raise Invalid("one assignment per point")
    return sum(_dist(p, facilities[a]) for p, a in zip(points, assignment, strict=True)) / len(
        points
    )


def loads(assignment: list[int], count: int) -> list[int]:
    out = [0] * count
    for a in assignment:
        out[a] += 1
    return out


def displaced(points: list[Point], facilities: list[Point], assignment: list[int]) -> int:
    first = nearest(points, facilities)
    return sum(1 for a, b in zip(assignment, first, strict=True) if a != b)


def improve(
    points: list[Point], facilities: list[Point], assignment: list[int], rounds: int = 5
) -> tuple[list[int], int]:
    # swap pairs of points between facilities while the summed distance falls
    current = list(assignment)
    swaps = 0
    for _ in range(rounds):
        changed = False
        by_facility: dict[int, list[int]] = {}
        for i, a in enumerate(current):
            by_facility.setdefault(a, []).append(i)
        for i in range(len(points)):
            fi = current[i]
            best_gain, best_j = 1e-12, -1
            for fj, members in by_facility.items():
                if fj == fi:
                    continue
                for j in members:
                    before = _dist(points[i], facilities[fi]) + _dist(points[j], facilities[fj])
                    after = _dist(points[i], facilities[fj]) + _dist(points[j], facilities[fi])
                    if before - after > best_gain:
                        best_gain, best_j = before - after, j
            if best_j >= 0:
                fj = current[best_j]
                by_facility[fi].remove(i)
                by_facility[fj].remove(best_j)
                by_facility[fi].append(best_j)
                by_facility[fj].append(i)
                current[i], current[best_j] = fj, fi
                swaps += 1
                changed = True
        if not changed:
            break
    return current, swaps


def fair_capacities(n: int, count: int, slack: float) -> list[int]:
    if slack < 1.0:
        raise Invalid("slack must be at least 1")
    each = math.ceil(n * slack / count)
    return [each] * count


def uniform(n: int, rng: random.Random, side: float = 100.0) -> list[Point]:
    return [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]


def clustered(n: int, rng: random.Random, side: float = 100.0) -> list[Point]:
    out = []
    for _ in range(n):
        if rng.random() < 0.6:
            out.append((rng.gauss(30, 8), rng.gauss(30, 8)))
        else:
            out.append((rng.uniform(0, side), rng.uniform(0, side)))
    return out
