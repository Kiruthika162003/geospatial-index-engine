"""Edge matching loses 2 of 40 pairs to strangers on a sparse edge and 56 of 200 on a dense one.

Lines crossing a map sheet boundary end on both sheets displaced
by digitising jitter, and edge matching pairs the ends within a
tolerance, here greedily nearest first with each end used once.
Forty crossing lines and ten strays a side over 1000 units, with
jitter 0.5 giving a gap between true ends of deviation 0.707, pair
11.5, 19.4, 31.9 and 37.7 of the 40 truly at tolerances 0.25, 0.5,
1 and 2, near the normal law's 40 less 28.9, 19.2, 6.3 and 0.2
missed, with 0.6, 1.0, 1.7 and 2.15 false pairs. The guess that a
tolerance a few deviations wide recovers every pair was wrong: at
tolerances of 5, 10 and 20 the true pairs stay at 38.0 while the
false pairs rise to 2.75, 3.2 and 4.55, since a stranger closer
than the true partner takes the end and the partner is lost, and
the floor of 2 misses never clears. Jitter 2 leaves the floor at
9.2 of 40 with 11.8 false pairs at tolerance 20.

The floor grows with density: 200 crossings and 50 strays a side,
250 ends per 1000 units, pair 143.6 of the 200 at best with jitter
0.5, 56 lost, and 75.2 with jitter 2, 125 lost, while the false
pairs climb to 88 and 156 at tolerance 20; the chance of a stranger
within the tolerance, 1 - e to the minus 2 times density times
tolerance, reads 0.63 at tolerance 2 on the dense edge against
0.18 on the sparse one. Five lines with no strays pair exactly.
"""

from __future__ import annotations

import random
from math import erf, sqrt

from atlas.errors import Invalid

End = tuple[float, str]


def match(left: list[End], right: list[End], tolerance: float) -> list[tuple[int, int]]:
    # greedy nearest pairing of line ends along a shared edge, each end used once
    if tolerance < 0:
        raise Invalid("the tolerance must not be negative")
    pairs = []
    candidates = []
    for i, (y, _) in enumerate(left):
        for j, (z, _) in enumerate(right):
            if abs(y - z) <= tolerance:
                candidates.append((abs(y - z), i, j))
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    for _, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        pairs.append((i, j))
    return pairs


def score(pairs: list[tuple[int, int]], left: list[End], right: list[End]) -> dict[str, int]:
    true_pairs = sum(1 for i, j in pairs if left[i][1] == right[j][1])
    false_pairs = len(pairs) - true_pairs
    shared = len({name for _, name in left} & {name for _, name in right})
    return {"true": true_pairs, "false": false_pairs, "missed": shared - true_pairs}


def sheet_edges(
    crossings: int, strays_each: int, jitter: float, rng: random.Random, length: float = 1000.0
) -> tuple[list[End], list[End]]:
    # crossing lines appear on both sheets displaced by jitter; strays on one sheet only
    if crossings < 0 or strays_each < 0 or jitter < 0:
        raise Invalid("counts and jitter must not be negative")
    left, right = [], []
    for k in range(crossings):
        y = rng.uniform(0, length)
        left.append((y + rng.gauss(0, jitter), f"line{k}"))
        right.append((y + rng.gauss(0, jitter), f"line{k}"))
    for k in range(strays_each):
        left.append((rng.uniform(0, length), f"leftstray{k}"))
        right.append((rng.uniform(0, length), f"rightstray{k}"))
    return left, right


def sweep(
    crossings: int,
    strays_each: int,
    jitter: float,
    tolerances: list[float],
    seed: int,
    draws: int = 20,
) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    for tolerance in tolerances:
        totals = {"true": 0, "false": 0, "missed": 0}
        for k in range(draws):
            left, right = sheet_edges(crossings, strays_each, jitter, random.Random(seed + k))
            for key, value in score(match(left, right, tolerance), left, right).items():
                totals[key] += value
        out[tolerance] = {key: value / draws for key, value in totals.items()}
    return out


def gap_law(jitter: float) -> float:
    # the standard deviation of the gap between a crossing line's two ends
    return jitter * 2**0.5


def stranger_law(density: float, tolerance: float) -> float:
    # the chance a given end has a stranger within the tolerance on the other sheet
    return 1 - 2.718281828459045 ** (-2 * density * tolerance)


def miss_law(jitter: float, tolerance: float) -> float:
    # the share of true pairs whose gap exceeds the tolerance, for a normal gap
    if jitter == 0:
        return 0.0
    return 1 - erf(tolerance / (gap_law(jitter) * sqrt(2)))
