"""Fat boxes cut index updates to 0.8 v/m a step when straight and (v/m) squared on a walk.

A spatial index of moving objects can hold each object in a box
fattened by a margin and touch the index only when the object
leaves its box, paying for the rest with false candidates. Five
hundred objects over a 1000 square walking straight at speed 1 need
0.158, 0.083, 0.042, 0.0154 and 0.0061 updates per object and step
at margins of 5, 10, 20, 50 and 100, about 0.8 of the law v over m,
0.2 down to 0.01, up to a margin of 50, since an object leaves
through the nearer face along its heading, and 0.6 at 100, where a
run of 200 steps has not settled; at speed 3 the readings are 0.45
to 0.026, the same 0.8 of 0.6 to 0.03. The guess that a random walk leaves its
box at the same rate was wrong by an order: it wanders, and the
updates read 0.0286, 0.0066, 0.0007 and then none at speed 1 and
0.206, 0.060, 0.016, 0.0015 and none at speed 3, falling with the
square of the margin as the law (v over m) squared says, so a
margin of 20 at speed 3 costs an update every 63 steps against
every 8 for straight motion.

The price is the band around every query: a 200-wide query meets
2.3, 4.8, 10.6, 29.6 and 65.8 fat boxes per query beyond its true
hits at the five margins, against the band law n((q + 2m) squared -
q squared) over the extent of 2.05, 4.2, 8.8, 25 and 60, a little
over it because a fat box is centred where the object was at its
last update, not where it is.
"""

from __future__ import annotations

import math
import random

from atlas.bbox import BBox
from atlas.errors import Invalid

Point = tuple[float, float]


class FatBoxIndex:
    # every object holds a box enlarged by a margin; the index is touched only when it leaves it
    def __init__(self, positions: list[Point], margin: float) -> None:
        if margin < 0:
            raise Invalid("the margin must not be negative")
        self.margin = margin
        self.boxes = [self._fat(p) for p in positions]
        self.updates = 0

    def _fat(self, p: Point) -> BBox:
        m = self.margin
        return BBox(p[0] - m, p[1] - m, p[0] + m, p[1] + m)

    def move(self, index: int, p: Point) -> bool:
        if not self.boxes[index].contains_point(p[0], p[1]):
            self.boxes[index] = self._fat(p)
            self.updates += 1
            return True
        return False

    def query(self, box: BBox, positions: list[Point]) -> tuple[list[int], int]:
        # candidates whose fat box meets the query, then the true positions checked
        candidates = [i for i, b in enumerate(self.boxes) if b.intersects(box)]
        hits = [i for i in candidates if box.contains_point(*positions[i])]
        return hits, len(candidates) - len(hits)


def random_walk(
    positions: list[Point], speed: float, rng: random.Random, side: float
) -> list[Point]:
    out = []
    for x, y in positions:
        a = rng.uniform(0, 2 * math.pi)
        nx = min(max(x + speed * math.cos(a), 0.0), side)
        ny = min(max(y + speed * math.sin(a), 0.0), side)
        out.append((nx, ny))
    return out


def straight_walk(
    positions: list[Point], headings: list[float], speed: float, side: float
) -> list[Point]:
    out = []
    for (x, y), a in zip(positions, headings, strict=True):
        out.append(((x + speed * math.cos(a)) % side, (y + speed * math.sin(a)) % side))
    return out


def simulate(
    n: int,
    margin: float,
    speed: float,
    steps: int,
    rng: random.Random,
    straight: bool,
    side: float = 1000.0,
) -> dict[str, float]:
    positions = [(rng.uniform(0, side), rng.uniform(0, side)) for _ in range(n)]
    headings = [rng.uniform(0, 2 * math.pi) for _ in range(n)]
    index = FatBoxIndex(positions, margin)
    query = BBox(400.0, 400.0, 600.0, 600.0)
    false_total = 0
    for _ in range(steps):
        positions = (
            straight_walk(positions, headings, speed, side)
            if straight
            else random_walk(positions, speed, rng, side)
        )
        for i, p in enumerate(positions):
            index.move(i, p)
        _, false = index.query(query, positions)
        false_total += false
    return {
        "updates_per_object_step": index.updates / (n * steps),
        "false_candidates_per_query": false_total / steps,
    }


def update_law_straight(margin: float, speed: float) -> float:
    # an object walking straight leaves a box of half-width m in about m over v steps
    if speed <= 0:
        raise Invalid("the speed must be positive")
    return speed / margin if margin > 0 else 1.0


def update_law_random(margin: float, speed: float) -> float:
    # a random walk needs about (m over v) squared steps to reach the box's edge
    if speed <= 0:
        raise Invalid("the speed must be positive")
    return (speed / margin) ** 2 if margin > 0 else 1.0


def false_candidate_law(
    n: int, margin: float, query_side: float, side: float = 1000.0
) -> float:
    # fat boxes meeting the query beyond the true hits: the band of width m around it
    band = (query_side + 2 * margin) ** 2 - query_side**2
    return n * band / (side * side)
