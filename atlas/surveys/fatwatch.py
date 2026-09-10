"""Fat boxes around straight and wandering movers read for index updates against the laws.

The drill moves 500 objects over a 1000 square for 200 steps, each
held in a box fattened by a margin of 20, and counts how often an
object leaves its box, once at speed 3 walking straight and once at
speed 3 in a random walk. The guess before measuring was that the
walk leaves its box about as often as the straight mover, since it
covers the same ground each step. The measurement refutes that by
an order: the straight mover leaves 0.126 times per object and
step, 0.84 of the law v over m, while the walk leaves 0.016 times,
an eighth as often, 7.95 times fewer, near the law (v over m) squared, because it
wanders about its box instead of crossing it. The survey keeps the
same-rate guess beside the two readings.
"""

from __future__ import annotations

import random

from atlas.bboxcache import simulate, update_law_random, update_law_straight
from atlas.surveys.survey import Survey


def run() -> Survey:
    straight = simulate(500, 20.0, 3.0, 200, random.Random(800), True)
    walk = simulate(500, 20.0, 3.0, 200, random.Random(800), False)
    readings = {
        "straight_updates": round(straight["updates_per_object_step"], 4),
        "straight_law": round(update_law_straight(20.0, 3.0), 4),
        "walk_updates": round(walk["updates_per_object_step"], 4),
        "walk_law": round(update_law_random(20.0, 3.0), 4),
        "ratio_straight_over_walk": round(
            straight["updates_per_object_step"] / walk["updates_per_object_step"], 2
        ),
    }
    holds = (
        0.8 < straight["updates_per_object_step"] / update_law_straight(20.0, 3.0) < 0.9
        and walk["updates_per_object_step"] < update_law_random(20.0, 3.0)
        and straight["updates_per_object_step"] > 6 * walk["updates_per_object_step"]
    )
    return Survey(
        surveyor="fatwatch",
        finding=(
            "a straight mover left its fat box 0.126 times per step at 0.84 of v over m "
            "while a random walk of the same speed left it 0.016 times, an eighth as often"
        ),
        readings=readings,
        holds=holds,
    )
