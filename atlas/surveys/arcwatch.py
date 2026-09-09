"""Two hundred great-circle arcs, their Clairaut boxes checked against dense sampling.

The drill draws two hundred random arcs between latitudes -70 and
70, computes each arc's bounding box from Clairaut's constant, and
compares it with the box of a thousand points sampled along the
arc, which is the plain truth. The guess before measuring was that
the endpoint box would do for most routes and the vertex bulge
would matter only for a few polar ones. The measurement refutes
that: the Clairaut box matched the sampled box within the sampling
step, 0.002 degrees, on every arc, and the endpoint box missed the
arc by more than a degree of latitude on 70 of the 200, about a
third, by 44 degrees on the worst, so a spatial index filed on
endpoint boxes would lose a large share of long routes. The survey
keeps the endpoints-will-do guess beside the measured miss rate.
"""

from __future__ import annotations

import random

from atlas.arcbbox import arc_box, endpoint_box, sampled_box
from atlas.surveys.survey import Survey


def run() -> Survey:
    rng = random.Random(225)
    worst_gap = 0.0
    missed = 0
    worst_miss = 0.0
    for _ in range(200):
        a = (rng.uniform(-70, 70), rng.uniform(-180, 180))
        b = (rng.uniform(-70, 70), rng.uniform(-180, 180))
        box, sampled, plain = arc_box(a, b), sampled_box(a, b, 1000), endpoint_box(a, b)
        worst_gap = max(worst_gap, abs(box[0] - sampled[0]), abs(box[2] - sampled[2]))
        miss = max(sampled[2] - plain[2], plain[0] - sampled[0])
        missed += miss > 1.0
        worst_miss = max(worst_miss, miss)
    readings = {
        "arcs": 200,
        "worst_box_gap_deg": round(worst_gap, 6),
        "endpoint_box_misses_over_1_deg": missed,
        "worst_endpoint_miss_deg": round(worst_miss, 2),
    }
    holds = worst_gap < 1e-2 and missed > 40
    return Survey(
        surveyor="arcwatch",
        finding=(
            "Clairaut boxes matched dense sampling within the sampling step on all "
            "200 arcs while endpoint boxes missed about a third by over a degree, "
            "44 degrees at worst"
        ),
        readings=readings,
        holds=holds,
    )
