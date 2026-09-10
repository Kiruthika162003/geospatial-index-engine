"""A grid index's ring search read under two stop rules on two thousand points.

The drill builds a grid index over 2000 uniform points with cells
of 40 units, 3.2 points a cell, and answers 1000 nearest-neighbour
queries under the tempting rule, stop at the first ring that holds
a point, and the safe rule, search on until the best distance is
within the rings' guaranteed reach. The guess before measuring was
that with three points a cell the query's own cell would nearly
always hold the nearest, so the tempting rule would fail on a few
percent of queries at most. The measurement refutes that: the
tempting rule returns the wrong point on 31.1 percent of the
queries, since a point next door is often closer than the one in
the query's own cell, while the safe rule is never wrong and costs
exactly one ring on average. The survey keeps the few-percent guess
beside the 31.
"""

from __future__ import annotations

import random

from atlas.spiralsearch import error_rate, uniform
from atlas.surveys.survey import Survey


def run() -> Survey:
    points = uniform(2000, random.Random(790))
    queries = uniform(1000, random.Random(791))
    read = error_rate(points, 40.0, queries)
    readings = {
        "naive_wrong_share": round(read["naive_wrong"], 4),
        "safe_wrong_share": round(read["safe_wrong"], 4),
        "naive_rings": round(read["naive_rings"], 4),
        "safe_rings": round(read["safe_rings"], 4),
    }
    holds = (
        read["safe_wrong"] == 0.0
        and 0.25 < read["naive_wrong"] < 0.35
        and read["safe_rings"] == 1.0
    )
    return Survey(
        surveyor="ringwatch",
        finding=(
            "the first-ring stop rule returned the wrong nearest neighbour on 31 percent "
            "of 1000 queries at 3.2 points a cell while the safe rule was never wrong at "
            "exactly one ring"
        ),
        readings=readings,
        holds=holds,
    )
