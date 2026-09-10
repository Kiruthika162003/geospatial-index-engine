"""A Morton-order sweep join read for recall against a brute-force join on a thousand points.

The drill sorts 1000 uniform points by Morton key and pairs each
with the next 8 in key order, keeping the pairs within 10 units,
then reads the result against every pair the brute-force join
finds. The guess before measuring was that the pairs the sweep
misses are the ones straddling the curve's central seam, the
quadrant boundary that the Z order jumps across, so the miss share
would match the seam share. The measurement refutes that: the seam
holds 1.3 percent of the true pairs while the sweep misses 5.2
percent, four times as many, because every level of the curve has
its own seams and the lower ones are far more numerous. The survey
keeps the seam guess beside the two shares and checks that the
sweep invents no pair.
"""

from __future__ import annotations

import random

from atlas.mortonjoin import brute_join, recall, seam_share, sweep_join, uniform
from atlas.surveys.survey import Survey


def run() -> Survey:
    points = uniform(1000, random.Random(811))
    found, pairs, compares = recall(points, 1000.0, 10.0, 8)
    seam = seam_share(points, 1000.0, 10.0)
    invented = len(sweep_join(points, 1000.0, 10.0, 8) - brute_join(points, 10.0))
    readings = {
        "true_pairs": pairs,
        "recall_at_window_8": round(found, 4),
        "missed_share": round(1 - found, 4),
        "seam_share": round(seam, 4),
        "comparisons": compares,
        "invented_pairs": invented,
    }
    holds = pairs == 153 and 0.94 < found < 0.96 and (1 - found) > 3 * seam and invented == 0
    return Survey(
        surveyor="joinwatch",
        finding=(
            "a Morton sweep with a window of 8 found 94.8 percent of the 153 pairs within "
            "10 units and invented none, missing four times the 1.3 percent that straddle "
            "the central seam"
        ),
        readings=readings,
        holds=holds,
    )
