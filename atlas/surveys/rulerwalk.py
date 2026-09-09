"""Dividers walked along the Koch curve at five spans, the step counts and the slope read.

The drill builds a six-level Koch curve and walks dividers along it
at spans of a third to the first through fifth powers, counting the
steps, then fits the Richardson slope of log length against log
span. The guess before measuring was that a divider walk on a
polyline would drift from the ideal staircase by a step or two at
each span, since the dividers land part-way along segments and
rounding accumulates over a thousand steps. The measurement refutes
that: the walk took exactly 4, 16, 64, 256, and 1024 steps, the
powers of four the construction demands, with leftovers at the
level of rounding, and the fitted slope read one minus the Koch
dimension to four places. The survey keeps the drift guess beside
the exact counts, because the walk finds each step by intersecting
a circle with the line ahead rather than by accumulating lengths,
so nothing accumulates.
"""

from __future__ import annotations

import math

from atlas.coastline import divider_walk, koch, richardson_slope
from atlas.surveys.survey import Survey


def run() -> Survey:
    curve = koch(6)
    counts = []
    worst_leftover = 0.0
    for p in (1, 2, 3, 4, 5):
        steps, leftover = divider_walk(curve, 3.0**-p)
        counts.append(steps)
        worst_leftover = max(worst_leftover, leftover)
    slope = richardson_slope(curve, [3.0**-p for p in (1, 2, 3, 4, 5)])
    expected = 1 - math.log(4) / math.log(3)
    readings = {
        "steps_at_powers_of_a_third": counts,
        "powers_of_four": [4**p for p in (1, 2, 3, 4, 5)],
        "worst_leftover": worst_leftover,
        "richardson_slope": round(slope, 4),
        "one_minus_koch_dimension": round(expected, 4),
    }
    holds = counts == [4, 16, 64, 256, 1024] and abs(slope - expected) < 1e-3
    return Survey(
        surveyor="rulerwalk",
        finding=(
            "divider walks on the Koch curve took exactly 4, 16, 64, 256, and 1024 "
            "steps at powers of a third and fitted a slope of -0.2619, the Koch "
            "dimension to four places"
        ),
        readings=readings,
        holds=holds,
    )
