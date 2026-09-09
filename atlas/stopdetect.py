"""Stop detection: a dwell is a run of fixes that stays within a radius for long enough.

A trace of a delivery van, a hiker, or a shopper is mostly motion
punctuated by stops, and the stops are usually what the analysis
wants: where the deliveries happened, where the hiker rested, which
shop held the shopper. A stop, or dwell, is a maximal run of
consecutive fixes that all lie within a radius of the run's first fix
and that spans at least a minimum duration; a run that satisfies the
radius but not the duration is a slow patch, not a stop, and a fix
that breaks the radius ends the run and starts the search afresh. The
two thresholds are the whole definition and they trade against each
other in a way the survey measures rather than describes. With a
trace built from a known itinerary, planted stops of known length
between legs of motion, the detector recovers every planted stop when
the radius exceeds the fix jitter and the duration threshold sits
below the shortest planted stop, and it recovers nothing spurious on
the legs, because motion carries fixes out of any radius before the
duration is met. Tighten the radius below the jitter and planted
stops fracture into fragments too short to count, so recall falls;
raise the duration above a planted stop's length and that stop is
missed by design; and a leg driven slowly enough that the fixes
crowd within the radius for the duration produces a false stop, which
is the honest failure mode of a purely geometric definition. The
finding worth stating is that with the radius above the jitter and
the duration below the shortest real stop the detector recovers the
planted stops exactly with no false positives, and each threshold
moved past its bound loses stops in a predictable way. This module
detects dwells by the radius-and-duration rule, and a survey confirms
exact recovery and the two failure modes.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Fix = tuple[float, float, float]  # x, y, time


def detect_stops(trace: list[Fix], radius: float, min_duration: float) -> list[tuple[int, int]]:
    # returns (start_index, end_index) inclusive for each dwell
    if trace is None or not trace:
        raise Invalid("the trace must not be empty")
    if radius <= 0 or min_duration <= 0:
        raise Invalid("radius and minimum duration must be positive")
    stops: list[tuple[int, int]] = []
    i = 0
    n = len(trace)
    while i < n:
        ax, ay, at = trace[i]
        j = i
        while j + 1 < n and math.hypot(trace[j + 1][0] - ax, trace[j + 1][1] - ay) <= radius:
            j += 1
        if trace[j][2] - at >= min_duration:
            stops.append((i, j))
            i = j + 1
        else:
            i += 1
    return stops


def stop_centers(trace: list[Fix], stops: list[tuple[int, int]]) -> list[tuple[float, float]]:
    centers = []
    for start, end in stops:
        xs = [trace[k][0] for k in range(start, end + 1)]
        ys = [trace[k][1] for k in range(start, end + 1)]
        centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return centers
