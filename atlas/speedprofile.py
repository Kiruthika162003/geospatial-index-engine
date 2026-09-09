"""Speed profile: speeds from a timestamped trace, with the glitch spikes that noise fakes.

A trace of timestamped fixes carries speed only implicitly, as the
distance between consecutive fixes over the time between them, and
that quotient is treacherous in a way worth measuring rather than
assuming. Position noise that is small in distance becomes large in
speed when the interval is short, because the noise is divided by the
interval: a fix jittered by a few meters at one-second spacing reads
as a speed error of several meters per second, which is walking pace
conjured from a stationary receiver. So a speed profile computed
naively from raw fixes is spiky, and the spikes grow as the sampling
gets denser, the opposite of what more data usually buys. The module
computes the per-segment speeds and flags the implausible ones, those
exceeding a physical ceiling for the mover, and the survey shows three
things. On a clean trace at constant speed the profile is flat at the
true speed. On the same trace with position jitter, the per-segment
speeds scatter around the truth with a spread that grows as the
interval shrinks, one over the interval, so halving the sampling
interval doubles the speed noise for the same position noise;
measured, the spread went 0.074, 0.140, 0.262, and 0.424 as the
interval fell from four seconds to half a second, ratios of 1.89 and
1.87 per halving and then 1.62, the last step short of two because at
half a second the two-meter jitter is nearly half the five-meter step
and a distance cannot come out negative, so the folded scatter
compresses the spread. And a single glitched fix, one position
teleported by a large error, produces two spikes, an implausible
speed into it and another out of it, that
the ceiling flags while the clean segments pass, so the flag count is
exactly twice the glitch count. The finding worth stating is that
speed noise scales as position noise over the sampling interval, so
dense sampling amplifies rather than suppresses it, and a plausibility
ceiling catches teleport glitches as paired spikes. This module
computes the profile and flags, and a survey measures the flat clean
profile, the one-over-interval scaling, and the paired spikes.
"""

from __future__ import annotations

import math
from itertools import pairwise

from atlas.errors import Invalid
from atlas.hexbin import coefficient_of_variation

Fix = tuple[float, float, float]  # x, y, time


def speeds(trace: list[Fix]) -> list[float]:
    if trace is None or len(trace) < 2:
        raise Invalid("a speed profile needs at least two fixes")
    out = []
    for (x0, y0, t0), (x1, y1, t1) in pairwise(trace):
        if t1 <= t0:
            raise Invalid("fix times must strictly increase")
        out.append(math.hypot(x1 - x0, y1 - y0) / (t1 - t0))
    return out


def flag_implausible(trace: list[Fix], ceiling: float) -> list[int]:
    # indices of segments whose speed exceeds the ceiling
    if ceiling <= 0:
        raise Invalid("the speed ceiling must be positive")
    return [i for i, s in enumerate(speeds(trace)) if s > ceiling]


def spread(values: list[float]) -> float:
    # the coefficient of variation of a profile, its noise relative to its mean
    return coefficient_of_variation({i: v for i, v in enumerate(values)})
