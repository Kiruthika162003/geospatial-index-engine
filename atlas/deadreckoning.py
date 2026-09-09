"""Dead reckoning: position from heading and speed over time, and how its error grows.

Without a fix from outside, a navigator advances the last known
position by the heading and speed held over each interval, and the
sum of those legs is the reckoned position. The method is exact when
the inputs are, which the survey confirms by reckoning along a
known path and recovering its end to floating precision, and its
value lies in what it does when the inputs are wrong, because the
two kinds of wrongness grow differently. A constant heading bias,
the compass reading two degrees off all the way, bends the whole
track by that angle, so the position error grows in proportion to
distance run: at two degrees it is about 3.5 percent of the
distance, since the chord of a two-degree arc is 2 sin(1 degree)
times the radius. Random heading noise, a fresh error each interval,
partly cancels: the errors add like a random walk, so the position
error grows as the square root of the number of legs, and after a
hundred legs of one-degree noise the track ends about a tenth as
far off as a one-degree bias would leave it. The survey measures
both laws by running many trials: the bias error fraction was
0.01745, 0.0349, and 0.0872 at 1, 2, and 5 degrees, matching the
chord to five figures at 25 and 100 legs alike; the noise error over
500 trials of ten-unit legs was 0.95, 1.84, and 3.64 at 25, 100,
and 400 legs, ratios 1.94 and 1.97 per quadrupling against the
square root's 2, and the ratio of the hundred-leg noise error to
the one-degree bias error was 0.106. The cross-track spread is
where the error lives, 0.95 against 0.04 along track at 25 legs,
the along-track part being second order since a small heading error
barely shortens a leg, though it grows faster, reaching 0.61
against 3.58 by 400 legs. A speed bias, by contrast, puts its error
entirely along track, 20 units for 2 percent over a 1000-unit run
with 2e-12 across, and grows linearly like the heading bias. The
finding worth stating is that a compass
bias costs distance times the angle and never averages out, while
random heading noise costs the square root of the leg count and
mostly cross-track, so a reckoning track is trusted by its
calibration more than by its resolution. This module advances
positions by legs, and a survey measures the exactness and the
two growth laws.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from atlas.errors import Invalid

Point = tuple[float, float]
Leg = tuple[float, float, float]  # heading degrees clockwise from north, speed, duration


def advance(position: Point, heading_deg: float, speed: float, duration: float) -> Point:
    if speed < 0 or duration < 0:
        raise Invalid("speed and duration cannot be negative")
    r = math.radians(heading_deg)
    run = speed * duration
    return (position[0] + run * math.sin(r), position[1] + run * math.cos(r))


def reckon(start: Point, legs: Iterable[Leg]) -> Point:
    position = start
    for heading, speed, duration in legs:
        position = advance(position, heading, speed, duration)
    return position


def track(start: Point, legs: Iterable[Leg]) -> list[Point]:
    positions = [start]
    for heading, speed, duration in legs:
        positions.append(advance(positions[-1], heading, speed, duration))
    return positions


def distance_run(legs: Iterable[Leg]) -> float:
    return sum(speed * duration for _, speed, duration in legs)


def bias_error_fraction(bias_deg: float) -> float:
    # the end error per unit distance under a constant heading bias: the chord of the arc
    return 2.0 * math.sin(math.radians(bias_deg) / 2.0)


def with_heading_bias(legs: Sequence[Leg], bias_deg: float) -> list[Leg]:
    return [(h + bias_deg, s, d) for h, s, d in legs]


def with_heading_noise(legs: Sequence[Leg], sigma_deg: float, rng) -> list[Leg]:
    return [(h + rng.gauss(0, sigma_deg), s, d) for h, s, d in legs]


def end_error(start: Point, truth: Sequence[Leg], perturbed: Sequence[Leg]) -> float:
    tx, ty = reckon(start, truth)
    px, py = reckon(start, perturbed)
    return math.hypot(px - tx, py - ty)


def split_error(
    start: Point, truth: Sequence[Leg], perturbed: Sequence[Leg]
) -> tuple[float, float]:
    # the end error resolved along and across the true track's overall direction
    tx, ty = reckon(start, truth)
    px, py = reckon(start, perturbed)
    dx, dy = tx - start[0], ty - start[1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        raise Invalid("the true track does not move; no direction to resolve along")
    ux, uy = dx / norm, dy / norm
    ex, ey = px - tx, py - ty
    along = ex * ux + ey * uy
    across = ex * -uy + ey * ux
    return (along, across)
