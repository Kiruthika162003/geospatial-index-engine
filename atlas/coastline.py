"""Coastline paradox: measuring a coast with dividers, and reading its dimension from the ruler.

How long is the coast of Britain depends on the ruler, and
Richardson measured it with dividers: set the dividers to a
span, walk them along the coast counting steps, and the length
is the span times the count, which grows as the span shrinks
because a shorter span follows inlets a longer one strides
across. On a log-log plot of length against span the points fall
on a line whose slope is one minus the fractal dimension, so a
smooth coast has slope zero, its length settling to a limit, and
a fractal one has a negative slope with no limit at all. The
module walks dividers along a polyline exactly, finding each
next step by intersecting the circle of the span with the line
ahead, and the survey calibrates the walk on curves whose
dimension is known. On a unit circle of 3600 segments the walk
took 6, 12, 62, and 314 steps at spans of 1, 0.5, 0.1, and 0.02,
the chord count the arcsine predicts, lengths of 6.0, 6.0, 6.2,
and 6.28 rising toward the circumference of 6.283 and a slope of
-0.013, near zero. On the six-level Koch curve the length at
span s is the classical staircase: at spans of a third to the
first through fifth powers the walk took exactly 4, 16, 64, 256,
and 1024 steps with leftovers of 1e-16, and the slope fitted at
those spans read -0.2619, log 4 over log 3 less one to four
places. At spans between the powers, where the dividers land
part-way along segments, the staircase shows: spans of 0.5 and
0.25 both read length 1.0, 0.15 read 1.2, and 0.02 and 0.01 both
read 2.56, so the lengths sit 8 to 30 percent below the fitted
line and a fit through such spans read -0.280 rather than
-0.262, so a Richardson slope is only as good as the spans it is
read at. On a random walk of 5000 steps the slope read -1.27,
steeper than the guessed minus one, since a walk's trace fills
the plane and the divider length grows faster than one over the
span. The first version of the walk lost every step that landed
exactly on a vertex, reading 5 steps round the circle at span 1
and zero on the Koch curve at a twenty-seventh, until the crossing
test was moved to the far endpoint's distance with the root
clamped. The finding worth stating is that divider walking
recovers the Koch dimension from the slope at power-of-three
spans to four places and reads a circle as dimension one, that
the staircase between the Koch scales pulls a careless fit off by
0.02, and that the length of a fractal coast is a function of the
ruler with no limit. This
module walks dividers along polylines, and a survey measures the
Richardson slope on a circle, the Koch curve, and a walk.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def _circle_line_exit(center: Point, span: float, a: Point, b: Point) -> float:
    # the t in [0, 1] where the segment a-b, starting inside the circle of radius span and
    # ending on or outside it, crosses the circle; the larger root, clamped against rounding
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - center[0], a[1] - center[1]
    qa = dx * dx + dy * dy
    if qa == 0:
        return 1.0
    qb = 2 * (fx * dx + fy * dy)
    qc = fx * fx + fy * fy - span * span
    disc = max(0.0, qb * qb - 4 * qa * qc)
    t = (-qb + math.sqrt(disc)) / (2 * qa)
    return max(0.0, min(1.0, t))


def divider_walk(line: Sequence[Point], span: float) -> tuple[int, float]:
    # steps of exactly `span` along the line and the leftover distance at the end
    if span <= 0:
        raise Invalid("the span must be positive")
    if len(line) < 2:
        raise Invalid("a line needs at least two points")
    current = line[0]
    seg = 0
    steps = 0
    while True:
        found = False
        for i in range(seg, len(line) - 1):
            a = current if i == seg else line[i]
            b = line[i + 1]
            # the far end reaches the span: the crossing lies on this segment
            if math.dist(current, b) >= span * (1 - 1e-12):
                t = _circle_line_exit(current, span, a, b)
                current = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                seg = i
                steps += 1
                found = True
                break
        if not found:
            break
    leftover = sum(
        math.dist(line[i], line[i + 1]) for i in range(seg + 1, len(line) - 1)
    ) + math.dist(current, line[seg + 1])
    return steps, leftover


def divider_length(line: Sequence[Point], span: float) -> float:
    steps, _ = divider_walk(line, span)
    return steps * span


def richardson_slope(line: Sequence[Point], spans: Sequence[float]) -> float:
    # the slope of log length against log span; one minus the dimension
    if len(spans) < 2:
        raise Invalid("need at least two spans")
    xs = [math.log(s) for s in spans]
    ys = [math.log(divider_length(line, s)) for s in spans]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def circle(points: int, radius: float = 1.0) -> list[Point]:
    angles = [2 * math.pi * k / points for k in range(points + 1)]
    return [(radius * math.cos(a), radius * math.sin(a)) for a in angles]


def chord_length(steps: int, radius: float, span: float) -> float:
    # the length a divider walk of `steps` along a circle reads: steps times the chord
    return steps * span if span <= 2 * radius else 0.0


def koch(levels: int) -> list[Point]:
    points: list[Point] = [(0.0, 0.0), (1.0, 0.0)]
    for _ in range(levels):
        refined: list[Point] = [points[0]]
        for (x1, y1), (x2, y2) in itertools.pairwise(points):
            dx, dy = (x2 - x1) / 3, (y2 - y1) / 3
            a = (x1 + dx, y1 + dy)
            half_root3 = math.sqrt(3) / 2
            peak = (a[0] + dx / 2 - dy * half_root3, a[1] + dx * half_root3 + dy / 2)
            refined.extend([a, peak, (x1 + 2 * dx, y1 + 2 * dy), (x2, y2)])
        points = refined
    return points
