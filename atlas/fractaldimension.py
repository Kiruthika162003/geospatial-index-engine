"""Fractal dimension: box counting on coastlines, and how close it gets to the true value.

A coastline's measured length grows as the ruler shrinks, because
each finer ruler follows bays and headlands the coarser one cut
across, and the rate of growth is the fractal dimension: a straight
line has dimension 1, a filled region 2, and a wiggly coast
something between. Box counting estimates it by laying grids of
box size s over the shape, counting the boxes N(s) the shape
touches, and reading the dimension as the slope of log N against
log(1/s), since a shape of dimension D touches about s to the minus
D boxes. The survey calibrates the estimate on figures whose
dimension is known exactly. The Koch curve, built by replacing each
segment with four of a third the length, has dimension log 4 over
log 3, 1.2619. Box counting on a five-level curve with boxes from
0.25 down to three times the shortest segment read 1.2628, an
error of 0.001; from 0.5 down to the shortest segment it read
1.249 at four, five, and six levels alike, an error of 0.013 that
the extra levels did not remove, since the finest boxes at the
segment scale see straight pieces. A straight line read 0.973 and
a filled square of a million grid points read 1.909 rather than
2.0 with boxes from 0.25 down, the shortfall being an edge effect:
the square's boundary touches an extra row and column of the
coarse boxes, inflating the coarse counts and flattening the
slope, and it fades as the coarsest box shrinks, 1.958 from 0.1,
1.973 from 0.05, 1.986 from 0.02, with the line likewise climbing
to 0.996. A random walk of 20000
steps, whose trace has dimension 2 in the limit, read 1.65. The
survey also records the failure modes rather than hiding them:
boxes finer than the point spacing count points rather than the
curve and the slope fell to 0.32, boxes larger than the whole
shape are stuck at one and the slope fell to 0.0, the raw Koch
vertices without densifying read 0.02 at fine scales, and a range
starting coarser than the curve's features read 1.06. The finding
worth stating is that box counting recovers the Koch dimension to
within 0.001 to 0.03 when the scale range is chosen inside the
curve's features, and drifts by tenths to unity when it is not, so
the estimate is as good as the choice of scales and no better.
This module counts boxes and fits
the dimension, and a survey calibrates it on the Koch curve, a
line, and a square.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from atlas.errors import Invalid

Point = tuple[float, float]


def koch_curve(levels: int, start: Point = (0.0, 0.0), end: Point = (1.0, 0.0)) -> list[Point]:
    if levels < 0:
        raise Invalid("levels cannot be negative")
    points = [start, end]
    for _ in range(levels):
        refined: list[Point] = [points[0]]
        for (x1, y1), (x2, y2) in itertools.pairwise(points):
            dx, dy = (x2 - x1) / 3.0, (y2 - y1) / 3.0
            a = (x1 + dx, y1 + dy)
            b = (x1 + 2 * dx, y1 + 2 * dy)
            # the peak: rotate the middle third by 60 degrees about its start
            cos60, sin60 = 0.5, math.sqrt(3) / 2
            peak = (a[0] + dx * cos60 - dy * sin60, a[1] + dx * sin60 + dy * cos60)
            refined.extend([a, peak, b, (x2, y2)])
        points = refined
    return points


def densify(points: Sequence[Point], step: float) -> list[Point]:
    # add points along each segment so no gap exceeds the step, so boxes see the curve
    if step <= 0:
        raise Invalid("the step must be positive")
    out: list[Point] = [points[0]]
    for (x1, y1), (x2, y2) in itertools.pairwise(points):
        length = math.hypot(x2 - x1, y2 - y1)
        pieces = max(1, math.ceil(length / step))
        for i in range(1, pieces + 1):
            t = i / pieces
            out.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return out


def box_count(points: Sequence[Point], size: float) -> int:
    if size <= 0:
        raise Invalid("the box size must be positive")
    return len({(math.floor(x / size), math.floor(y / size)) for x, y in points})


def fit_dimension(points: Sequence[Point], sizes: Sequence[float]) -> float:
    # least-squares slope of log N against log(1 / s)
    if len(sizes) < 2:
        raise Invalid("need at least two box sizes to fit a slope")
    xs = [math.log(1.0 / s) for s in sizes]
    ys = [math.log(box_count(points, s)) for s in sizes]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise Invalid("the box sizes must differ")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def geometric_sizes(largest: float, smallest: float, count: int) -> list[float]:
    if count < 2 or largest <= smallest or smallest <= 0:
        raise Invalid("sizes need a positive range and at least two entries")
    ratio = (smallest / largest) ** (1.0 / (count - 1))
    return [largest * ratio**i for i in range(count)]


def koch_dimension() -> float:
    return math.log(4) / math.log(3)


def filled_square(side: float, step: float) -> list[Point]:
    n = math.ceil(side / step)
    return [(i * step, j * step) for i in range(n + 1) for j in range(n + 1)]
