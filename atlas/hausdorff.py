"""Hausdorff distance: how far two shapes stray from each other, ruled by the worst point.

Asking how different two shapes are, two coastlines, two traced
routes, two building footprints, needs a distance between point sets
rather than between points. The Hausdorff distance is the classical
answer and it is built from a max of mins. The directed distance from
set A to set B takes each point of A, finds its nearest point in B,
and reports the largest of those nearest distances: how far the worst
placed point of A is from B. That directed distance is not symmetric,
and the asymmetry is real rather than a technicality. A short curve
lying along part of a long one is close to it in the directed sense,
every point of the short curve has a near neighbor on the long one,
while the long curve is far from the short one, because its far end
has no near neighbor at all. The full Hausdorff distance takes the
larger of the two directed distances and is symmetric, a true metric
on closed sets. Its defining character, worth measuring rather than
describing, is that a single outlier dominates: move one point of one
set far away and the Hausdorff distance jumps to that point's distance
no matter how well the other thousand points match, because it is a
maximum, not an average. That makes it the right measure when the
question is whether every part of one shape is near the other, and the
wrong one when a few stray points should not decide the verdict. The
finding worth stating is that the directed distances differ, sometimes
by a factor of many, the symmetric distance is their max, and one
displaced point sets the whole value, so Hausdorff answers the
worst-case question and only that one. This module computes directed
and symmetric Hausdorff distances between point sets, and a survey
measures the asymmetry and the outlier's dominance.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Point = tuple[float, float]


def _nearest(p: Point, points: list[Point]) -> float:
    return min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in points)


def directed(a: list[Point], b: list[Point]) -> float:
    if a is None or b is None or not a or not b:
        raise Invalid("both point sets must be non-empty")
    return max(_nearest(p, b) for p in a)


def hausdorff(a: list[Point], b: list[Point]) -> float:
    return max(directed(a, b), directed(b, a))
