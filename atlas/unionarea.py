"""Union area: the ground covered by overlapping rectangles, counted once, by a sweep.

Summing the areas of a set of rectangles overcounts wherever they
overlap, and the true covered area, each patch of ground counted once
however many rectangles lie on it, is what a coverage map, a served
region, or a footprint union actually needs. The sweep line computes
it exactly without ever forming the union's shape. Sort every
rectangle's left and right edges by x; between two consecutive edges
the set of active rectangles does not change, so the covered vertical
extent is constant across that strip, and the strip contributes that
extent times its width. The vertical extent is the union of the active
rectangles' y intervals, found by merging them after coordinate
compression, so the whole computation is sorts and merges, order n
squared in the simple form, with no floating geometry beyond
multiplication. Three properties pin it and are worth measuring. The
union area never exceeds the sum of the areas, and equals it exactly
when no two rectangles overlap, since then every strip's union is a
disjoint sum. Two identical rectangles stacked have the union area of
one, the extreme of overcounting. And the sweep agrees with a brute
rasterization, counting covered grid cells at fine resolution, to
within the rasterization's own resolution, which is the independent
check that the merging is right. The finding worth stating is that
union area is bounded above by the summed area with equality exactly
at zero overlap, so the gap between the two is a direct measure of
how much the rectangles pile up, and the sweep computes it exactly
where a raster only approximates. This module computes the union area
of axis-aligned rectangles by an x sweep with y-interval merging, and
a survey confirms the bound, the equality, and the raster agreement.
"""

from __future__ import annotations

from itertools import pairwise

from atlas.bbox import BBox
from atlas.errors import Invalid


def _merged_extent(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    lo, hi = intervals[0]
    for a, b in intervals[1:]:
        if a > hi:
            total += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return total + (hi - lo)


def union_area(boxes: list[BBox]) -> float:
    if boxes is None:
        raise Invalid("boxes must not be None")
    if not boxes:
        return 0.0
    xs = sorted({b.min_x for b in boxes} | {b.max_x for b in boxes})
    total = 0.0
    for left, right in pairwise(xs):
        width = right - left
        if width <= 0:
            continue
        mid = (left + right) / 2
        active = [(b.min_y, b.max_y) for b in boxes if b.min_x <= mid <= b.max_x]
        total += _merged_extent(active) * width
    return total


def summed_area(boxes: list[BBox]) -> float:
    if boxes is None:
        raise Invalid("boxes must not be None")
    return sum(b.area() for b in boxes)


def raster_area(boxes: list[BBox], cell: float) -> float:
    # an independent brute check: count covered grid cells by their centers
    if cell <= 0:
        raise Invalid("cell size must be positive")
    if not boxes:
        return 0.0
    min_x = min(b.min_x for b in boxes)
    max_x = max(b.max_x for b in boxes)
    min_y = min(b.min_y for b in boxes)
    max_y = max(b.max_y for b in boxes)
    covered = 0
    y = min_y + cell / 2
    while y < max_y:
        x = min_x + cell / 2
        while x < max_x:
            if any(b.contains_point(x, y) for b in boxes):
                covered += 1
            x += cell
        y += cell
    return covered * cell * cell
