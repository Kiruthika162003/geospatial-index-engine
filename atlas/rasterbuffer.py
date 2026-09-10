"""A raster buffer undershoots the true ring by a third when the ring is a cell wide, not over.

A raster buffer keeps the cells whose centres lie within the
distance of a set cell's centre, read here from the exact distance
transform. The guess before measuring was that the rule overshoots,
reaching half a cell past the true edge on average, 2.9 percent of
a disc of radius 30 buffered by 5 at cells of 1 unit. The
measurement found the opposite sign at every cell and distance: the
buffered disc reads 8.1, 6.7 and 2.0 percent under the true area at
cells of 4 units for distances of 5, 10 and 20, 3.0, 0.6 and 0.9
percent under at cells of 2, 0.71, 0.67 and 0.62 under at cells of
1, and 0.45, 0.44 and 0.40 under at cells of 0.5, since the distance
is measured from cell centres that sit inside the shape's true edge
and the rasterised disc itself is already within 0.3 percent of its
area. The ring alone, the buffer less the disc, fares worst when it
is thin: 31, 15.6 and 3.2 percent under at cells of 4, 12.2, 1.8 and
1.6 under at cells of 2, 2.1, 1.2 and 0.85 under at cells of 1, and
1.2, 0.78 and 0.53 under at cells of 0.5.

A one-cell line buffered by 5 and 10 reads 12 and 20 wide at cells
of 4 against the 14 and 24 that two distances and the line's own
cell would give, since only whole cells within the distance count,
and exactly 11 and 21 at cells of 1.
"""

from __future__ import annotations

import math

from atlas.distancetransform import exact
from atlas.errors import Invalid

Mask = list[list[bool]]


def disc_mask(size: int, cell: float, radius: float) -> Mask:
    if size < 3 or cell <= 0 or radius <= 0:
        raise Invalid("the grid needs three cells, a positive cell and a positive radius")
    centre = (size - 1) / 2 * cell
    return [
        [
            math.hypot(
                (c + 0.5) * cell - cell / 2 - centre, (r + 0.5) * cell - cell / 2 - centre
            )
            <= radius
            for c in range(size)
        ]
        for r in range(size)
    ]


def raster_buffer(mask: Mask, distance: float, cell: float) -> Mask:
    # cells whose centre lies within the distance of a set cell's centre
    if distance < 0 or cell <= 0:
        raise Invalid("distance must not be negative and the cell must be positive")
    if not any(any(row) for row in mask):
        raise Invalid("the mask is empty")
    dist = exact(mask)
    return [[d * cell <= distance for d in row] for row in dist]


def mask_area(mask: Mask, cell: float) -> float:
    return sum(sum(1 for v in row if v) for row in mask) * cell * cell


def true_buffered_disc(radius: float, distance: float) -> float:
    return math.pi * (radius + distance) ** 2


def ring_area(radius: float, distance: float) -> float:
    return true_buffered_disc(radius, distance) - math.pi * radius * radius


def buffer_error(size: int, cell: float, radius: float, distance: float) -> dict[str, float]:
    mask = disc_mask(size, cell, radius)
    buffered = raster_buffer(mask, distance, cell)
    disc = mask_area(mask, cell)
    total = mask_area(buffered, cell)
    return {
        "disc_error": disc / (math.pi * radius * radius) - 1,
        "buffer_error": total / true_buffered_disc(radius, distance) - 1,
        "ring_error": (total - disc) / ring_area(radius, distance) - 1,
    }


def overshoot_guess(radius: float, distance: float, cell: float) -> float:
    # the guess before measuring: the centre-to-centre rule reaches half a cell past the true
    # edge, so the buffer overshoots; the measurement found the opposite sign
    outer = radius + distance
    return ((outer + cell / 2) ** 2 - outer**2) / outer**2


def line_mask(size: int, row: int) -> Mask:
    if not 0 <= row < size:
        raise Invalid("the line must lie on the grid")
    return [[r == row for _ in range(size)] for r in range(size)]


def line_buffer_width(size: int, row: int, distance: float, cell: float) -> float:
    buffered = raster_buffer(line_mask(size, row), distance, cell)
    return sum(1 for r in range(size) if buffered[r][size // 2]) * cell
