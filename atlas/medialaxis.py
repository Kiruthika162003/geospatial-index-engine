"""A raster medial axis of a disc keeps 74 percent of its cells, and one notch sprouts 169 more.

The medial axis of a shape is the set of centres of maximal inscribed
discs. On a raster the exact distance transform gives each inside
cell the radius of its disc, and a cell is on the axis when no
neighbour's disc contains its own, that is when no neighbour's
distance reaches its own distance plus the step between them. On a
100 by 41 rectangle this reads 140 cells: the 59-cell spine and four
diagonal arms of 20, one component, two junctions and four endpoints,
with a reach of 21. An even height doubles the spine, since two rows
tie for the largest disc: 100 by 40 reads 200 cells and 100 by 10
reads 200 as well. A 61 by 61 square reads 121 cells, an X of four
arms meeting at one junction; 60 by 60 reads 120 with a four-cell
junction. A line one cell thick is its own axis, 100 cells.

The guess that a disc's axis is its centre was wrong on a raster. The
boundary is a staircase, so a disc of radius 5, 10, 20 and 40 keeps
45, 197, 841 and 3713 cells as maximal discs, 74 percent of the
5027-cell disc of radius 40, all in one component. Pruning by disc
radius recovers the centre: at one below the radius 5 cells remain
for radii 10, 20 and 40 and 9 for radius 5; at two below, 13 remain
for radii 5, 20 and 40 and 21 for radius 10, the staircase's own
pattern at each size.

One cell removed from the flat edge of the 100 by 41 rectangle adds
169 cells to the axis, the same at columns 30, 50 and 51 and 96 at
column 15 near a corner, since every disc tangent to the notch's
corners is maximal. Pruning does not remove the branch the way it
removes the disc's staircase: thresholds of 2, 3, 5 and 10 leave
167, 161, 153 and 115 of the extra 169, because the branch's discs
are as large as the spine's. A hole at the rectangle's centre adds
539 cells. An L of arm 60 and thickness 11 reads 144 cells in one
component with 5 endpoints and a reach of 7; thickness 10 reads 233,
the doubled spine again.
"""

from __future__ import annotations

import math

from atlas.distancetransform import exact
from atlas.errors import Invalid

Mask = list[list[bool]]
Cell = tuple[int, int]
Field = list[list[float]]


def inside_distance(mask: Mask) -> Field:
    if not mask or not mask[0]:
        raise Invalid("the mask is empty")
    if not any(any(row) for row in mask):
        raise Invalid("the mask has no shape")
    background = [[not v for v in row] for row in mask]
    if not any(any(row) for row in background):
        raise Invalid("the shape fills the whole grid, so it has no boundary")
    return exact(background)


def _at(dist: Field, r: int, c: int) -> float:
    if 0 <= r < len(dist) and 0 <= c < len(dist[0]):
        return dist[r][c]
    return 0.0


def _maximal(dist: Field, r: int, c: int, tolerance: float) -> bool:
    # the disc at this cell is inside no neighbour's disc
    d = dist[r][c]
    if d <= 0:
        return False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and _at(dist, r + dr, c + dc) >= d + math.hypot(dr, dc) - tolerance:
                return False
    return True


def ridge(mask: Mask, tolerance: float = 1e-9) -> list[Cell]:
    dist = inside_distance(mask)
    return [
        (r, c)
        for r, row in enumerate(mask)
        for c, inside in enumerate(row)
        if inside and _maximal(dist, r, c, tolerance)
    ]


def prune(axis: list[Cell], dist: Field, minimum: float) -> list[Cell]:
    if minimum < 0:
        raise Invalid("the pruning distance must not be negative")
    return [(r, c) for r, c in axis if dist[r][c] >= minimum]


def _degree(cells: set[Cell], r: int, c: int) -> int:
    return sum(
        1 for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc) and (r + dr, c + dc) in cells
    )


def junctions(axis: list[Cell]) -> list[Cell]:
    cells = set(axis)
    return [(r, c) for r, c in axis if _degree(cells, r, c) >= 3]


def endpoints(axis: list[Cell]) -> list[Cell]:
    cells = set(axis)
    return [(r, c) for r, c in axis if _degree(cells, r, c) <= 1]


def components(axis: list[Cell]) -> int:
    cells = set(axis)
    seen: set[Cell] = set()
    count = 0
    for start in axis:
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    other = (r + dr, c + dc)
                    if other in cells and other not in seen:
                        seen.add(other)
                        stack.append(other)
    return count


def reach(axis: list[Cell], dist: Field) -> float:
    if not axis:
        raise Invalid("an empty axis has no reach")
    return max(dist[r][c] for r, c in axis)


def rectangle(width: int, height: int, pad: int = 2) -> Mask:
    if width < 1 or height < 1:
        raise Invalid("the rectangle needs a positive width and height")
    rows, cols = height + 2 * pad, width + 2 * pad
    out = []
    for r in range(rows):
        out.append([pad <= r < pad + height and pad <= c < pad + width for c in range(cols)])
    return out


def disc(radius: int, pad: int = 2) -> Mask:
    if radius < 1:
        raise Invalid("the disc needs a positive radius")
    size = 2 * radius + 1 + 2 * pad
    centre = radius + pad
    return [
        [math.hypot(r - centre, c - centre) <= radius for c in range(size)] for r in range(size)
    ]


def notched(mask: Mask, r: int, c: int) -> Mask:
    if not (0 <= r < len(mask) and 0 <= c < len(mask[0])) or not mask[r][c]:
        raise Invalid("the notch must sit on the shape")
    out = [list(row) for row in mask]
    out[r][c] = False
    return out


def l_shape(arm: int, thickness: int, pad: int = 2) -> Mask:
    if thickness < 1 or arm <= thickness:
        raise Invalid("the arm must be longer than the thickness")
    size = arm + 2 * pad
    out = [[False] * size for _ in range(size)]
    for r in range(pad, pad + arm):
        for c in range(pad, pad + arm):
            if r < pad + thickness or c < pad + thickness:
                out[r][c] = True
    return out


def expected_rectangle_cells(width: int, height: int) -> int:
    short, long = min(width, height), max(width, height)
    return (long - short) + 4 * (short // 2)
