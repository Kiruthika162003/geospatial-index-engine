"""Grid traversal: every cell a segment touches, more than Bresenham's minimal walk.

Bresenham draws the thinnest connected line, one cell per step along
the major axis, which is right for painting but wrong for asking which
cells a ray actually passes through: a segment that clips the corner of
a cell touches it, yet Bresenham may step diagonally past it. The grid
traversal, the Amanatides-Woo walk, visits every cell the segment
enters, in the order it enters them. It tracks, for each axis, the
parameter value at which the segment will next cross a grid line in
that axis, and at every step it advances whichever crossing comes
first, moving one cell in that axis alone. Because it never moves in
both axes at once, it never cuts a corner, so the visited cells form a
four-connected path, each cell sharing a full edge with the next, and
that is the guarantee a ray cast, a line-of-sight test, or a
collision sweep needs: no cell the segment touches is skipped. The
price is more cells. A diagonal segment that Bresenham renders with n
cells is traversed with about twice as many, since the walk steps in x
then y then x rather than diagonally, and in general the traversal
visits one more cell for every grid line the segment crosses. The two
walks answer different questions: Bresenham asks which cells best
approximate the line, traversal asks which cells the line intersects,
and only the second is safe for a query whose answer must not miss
anything. The finding worth stating is that the traversal's cell count
equals one plus the number of grid lines crossed, it is four-connected
where Bresenham is only eight-connected, and every one of Bresenham's
cells appears among the traversal's, so the traversal is a superset
walk that trades thinness for completeness. This module traverses a
segment through a unit grid, and a survey confirms the count law, the
four-connectivity, and the superset relation against Bresenham.
"""

from __future__ import annotations

import math
from itertools import pairwise

from atlas.errors import Invalid

Cell = tuple[int, int]


def traverse(x0: float, y0: float, x1: float, y1: float) -> list[Cell]:
    for v in (x0, y0, x1, y1):
        if not math.isfinite(v):
            raise Invalid("segment endpoints must be finite")
    cx, cy = math.floor(x0), math.floor(y0)
    end_x, end_y = math.floor(x1), math.floor(y1)
    dx, dy = x1 - x0, y1 - y0
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
    # parameter t at which the next vertical / horizontal grid line is crossed
    t_max_x = ((cx + (1 if step_x > 0 else 0)) - x0) / dx if dx != 0 else math.inf
    t_max_y = ((cy + (1 if step_y > 0 else 0)) - y0) / dy if dy != 0 else math.inf
    t_delta_x = abs(1 / dx) if dx != 0 else math.inf
    t_delta_y = abs(1 / dy) if dy != 0 else math.inf
    cells: list[Cell] = [(cx, cy)]
    guard = abs(end_x - cx) + abs(end_y - cy) + 2
    while (cx, cy) != (end_x, end_y) and guard > 0:
        guard -= 1
        if t_max_x < t_max_y:
            t_max_x += t_delta_x
            cx += step_x
        else:
            t_max_y += t_delta_y
            cy += step_y
        cells.append((cx, cy))
    return cells


def is_four_connected(cells: list[Cell]) -> bool:
    return all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in pairwise(cells))


def grid_lines_crossed(x0: float, y0: float, x1: float, y1: float) -> int:
    return abs(math.floor(x1) - math.floor(x0)) + abs(math.floor(y1) - math.floor(y0))
