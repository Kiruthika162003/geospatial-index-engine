"""A 45 degree valley floor sees 0.46 of the sky by the raster fan, 0.50 by the law, not cos 45.

The sky view factor is one minus the mean, over a fan of directions,
of the sine of the horizon angle, and on a raster the horizon in a
direction is the steepest line from the cell to any cell along a
rounded ray. A plane reads 1.0. The guess that a valley floor sees
the cosine of its wall angle, a canyon rule of thumb, was wrong: with
walls of slope 1 the floor reads 0.5345, 0.4884, 0.4626 and 0.4567
with fans of 8, 16, 32 and 64 directions, against the hemisphere
integral over an infinite V valley of 0.5 and the cosine guess of
0.7071. Walls of slope 0.25, 0.5 and 2 read 0.8274, 0.675 and 0.2493
at 64 directions against laws of 0.844, 0.7048 and 0.2952. The fan
converges below the law because the rounded ray steps sideways onto
cells off the ideal line, and the maximum over the steps keeps the
steepest of those overestimates; the eight direction fan, having no
oblique rays, happens to read closest. The search depth does not
matter in a V valley, since the first cell up the wall is the
horizon: 0.4626 at 4, 8, 16 and 32 steps. Climbing the wall of the
slope 1 valley reads 0.5619, 0.6252, 0.6961 and 0.7177 at 5, 10, 20
and 25 cells from the axis.

A trench 5 deep and 5 wide reads 0.3315 at its centre and 0.3035 at
the wall foot against a cosine guess of 0.5145; 10 deep and 5 wide
0.1785; 10 deep and 21 wide 0.5523 at the centre and 0.4303 at the
foot. A round pit 5 deep and 5 in radius reads 0.3293, between the
law for a rim at 5 of 0.2929 and at 6 of 0.3598, since the raster rim
sits between the two; 10 deep at radius 5 reads 0.1252 and 10 deep
at radius 20 0.5631. A 31 by 31 grid with 16 directions and 32 steps
takes 0.09 seconds.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Grid = list[list[float]]


def _check(dem: Grid) -> tuple[int, int]:
    if not dem or not dem[0]:
        raise Invalid("the elevation grid is empty")
    return len(dem), len(dem[0])


def horizon_angle(
    dem: Grid, r: int, c: int, angle: float, cell: float = 1.0, max_steps: int = 64
) -> float:
    rows, cols = _check(dem)
    if not (0 <= r < rows and 0 <= c < cols):
        raise Invalid("the cell must lie on the grid")
    if cell <= 0 or max_steps < 1:
        raise Invalid("cell and max_steps must be positive")
    z0 = dem[r][c]
    dr, dc = -math.cos(angle), math.sin(angle)
    best = 0.0
    for k in range(1, max_steps + 1):
        rr, cc = round(r + k * dr), round(c + k * dc)
        if not (0 <= rr < rows and 0 <= cc < cols):
            break
        rise = dem[rr][cc] - z0
        run = math.hypot(rr - r, cc - c) * cell
        if run > 0:
            best = max(best, math.atan2(rise, run))
    return best


def sky_view(
    dem: Grid, r: int, c: int, directions: int = 16, cell: float = 1.0, max_steps: int = 64
) -> float:
    if directions < 1:
        raise Invalid("at least one direction is needed")
    total = 0.0
    for k in range(directions):
        angle = 2 * math.pi * k / directions
        total += math.sin(horizon_angle(dem, r, c, angle, cell, max_steps))
    return 1 - total / directions


def sky_view_grid(
    dem: Grid, directions: int = 16, cell: float = 1.0, max_steps: int = 64
) -> Grid:
    rows, cols = _check(dem)
    return [
        [sky_view(dem, r, c, directions, cell, max_steps) for c in range(cols)]
        for r in range(rows)
    ]


def mean_interior(grid: Grid, margin: int) -> float:
    rows, cols = _check(grid)
    if 2 * margin >= min(rows, cols):
        raise Invalid("the margin eats the whole grid")
    inner_rows = range(margin, rows - margin)
    inner_cols = range(margin, cols - margin)
    values = [grid[r][c] for r in inner_rows for c in inner_cols]
    return sum(values) / len(values)


def plane(size: int, slope: float = 0.0) -> Grid:
    if size < 3:
        raise Invalid("a grid needs three cells a side")
    return [[slope * c for c in range(size)] for _ in range(size)]


def v_valley(size: int, wall_slope: float) -> Grid:
    if size < 3 or wall_slope < 0:
        raise Invalid("a valley needs three cells a side and a non-negative wall slope")
    centre = size // 2
    return [[wall_slope * abs(c - centre) for c in range(size)] for _ in range(size)]


def trench(size: int, depth: float, half_width: int) -> Grid:
    if half_width < 0 or depth < 0:
        raise Invalid("depth and half width must not be negative")
    centre = size // 2
    row = [0.0 if abs(c - centre) <= half_width else depth for c in range(size)]
    return [list(row) for _ in range(size)]


def pit(size: int, depth: float, radius: int) -> Grid:
    centre = size // 2
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            row.append(0.0 if math.hypot(r - centre, c - centre) <= radius else depth)
        out.append(row)
    return out


def valley_floor_law(wall_slope: float, directions: int = 3600) -> float:
    # the horizon toward angle theta of an infinite V valley whose axis runs north-south
    total = 0.0
    for k in range(directions):
        theta = 2 * math.pi * k / directions
        total += math.sin(math.atan(wall_slope * abs(math.sin(theta))))
    return 1 - total / directions


def canyon_guess(wall_angle_deg: float) -> float:
    return math.cos(math.radians(wall_angle_deg))


def pit_law(depth: float, radius: float) -> float:
    return 1 - math.sin(math.atan2(depth, radius))
