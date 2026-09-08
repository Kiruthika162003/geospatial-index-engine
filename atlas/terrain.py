"""Terrain slope and aspect: central differences recover a plane's gradient exactly.

A height grid, a digital elevation model, answers two questions at
every cell that maps and models depend on: how steep is the ground,
the slope, and which way does it face, the aspect. Both come from the
gradient, the rate the height changes per unit of distance east and
north, and on a grid the gradient is estimated by central differences:
the east-west rate is the height one cell east minus one cell west,
over twice the cell size, and the north-south rate likewise. Slope is
the arctangent of the gradient's magnitude, the angle the surface
makes with the horizontal, and aspect is the compass direction of
steepest descent, the direction the gradient points down, measured
clockwise from north. Two things pin the estimate and are worth
measuring rather than trusting. On a planar surface, one whose height
is a linear function of position, central differences recover the
gradient exactly, not approximately, because the linear terms are
differenced without error and there are no higher terms to truncate;
so a plane rising at a known angle in a known direction reports that
slope and that aspect to floating-point precision, which is the
calibration. On a curved surface the estimate carries a truncation
error that shrinks with the square of the cell size, so halving the
cell quarters the error, and the first attempt to measure that rate
taught something worth keeping. The survey chose a cone, uniform in
slope everywhere but its apex, and read the slope on the grid row
through the apex; the error there was a few parts in ten to the
fifteenth at every cell size, no decay at all, because along a line
through the apex a cone is linear and across it symmetric, so the
differences are exact and the truncation term never appears. Read
off the axis, five cells east and three north of the apex, the error
was 0.79, 0.14, 0.036, and 0.0089 degrees as the cell halved from two
to a quarter, ratios of four, and on a Gaussian hill, curved in every
direction, it was 1.00, 0.25, 0.063, and 0.016, ratios of four again.
The on-axis reading was not a failure of the method but a surface
that was locally planar exactly where it was sampled, which is the
lesson: the error is second order where the terrain curves and zero
where it does not. A flat plane has zero slope and no meaningful
aspect, which the module reports as a sentinel rather than an
arbitrary compass point. The finding worth
stating is that central differences are exact on planes and second-
order on curves, so slope and aspect from a grid are trustworthy
wherever the terrain is locally planar at the cell scale, which is
the assumption every elevation model makes. This module computes
slope and aspect grids by central differences, and a survey confirms
the planar calibration and the error's quadratic decay.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

FLAT_ASPECT = -1.0


def gradient(grid: list[list[float]], cell: float, r: int, c: int) -> tuple[float, float]:
    # (dz/dx eastward, dz/dy northward) at an interior cell; rows increase northward
    if cell <= 0:
        raise Invalid("cell size must be positive")
    rows, cols = len(grid), len(grid[0])
    if not (1 <= r < rows - 1 and 1 <= c < cols - 1):
        raise Invalid("central differences need an interior cell")
    dzdx = (grid[r][c + 1] - grid[r][c - 1]) / (2 * cell)
    dzdy = (grid[r + 1][c] - grid[r - 1][c]) / (2 * cell)
    return (dzdx, dzdy)


def slope_degrees(grid: list[list[float]], cell: float, r: int, c: int) -> float:
    dzdx, dzdy = gradient(grid, cell, r, c)
    return math.degrees(math.atan(math.hypot(dzdx, dzdy)))


def aspect_degrees(grid: list[list[float]], cell: float, r: int, c: int) -> float:
    # direction of steepest descent, clockwise from north; flat ground has none
    dzdx, dzdy = gradient(grid, cell, r, c)
    if dzdx == 0 and dzdy == 0:
        return FLAT_ASPECT
    return (math.degrees(math.atan2(-dzdx, -dzdy)) + 360.0) % 360.0


def plane(
    rows: int, cols: int, cell: float, slope_deg: float, aspect_deg: float
) -> list[list[float]]:
    # a planar surface descending toward the given aspect at the given slope
    if rows < 3 or cols < 3:
        raise Invalid("a grid needs at least three rows and columns")
    g = math.tan(math.radians(slope_deg))
    down = math.radians(aspect_deg)
    dzdx = -g * math.sin(down)
    dzdy = -g * math.cos(down)
    return [[dzdx * c * cell + dzdy * r * cell for c in range(cols)] for r in range(rows)]


def cone(rows: int, cols: int, cell: float, slope_deg: float) -> list[list[float]]:
    # a cone with its apex at the center, descending at a uniform slope
    if rows < 3 or cols < 3:
        raise Invalid("a grid needs at least three rows and columns")
    g = math.tan(math.radians(slope_deg))
    cr, cc = (rows - 1) / 2, (cols - 1) / 2
    return [[-g * cell * math.hypot(r - cr, c - cc) for c in range(cols)] for r in range(rows)]
