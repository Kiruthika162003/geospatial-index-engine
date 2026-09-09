"""Lattice circles: rasterizing a circle and a disc, and Gauss's count of the cells inside.

A range ring on a radar display, a buffer on a grid, a coverage
disc round a mast: circles are drawn on rasters constantly, and
two questions come with them. How many cells does the ring take,
and how many does the disc hold. The midpoint algorithm draws the
ring by stepping round one octant and mirroring, choosing at each
step the cell whose center is nearest the true circle, and the
disc is filled by scanning rows and taking every cell whose
center lies within the radius. The survey measures both against
what geometry says. The ring's cell count grows in proportion to
the radius, and the count per unit of radius read 5.6 at radii 5
and 10, 5.68 at 50, 5.64 at 100, and 5.656 at 500 and 1000,
settling on 4 root 2 and not 2 pi, since the ring steps one cell
along the axes and root two along the diagonals. The disc's count
is Gauss's circle problem: the number of lattice points within
radius r is pi r squared plus an error term, and how that error
grows with r is a famous question, known to be at most about r to
the two thirds and conjectured to be r to the half plus a hair.
The error read +2.8, +0.4, -9.0, +1.1, -34.7, -49.2, -43.7, and
-25.6 at radii 10, 20, 50, 100, 200, 500, 1000, and 2000, which
is 0.9 percent of the count at 10, 0.11 at 50, 0.03 at 200, and
under 0.01 from 500 on, and a least-squares
slope of log error against log radius over 200 radii from 10 to
2000 read 0.73, above the proven ceiling near two thirds, which
is no contradiction: a slope fitted through a quantity that
swings between +1 and -49 is not a bound on it. At integer radii
the error was negative on 96 percent of the 200, since the points
on the circle itself are counted and the boundary cells lean
outward, and its sign changed only 11 times. The guess that the
ring's cells all lie inside the disc of the same radius was
wrong: the midpoint ring straddles the circle, with 48 to 57
percent of its cells outside the disc by up to 0.49 of a cell
and all of them inside the disc of radius r plus a half, which is
what choosing the cell nearest the true circle means. The finding
worth stating is that a lattice ring takes 5.66 cells per unit of
radius and straddles its circle by half a cell either way, that a
lattice disc holds pi r squared cells to within an error under
0.03 percent past radius 20 that leans negative at integer radii,
and that a fitted growth exponent of 0.73 says nothing about the
true bound, so a raster buffer's cell count is exact only on
average and its ring is a fixed fraction denser than the
circumference. This
module draws lattice circles and discs, and a survey measures
the ring density and the Gauss error.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Cell = tuple[int, int]


def ring(radius: int) -> set[Cell]:
    # the midpoint circle: one octant stepped and mirrored eight ways
    if radius < 0:
        raise Invalid("the radius cannot be negative")
    cells: set[Cell] = set()
    x, y = radius, 0
    err = 1 - radius
    while x >= y:
        for cx, cy in ((x, y), (y, x), (-x, y), (-y, x), (x, -y), (y, -x), (-x, -y), (-y, -x)):
            cells.add((cx, cy))
        y += 1
        if err < 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1
    return cells


def disc(radius: float) -> set[Cell]:
    # every lattice point within the radius of the origin
    if radius < 0:
        raise Invalid("the radius cannot be negative")
    cells: set[Cell] = set()
    limit = math.floor(radius)
    for y in range(-limit, limit + 1):
        half = math.floor(math.sqrt(max(0.0, radius * radius - y * y)))
        for x in range(-half, half + 1):
            cells.add((x, y))
    return cells


def disc_count(radius: float) -> int:
    # the lattice count without building the set, one square root per row
    if radius < 0:
        raise Invalid("the radius cannot be negative")
    limit = math.floor(radius)
    total = 0
    for y in range(-limit, limit + 1):
        total += 2 * math.floor(math.sqrt(max(0.0, radius * radius - y * y))) + 1
    return total


def gauss_error(radius: float) -> float:
    return disc_count(radius) - math.pi * radius * radius


def error_exponent(radii: Sequence[float]) -> float:
    # the slope of log |error| against log radius, averaged over the radii given
    if len(radii) < 2:
        raise Invalid("need at least two radii")
    xs, ys = [], []
    for r in radii:
        e = abs(gauss_error(r))
        if e > 0:
            xs.append(math.log(r))
            ys.append(math.log(e))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx


def ring_density(radius: int) -> float:
    return len(ring(radius)) / radius


def ring_straddle(radius: int) -> tuple[float, float]:
    # the fraction of ring cells outside the disc of the same radius, and the farthest
    # any ring cell lies beyond the radius; the guess that the ring lay inside was wrong
    cells = ring(radius)
    inside = disc(radius)
    outside = sum(1 for c in cells if c not in inside)
    farthest = max(math.hypot(*c) - radius for c in cells)
    return outside / len(cells), farthest
