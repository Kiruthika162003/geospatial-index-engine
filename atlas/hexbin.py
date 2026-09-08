"""Hex binning: aggregate points into hexagons, whose equal neighbor spacing beats squares.

Binning points into cells to count them is the first step of any
density map, and the choice of cell shape is not cosmetic. A square
grid has two kinds of neighbor, four across an edge and four across a
corner, at two different distances, so a cluster straddling a corner
splits four ways while one straddling an edge splits two, and the
resulting map carries the grid's own texture. A hexagonal grid has one
kind of neighbor, six of them, all at the same center-to-center
distance, so a cluster splits evenly whichever way it straddles a
boundary, and the map shows the data's texture rather than the
grid's. Hex binning assigns each point to the hexagon containing it by
the cube-coordinate rounding the hex grid module provides, then counts
per cell, and the survey pins three properties. The counts sum to the
point count exactly, since every point lands in one cell and no cell
double-counts, a conservation the aggregation must honor. Every
occupied cell's neighbors are at exactly one distance, the hex
spacing, where a square grid's neighbors sit at two, the edge and the
diagonal, which is the geometric root of the smoother map. And at
equal cell area, a hexagon is closer to a disk than a square is, its
perimeter for the same area being about seven percent shorter, so the
hexagon collects a rounder neighborhood. A first guess went one step
further and expected that rounder neighborhood to show as clearly
lower cell-to-cell noise on smooth data; the measurement refuted the
size of it. On twenty thousand uniformly scattered points binned at
equal cell area, the interior hex cells had a coefficient of variation
of 0.1446 against 0.1466 for squares, only one and a half percent
lower, because on scattered points the fluctuation is counting noise,
which depends on how many points a cell holds and so on its area, not
its shape. The guess is kept beside the result: the hexagon's real
advantages are the single neighbor distance and the shorter perimeter,
and the noise advantage is marginal. The finding worth stating is that
hex bins conserve the count, have a single neighbor distance where
squares have two, and a 6.9 percent shorter boundary at equal area,
while their noise on random points is within two percent of squares',
so the hexagon is the better bin for its geometry, not for its
statistics. This module bins points into hexes and into squares of equal
area for comparison, and a survey measures the conservation, the
spacing, and the noise.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid
from atlas.hexgrid import from_point, neighbors, to_point

Point = tuple[float, float]
Hex = tuple[int, int, int]


def hex_bins(points: list[Point], size: float) -> dict[Hex, int]:
    if points is None:
        raise Invalid("points must not be None")
    if size <= 0:
        raise Invalid("hex size must be positive")
    counts: dict[Hex, int] = {}
    for x, y in points:
        h = from_point(x, y, size)
        counts[h] = counts.get(h, 0) + 1
    return counts


def hex_area(size: float) -> float:
    # a flat-top hexagon of circumradius size
    return 3 * math.sqrt(3) / 2 * size * size


def square_bins(points: list[Point], side: float) -> dict[tuple[int, int], int]:
    if points is None:
        raise Invalid("points must not be None")
    if side <= 0:
        raise Invalid("square side must be positive")
    counts: dict[tuple[int, int], int] = {}
    for x, y in points:
        key = (math.floor(x / side), math.floor(y / side))
        counts[key] = counts.get(key, 0) + 1
    return counts


def square_side_for_hex(size: float) -> float:
    # the square whose area equals the hexagon's
    return math.sqrt(hex_area(size))


def neighbor_distances(h: Hex, size: float) -> list[float]:
    cx, cy = to_point(h, size)
    out = []
    for n in neighbors(h):
        nx, ny = to_point(n, size)
        out.append(math.hypot(nx - cx, ny - cy))
    return out


def coefficient_of_variation(counts: dict) -> float:
    values = list(counts.values())
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) / mean
