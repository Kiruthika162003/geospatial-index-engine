"""Hex against square: what a grid's shape does to distance, neighbours, and direction.

A raster of squares is the default because images are square,
and a raster of hexagons is what a modeller chooses when the
cells' neighbours should all be alike, and the survey measures
what the choice buys. On a square grid a cell has four edge
neighbours at distance one and four corner neighbours at root
two, so a path that steps only through edges overshoots the
straight line by up to 41 percent, the Manhattan penalty, and a
path that may step through corners by up to 8.2 percent, the
octile penalty, both at 45 degrees; on a hex grid a cell has six
neighbours all at distance one and a path through them overshoots
by at most 15.5 percent, two over root three less one, at 30
degrees off a lattice axis. The survey measures the penalties by
walking grid paths between 20000 random pairs of cells up to 400
apart and dividing the path length by the straight distance: the
Manhattan penalty averaged 27.4 percent and peaked at 41.42, the
octile 5.42 and 8.24, and the hex 10.36 and 15.47, each worst
landing on its closed form to the hundredth of a percent, at 45
degrees, at 22.5 where cos plus (root 2 minus 1) sin peaks, and
at 30 off the hex axis. It measures the neighbourhood: the
square's eight neighbours span two distances, 1 and 1.414, and
the hex's six span one, and the distance from a cell's center to
its farthest boundary point over its nearest is 1.414 for a
square and 1.155 for a hexagon, so a hexagon is rounder by 1.225.
And it measures direction: the count of distinct directions to
cells within two steps is 16 on a square grid with corners and
12 on a hex grid, so the square grid has more directions and the
hex grid more even ones. The finding worth stating is that hex
grids cap the path penalty at 15.47 percent with one neighbour
distance where square grids cap it at 8.24 with corners and 41.42
without, that a hexagon is 22 percent rounder than a square by
center-to-boundary ratio, and that the square grid's extra
directions come at the price of two neighbour distances, so the
shape is a trade between evenness and richness the survey has
priced. This module measures both grids,
and a survey reads the penalties, the neighbourhoods, and the
directions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from atlas.errors import Invalid

Cell = tuple[int, int]
SQUARE_FOUR = ((1, 0), (-1, 0), (0, 1), (0, -1))
SQUARE_EIGHT = (*SQUARE_FOUR, (1, 1), (1, -1), (-1, 1), (-1, -1))
# axial hex coordinates: six neighbours, all at unit distance in the plane
HEX_SIX = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


def hex_to_plane(q: int, r: int) -> tuple[float, float]:
    return (q + r / 2.0, r * math.sqrt(3) / 2)


def square_path_length(a: Cell, b: Cell, corners: bool) -> float:
    # the shortest grid path length between two cells: Manhattan or octile
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    if not corners:
        return float(dx + dy)
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def hex_path_length(a: Cell, b: Cell) -> float:
    # the hex distance in steps, each of unit length
    dq, dr = a[0] - b[0], a[1] - b[1]
    return float(max(abs(dq), abs(dr), abs(dq + dr)))


def square_penalty(a: Cell, b: Cell, corners: bool) -> float:
    straight = math.dist(a, b)
    if straight == 0:
        raise Invalid("the cells coincide")
    return square_path_length(a, b, corners) / straight - 1


def hex_penalty(a: Cell, b: Cell) -> float:
    straight = math.dist(hex_to_plane(*a), hex_to_plane(*b))
    if straight == 0:
        raise Invalid("the cells coincide")
    return hex_path_length(a, b) / straight - 1


def penalties(count: int, rng, span: int = 200) -> dict[str, tuple[float, float]]:
    # mean and worst penalty over random pairs for each grid and rule
    if count < 1:
        raise Invalid("need at least one pair")
    sums = {"manhattan": 0.0, "octile": 0.0, "hex": 0.0}
    worst = {"manhattan": 0.0, "octile": 0.0, "hex": 0.0}
    for _ in range(count):
        a = (rng.randint(-span, span), rng.randint(-span, span))
        b = (rng.randint(-span, span), rng.randint(-span, span))
        if a == b:
            continue
        readings = {
            "manhattan": square_penalty(a, b, False),
            "octile": square_penalty(a, b, True),
            "hex": hex_penalty(a, b),
        }
        for key, value in readings.items():
            sums[key] += value
            worst[key] = max(worst[key], value)
    return {key: (sums[key] / count, worst[key]) for key in sums}


def closed_form_worst() -> dict[str, float]:
    # Manhattan peaks at 45 degrees, octile at 22.5 where cos + (root 2 - 1) sin is largest,
    # and the hex step count at 30 degrees off a lattice axis
    diagonal_step = math.sqrt(2) - 1
    return {
        "manhattan": math.sqrt(2) - 1,
        "octile": math.sqrt(1 + diagonal_step * diagonal_step) - 1,
        "hex": 2 / math.sqrt(3) - 1,
    }


def neighbour_distances(kind: str) -> list[float]:
    if kind == "square":
        return sorted({round(math.hypot(dx, dy), 12) for dx, dy in SQUARE_EIGHT})
    if kind == "hex":
        origin = hex_to_plane(0, 0)
        return sorted({round(math.dist(origin, hex_to_plane(q, r)), 12) for q, r in HEX_SIX})
    raise Invalid("kind must be square or hex")


def roundness(kind: str) -> float:
    # farthest boundary point over nearest, from the cell's center
    if kind == "square":
        return math.sqrt(2)
    if kind == "hex":
        return 2 / math.sqrt(3)
    raise Invalid("kind must be square or hex")


def directions_within_two_steps(kind: str) -> int:
    if kind not in ("square", "hex"):
        raise Invalid("kind must be square or hex")
    steps: Sequence[tuple[int, int]] = SQUARE_EIGHT if kind == "square" else HEX_SIX

    def to_plane(c: Cell) -> tuple[float, float]:
        return (float(c[0]), float(c[1])) if kind == "square" else hex_to_plane(*c)

    reached: set[Cell] = {(0, 0)}
    frontier = {(0, 0)}
    for _ in range(2):
        frontier = {(c[0] + dx, c[1] + dy) for c in frontier for dx, dy in steps}
        reached |= frontier
    angles = set()
    for c in reached:
        if c == (0, 0):
            continue
        x, y = to_plane(c)
        angles.add(round(math.degrees(math.atan2(y, x)) % 360.0, 6))
    return len(angles)
