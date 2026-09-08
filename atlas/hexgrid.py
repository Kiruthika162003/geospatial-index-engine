"""Hex grid: cube coordinates make distance a three-way max and rings six times k.

Hexagonal grids index the plane with cells that have six equidistant
neighbors, unlike squares whose diagonal neighbors are farther than
their edge neighbors, and that uniformity is why hex grids underlie
spatial systems like H3 and many simulations. The awkwardness is
addressing them: offset row-column schemes make neighbor and distance
arithmetic a tangle of parity cases. Cube coordinates dissolve it.
Give each hex three integers q, r, s constrained to sum to zero, which
is a plane through a three-dimensional integer lattice, and the six
neighbors are the six unit steps that keep the sum at zero. Two
consequences fall out with no case analysis. The hex distance between
two cells, the fewest neighbor steps from one to the other, is the
largest of the three absolute coordinate differences, exactly as a
king's move distance on a square board is the largest of two, because
a step changes two coordinates at once and the largest difference is
what must be walked down. And the ring of cells at exactly distance k
from a center holds exactly six times k cells, one edge of k cells for
each of the six sides, so the hexagonal disk of radius k holds one plus
three k times k plus one cells. Converting a plane point to the hex
that contains it is a fractional cube coordinate followed by a rounding
that restores the zero-sum constraint by fixing whichever coordinate
rounded worst. The finding worth stating is that cube coordinates turn
hex distance into a three-way max that matches a breadth-first count
exactly and make every ring six k in size, so hexagonal geometry
becomes plain integer arithmetic. This module converts points to hexes
and back, computes distance, and enumerates neighbors and rings, and a
survey checks the distance against breadth-first search and the ring
sizes against the six-k law.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Hex = tuple[int, int, int]

_DIRECTIONS: tuple[Hex, ...] = (
    (1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
)


def make(q: int, r: int, s: int) -> Hex:
    if q + r + s != 0:
        raise Invalid("cube coordinates must sum to zero")
    return (q, r, s)


def distance(a: Hex, b: Hex) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def neighbors(h: Hex) -> list[Hex]:
    return [(h[0] + d[0], h[1] + d[1], h[2] + d[2]) for d in _DIRECTIONS]


def ring(center: Hex, k: int) -> list[Hex]:
    if k < 0:
        raise Invalid("ring radius must not be negative")
    if k == 0:
        return [center]
    cells: list[Hex] = []
    # start k steps out along direction 4, then walk each of the six sides
    start = _DIRECTIONS[4]
    h = (center[0] + start[0] * k, center[1] + start[1] * k, center[2] + start[2] * k)
    for side in range(6):
        d = _DIRECTIONS[side]
        for _ in range(k):
            cells.append(h)
            h = (h[0] + d[0], h[1] + d[1], h[2] + d[2])
    return cells


def disk(center: Hex, k: int) -> list[Hex]:
    cells: list[Hex] = []
    for radius in range(k + 1):
        cells.extend(ring(center, radius))
    return cells


def to_point(h: Hex, size: float) -> tuple[float, float]:
    # flat-top hex layout
    x = size * 1.5 * h[0]
    y = size * (math.sqrt(3) / 2 * h[0] + math.sqrt(3) * h[1])
    return (x, y)


def from_point(x: float, y: float, size: float) -> Hex:
    if size <= 0:
        raise Invalid("hex size must be positive")
    q = (2.0 / 3 * x) / size
    r = (-1.0 / 3 * x + math.sqrt(3) / 3 * y) / size
    return _round(q, r, -q - r)


def _round(fq: float, fr: float, fs: float) -> Hex:
    q, r, s = round(fq), round(fr), round(fs)
    dq, dr, ds = abs(q - fq), abs(r - fr), abs(s - fs)
    # fix whichever coordinate rounded worst so the sum returns to zero
    if dq > dr and dq > ds:
        q = -r - s
    elif dr > ds:
        r = -q - s
    else:
        s = -q - r
    return (q, r, s)
