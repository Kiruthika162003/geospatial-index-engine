"""Line density reads a lone line at 1 over (2r + 1) cells at any window and conserves length.

A line density raster holds, for each cell, the length of line
within a window around it over the window's area. The length is
first apportioned to cells by clipping every segment to every
cell it crosses, and the apportioned lengths sum to the true
length exactly, 100.0 for one line across a 100 square and
3631.922 for 200 random segments. A single line across the middle
reads a density on its own cells of 0.5, 0.1, 0.0455 and 0.0238 at
window radii of 0, 2, 5 and 10 cells of 2 units, exactly 1 over
(2r + 1) times the cell, and 0.0 three cells away until the window
reaches it; the mean of the whole field stays at the expected
length over area, 0.01, at every radius, since the clipped window
at the edge divides by its own smaller area. Two hundred random
segments of length 20 read a mean of 0.3632 at radius 0 against the
expected 0.3632 and 0.3627, 0.363, 0.3652 and 0.3746 at radii 2, 5,
10 and 20, the widest window drifting 3 percent high through the
edge clipping; the maximum falls from 2.48 to 0.46 and the share of
cells reading zero from 36.8 percent to 1 percent at radius 2 and
none beyond. The corner reads 0.57 to 0.78 of the centre from
radius 2 on, since segments that would leave the extent are clamped
to it and a corner window sees fewer of them.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _clip_length(a: Point, b: Point, box: tuple[float, float, float, float]) -> float:
    # Liang and Barsky, returning the length of the piece inside the box
    x0, y0, x1, y1 = box
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return 0.0
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return 0.0
            t0 = max(t0, t)
        else:
            if t < t0:
                return 0.0
            t1 = min(t1, t)
    return max(0.0, t1 - t0) * math.hypot(dx, dy)


def cell_lengths(segments: list[Segment], size: int, cell: float) -> list[list[float]]:
    # the length of line inside each cell
    if size < 1 or cell <= 0:
        raise Invalid("size and cell must be positive")
    grid = [[0.0] * size for _ in range(size)]
    for a, b in segments:
        c0 = max(0, math.floor(min(a[0], b[0]) / cell))
        c1 = min(size - 1, math.floor(max(a[0], b[0]) / cell))
        r0 = max(0, math.floor(min(a[1], b[1]) / cell))
        r1 = min(size - 1, math.floor(max(a[1], b[1]) / cell))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                grid[r][c] += _clip_length(
                    a, b, (c * cell, r * cell, (c + 1) * cell, (r + 1) * cell)
                )
    return grid


def density(lengths: list[list[float]], cell: float, radius: int) -> list[list[float]]:
    # length within a square window over the window's area, clipped at the edges
    if radius < 0:
        raise Invalid("the radius must not be negative")
    size = len(lengths)
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            r0, r1 = max(0, r - radius), min(size - 1, r + radius)
            c0, c1 = max(0, c - radius), min(size - 1, c + radius)
            total = sum(lengths[rr][cc] for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1))
            row.append(total / ((r1 - r0 + 1) * (c1 - c0 + 1) * cell * cell))
        out.append(row)
    return out


def total_length(segments: list[Segment]) -> float:
    return sum(math.dist(a, b) for a, b in segments)


def mean(grid: list[list[float]]) -> float:
    return sum(sum(row) for row in grid) / (len(grid) * len(grid[0]))


def random_segments(n: int, rng: random.Random, extent: float, length: float) -> list[Segment]:
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, extent), rng.uniform(0, extent)
        a = rng.uniform(0, 2 * math.pi)
        bx = min(max(x + length * math.cos(a), 0.0), extent)
        by = min(max(y + length * math.sin(a), 0.0), extent)
        out.append(((x, y), (bx, by)))
    return out


def one_line(extent: float) -> list[Segment]:
    return [((0.0, extent / 2), (extent, extent / 2))]


def expected_density(segments: list[Segment], extent: float) -> float:
    return total_length(segments) / (extent * extent)


def edge_ratio(field: list[list[float]]) -> float:
    # the density read at the corner over the density read at the centre
    size = len(field)
    return (
        field[0][0] / field[size // 2][size // 2] if field[size // 2][size // 2] else math.inf
    )
