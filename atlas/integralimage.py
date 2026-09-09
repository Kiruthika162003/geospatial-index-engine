"""A summed-area table answers a box sum in four lookups; single precision drifts 0.4 percent.

A summed-area table holds, at each corner, the sum of everything
above and to the left, so any box sum is four lookups and a window
mean over a 128 grid costs the same at every radius: 5 milliseconds
for the field at radii 1, 3, 7 and 15 against 15, 38, 126 and 446
for the naive sums, speedups of 2.6, 7.6, 25 and 90, from a table
built in 1.4 milliseconds. On integer-valued grids the table is
exact, and on uniform random doubles the worst box sum differs from
the naive sum by 3.4e-13, 1.9e-12, 6.6e-12 and 3.0e-11 at sides 64,
128, 256 and 512, where the corner total reaches 131,225.

The guess that single precision would serve for a table of small
values was wrong: rounding every partial sum to 32 bits, as a
float32 raster library does, puts the worst error of a box sum at
2.4e-4, 8.0e-4, 3.3e-3 and 1.68e-2 at the same four sides, since a
small box is the difference of four numbers as large as the corner
total, whose last place is 8e-3 at 131,225. On the 512 grid that is
0.37 percent of a nine-cell sum, and the error grows about fourfold
per doubling of the side. The mean field from the double table
differs from the naive field by at most 3.2e-13 at radius 1 and
6.3e-15 at radius 15. The local variance of uniform noise over a
radius 3 window reads 0.0811 against the population law of one
twelfth, 0.0833, the small-window bias of 48 over 49.
"""

from __future__ import annotations

import random
import struct
import time

from atlas.errors import Invalid

Grid = list[list[float]]


def _check(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return len(grid), len(grid[0])


def _single(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


class SummedArea:
    def __init__(self, grid: Grid, single: bool = False) -> None:
        rows, cols = _check(grid)
        self.rows, self.cols = rows, cols
        narrow = _single if single else (lambda v: v)
        table = [[0.0] * (cols + 1) for _ in range(rows + 1)]
        for r in range(rows):
            running = 0.0
            for c in range(cols):
                running = narrow(running + grid[r][c])
                table[r + 1][c + 1] = narrow(table[r][c + 1] + running)
        self.table = table

    def box_sum(self, r0: int, c0: int, r1: int, c1: int) -> float:
        # inclusive corners
        if not (0 <= r0 <= r1 < self.rows and 0 <= c0 <= c1 < self.cols):
            raise Invalid("the box must lie on the grid with ordered corners")
        t = self.table
        return t[r1 + 1][c1 + 1] - t[r0][c1 + 1] - t[r1 + 1][c0] + t[r0][c0]

    def window_mean(self, r: int, c: int, radius: int) -> float:
        r0, r1 = max(0, r - radius), min(self.rows - 1, r + radius)
        c0, c1 = max(0, c - radius), min(self.cols - 1, c + radius)
        return self.box_sum(r0, c0, r1, c1) / ((r1 - r0 + 1) * (c1 - c0 + 1))

    def mean_field(self, radius: int) -> Grid:
        return [
            [self.window_mean(r, c, radius) for c in range(self.cols)] for r in range(self.rows)
        ]


def naive_box_sum(grid: Grid, r0: int, c0: int, r1: int, c1: int) -> float:
    return sum(grid[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))


def naive_mean_field(grid: Grid, radius: int) -> Grid:
    rows, cols = _check(grid)
    out = []
    for r in range(rows):
        row = []
        for c in range(cols):
            r0, r1 = max(0, r - radius), min(rows - 1, r + radius)
            c0, c1 = max(0, c - radius), min(cols - 1, c + radius)
            row.append(naive_box_sum(grid, r0, c0, r1, c1) / ((r1 - r0 + 1) * (c1 - c0 + 1)))
        out.append(row)
    return out


def worst_difference(a: Grid, b: Grid) -> float:
    rows = zip(a, b, strict=True)
    return max(abs(x - y) for ra, rb in rows for x, y in zip(ra, rb, strict=True))


def random_grid(size: int, rng: random.Random, low: float = 0.0, high: float = 1.0) -> Grid:
    return [[rng.uniform(low, high) for _ in range(size)] for _ in range(size)]


def integer_grid(size: int, rng: random.Random, high: int = 1000) -> Grid:
    return [[float(rng.randint(0, high)) for _ in range(size)] for _ in range(size)]


def timings(grid: Grid, radius: int) -> tuple[float, float, float]:
    t0 = time.perf_counter()
    table = SummedArea(grid)
    built = time.perf_counter() - t0
    t0 = time.perf_counter()
    table.mean_field(radius)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    naive_mean_field(grid, radius)
    slow = time.perf_counter() - t0
    return built, fast, slow


def local_variance(grid: Grid, radius: int) -> Grid:
    table = SummedArea(grid)
    squares = SummedArea([[v * v for v in row] for row in grid])
    out = []
    for r in range(table.rows):
        row = []
        for c in range(table.cols):
            mean = table.window_mean(r, c, radius)
            row.append(max(0.0, squares.window_mean(r, c, radius) - mean * mean))
        out.append(row)
    return out


def count_in_window(mask: list[list[bool]], r: int, c: int, radius: int) -> int:
    table = SummedArea([[1.0 if v else 0.0 for v in row] for row in mask])
    r0, r1 = max(0, r - radius), min(table.rows - 1, r + radius)
    c0, c1 = max(0, c - radius), min(table.cols - 1, c + radius)
    return round(table.box_sum(r0, c0, r1, c1))
