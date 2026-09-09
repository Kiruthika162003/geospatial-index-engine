"""The Hilbert walk makes 128 runs where rows make 1024, and 129 where rows make 16.

Run-length coding a mask depends on the walk that strings the cells
together. The guess that a Hilbert walk, which keeps neighbours
close, makes fewer runs than rows on any shape was wrong. On a 64 by
64 grid vertical stripes of period 8 code to 1024 runs in row order
and 128 in Hilbert order, eight times fewer, while the same stripes
laid horizontally code to 16 runs in row order and 129 in Hilbert
order, eight times more, with 1088 boundary edges either way; the
Hilbert walk simply cannot exploit a grain. On discs the two walks
trade places: a disc of radius 20 codes to 81 row runs and 67
Hilbert runs, radius 30 to 121 and 111, but discs of radius 0.4
times the side at sides 16, 32, 64, 128 and 256 code to 25, 53, 105,
205 and 409 row runs against 27, 59, 107, 223 and 395 Hilbert runs,
ratios 1.08, 1.11, 1.02, 1.09 and 0.97. Row runs on a disc are half
the boundary edges, 0.521 falling to 0.501 of them, since each row
crossing the disc opens one run at one edge and closes it at
another. Twelve blobs code to 211 and 205.

Noise defeats both walks alike: at densities 0.01, 0.05, 0.2 and 0.5
the row runs read 89, 369, 1336 and 2060 and the Hilbert runs 89,
371, 1328 and 2042, against the law 1 + (n squared - 1) 2 p (1 - p)
of 82, 390, 1311 and 2049. With a bit and a 16-bit count per run
the coded mask passes the 512 raw bytes of the grid at 5 percent
density, 785 bytes, and reaches 4378 at half. Decoding the Hilbert
runs returns every mask exactly.
"""

from __future__ import annotations

import math
import random

from atlas import hilbert
from atlas.errors import Invalid

Mask = list[list[bool]]
Run = tuple[bool, int]


def _check(mask: Mask) -> tuple[int, int]:
    if not mask or not mask[0]:
        raise Invalid("the mask is empty")
    return len(mask), len(mask[0])


def row_order(rows: int, cols: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(rows) for c in range(cols)]


def hilbert_order(rows: int, cols: int) -> list[tuple[int, int]]:
    if rows != cols or rows & (rows - 1):
        raise Invalid("the Hilbert walk needs a square grid whose side is a power of two")
    order = int(math.log2(rows))
    cells = []
    for d in range(rows * cols):
        x, y = hilbert.decode(d, order)
        cells.append((y, x))
    return cells


def encode(mask: Mask, walk: list[tuple[int, int]]) -> list[Run]:
    rows, cols = _check(mask)
    if len(walk) != rows * cols:
        raise Invalid("the walk must visit every cell once")
    runs: list[Run] = []
    for r, c in walk:
        value = mask[r][c]
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


def decode(runs: list[Run], walk: list[tuple[int, int]], rows: int, cols: int) -> Mask:
    if sum(n for _, n in runs) != len(walk):
        raise Invalid("the runs must cover the walk exactly")
    mask = [[False] * cols for _ in range(rows)]
    index = 0
    for value, count in runs:
        for _ in range(count):
            r, c = walk[index]
            mask[r][c] = value
            index += 1
    return mask


def run_count(mask: Mask, walk: list[tuple[int, int]]) -> int:
    return len(encode(mask, walk))


def boundary_length(mask: Mask) -> int:
    # the number of horizontal and vertical edges between a true cell and a false or outside one
    rows, cols = _check(mask)
    edges = 0
    for r in range(rows):
        for c in range(cols):
            if not mask[r][c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols) or not mask[rr][cc]:
                    edges += 1
    return edges


def bytes_needed(runs: list[Run], count_bits: int = 16) -> int:
    return math.ceil(len(runs) * (1 + count_bits) / 8)


def raw_bytes(rows: int, cols: int) -> int:
    return math.ceil(rows * cols / 8)


def disc(size: int, radius: float) -> Mask:
    centre = (size - 1) / 2
    return [
        [math.hypot(r - centre, c - centre) <= radius for c in range(size)] for r in range(size)
    ]


def stripes(size: int, period: int, vertical: bool) -> Mask:
    if period < 2:
        raise Invalid("a stripe needs a period of two")
    return [
        [((c if vertical else r) % period) < period // 2 for c in range(size)]
        for r in range(size)
    ]


def noise(size: int, density: float, rng: random.Random) -> Mask:
    return [[rng.random() < density for _ in range(size)] for _ in range(size)]


def blobs(size: int, count: int, radius: float, rng: random.Random) -> Mask:
    centres = [(rng.uniform(0, size), rng.uniform(0, size)) for _ in range(count)]
    return [
        [any(math.hypot(r - cy, c - cx) <= radius for cx, cy in centres) for c in range(size)]
        for r in range(size)
    ]


def expected_noise_runs(size: int, density: float) -> float:
    # a run starts at the first cell and wherever a cell differs from its predecessor
    return 1 + (size * size - 1) * 2 * density * (1 - density)
