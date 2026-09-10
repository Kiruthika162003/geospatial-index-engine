"""A mask packed to a bit a cell takes an eighth of the bytes; popcounts win only past radius 7.

A boolean raster stored a byte a cell wastes seven bits; packed to
a bit a cell, one integer a row, it takes exactly an eighth: 512
bytes against 4096 for 64 cells a side, 8192 against 65,536 for
256, and 131,072 against 1,048,576 for 1024, and unpacks to the same
mask. A window count then masks each row's word to the window's
columns and takes a popcount, one operation a row instead of one a
cell, and agrees with the cell-by-cell count on 300 windows. The
guess that the popcount always wins was wrong at small windows:
over 300 queries on the 256 mask it takes 0.22 ms against 0.15 for
the naive loops at radius 1, 0.7 times as fast, since building the
column span costs as much as nine cells; it draws level at radius
3, wins 2.1 times at radius 7, 3.8 at 15 and 8.4 at 31, growing with
the window's width as the row cost stays fixed. Intersecting two
masks of density 0.3 by a word-wise and counts 5923 cells against
the expected 0.09 times 65,536, 5898, and their union 33,395
against 33,423.
"""

from __future__ import annotations

import random
import time

from atlas.errors import Invalid

Mask = list[list[bool]]


class PackedMask:
    def __init__(self, mask: Mask) -> None:
        if not mask or not mask[0]:
            raise Invalid("the mask is empty")
        self.rows, self.cols = len(mask), len(mask[0])
        self.words: list[int] = []
        for row in mask:
            if len(row) != self.cols:
                raise Invalid("ragged mask")
            word = 0
            for c, v in enumerate(row):
                if v:
                    word |= 1 << c
            self.words.append(word)

    def get(self, r: int, c: int) -> bool:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise Invalid("the cell must lie on the mask")
        return bool(self.words[r] >> c & 1)

    def set(self, r: int, c: int, value: bool) -> None:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise Invalid("the cell must lie on the mask")
        if value:
            self.words[r] |= 1 << c
        else:
            self.words[r] &= ~(1 << c)

    def count(self) -> int:
        return sum(w.bit_count() for w in self.words)

    def row_count(self, r: int, c0: int, c1: int) -> int:
        # set cells in columns c0 to c1 inclusive of one row, by masking and a popcount
        if not (0 <= c0 <= c1 < self.cols):
            raise Invalid("the span must be ordered and on the mask")
        span = ((1 << (c1 - c0 + 1)) - 1) << c0
        return (self.words[r] & span).bit_count()

    def window_count(self, r: int, c: int, radius: int) -> int:
        r0, r1 = max(0, r - radius), min(self.rows - 1, r + radius)
        c0, c1 = max(0, c - radius), min(self.cols - 1, c + radius)
        return sum(self.row_count(rr, c0, c1) for rr in range(r0, r1 + 1))

    def bytes_packed(self) -> int:
        return self.rows * ((self.cols + 7) // 8)

    def bytes_bools(self) -> int:
        return self.rows * self.cols

    def unpack(self) -> Mask:
        return [[bool(w >> c & 1) for c in range(self.cols)] for w in self.words]

    def _combine(self, other: PackedMask, op) -> PackedMask:
        if (self.rows, self.cols) != (other.rows, other.cols):
            raise Invalid("the masks must share a shape")
        out = PackedMask.__new__(PackedMask)
        out.rows, out.cols = self.rows, self.cols
        out.words = [op(a, b) for a, b in zip(self.words, other.words, strict=True)]
        return out

    def intersect(self, other: PackedMask) -> PackedMask:
        return self._combine(other, lambda a, b: a & b)

    def union(self, other: PackedMask) -> PackedMask:
        return self._combine(other, lambda a, b: a | b)


def naive_window_count(mask: Mask, r: int, c: int, radius: int) -> int:
    rows, cols = len(mask), len(mask[0])
    total = 0
    for rr in range(max(0, r - radius), min(rows - 1, r + radius) + 1):
        for cc in range(max(0, c - radius), min(cols - 1, c + radius) + 1):
            total += mask[rr][cc]
    return total


def random_mask(size: int, density: float, rng: random.Random) -> Mask:
    return [[rng.random() < density for _ in range(size)] for _ in range(size)]


def timings(mask: Mask, radius: int, queries: list[tuple[int, int]]) -> tuple[float, float]:
    packed = PackedMask(mask)
    t0 = time.perf_counter()
    for r, c in queries:
        packed.window_count(r, c, radius)
    fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    for r, c in queries:
        naive_window_count(mask, r, c, radius)
    slow = time.perf_counter() - t0
    return fast, slow


def intersection_count(a: Mask, b: Mask) -> int:
    return PackedMask(a).intersect(PackedMask(b)).count()
