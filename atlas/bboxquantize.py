"""Quantized boxes never miss: 8 bits fatten a 1-unit box 24-fold and make 13 percent false.

An R-tree node can store its boxes as small integers on a grid over
the node's extent, rounding each low edge down and each high edge
up, so the stored box always contains the true one. Over a 1000
square with 2000 boxes and 200 queries of side 50, no quantization
tried ever missed a true hit, and every fattened box contained its
original. The price is fat: a box of side s on a grid of cell c
grows to (s + c) squared over s squared on average, since the floor
and the ceiling each add half a cell; the guess of one and a half
cells was high. Measured against that law, 1-unit boxes inflate
4604-fold at 4 bits against 4579, 23.95-fold at 8 bits against
24.22, 1.55 at 12 and 1.03 at 16; 5-unit boxes 203, 3.17, 1.10 and
1.01; 20-unit boxes 18.4, 1.43, 1.02 and 1.00; 100-unit boxes 2.74,
1.08, 1.00 and 1.00.

Fat boxes make false hits. Among the quantized hits of the 50-wide
queries the false share reads 80.7, 41.1, 13.1, 4.0, 1.3 and 0.19
percent for 1-unit boxes at 4, 6, 8, 10, 12 and 16 bits, 79.5, 39.0,
12.1, 4.2, 1.0 and 0.17 for 5-unit boxes, 73.3, 32.7, 9.3, 2.6, 0.66
and 0.05 for 20-unit boxes, and 49.3, 16.9, 4.5, 1.15, 0.29 and 0.02
for 100-unit boxes. A box costs 4 bytes at 8 bits and 8 at 16
against 32 for four doubles; a 1-unit cell over 1000 needs 10 bits
and a 0.01 cell 17.
"""

from __future__ import annotations

import math
import random

from atlas.bbox import BBox
from atlas.errors import Invalid


class Quantizer:
    def __init__(self, extent: BBox, bits: int) -> None:
        if not 1 <= bits <= 32:
            raise Invalid("bits must lie between 1 and 32")
        if extent.max_x <= extent.min_x or extent.max_y <= extent.min_y:
            raise Invalid("the extent must have area")
        self.extent = extent
        self.bits = bits
        self.steps = (1 << bits) - 1
        self.sx = (extent.max_x - extent.min_x) / self.steps
        self.sy = (extent.max_y - extent.min_y) / self.steps

    def encode(self, box: BBox) -> tuple[int, int, int, int]:
        if not self.extent.contains_box(box):
            raise Invalid("the box must lie within the extent")
        x0 = math.floor((box.min_x - self.extent.min_x) / self.sx)
        y0 = math.floor((box.min_y - self.extent.min_y) / self.sy)
        x1 = math.ceil((box.max_x - self.extent.min_x) / self.sx)
        y1 = math.ceil((box.max_y - self.extent.min_y) / self.sy)
        clamp = lambda v: min(max(v, 0), self.steps)  # noqa: E731
        return clamp(x0), clamp(y0), clamp(x1), clamp(y1)

    def decode(self, code: tuple[int, int, int, int]) -> BBox:
        x0, y0, x1, y1 = code
        return BBox(
            self.extent.min_x + x0 * self.sx,
            self.extent.min_y + y0 * self.sy,
            self.extent.min_x + x1 * self.sx,
            self.extent.min_y + y1 * self.sy,
        )

    def fatten(self, box: BBox) -> BBox:
        return self.decode(self.encode(box))

    def bytes_per_box(self) -> float:
        return 4 * self.bits / 8


def contains_original(fat: BBox, box: BBox) -> bool:
    return fat.contains_box(box)


def inflation(fat: BBox, box: BBox) -> float:
    if box.area() == 0:
        return math.inf
    return fat.area() / box.area()


def false_hit_rate(
    boxes: list[BBox], queries: list[BBox], quantizer: Quantizer
) -> tuple[float, int, int, int]:
    # among quantized hits, the share whose true boxes do not intersect the query
    fat = [quantizer.fatten(b) for b in boxes]
    true_hits = fat_hits = 0
    misses = 0
    for q in queries:
        for box, big in zip(boxes, fat, strict=True):
            truth = box.intersects(q)
            guess = big.intersects(q)
            true_hits += truth
            fat_hits += guess
            if truth and not guess:
                misses += 1
    false = fat_hits - true_hits
    return (false / fat_hits if fat_hits else 0.0), true_hits, fat_hits, misses


def random_boxes(n: int, rng: random.Random, size: float, side: float = 1000.0) -> list[BBox]:
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, side - size), rng.uniform(0, side - size)
        out.append(BBox(x, y, x + size, y + size))
    return out


def mean_inflation(boxes: list[BBox], quantizer: Quantizer) -> float:
    return sum(inflation(quantizer.fatten(b), b) for b in boxes) / len(boxes)


def expected_inflation(size: float, cell: float) -> float:
    # floor on the low side and ceil on the high side each add half a cell on average
    return ((size + cell) / size) ** 2


def bits_for_cell(extent_side: float, cell: float) -> int:
    return math.ceil(math.log2(extent_side / cell + 1))
