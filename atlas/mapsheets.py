"""A box a fifth of a sheet wide falls outside a single map sheet 36 percent of the time.

A map sheet index names the sheets of a regular grid, A001 in the
first row and column, AB012 in row 28, AAA004 in row 703, and the
names parse back exactly. A box dropped at random on a grid of
50-unit sheets sits within one sheet with probability (1 - w/W)(1 -
h/H): a box of side 5 reads 80.6 percent on one sheet, 18.3 on two
and 1.1 on four against a law of 81.0, side 10 reads 63.8, 31.9 and
4.3 against 64.0, side 20 reads 35.7, 48.0 and 16.3 against 36.0,
side 30 reads 16.0, 47.5 and 36.5 against 16.0, and a box the size
of a sheet always touches four. The mean count of sheets reads
1.217, 1.449, 1.968, 2.570 and 4.0 against the law (1 + w/W)(1 +
h/H) of 1.21, 1.44, 1.96, 2.56 and 4.0. The guess that subdividing
the sheets helps a box of their new size was wrong: a box of side
10 on sheets of side 10 touches four sheets every time, since it
always spans an edge.

A route of four legs across 200 units on 50-unit sheets is walked
at a quarter of a sheet a step and touches nine sheets in the order
the walk meets them.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

from atlas.errors import Invalid

Sheet = tuple[int, int]


class SheetGrid:
    def __init__(
        self, width: float, height: float, origin: tuple[float, float] = (0.0, 0.0)
    ) -> None:
        if width <= 0 or height <= 0:
            raise Invalid("sheets need positive width and height")
        self.width, self.height = width, height
        self.origin = origin

    def sheet_of(self, x: float, y: float) -> Sheet:
        return (
            math.floor((x - self.origin[0]) / self.width),
            math.floor((y - self.origin[1]) / self.height),
        )

    def name(self, sheet: Sheet) -> str:
        col, row = sheet
        letters = ""
        n = row
        while True:
            letters = chr(ord("A") + n % 26) + letters
            n = n // 26 - 1
            if n < 0:
                break
        return f"{letters}{col + 1:03d}"

    def parse(self, name: str) -> Sheet:
        letters = name.rstrip("0123456789")
        digits = name[len(letters) :]
        if not letters or not digits or not letters.isalpha() or not letters.isupper():
            raise Invalid("a sheet name is capital letters followed by digits")
        row = 0
        for ch in letters:
            row = row * 26 + (ord(ch) - ord("A") + 1)
        return int(digits) - 1, row - 1

    def sheets_covering(self, x0: float, y0: float, x1: float, y1: float) -> list[Sheet]:
        if x0 > x1 or y0 > y1:
            raise Invalid("the box must be ordered")
        c0, r0 = self.sheet_of(x0, y0)
        c1, r1 = self.sheet_of(x1, y1)
        return [(c, r) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def bounds(self, sheet: Sheet) -> tuple[float, float, float, float]:
        col, row = sheet
        x0 = self.origin[0] + col * self.width
        y0 = self.origin[1] + row * self.height
        return x0, y0, x0 + self.width, y0 + self.height


def straddle_histogram(
    grid: SheetGrid,
    box_w: float,
    box_h: float,
    rng: random.Random,
    trials: int,
    span: float = 1000.0,
) -> dict[int, float]:
    if trials < 1:
        raise Invalid("at least one trial is needed")
    counts: dict[int, int] = {}
    for _ in range(trials):
        x, y = rng.uniform(0, span), rng.uniform(0, span)
        n = len(grid.sheets_covering(x, y, x + box_w, y + box_h))
        counts[n] = counts.get(n, 0) + 1
    return {k: v / trials for k, v in sorted(counts.items())}


def one_sheet_law(box_w: float, box_h: float, sheet_w: float, sheet_h: float) -> float:
    # a box sits in one sheet when its corner falls in the sheet's inner rectangle
    if box_w >= sheet_w or box_h >= sheet_h:
        return 0.0
    return (1 - box_w / sheet_w) * (1 - box_h / sheet_h)


def expected_sheets(box_w: float, box_h: float, sheet_w: float, sheet_h: float) -> float:
    return (1 + box_w / sheet_w) * (1 + box_h / sheet_h)


def hierarchical(grid: SheetGrid, split: int) -> SheetGrid:
    if split < 2:
        raise Invalid("a subdivision needs at least two parts")
    return SheetGrid(grid.width / split, grid.height / split, grid.origin)


def sheets_for_route(grid: SheetGrid, points: list[tuple[float, float]]) -> list[Sheet]:
    if len(points) < 2:
        raise Invalid("a route needs two points")
    seen: list[Sheet] = []
    for (x0, y0), (x1, y1) in pairwise(points):
        steps = max(
            1, math.ceil(math.dist((x0, y0), (x1, y1)) / min(grid.width, grid.height) * 4)
        )
        for k in range(steps + 1):
            t = k / steps
            sheet = grid.sheet_of(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
            if sheet not in seen:
                seen.append(sheet)
    return seen
