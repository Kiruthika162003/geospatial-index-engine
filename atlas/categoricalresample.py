"""Coarsening a class raster: nearest keeps every class within 6 percent, majority erases one.

A class raster coarsened by a factor can take the centre cell of
each block or the block's majority. On a 96-cell landscape of
four blocky classes with a fifth class sprinkled over 5.1 percent
of the cells in 559 patches, nearest sampling changes the class
shares by 0.4, 1.0, 2.4, 2.2 and 5.7 percent at factors 2, 3, 4, 6
and 8, loses no class, and keeps the sprinkled class at 5.3, 5.7,
6.1, 6.3 and 4.9 percent, since each sampled cell is sprinkled with
the same chance as any other. Majority sampling loses the sprinkled
class at every factor, reading 0.0 against a binomial law of
0.00048 at factor 2 and 3e-5 at 3, and changes the shares by the
whole 5.1 percent, 9.6 at factor 4; over ten landscapes at factor 4
the share error reads 2.7 percent for nearest and 9.7 for majority.
Majority does make the cleaner map: 131, 131, 116, 131 and 80
patches against nearest's 243, 183, 167, 144 and 85.

The guess that nearest sampling keeps thin features was wrong: it
keeps them by the luck of phase. Four one-cell roads holding 4.17
percent of a scene read 2.08 percent at factor 2, 0.0 at factor 3
and 8.33 at factor 4 under nearest sampling, according to whether
the sampled column of each block lands on a road, while majority
reads 0.0 at every factor.
"""

from __future__ import annotations

import math
import random

from atlas.errors import Invalid

Classes = list[list[int]]


def _check(grid: Classes, factor: int) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the raster is empty")
    rows, cols = len(grid), len(grid[0])
    if factor < 1 or rows % factor or cols % factor:
        raise Invalid("the factor must divide both sides")
    return rows, cols


def nearest(grid: Classes, factor: int) -> Classes:
    rows, cols = _check(grid, factor)
    half = factor // 2
    return [
        [grid[r + half][c + half] for c in range(0, cols, factor)]
        for r in range(0, rows, factor)
    ]


def majority(grid: Classes, factor: int) -> Classes:
    rows, cols = _check(grid, factor)
    out = []
    for r in range(0, rows, factor):
        row = []
        for c in range(0, cols, factor):
            counts: dict[int, int] = {}
            for rr in range(r, r + factor):
                for cc in range(c, c + factor):
                    counts[grid[rr][cc]] = counts.get(grid[rr][cc], 0) + 1
            row.append(max(sorted(counts), key=lambda k: counts[k]))
        out.append(row)
    return out


def shares(grid: Classes) -> dict[int, float]:
    cells = [v for row in grid for v in row]
    out: dict[int, float] = {}
    for v in cells:
        out[v] = out.get(v, 0.0) + 1
    return {k: n / len(cells) for k, n in sorted(out.items())}


def share_error(fine: Classes, coarse: Classes) -> float:
    a, b = shares(fine), shares(coarse)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)) / 2


def lost_classes(fine: Classes, coarse: Classes) -> int:
    return len(set(shares(fine)) - set(shares(coarse)))


def fragments(grid: Classes) -> int:
    # the number of four-connected patches
    rows, cols = len(grid), len(grid[0])
    seen = [[False] * cols for _ in range(rows)]
    count = 0
    for r in range(rows):
        for c in range(cols):
            if seen[r][c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if (
                        0 <= yy < rows
                        and 0 <= xx < cols
                        and not seen[yy][xx]
                        and grid[yy][xx] == grid[r][c]
                    ):
                        seen[yy][xx] = True
                        stack.append((yy, xx))
    return count


def landscape(
    size: int, rng: random.Random, kinds: int = 4, patch: int = 6, sprinkle: float = 0.05
) -> Classes:
    # blocky patches with a sprinkled minority class of index kinds
    if kinds < 2 or patch < 1:
        raise Invalid("a landscape needs two kinds and a positive patch")
    blocks = [
        [rng.randrange(kinds) for _ in range(size // patch + 1)]
        for _ in range(size // patch + 1)
    ]
    out = []
    for r in range(size):
        row = []
        for c in range(size):
            row.append(kinds if rng.random() < sprinkle else blocks[r // patch][c // patch])
        out.append(row)
    return out


def road_scene(size: int, rng: random.Random, roads: int = 4) -> Classes:
    # a background with one-cell-wide roads of class 9
    out = [[rng.randrange(3) // 3 for _ in range(size)] for _ in range(size)]
    columns = sorted(rng.sample(range(size), roads))
    for r in range(size):
        for c in columns:
            out[r][c] = 9
    return out


def majority_survival_law(factor: int, sprinkle: float) -> float:
    # a sprinkled class wins a block of factor squared cells only with more than half of them
    if factor < 1 or not 0 <= sprinkle <= 1:
        raise Invalid("factor must be positive and sprinkle a fraction")
    n = factor * factor
    total = 0.0
    for k in range(n // 2 + 1, n + 1):
        total += math.comb(n, k) * sprinkle**k * (1 - sprinkle) ** (n - k)
    return total
