"""A one-cell misregistration reads a 2 percent land-cover change as 10.7; no filter undoes it.

Change between two class rasters is read cell by cell, and any
misregistration between them fakes change along every patch edge.
On 128-cell scenes of square patches in four classes, shifting the
second raster by 1, 2 and 4 cells fakes change on 18.8, 37.6 and
75.2 percent of the cells with 4-cell patches, 9.1, 18.2 and 36.3
with 8-cell patches, 5.1, 10.2 and 20.3 with 16 and 2.3, 4.7 and 9.4
with 32, which is the shift over the patch side times the three
quarters of neighbouring patches that differ, within 3 percent of
the law at every size; a diagonal shift of one cell reads 1.74, 1.83
and 1.97 times the axial one on 4, 8 and 32-cell patches. Cohen's
kappa falls from 1.0 to 0.749 for a one-cell
shift on 4-cell patches and to -0.004 for a four-cell shift, pure
chance, while 32-cell patches keep 0.968 under a one-cell shift.

The guess that a majority filter cleans the salt and pepper of
misregistration was wrong: with 8-cell patches a real change of 2,
5 and 10 percent reads 1.8, 4.8 and 9.7 percent aligned and 10.65,
13.4 and 17.7 percent misregistered by one cell, and a 3 by 3
majority filter on both rasters reads 9.1, 9.2 and 9.5, since it
smooths the scattered real changes away while the shifted edges,
being whole lines, survive it. Kappa reads 0.976, 0.936 and 0.870
aligned and 0.858, 0.821 and 0.764 misregistered; a raster against
itself reads 1.0 and against an unrelated scene -0.019.
"""

from __future__ import annotations

import random

from atlas.errors import Invalid

Classes = list[list[int]]


def _check(a: Classes, b: Classes) -> tuple[int, int]:
    if not a or not a[0]:
        raise Invalid("the raster is empty")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise Invalid("the rasters must share a shape")
    return len(a), len(a[0])


def confusion(before: Classes, after: Classes) -> dict[tuple[int, int], int]:
    rows, cols = _check(before, after)
    table: dict[tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            key = (before[r][c], after[r][c])
            table[key] = table.get(key, 0) + 1
    return table


def change_fraction(before: Classes, after: Classes) -> float:
    rows, cols = _check(before, after)
    changed = sum(1 for r in range(rows) for c in range(cols) if before[r][c] != after[r][c])
    return changed / (rows * cols)


def kappa(table: dict[tuple[int, int], int]) -> float:
    total = sum(table.values())
    if total == 0:
        raise Invalid("an empty table has no agreement")
    labels = {k[0] for k in table} | {k[1] for k in table}
    agree = sum(v for (a, b), v in table.items() if a == b) / total
    row_sums = {label: sum(v for (a, _), v in table.items() if a == label) for label in labels}
    col_sums = {label: sum(v for (_, b), v in table.items() if b == label) for label in labels}
    chance = sum(row_sums[label] * col_sums[label] for label in labels) / (total * total)
    if chance == 1:
        return 1.0
    return (agree - chance) / (1 - chance)


def shifted(classes: Classes, dr: int, dc: int) -> Classes:
    rows, cols = len(classes), len(classes[0])
    return [
        [classes[(r - dr) % rows][(c - dc) % cols] for c in range(cols)] for r in range(rows)
    ]


def patches(size: int, patch: int, kinds: int, rng: random.Random) -> Classes:
    if patch < 1 or kinds < 2:
        raise Invalid("patches need a positive size and two kinds")
    blocks = [
        [rng.randrange(kinds) for _ in range(size // patch + 1)]
        for _ in range(size // patch + 1)
    ]
    return [[blocks[r // patch][c // patch] for c in range(size)] for r in range(size)]


def edge_fraction(classes: Classes) -> float:
    # cells whose right or lower neighbour differs, over the cells
    rows, cols = len(classes), len(classes[0])
    edges = 0
    for r in range(rows):
        for c in range(cols):
            if (
                classes[r][c] != classes[r][(c + 1) % cols]
                or classes[r][c] != classes[(r + 1) % rows][c]
            ):
                edges += 1
    return edges / (rows * cols)


def real_change(classes: Classes, fraction: float, kinds: int, rng: random.Random) -> Classes:
    if not 0 <= fraction <= 1:
        raise Invalid("the fraction must lie in [0, 1]")
    out = [list(row) for row in classes]
    for r in range(len(out)):
        for c in range(len(out[0])):
            if rng.random() < fraction:
                out[r][c] = (out[r][c] + rng.randrange(1, kinds)) % kinds
    return out


def spurious_change_law(patch: int, shift: int, kinds: int) -> float:
    # a shift of s cells on square patches of side p flips the cells within s of a patch edge
    # in the shifted direction, and a share (k - 1) / k of neighbouring patches differ
    if patch < 1 or kinds < 2:
        raise Invalid("patches need a positive size and two kinds")
    return min(1.0, shift / patch) * (kinds - 1) / kinds


def majority_filter(classes: Classes, radius: int = 1) -> Classes:
    rows, cols = len(classes), len(classes[0])
    out = [list(row) for row in classes]
    for r in range(rows):
        for c in range(cols):
            counts: dict[int, int] = {}
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        counts[classes[rr][cc]] = counts.get(classes[rr][cc], 0) + 1
            out[r][c] = max(counts, key=lambda k: (counts[k], k == classes[r][c]))
    return out
