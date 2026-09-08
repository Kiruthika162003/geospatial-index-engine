"""Strahler stream order: a stream's rank rises only where two streams of equal rank meet.

A drainage network has a hierarchy, headwater rills feeding brooks
feeding rivers, and the Strahler order names the levels by one rule.
A stream with no tributaries is order one. Where two streams join, if
they have the same order the result is one order higher, and if they
differ the result keeps the larger of the two; a stream absorbing a
lesser tributary does not grow in rank. The rule makes order a
measure of branching depth rather than of length or flow: a long
river fed only by first-order brooks stays order two no matter how
many it collects, while a symmetric tree of confluences climbs one
order per level. On a D8 drainage grid the network is the tree of
cells that drain into each other, and the order of every cell is
computed from its inflowing neighbors by the rule, in topological
order from the headwaters down, exactly as flow accumulation is. Three
properties pin the computation and are worth measuring rather than
assuming. A perfectly symmetric binary confluence tree of depth k has
outlet order k plus one, since every join is between equals, which is
the calibration. A chain of cells with a single line of tributaries
each of order one stays at order two all the way down, since every
join is between unequals, which is the rule's other half. And order
is bounded by the logarithm of the cell count, base two, because
reaching order k needs at least two to the k minus one headwater
cells, so no basin of n cells can exceed order one plus log two of n,
which the survey checks on random rough surfaces. The finding worth
stating is that Strahler order climbs one per level on a symmetric
tree, stays flat on a chain of unequal joins, and never exceeds the
logarithmic bound, so it measures branching structure and nothing
else. This module computes stream order over D8 directions, and a
survey confirms the three properties.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid
from atlas.flowdirection import NOWHERE


def stream_order(directions: list[list[tuple[int, int]]]) -> list[list[int]]:
    if directions is None or not directions or not directions[0]:
        raise Invalid("the direction grid must not be empty")
    rows, cols = len(directions), len(directions[0])
    inflow = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            target = directions[r][c]
            if target != NOWHERE:
                inflow[target[0]][target[1]] += 1
    # the two largest orders arriving at each cell decide its own order
    best = [[0] * cols for _ in range(rows)]
    second = [[0] * cols for _ in range(rows)]
    order = [[1] * cols for _ in range(rows)]
    pending = [[inflow[r][c] for c in range(cols)] for r in range(rows)]
    ready = [(r, c) for r in range(rows) for c in range(cols) if inflow[r][c] == 0]
    while ready:
        r, c = ready.pop()
        if best[r][c] == 0:
            order[r][c] = 1
        elif best[r][c] == second[r][c]:
            order[r][c] = best[r][c] + 1
        else:
            order[r][c] = best[r][c]
        target = directions[r][c]
        if target == NOWHERE:
            continue
        tr, tc = target
        o = order[r][c]
        if o > best[tr][tc]:
            second[tr][tc] = best[tr][tc]
            best[tr][tc] = o
        elif o > second[tr][tc]:
            second[tr][tc] = o
        pending[tr][tc] -= 1
        if pending[tr][tc] == 0:
            ready.append((tr, tc))
    if any(pending[r][c] != 0 for r in range(rows) for c in range(cols)):
        raise Invalid("the drainage graph contains a cycle")
    return order


def order_bound(cell_count: int) -> int:
    # reaching order k needs 2^(k-1) headwaters, so order <= 1 + log2(n)
    if cell_count < 1:
        raise Invalid("cell count must be positive")
    return 1 + int(math.log2(cell_count))
