"""D8 flow direction and accumulation: every cell drains somewhere; the outlet collects it all.

Where does rain that lands on a cell go? The D8 answer, the standard
of hydrology on grids, sends it to whichever of the eight neighbors
lies steepest downhill, the drop in height divided by the distance to
that neighbor, one cell straight and root two diagonally, so a
diagonal neighbor must drop more to win. A cell with no lower
neighbor is a pit or the outlet and drains nowhere. Following the
directions from every cell traces the drainage network, and flow
accumulation counts, for each cell, how many cells drain through it,
itself included, which is the quantity that reveals rivers: cells
with high accumulation are the channels, and the outlet of a basin
accumulates every cell in it. Three properties pin the computation
and are worth measuring rather than assuming. On an inverted cone,
a bowl with its lowest point at the center, every cell drains to
that center, no pit interrupts, and the center's accumulation is
exactly the cell count, since every cell reaches it. On a tilted
plane every cell drains toward the low edge and each edge cell's
accumulation is the length of its column plus the diagonal spill,
which the survey checks by totals: the accumulations of the cells
that drain nowhere sum to the cell count, because every cell's water
ends in exactly one such sink. And the drainage graph is a forest,
free of cycles, since water only ever moves downhill, so the walk
from any cell reaches a sink in at most as many steps as there are
cells, which the survey confirms by walking every cell to its end.
The finding worth stating is that D8 partitions the grid into
drainage basins whose sink accumulations sum to the total cell count,
with a bowl draining entirely to its center, so accumulation is a
conserved count of contributing area. This module computes D8
directions and accumulations, and a survey confirms the conservation,
the bowl, and the absence of cycles.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
)
NOWHERE = (-1, -1)


def flow_directions(grid: list[list[float]]) -> list[list[tuple[int, int]]]:
    if grid is None or not grid or not grid[0]:
        raise Invalid("the grid must not be empty")
    rows, cols = len(grid), len(grid[0])
    out: list[list[tuple[int, int]]] = [[NOWHERE] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            best_drop = 0.0
            best = NOWHERE
            for dr, dc in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                distance = math.sqrt(2) if dr and dc else 1.0
                drop = (grid[r][c] - grid[nr][nc]) / distance
                if drop > best_drop:
                    best_drop = drop
                    best = (nr, nc)
            out[r][c] = best
    return out


def flow_accumulation(directions: list[list[tuple[int, int]]]) -> list[list[int]]:
    rows, cols = len(directions), len(directions[0])
    acc = [[1] * cols for _ in range(rows)]
    # count how many cells flow into each cell, then drain in topological order
    inflow = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            target = directions[r][c]
            if target != NOWHERE:
                inflow[target[0]][target[1]] += 1
    ready = [(r, c) for r in range(rows) for c in range(cols) if inflow[r][c] == 0]
    while ready:
        r, c = ready.pop()
        target = directions[r][c]
        if target == NOWHERE:
            continue
        tr, tc = target
        acc[tr][tc] += acc[r][c]
        inflow[tr][tc] -= 1
        if inflow[tr][tc] == 0:
            ready.append((tr, tc))
    return acc


def sinks(directions: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(len(directions))
        for c in range(len(directions[0]))
        if directions[r][c] == NOWHERE
    ]


def path_to_sink(directions: list[list[tuple[int, int]]], start: tuple[int, int]) -> int:
    # the number of steps from a cell to its sink; refuses to loop forever
    limit = len(directions) * len(directions[0])
    steps = 0
    cell = start
    while directions[cell[0]][cell[1]] != NOWHERE:
        cell = directions[cell[0]][cell[1]]
        steps += 1
        if steps > limit:
            raise Invalid("the drainage graph contains a cycle")
    return steps
