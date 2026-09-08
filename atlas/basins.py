"""Drainage basins: label every cell by the sink it drains to, one basin per sink, no gaps.

Once every cell has a D8 flow direction, the grid falls into drainage
basins: the set of cells whose water ends at the same sink is that
sink's basin, the catchment a river gauge, a reservoir, or a flood
model reasons about. Labeling them is a walk: follow each cell's
direction until a sink, and give the cell that sink's label, with
memoization so a cell whose downstream neighbor is already labeled
inherits the label in one step rather than re-walking the chain. The
result partitions the grid, and the partition has properties worth
measuring rather than assuming. Every cell gets exactly one label, so
the basins are disjoint and cover the grid, and the number of basins
equals the number of sinks, one each. Each basin's cell count equals
its sink's flow accumulation, since accumulation counts exactly the
cells that drain through the sink and the sink is the last cell on
every path to it, which is the cross-check between the two
computations, labeling and counting, that must agree cell for cell.
And a basin is connected under eight-neighbor adjacency, because
every cell in it reaches the sink by a chain of neighbor steps that
all stay in the basin, so no basin is two islands. On a bowl there is
one basin holding every cell; on a surface with two separated bowls
there are two basins, split along the ridge between them, and the
survey checks that the split follows the ridge by testing that the
cells on each side label to the bowl on that side. The finding worth
stating is that basin labeling and flow accumulation are two views of
one partition, agreeing in count and in size for every sink, with
each basin a connected patch, so the catchment map is consistent with
the drainage counts. This module labels basins from flow directions,
and a survey confirms the cover, the size agreement, and the
connectivity.
"""

from __future__ import annotations

from atlas.errors import Invalid
from atlas.flowdirection import NOWHERE


def label_basins(directions: list[list[tuple[int, int]]]) -> list[list[int]]:
    if directions is None or not directions or not directions[0]:
        raise Invalid("the direction grid must not be empty")
    rows, cols = len(directions), len(directions[0])
    labels = [[-1] * cols for _ in range(rows)]
    sink_ids: dict[tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            if labels[r][c] != -1:
                continue
            chain = []
            cell = (r, c)
            while labels[cell[0]][cell[1]] == -1 and cell not in chain:
                chain.append(cell)
                nxt = directions[cell[0]][cell[1]]
                if nxt == NOWHERE:
                    break
                cell = nxt
            if labels[cell[0]][cell[1]] != -1:
                label = labels[cell[0]][cell[1]]
            elif directions[cell[0]][cell[1]] == NOWHERE:
                label = sink_ids.setdefault(cell, len(sink_ids))
            else:
                raise Invalid("the drainage graph contains a cycle")
            for cr, cc in chain:
                labels[cr][cc] = label
    return labels


def basin_sizes(labels: list[list[int]]) -> dict[int, int]:
    sizes: dict[int, int] = {}
    for row in labels:
        for label in row:
            sizes[label] = sizes.get(label, 0) + 1
    return sizes


def is_connected(labels: list[list[int]], label: int) -> bool:
    rows, cols = len(labels), len(labels[0])
    cells = [(r, c) for r in range(rows) for c in range(cols) if labels[r][c] == label]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if not ((dr or dc) and 0 <= nr < rows and 0 <= nc < cols):
                    continue
                if labels[nr][nc] == label and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == len(cells)
