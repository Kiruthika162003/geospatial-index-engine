"""Sea level: which cells flood when the sea rises, by connection and by the bathtub rule.

A map of what a meter of sea-level rise floods is drawn two ways.
The bathtub rule colours every cell whose height is below the new
sea level, which is quick and wrong: a valley floor behind a
ridge is below sea level and stays dry, since the sea cannot
reach it. The connected rule floods only the cells below the
level that are joined to the sea, cell by cell through other
flooded cells, which is a flood fill from the coast, and the
difference between the two is the enclosed low ground the
bathtub overstates. The survey measures the overstatement on
terrain built to have it, a 30 by 30 grid with a coastal plain
rising half a unit per row from the sea at row 0, a ridge of
height 12 at row 20, and a basin of 270 cells at height 4 behind
it, lower than the plain's top of 9.5. At a level of 2 both
rules flooded 120 cells; at 5 the bathtub flooded 570 and the
connected rule 300, the difference being all 270 basin cells; at
9.9, 10.5, and 12.0 the bathtub read 870 and the connected rule
600, the basin still dry behind the ridge; and at 12.1, the sea
topping the ridge, both read 900, the whole grid. The
overstatement was exactly the basin's area below the level at
every level, and over 200 levels from 0 to 20 the connected area
never exceeded the bathtub's and both rose monotonically. The
rule of connection was measured with a one-cell gap of height 6
in the ridge whose cell behind was kept high, so the way through
is diagonal: at levels 5 and 7 the sea had not yet crossed the
plain and both connectivities read alike, 300 and 420; at 11,
above the plain's top and below the ridge, four-connectivity read
601, the gap cell alone, and eight read 870, the basin flooded
through the diagonal; and at 12.5 both read 900. The finding
worth stating is that the bathtub rule overstates flooding by
exactly the enclosed low ground, 270 cells here, until the sea
tops the enclosing ridge, that the two rules agree above it, and
that a diagonal gap floods a basin under eight-connectivity and
not four across the band of levels between the plain and the
ridge, so a flood map is a statement about connection as much as
about height. This module floods a
height grid by both rules, and a survey measures the
overstatement and the connectivity.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from atlas.errors import Invalid

Grid = Sequence[Sequence[float]]


def bathtub(grid: Grid, level: float) -> list[list[bool]]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return [[h < level for h in row] for row in grid]


def connected(
    grid: Grid, level: float, sea_cells: Sequence[tuple[int, int]], eight: bool = False
) -> list[list[bool]]:
    # cells below the level reachable from the sea cells through cells below the level
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    rows, cols = len(grid), len(grid[0])
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if eight:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    flooded = [[False] * cols for _ in range(rows)]
    queue: deque[tuple[int, int]] = deque()
    for r, c in sea_cells:
        if not (0 <= r < rows and 0 <= c < cols):
            raise Invalid("a sea cell lies off the grid")
        if grid[r][c] < level and not flooded[r][c]:
            flooded[r][c] = True
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in steps:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            if not flooded[rr][cc] and grid[rr][cc] < level:
                flooded[rr][cc] = True
                queue.append((rr, cc))
    return flooded


def area(mask: Sequence[Sequence[bool]]) -> int:
    return sum(1 for row in mask for v in row if v)


def overstatement(grid: Grid, level: float, sea_cells: Sequence[tuple[int, int]]) -> int:
    return area(bathtub(grid, level)) - area(connected(grid, level, sea_cells))


def coast_with_basin(size: int, plain_slope: float, ridge_height: float, basin_depth: float):
    # rows run inland from the sea at row 0: a plain rising with the row, a ridge at
    # two thirds of the way, and a basin behind it sunk below the ridge
    grid = []
    ridge_row = 2 * size // 3
    for r in range(size):
        if r < ridge_row:
            height = plain_slope * r
        elif r == ridge_row:
            height = ridge_height
        else:
            height = ridge_height - basin_depth
        grid.append([height] * size)
    return grid, [(0, c) for c in range(size)]


def with_diagonal_gap(grid: Sequence[Sequence[float]], gap_height: float):
    # lower one ridge cell and shift the low cell behind it by a column, so the gap is
    # passable only diagonally
    out = [list(row) for row in grid]
    ridge_row = next(r for r in range(len(out)) if out[r][0] > out[min(r + 1, len(out) - 1)][0])
    out[ridge_row][0] = gap_height
    for c in range(1, len(out[0])):
        out[ridge_row][c] = grid[ridge_row][c]
    # block the orthogonal way through: the cell straight behind the gap stays high
    out[ridge_row + 1][0] = grid[ridge_row][1]
    return out
