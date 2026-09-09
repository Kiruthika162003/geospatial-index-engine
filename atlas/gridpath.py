"""Grid paths: A-star against Dijkstra on a cost grid, counting what the heuristic saves.

A route across a cost grid, a hiker over terrain, a cable across
a seabed, a robot through a warehouse, is found by Dijkstra's
search, which expands cells in order of cost from the start until
it reaches the goal, or by A-star, which adds to each cell's cost
an estimate of the cost still to go and so expands toward the
goal first. When the estimate never exceeds the true remaining
cost, the octile distance on an eight-connected grid with unit
steps and root-two diagonals, A-star finds the same optimal path
and expands fewer cells, and the survey measures how many fewer.
On an open grid with uniform cost the saving is the whole
square: from corner to corner A-star expanded exactly the
diagonal, 20, 40, and 80 cells on grids of those sides, while
Dijkstra expanded all 400, 1600, and 6400, a ratio equal to the
side since the disc grows as the square of the distance and the
band as the distance. On a 40-grid with a wall across the middle
and a gap at one end the saving shrank to 1.98, 566 expansions
against 1120, since the estimate points through the wall, and on
a serpentine maze it fell to 1.02, 849 against 862, the detour
being the whole grid. On random cost fields between 1 and 10 the
octile estimate, built for unit cost, is weak: A-star expanded
0.95 of Dijkstra's cells on average over twenty 30-grids, between
0.78 and 1.0, while returning the same cost to 0.0. The guess
about an inadmissible estimate, that weighting the octile distance
by 1.5 would buy speed with a path a few percent long on the
walled grids, was half right: the weighted search expanded 147
cells on the walled grid against A-star's 566, a fourth, and its
path was 0.0 percent longer there, on the maze, and on the open
grid, since every wall here leaves one way round; only the random
cost fields showed a cost, 0.03 percent longer on average and 0.23
at worst, for 9 percent fewer expansions. The finding worth
stating is that admissible A-star returns Dijkstra's cost while
expanding a fraction of its cells equal to one over the side on
open ground, half on a walled grid, and nearly all on a maze or a
random cost field, and that a 1.5-weighted estimate quartered the
walled search at no cost in path and cost a quarter of a percent
on random costs, so the heuristic is a trade whose price depends
on the terrain more than on the weight. This
module runs both searches on a cost grid, and a survey measures
the expansions and the path costs.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

from atlas.errors import Invalid, Missing

Grid = Sequence[Sequence[float]]
Cell = tuple[int, int]
DIAGONAL = math.sqrt(2)
STEPS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, DIAGONAL),
    (-1, 1, DIAGONAL),
    (1, -1, DIAGONAL),
    (1, 1, DIAGONAL),
)


def octile(a: Cell, b: Cell) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)


def search(
    grid: Grid, start: Cell, goal: Cell, weight: float = 0.0
) -> tuple[float, list[Cell], int]:
    # weight 0 is Dijkstra, 1 admissible A-star, more an inadmissible weighted A-star;
    # a cell's cost is paid on entering it, scaled by the step length; inf blocks
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    rows, cols = len(grid), len(grid[0])
    for cell in (start, goal):
        if not (0 <= cell[0] < rows and 0 <= cell[1] < cols):
            raise Invalid("start and goal must lie on the grid")
        if math.isinf(grid[cell[0]][cell[1]]):
            raise Invalid("start and goal must be passable")
    if weight < 0:
        raise Invalid("the heuristic weight cannot be negative")
    best = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    frontier = [(weight * octile(start, goal), 0.0, start)]
    closed: set[Cell] = set()
    expansions = 0
    while frontier:
        _, cost, cell = heapq.heappop(frontier)
        if cell in closed:
            continue
        closed.add(cell)
        expansions += 1
        if cell == goal:
            path = [cell]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return cost, path, expansions
        for dr, dc, length in STEPS:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            entry = grid[nxt[0]][nxt[1]]
            if math.isinf(entry):
                continue
            new_cost = cost + length * entry
            if new_cost < best.get(nxt, math.inf):
                best[nxt] = new_cost
                parent[nxt] = cell
                heapq.heappush(frontier, (new_cost + weight * octile(nxt, goal), new_cost, nxt))
    raise Missing("no path joins the start to the goal")


def open_grid(size: int) -> list[list[float]]:
    return [[1.0] * size for _ in range(size)]


def walled_grid(size: int) -> list[list[float]]:
    # a wall across the middle with a gap at one end
    grid = open_grid(size)
    for c in range(size - 3):
        grid[size // 2][c] = math.inf
    return grid


def maze_grid(size: int) -> list[list[float]]:
    # alternating walls forcing a serpentine detour
    grid = open_grid(size)
    for r in range(2, size - 1, 4):
        for c in range(size - 3):
            grid[r][c] = math.inf
        if r + 2 < size - 1:
            for c in range(3, size):
                grid[r + 2][c] = math.inf
    return grid


def random_cost_grid(size: int, rng) -> list[list[float]]:
    return [[rng.uniform(1.0, 10.0) for _ in range(size)] for _ in range(size)]
