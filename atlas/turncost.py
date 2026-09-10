"""A turn penalty of 0.1 halves the turns on noisy ground; rough ground charges 94 percent.

Routing over a cost surface with eight moves, the shortest path
wiggles wherever cheap cells lie off the line, and a penalty per 45
degrees of turn, carried in the search state as a heading, buys
straightness with travel cost. Between two cells 50 apart on a 61
grid whose costs are 1 plus noise of sigma 0.1, the free path makes
7 turns over a length of 52.5 at a travel cost of 49.80; a penalty
of 0.1 cuts it to 3 turns and 50.8 at 50.0, 0.3 to 2, and 1.0 to a
straight line at 51.34, 3 percent more than the wiggle. At sigma
0.3 the free path makes 22 turns over 59.7 at 42.01, penalties of
0.1, 0.3 and 1.0 read 13, 9 and 5 turns at 42.9, 43.3 and 44.7, and
3.0 forces the straight line at 54.03, 29 percent more. The guess
that a small penalty always buys straightness cheaply was wrong on
a rough surface: at sigma 0.6 the free path makes 33 turns over
64.5 at 30.37, a penalty of 1.0 still leaves 5 turns at 39.55, and
the straight line at penalty 10 costs 58.94, 94 percent more, since
the cheap cells it forgoes are worth more than the bends.

On an open grid the octile-optimal path between cells 50 by 30
apart can always be drawn with one turn, so the penalty changes the
shape but never the length: the free search returns 3 turns and a
penalty of 0.1 returns 1, both 62.43 long against a beeline of
58.31. A wall with a one-cell gap forces 2 turns at any penalty.
"""

from __future__ import annotations

import heapq
import math
import random
from itertools import pairwise

from atlas.errors import Invalid

Cell = tuple[int, int]
MOVES = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def route(
    size: int,
    start: Cell,
    goal: Cell,
    penalty: float,
    blocked: set[Cell] | None = None,
    cost: list[list[float]] | None = None,
) -> tuple[list[Cell], float]:
    # Dijkstra over (cell, heading) states, charging the penalty per 45 degrees of turn and
    # the entered cell's cost per unit of step
    if size < 2:
        raise Invalid("the grid needs two cells a side")
    if penalty < 0:
        raise Invalid("the penalty must not be negative")
    for r, c in (start, goal):
        if not (0 <= r < size and 0 <= c < size):
            raise Invalid("start and goal must lie on the grid")
    blocked = blocked or set()
    best: dict[tuple[Cell, int], float] = {}
    parent: dict[tuple[Cell, int], tuple[Cell, int] | None] = {}
    heap: list[tuple[float, Cell, int]] = []
    for h in range(len(MOVES)):
        best[(start, h)] = 0.0
        parent[(start, h)] = None
        heapq.heappush(heap, (0.0, start, h))
    final = None
    while heap:
        so_far, cell, heading = heapq.heappop(heap)
        if so_far > best.get((cell, heading), math.inf):
            continue
        if cell == goal:
            final = (cell, heading)
            break
        for h, (dr, dc) in enumerate(MOVES):
            nxt = (cell[0] + dr, cell[1] + dc)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in blocked:
                continue
            turn = _turn_units(heading, h) if cell != start else 0
            weight = cost[nxt[0]][nxt[1]] if cost is not None else 1.0
            step = math.hypot(dr, dc) * weight + penalty * turn
            candidate = so_far + step
            if candidate < best.get((nxt, h), math.inf):
                best[(nxt, h)] = candidate
                parent[(nxt, h)] = (cell, heading)
                heapq.heappush(heap, (candidate, nxt, h))
    if final is None:
        raise Invalid("the goal is unreachable")
    path = []
    state: tuple[Cell, int] | None = final
    while state is not None:
        path.append(state[0])
        state = parent[state]
    path.reverse()
    return path, best[final]


def _angle(index: int) -> float:
    dr, dc = MOVES[index]
    return math.atan2(dr, dc)


def _turn_units(before: int, after: int) -> int:
    diff = abs(_angle(before) - _angle(after))
    diff = min(diff, 2 * math.pi - diff)
    return round(diff / (math.pi / 4))


def turns(path: list[Cell]) -> int:
    count = 0
    for i in range(1, len(path) - 1):
        a = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        b = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if a != b:
            count += 1
    return count


def length(path: list[Cell]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairwise(path))


def noisy_cost(size: int, sigma: float, rng: random.Random) -> list[list[float]]:
    if sigma < 0:
        raise Invalid("sigma must not be negative")
    return [[max(0.2, 1 + rng.gauss(0, sigma)) for _ in range(size)] for _ in range(size)]


def travel_cost(path: list[Cell], cost: list[list[float]]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) * cost[b[0]][b[1]] for a, b in pairwise(path)
    )


def wall(size: int, column: int, gap: int) -> set[Cell]:
    return {(r, column) for r in range(size) if abs(r - gap) > 1}


def straightness(path: list[Cell]) -> float:
    if len(path) < 2:
        raise Invalid("a path needs two cells")
    return math.dist(path[0], path[-1]) / length(path)
