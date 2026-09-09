"""Sixteen moves beat first-order fast marching to radius 40; the two err at opposite angles.

Distance over a unit grid can be read by Dijkstra over a move set or
by fast marching, which solves the eikonal equation cell by cell
from the two nearest frozen neighbours. Measured against the
Euclidean distance on rings of radius 10, 40 and 75 about the centre
of a 161 grid, four moves read 41.42 percent high at worst on the
diagonals and 26 to 28 percent high on average, the Manhattan law
root two minus one; eight moves read 8.20 to 8.24 percent high at
worst near 22.5 degrees off an axis and 5.3 to 5.5 on average, the
octile law of 8.239; sixteen moves, the knight's moves added, read
2.69 to 2.75 at worst near 13 degrees and 1.4 on average. None of
these improve with distance, since the error is the metric's own.

The guess that fast marching beats every move set was wrong within
40 cells: its worst error on the rings reads 10.6, 7.31, 4.64, 2.85
and 1.78 percent at radii 5, 10, 20, 40 and 75, and its mean 6.37,
4.28, 2.71, 1.78 and 1.07, falling with distance as the first-order
scheme's own error at the source is spread over a longer path, so it
passes eight moves only beyond radius 10 and sixteen moves only
beyond 40. Its worst cells sit on the diagonals, where Dijkstra with
eight moves is exact, and Dijkstra's worst cells sit near the axes,
where fast marching is nearly exact. Around a wall with one gap the
path to a point 107.08 off the axis reads 112.43 with eight moves, 5
percent high, 107.08 with sixteen and 108.62 by fast marching. On
the 161 grid the four runs take 0.02, 0.03, 0.07 and 0.03 seconds.
"""

from __future__ import annotations

import heapq
import math

from atlas.errors import Invalid

Grid = list[list[float]]
Cell = tuple[int, int]

MOVES = {
    4: [(1, 0), (-1, 0), (0, 1), (0, -1)],
    8: [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc],
    16: [
        (dr, dc)
        for dr in (-2, -1, 0, 1, 2)
        for dc in (-2, -1, 0, 1, 2)
        if (dr or dc) and math.gcd(abs(dr), abs(dc)) == 1
    ],
}


def dijkstra(
    size: int, source: Cell, connectivity: int = 8, blocked: set[Cell] | None = None
) -> Grid:
    if size < 1:
        raise Invalid("the grid needs at least one cell")
    if connectivity not in MOVES:
        raise Invalid("connectivity must be 4, 8 or 16")
    sr, sc = source
    if not (0 <= sr < size and 0 <= sc < size):
        raise Invalid("the source must lie on the grid")
    blocked = blocked or set()
    dist = [[math.inf] * size for _ in range(size)]
    dist[sr][sc] = 0.0
    heap = [(0.0, sr, sc)]
    moves = [(dr, dc, math.hypot(dr, dc)) for dr, dc in MOVES[connectivity]]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        for dr, dc, length in moves:
            rr, cc = r + dr, c + dc
            if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in blocked:
                knight = abs(dr) == 2 or abs(dc) == 2
                if knight and (r + int(dr / 2), c + int(dc / 2)) in blocked:
                    continue
                nd = d + length
                if nd < dist[rr][cc]:
                    dist[rr][cc] = nd
                    heapq.heappush(heap, (nd, rr, cc))
    return dist


def fast_marching(size: int, source: Cell, blocked: set[Cell] | None = None) -> Grid:
    # first-order fast marching for |grad T| = 1 on a unit grid
    if size < 1:
        raise Invalid("the grid needs at least one cell")
    sr, sc = source
    if not (0 <= sr < size and 0 <= sc < size):
        raise Invalid("the source must lie on the grid")
    blocked = blocked or set()
    dist = [[math.inf] * size for _ in range(size)]
    frozen = [[False] * size for _ in range(size)]
    dist[sr][sc] = 0.0
    heap = [(0.0, sr, sc)]
    while heap:
        _, r, c = heapq.heappop(heap)
        if frozen[r][c]:
            continue
        frozen[r][c] = True
        for dr, dc in MOVES[4]:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < size and 0 <= cc < size) or frozen[rr][cc] or (rr, cc) in blocked:
                continue
            candidate = _update(dist, rr, cc, size)
            if candidate < dist[rr][cc]:
                dist[rr][cc] = candidate
                heapq.heappush(heap, (candidate, rr, cc))
    return dist


def _update(dist: Grid, r: int, c: int, size: int) -> float:
    a = min(dist[r - 1][c] if r > 0 else math.inf, dist[r + 1][c] if r + 1 < size else math.inf)
    b = min(dist[r][c - 1] if c > 0 else math.inf, dist[r][c + 1] if c + 1 < size else math.inf)
    if a == math.inf and b == math.inf:
        return math.inf
    if abs(a - b) >= 1.0:
        return min(a, b) + 1.0
    return (a + b + math.sqrt(2 - (a - b) ** 2)) / 2


Errors = list[tuple[float, float]]


def ring_errors(dist: Grid, source: Cell, radius: float, tolerance: float = 0.5) -> Errors:
    sr, sc = source
    out = []
    for r, row in enumerate(dist):
        for c, d in enumerate(row):
            true = math.hypot(r - sr, c - sc)
            if abs(true - radius) <= tolerance and true > 0:
                angle = math.degrees(math.atan2(c - sc, -(r - sr))) % 360
                out.append((angle, d / true - 1))
    return out


def worst_and_mean(errors: list[tuple[float, float]]) -> tuple[float, float, float]:
    if not errors:
        raise Invalid("no cells on the ring")
    values = [e for _, e in errors]
    index = max(range(len(values)), key=lambda i: abs(values[i]))
    return values[index], sum(values) / len(values), errors[index][0]


def octile_law(theta_deg: float) -> float:
    t = math.radians(theta_deg % 90)
    t = min(t, math.pi / 2 - t)
    return math.cos(t) + (math.sqrt(2) - 1) * math.sin(t) - 1


def manhattan_law(theta_deg: float) -> float:
    t = math.radians(theta_deg % 90)
    return math.cos(t) + math.sin(t) - 1


def wall_with_gap(size: int, column: int, gap_row: int) -> set[Cell]:
    return {(r, column) for r in range(size) if r != gap_row}
