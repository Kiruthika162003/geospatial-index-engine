"""Height above nearest drainage reads a V valley to 7.1; 0.1 of noise triples the floodplain.

Height above nearest drainage follows each cell's flow path to the
first stream cell, a cell whose flow accumulation passes a
threshold, and takes the height difference. On a 61-cell V valley
with walls of slope 0.5 and an axis falling 0.01 a row, a threshold
of 30 cells makes the axis and its two neighbours the stream, 4.9
percent of the cells, and the mean height reads 7.13 against 7.6
for the pure wall geometry, the difference being the stream cells
themselves. The floodplain within 0.5, 1, 2 and 5 of the stream
holds 7.7, 11.4, 18.0 and 37.7 percent of the cells against the
band law of 4.9, 8.2, 14.8 and 34.4, the stream's own width added.
The threshold is the whole story: at 5 cells 86.9 percent of the
valley counts as stream and the floodplain at 2 covers everything,
at 200 the stream is 1.56 percent and the floodplain 14.6, at 1000
1.21 and 13.9.

The guess that noise a fifth the size of a wall step would leave
the drainage alone was wrong: sigma 0.1 on the heights turns 18.2
percent of the valley into stream at the same threshold, since the
rows no longer flow straight to the axis but merge into rills that
pass 30 cells of accumulation on the way, the mean height falls to
2.45 and the floodplain at 2 rises to 49.7 percent; sigma 0.5 and 1
read 54.0 and 46.5. Filling the sinks first does not undo it, 50.4,
62.7 and 54.9, because the rills are convergence and not sinks, so
the threshold must grow with the noise. No cell reads below its
drainage.
"""

from __future__ import annotations

import random

from atlas.errors import Invalid
from atlas.flowdirection import NOWHERE, flow_accumulation, flow_directions

Grid = list[list[float]]
Cell = tuple[int, int]


def streams(dem: Grid, threshold: int) -> list[list[bool]]:
    if not dem or not dem[0]:
        raise Invalid("the elevation grid is empty")
    if threshold < 1:
        raise Invalid("the threshold must be at least one cell")
    accumulation = flow_accumulation(flow_directions(dem))
    return [[a >= threshold for a in row] for row in accumulation]


def hand(dem: Grid, threshold: int) -> tuple[Grid, list[list[bool]]]:
    # every cell's height above the stream cell its flow path first reaches;
    # directions hold the target cell of each step, NOWHERE for a sink
    directions = flow_directions(dem)
    channel = streams(dem, threshold)
    rows, cols = len(dem), len(dem[0])
    drain: list[list[float | None]] = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            path = []
            y, x = r, c
            steps = 0
            while drain[y][x] is None and not channel[y][x] and steps < rows * cols:
                path.append((y, x))
                target = directions[y][x]
                if target == NOWHERE:
                    break
                y, x = target
                steps += 1
            base = (
                dem[y][x]
                if channel[y][x]
                else (drain[y][x] if drain[y][x] is not None else dem[y][x])
            )
            for py, px in path:
                drain[py][px] = base
            if drain[r][c] is None:
                drain[r][c] = base
    heights = [[dem[r][c] - drain[r][c] for c in range(cols)] for r in range(rows)]
    return heights, channel


def floodplain_fraction(heights: Grid, level: float) -> float:
    cells = [v for row in heights for v in row]
    return sum(1 for v in cells if v <= level) / len(cells)


def stream_fraction(channel: list[list[bool]]) -> float:
    cells = [v for row in channel for v in row]
    return sum(1 for v in cells if v) / len(cells)


def v_valley(size: int, wall_slope: float, axis_slope: float = 0.01) -> Grid:
    centre = size // 2
    return [
        [wall_slope * abs(c - centre) + axis_slope * (size - r) for c in range(size)]
        for r in range(size)
    ]


def noisy(dem: Grid, sigma: float, rng: random.Random) -> Grid:
    return [[v + rng.gauss(0, sigma) for v in row] for row in dem]


def valley_law(size: int, wall_slope: float, level: float) -> float:
    # cells within level of the axis on a clean V valley: a band of half-width level over slope
    half = level / wall_slope
    return min(1.0, (2 * half + 1) / size)


def mean(values: Grid) -> float:
    cells = [v for row in values for v in row]
    return sum(cells) / len(cells)
