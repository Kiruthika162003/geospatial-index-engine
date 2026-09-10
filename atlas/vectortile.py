"""Vector tiles hold more vertices in total at zoom 10 than the raw lines, 92,000 to 40,000.

A vector tile pipeline clips each line to every tile it crosses and
simplifies the pieces to a tolerance in tile pixels, so the
tolerance in map units shrinks with the tile. Twenty wandering
lines of 2000 vertices, 40,000 in all, land in 1, 14, 77, 483, 3521
and 21,629 tiles at zooms 0, 2, 4, 6, 8 and 10. The guess that the
total across the tiles falls with zoom as simplification bites was
wrong: with no simplification it reads 40,000, 40,202, 40,822,
43,300, 53,042 and 92,268, since every tile crossing adds a vertex
on each side of the border, and with a tolerance of half a pixel
2467, 6947, 19,209, 35,421, 49,933 and 90,165, the same at the top
within 2 percent because a half pixel of a zoom 10 tile is under
the lines' own wiggle. Two pixels reads 640 at zoom 0 and 85,938 at
zoom 10. The largest tile holds all 40,000 vertices at zoom 0 and
22 at zoom 10 whatever the tolerance, and the mean tile 4.3, so the
zoom decides the per-tile load while the tolerance decides only the
low zooms. One line simplifies from 2000 vertices to 1022, 207 and
19 at tolerances of 1e-4, 1e-3 and 1e-2 of the extent, and the walk
takes about a tenth of a second per zoom.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

from atlas.errors import Invalid

Point = tuple[float, float]
Line = list[Point]
Box = tuple[float, float, float, float]


def _perpendicular(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return math.dist(p, a)
    return abs(dx * (a[1] - p[1]) - (a[0] - p[0]) * dy) / length


def simplify(line: Line, tolerance: float) -> Line:
    # Douglas and Peucker, with an explicit stack
    if tolerance < 0:
        raise Invalid("the tolerance must not be negative")
    if len(line) < 3:
        return list(line)
    keep = [False] * len(line)
    keep[0] = keep[-1] = True
    stack = [(0, len(line) - 1)]
    while stack:
        i, j = stack.pop()
        worst, at = 0.0, -1
        for k in range(i + 1, j):
            d = _perpendicular(line[k], line[i], line[j])
            if d > worst:
                worst, at = d, k
        if at >= 0 and worst > tolerance:
            keep[at] = True
            stack.append((i, at))
            stack.append((at, j))
    return [p for p, k in zip(line, keep, strict=True) if k]


def _clip_segment(a: Point, b: Point, box: Box) -> tuple[Point, Point] | None:
    # Liang and Barsky
    x0, y0, x1, y1 = box
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    return (a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy)


def clip(line: Line, box: Box) -> list[Line]:
    pieces: list[Line] = []
    current: Line = []
    for a, b in pairwise(line):
        piece = _clip_segment(a, b, box)
        if piece is None:
            if current:
                pieces.append(current)
                current = []
            continue
        start, end = piece
        if not current or current[-1] != start:
            if current:
                pieces.append(current)
            current = [start]
        current.append(end)
    if current:
        pieces.append(current)
    return pieces


def tile_box(zoom: int, x: int, y: int, extent: float = 1.0) -> Box:
    side = extent / (1 << zoom)
    return x * side, y * side, (x + 1) * side, (y + 1) * side


def tiles_at(
    zoom: int, lines: list[Line], pixel_tolerance: float, pixels: int = 256, extent: float = 1.0
) -> dict[tuple[int, int], int]:
    # vertices kept per tile after clipping and simplifying to a tolerance in tile pixels;
    # each segment is clipped only against the few tiles under its own box
    if zoom < 0 or pixels < 1 or pixel_tolerance < 0:
        raise Invalid("zoom and pixels must be non-negative, the tolerance too")
    n = 1 << zoom
    side = extent / n
    tolerance = pixel_tolerance * side / pixels
    per_tile: dict[tuple[int, int], list[Line]] = {}
    for line in lines:
        open_piece: dict[tuple[int, int], Line] = {}
        for a, b in pairwise(line):
            x_lo = max(0, math.floor(min(a[0], b[0]) / side))
            x_hi = min(n - 1, math.floor(max(a[0], b[0]) / side))
            y_lo = max(0, math.floor(min(a[1], b[1]) / side))
            y_hi = min(n - 1, math.floor(max(a[1], b[1]) / side))
            for x in range(x_lo, x_hi + 1):
                for y in range(y_lo, y_hi + 1):
                    piece = _clip_segment(a, b, tile_box(zoom, x, y, extent))
                    if piece is None:
                        continue
                    head, tail = piece
                    current = open_piece.get((x, y))
                    if current is not None and current[-1] == head:
                        current.append(tail)
                    else:
                        if current is not None:
                            per_tile.setdefault((x, y), []).append(current)
                        open_piece[(x, y)] = [head, tail]
        for tile, current in open_piece.items():
            per_tile.setdefault(tile, []).append(current)
    return {
        tile: sum(len(simplify(piece, tolerance)) for piece in pieces)
        for tile, pieces in per_tile.items()
    }


def summary(counts: dict[tuple[int, int], int]) -> dict[str, float]:
    if not counts:
        raise Invalid("no tiles hold anything")
    values = list(counts.values())
    return {
        "tiles": float(len(values)),
        "vertices": float(sum(values)),
        "largest": float(max(values)),
        "mean": sum(values) / len(values),
    }


def wiggly_line(n: int, rng: random.Random, extent: float = 1.0) -> Line:
    x, y = rng.uniform(0.1, 0.9) * extent, rng.uniform(0.1, 0.9) * extent
    heading = rng.uniform(0, 2 * math.pi)
    out = [(x, y)]
    step = extent / n
    for _ in range(n - 1):
        heading += rng.gauss(0, 0.5)
        x = min(max(x + step * math.cos(heading), 0.0), extent)
        y = min(max(y + step * math.sin(heading), 0.0), extent)
        out.append((x, y))
    return out


def raw_vertices(lines: list[Line]) -> int:
    return sum(len(line) for line in lines)
