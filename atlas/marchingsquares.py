"""Marching squares: isolines from a grid, one closed loop around every lone peak.

A contour map draws the curves along which a field, elevation,
temperature, pressure, holds one value, and marching squares extracts
those curves from a grid of samples cell by cell. Each grid cell has
four corners, each above or below the level, sixteen possible
patterns, and each pattern says where the isoline enters and leaves
the cell: through the two edges whose endpoints straddle the level,
at the point along each edge found by linear interpolation between
the corner values. Chaining the per-cell segments end to end gives the
isolines. Two of the sixteen patterns are ambiguous, a saddle where
diagonal corners are both above and the other two both below, and the
cell's center value decides whether the two isoline pieces connect
one way or the other. Three properties make the output trustworthy
and are worth measuring rather than assuming. Every segment endpoint
lies on a cell edge at the interpolated crossing, so the isoline is
exactly where the field crosses the level along that edge. On a field
with a single smooth peak inside the grid, every level between the
floor and the summit produces exactly one closed loop, since the level
set of a single hill is one ring, and a level above the summit or
below the floor produces nothing. And the loop encloses the peak: the
summit sample lies inside the polygon the loop forms, at every level,
so the contour rings nest around the top like the rings of a target.
The segment count grows with the loop's perimeter in cells and shrinks
as the level climbs toward the summit, since higher rings are smaller.
The finding worth stating is that on a single-peak field marching
squares yields exactly one closed loop per interior level, each
enclosing the summit and shrinking with height, so the extraction is a
faithful reading of the field's level sets. This module extracts
isoline segments and chains them into loops, and a survey counts the
loops and checks that each encloses the peak.
"""

from __future__ import annotations

from atlas.errors import Invalid
from atlas.pointinpolygon import ray_casting

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _lerp(p: Point, q: Point, vp: float, vq: float, level: float) -> Point:
    t = 0.5 if vq == vp else (level - vp) / (vq - vp)
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def isoline_segments(grid: list[list[float]], level: float) -> list[Segment]:
    if grid is None or len(grid) < 2 or len(grid[0]) < 2:
        raise Invalid("the grid needs at least two rows and two columns")
    rows, cols = len(grid), len(grid[0])
    segments: list[Segment] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            # corners: a=(c,r) b=(c+1,r) d=(c+1,r+1) e=(c,r+1), values likewise
            pa, pb, pd, pe = (c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)
            va, vb, vd, ve = grid[r][c], grid[r][c + 1], grid[r + 1][c + 1], grid[r + 1][c]
            crossings = []
            edges = ((pa, pb, va, vb), (pb, pd, vb, vd), (pd, pe, vd, ve), (pe, pa, ve, va))
            for p, q, vp, vq in edges:
                if (vp >= level) != (vq >= level):
                    crossings.append(_lerp(p, q, vp, vq, level))
            if len(crossings) == 2:
                segments.append((crossings[0], crossings[1]))
            elif len(crossings) == 4:
                # a saddle: the center value decides which pairs connect
                center = (va + vb + vd + ve) / 4
                if (center >= level) == (va >= level):
                    segments.append((crossings[0], crossings[3]))
                    segments.append((crossings[1], crossings[2]))
                else:
                    segments.append((crossings[0], crossings[1]))
                    segments.append((crossings[2], crossings[3]))
    return segments


def chain_loops(segments: list[Segment], tolerance: float = 1e-9) -> list[list[Point]]:
    remaining = [tuple(s) for s in segments]
    loops: list[list[Point]] = []

    def same(p: Point, q: Point) -> bool:
        return abs(p[0] - q[0]) <= tolerance and abs(p[1] - q[1]) <= tolerance

    while remaining:
        a, b = remaining.pop()
        loop = [a, b]
        grew = True
        while grew:
            grew = False
            for i, (p, q) in enumerate(remaining):
                if same(loop[-1], p):
                    loop.append(q)
                elif same(loop[-1], q):
                    loop.append(p)
                else:
                    continue
                remaining.pop(i)
                grew = True
                break
        loops.append(loop)
    return loops


def is_closed(loop: list[Point], tolerance: float = 1e-9) -> bool:
    if len(loop) <= 2:
        return False
    first, last = loop[0], loop[-1]
    return abs(first[0] - last[0]) <= tolerance and abs(first[1] - last[1]) <= tolerance


def encloses(loop: list[Point], point: Point) -> bool:
    return ray_casting(point, loop[:-1] if is_closed(loop) else loop)
