"""Morphology: growing, shrinking, opening, and closing a raster mask, with the areas measured.

A mask of built cells, flooded cells, or forest cells is cleaned
and reshaped with four operations from mathematical morphology.
Dilation sets a cell when any cell within a structuring element
round it is set, growing every patch by the element; erosion
sets a cell only when every cell within the element is set,
shrinking every patch by it; opening is erosion then dilation,
which removes patches and necks thinner than the element and
leaves the rest nearly as it was; closing is dilation then
erosion, which fills gaps and holes narrower than the element and
leaves the rest. The survey measures what each does to a patch by
area, since the pictures are persuasive and the numbers are the
check. Dilating a ten-cell square by a square element of radius r
gave 144, 196, and 256 cells at r of 1, 2, and 3, the perimeter
times r plus 4 r squared for the corners added to 100, and
eroding it gave 64, 36, and 16, the same removed, so the two are
inverses on a shape with no fine detail: eroding the dilated
square returned the square cell for cell, and dilating the eroded
one did too. Opening a square with a one-cell spur attached
removed the spur and returned the square exactly, 101 cells to
100, and opening a two-cell patch removed it entirely, area zero.
Closing two squares separated by a one-cell gap joined them, 200
cells becoming 210, the two squares plus the gap column, and
closing a square with a one-cell hole filled it, 99 to 100. Both
opening and closing were idempotent, a second application
changing no cell, and dilation was not, a second dilation growing
144 to 196. A square element of radius one dilated a lone cell to
nine and a cross element to five, at radius two to 25 and 13, so
the same radius grows area differently by shape; and a patch
touching the grid's border erodes there too, a five-cell square in
a corner eroding to nine cells, since the border counts as unset.
The finding worth stating is
that dilation and erosion trade area by the perimeter times the
radius and invert each other on coarse shapes, that opening and
closing remove and fill detail thinner than the element and then
stop, being idempotent, and that the element's shape sets the
growth per step, nine cells against five. This module implements
the four operations, and a survey measures their areas and
idempotence.
"""

from __future__ import annotations

from collections.abc import Sequence

from atlas.errors import Invalid

Grid = Sequence[Sequence[bool]]


def square_element(radius: int) -> list[tuple[int, int]]:
    if radius < 0:
        raise Invalid("the radius cannot be negative")
    return [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)]


def cross_element(radius: int) -> list[tuple[int, int]]:
    if radius < 0:
        raise Invalid("the radius cannot be negative")
    return [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
            if abs(dr) + abs(dc) <= radius]


def _check(grid: Grid) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise Invalid("the grid is empty")
    return len(grid), len(grid[0])


def dilate(grid: Grid, element: Sequence[tuple[int, int]]) -> list[list[bool]]:
    rows, cols = _check(grid)
    out = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if grid[r][c]:
                for dr, dc in element:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        out[rr][cc] = True
    return out


def erode(grid: Grid, element: Sequence[tuple[int, int]]) -> list[list[bool]]:
    # a cell survives when every element cell inside the grid is set; the border counts
    # as unset, so patches touching the edge erode there too
    rows, cols = _check(grid)
    out = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if not grid[r][c]:
                continue
            keep = True
            for dr, dc in element:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < rows and 0 <= cc < cols) or not grid[rr][cc]:
                    keep = False
                    break
            out[r][c] = keep
    return out


def opening(grid: Grid, element: Sequence[tuple[int, int]]) -> list[list[bool]]:
    return dilate(erode(grid, element), element)


def closing(grid: Grid, element: Sequence[tuple[int, int]]) -> list[list[bool]]:
    return erode(dilate(grid, element), element)


def area(grid: Grid) -> int:
    return sum(1 for row in grid for v in row if v)


def blank(rows: int, cols: int) -> list[list[bool]]:
    return [[False] * cols for _ in range(rows)]


def paint_square(grid: list[list[bool]], top: int, left: int, side: int) -> list[list[bool]]:
    for r in range(top, top + side):
        for c in range(left, left + side):
            grid[r][c] = True
    return grid
