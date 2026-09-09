"""Region quadtree: compressing a raster mask by splitting only where it is not uniform.

A land-cover mask at a million cells is mostly runs of the same
value, and the region quadtree stores it by asking one question
of each square: is it uniform? If so it is one leaf; if not it
splits into four quadrants and asks again, down to single cells.
A blank raster is one node, a checkerboard is a node per cell
plus the internal nodes above them, and a real map lies between,
with the node count governed by how much boundary it has. The
survey measures that governance. For a filled disc of radius r
cells in a raster of side 2r, the leaf count grows in proportion
to the perimeter, since every uniform square well inside or well
outside the disc is one leaf whatever its size and only squares
the boundary crosses keep splitting: the discs of side 32, 64,
128, and 256 took 244, 532, 1084, and 2020 leaves for 1024, 4096,
16384, and 65536 cells, the leaves growing 2.18, 2.04, and 1.86
times per doubling of the side while the cells grew four times.
For a random raster the tree is no better than the cells, every
2 by 2 block being non-uniform with high probability: a random
64-cell raster took 3733 leaves for 4096 cells, 91 percent, and a
checkerboard took all 4096 plus 1365 internal nodes, the worst
case, while a blank raster took one. For a raster of three
rectangles the tree is far smaller than the cells, and alignment
with the quadrant grid is everything: rectangles whose corners
sit on quadrant boundaries took 25 leaves at depth 3, and the
same rectangles shifted by one cell took 1471 at depth 7, 59
times as many, since the shift puts a boundary through every
large square. The tree decoded back to the raster cell for cell
at every size, and the node identity held throughout, the leaves
being three times the internal nodes plus one. The finding worth
stating is that a region quadtree's size follows the boundary
length rather than the area, doubling per doubling of the side
where the cells quadruple, that it cannot compress noise at all,
and that a one-cell shift can cost a factor of 59, so the
compression is a
statement about the map's boundaries and their placement. This
module builds and decodes region quadtrees, and a survey measures
their size on discs, noise, and rectangles.
"""

from __future__ import annotations

from collections.abc import Sequence

from atlas.errors import Invalid

Grid = Sequence[Sequence[bool]]


class _Node:
    __slots__ = ("children", "value")

    def __init__(self, value: bool | None, children: list | None = None):
        self.value = value
        self.children = children


class RegionQuadtree:
    def __init__(self, grid: Grid):
        if not grid or not grid[0]:
            raise Invalid("the grid is empty")
        side = len(grid)
        if any(len(row) != side for row in grid) or side & (side - 1):
            raise Invalid("the grid must be square with a power-of-two side")
        self.side = side
        self.root = self._build(grid, 0, 0, side)

    def _build(self, grid: Grid, top: int, left: int, size: int) -> _Node:
        first = grid[top][left]
        rows = range(top, top + size)
        uniform = all(grid[r][c] == first for r in rows for c in range(left, left + size))
        if uniform or size == 1:
            return _Node(first)
        half = size // 2
        return _Node(
            None,
            [
                self._build(grid, top, left, half),
                self._build(grid, top, left + half, half),
                self._build(grid, top + half, left, half),
                self._build(grid, top + half, left + half, half),
            ],
        )

    def leaves(self) -> int:
        return self._count(self.root, True)

    def internal(self) -> int:
        return self._count(self.root, False)

    def _count(self, node: _Node, leaves: bool) -> int:
        if node.children is None:
            return 1 if leaves else 0
        return (0 if leaves else 1) + sum(self._count(c, leaves) for c in node.children)

    def depth(self) -> int:
        def walk(node: _Node) -> int:
            if node.children is None:
                return 0
            return 1 + max(walk(c) for c in node.children)

        return walk(self.root)

    def decode(self) -> list[list[bool]]:
        out = [[False] * self.side for _ in range(self.side)]

        def fill(node: _Node, top: int, left: int, size: int) -> None:
            if node.children is None:
                for r in range(top, top + size):
                    for c in range(left, left + size):
                        out[r][c] = bool(node.value)
                return
            half = size // 2
            fill(node.children[0], top, left, half)
            fill(node.children[1], top, left + half, half)
            fill(node.children[2], top + half, left, half)
            fill(node.children[3], top + half, left + half, half)

        fill(self.root, 0, 0, self.side)
        return out


def disc(side: int) -> list[list[bool]]:
    center = (side - 1) / 2
    radius = side / 2 - 0.5
    return [
        [(c - center) ** 2 + (r - center) ** 2 <= radius * radius for c in range(side)]
        for r in range(side)
    ]


def rectangles(side: int, shift: int = 0) -> list[list[bool]]:
    # three rectangles whose corners sit on quadrant boundaries when the shift is zero
    boxes = [(0, 0, side // 2, side // 2), (side // 2, side // 4, side, side // 2 + side // 4)]
    boxes.append((side // 8, side // 2 + side // 8, side // 4 + side // 8, side))
    grid = [[False] * side for _ in range(side)]
    for top, left, bottom, right in boxes:
        for r in range(top + shift, min(bottom + shift, side)):
            for c in range(left + shift, min(right + shift, side)):
                grid[r][c] = True
    return grid


def random_grid(side: int, rng) -> list[list[bool]]:
    return [[rng.random() < 0.5 for _ in range(side)] for _ in range(side)]
