"""Grid index: a uniform spatial hash, constant-time on spread data, linear on clusters.

The simplest spatial index is a uniform grid: cover the plane with
square cells of a fixed size and hash each point to the cell that
contains it, keeping a bucket of points per occupied cell. A range or
radius query then visits only the cells the query region overlaps and
checks the points in them, skipping the rest of the plane for free.
When points are spread out and the cell size is chosen so that each
cell holds a small constant number of them, this is the fastest index
there is: a point lookup is a hash and a short scan, expected constant
time, with none of the tree traversal a k-d tree or R-tree pays. The
weakness is the same simplicity. The grid has no way to adapt to where
the points actually are, so if the data clusters, every point in the
cluster hashes to the same few cells and those buckets grow to hold a
large fraction of all the points; a query touching that cell then
scans nearly everything, degrading to linear time, and the empty cells
elsewhere waste space. This is the exact opposite failure mode of the
adaptive trees, which stay balanced under clustering but pay log-time
per query always. The cell size is the one knob and it trades directly:
smaller cells mean fewer points per bucket but more cells to store and
more to visit per query, larger cells the reverse. The finding worth
stating is that the grid's per-query cost is governed entirely by how
many points share a cell, which is a small constant for uniform data
and the whole cluster for clustered data, so the grid rewards uniform
spread and punishes clustering, the mirror image of the trees. This
module builds a grid index with insert and range query, and a survey
measures the bucket occupancy under uniform versus clustered inputs.
"""

from __future__ import annotations

import math

from atlas.bbox import BBox
from atlas.errors import Invalid

Point = tuple[float, float]


class GridIndex:
    def __init__(self, cell_size: float) -> None:
        if cell_size <= 0:
            raise Invalid("cell size must be positive")
        self._cell = cell_size
        self._buckets: dict[tuple[int, int], list[Point]] = {}
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _cell_of(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self._cell), math.floor(y / self._cell))

    def insert(self, point: Point) -> None:
        key = self._cell_of(*point)
        self._buckets.setdefault(key, []).append(point)
        self._size += 1

    def range_query(self, box: BBox) -> list[Point]:
        found: list[Point] = []
        i0, j0 = self._cell_of(box.min_x, box.min_y)
        i1, j1 = self._cell_of(box.max_x, box.max_y)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                bucket = self._buckets.get((i, j))
                if bucket:
                    found.extend(p for p in bucket if box.contains_point(*p))
        return found

    def occupied_cells(self) -> int:
        return len(self._buckets)

    def max_bucket(self) -> int:
        return max((len(b) for b in self._buckets.values()), default=0)

    def mean_bucket(self) -> float:
        if not self._buckets:
            return 0.0
        return self._size / len(self._buckets)
