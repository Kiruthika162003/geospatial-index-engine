"""Heatmap: kernel density on a grid, where every point deposits exactly one unit of mass.

A heatmap turns a scatter of points into a smooth surface of density
so that a viewer sees where things pile up rather than a cloud of
dots. Each point deposits a bump of mass, a kernel, centered on its
position and spread by a bandwidth, and the surface is the sum of the
bumps sampled on a grid. The kernel here is a Gaussian, whose value at
a grid cell is the exponential of minus half the squared distance
over the squared bandwidth, normalized so that the bump integrates to
one over the plane. Two properties define a well-behaved heatmap and
are worth measuring rather than assuming. First, mass is conserved:
because each kernel integrates to one, the total mass on the grid,
each cell's density times its area summed over all cells, equals the
number of points, provided the grid extends far enough past the
points that the bumps' tails are captured, and the survey measures
how close it comes with a margin of a few bandwidths. Second, the
bandwidth trades resolution for smoothness: a small bandwidth gives a
tall narrow spike at each point and a surface that is mostly zero, so
the peak density is high and two nearby points stay separate, while a
large bandwidth melts them into one broad hill, lowering the peak and
spreading the mass, which is the smoothing a viewer wants when the
points are noisy samples of an underlying pattern. The peak height
falls with the square of the bandwidth for an isolated point, since
the bump's volume is fixed at one and its footprint grows as the
bandwidth squared. The finding worth stating is that the grid mass
equals the point count to within the tail lost past the grid edge,
and the peak of an isolated point scales as one over the bandwidth
squared, so the heatmap is a conserved redistribution of the points,
not an invention. This module rasterizes a Gaussian kernel density on
a grid, and a survey measures mass conservation and the peak's
bandwidth law.
"""

from __future__ import annotations

import math

from atlas.errors import Invalid

Point = tuple[float, float]


class Heatmap:
    def __init__(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        cell: float,
        bandwidth: float,
    ) -> None:
        if cell <= 0 or bandwidth <= 0:
            raise Invalid("cell size and bandwidth must be positive")
        if min_x >= max_x or min_y >= max_y:
            raise Invalid("the grid must have positive extent")
        self.min_x, self.min_y = min_x, min_y
        self.cell = cell
        self.bandwidth = bandwidth
        self.cols = math.ceil((max_x - min_x) / cell)
        self.rows = math.ceil((max_y - min_y) / cell)
        self.grid = [[0.0] * self.cols for _ in range(self.rows)]
        self._norm = 1.0 / (2 * math.pi * bandwidth * bandwidth)
        self._reach = 4 * bandwidth  # beyond four bandwidths the Gaussian is negligible

    def add(self, point: Point) -> None:
        px, py = point
        c0 = max(0, int((px - self._reach - self.min_x) / self.cell))
        c1 = min(self.cols - 1, int((px + self._reach - self.min_x) / self.cell))
        r0 = max(0, int((py - self._reach - self.min_y) / self.cell))
        r1 = min(self.rows - 1, int((py + self._reach - self.min_y) / self.cell))
        inv = 1.0 / (2 * self.bandwidth * self.bandwidth)
        for r in range(r0, r1 + 1):
            cy = self.min_y + (r + 0.5) * self.cell
            for c in range(c0, c1 + 1):
                cx = self.min_x + (c + 0.5) * self.cell
                d2 = (cx - px) ** 2 + (cy - py) ** 2
                self.grid[r][c] += self._norm * math.exp(-d2 * inv)

    def total_mass(self) -> float:
        return sum(sum(row) for row in self.grid) * self.cell * self.cell

    def peak(self) -> float:
        return max(max(row) for row in self.grid)

    def value_at(self, point: Point) -> float:
        c = int((point[0] - self.min_x) / self.cell)
        r = int((point[1] - self.min_y) / self.cell)
        if not (0 <= c < self.cols and 0 <= r < self.rows):
            raise Invalid("point lies outside the grid")
        return self.grid[r][c]
