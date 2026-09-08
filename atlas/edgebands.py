"""Edge bands: index polygon edges by y so a point-in-polygon test walks only crossable edges.

The ray-casting point-in-polygon test walks every edge of the polygon
for every query, which is fine for a triangle and wasteful for a
coastline with ten thousand vertices, since a horizontal ray from a
query can only cross edges whose vertical span contains the query's y,
and most edges of a big polygon do not. Banding fixes that once, at
build time. Cut the polygon's y extent into a fixed number of
horizontal bands, and for each band record the edges whose vertical
span overlaps it; a query then finds its band by one division and
walks only that band's edge list, applying the same crossing-parity
rule as plain ray casting to exactly the edges that could cross its
ray. The parity rule is unchanged, so the answer is identical to the
full walk, and the survey checks that identity on every query it
makes. The saving is the fraction of edges skipped, which depends on
the polygon's shape and the band count: a polygon whose edges are
short relative to its height puts each edge in one or two bands, so a
query with many bands tests only a small slice of the edges, while a
polygon of a few tall edges gains little, because tall edges land in
many bands. There is a build cost, each edge inserted into every band
it spans, and a memory cost of the band lists, both paid once and
amortized over the queries, which is the usual trade for a static
index. The finding worth stating is that banding returns the identical
inside-or-outside verdict as the full ray cast while testing a small
fraction of the edges per query, a fraction that shrinks as the
polygon gets more finely vertexed, so it is the right structure for
many queries against a detailed polygon. This module builds banded
edges and answers point-in-polygon through them, and a survey confirms
the verdict against plain ray casting and measures the edges tested.
"""

from __future__ import annotations

from atlas.errors import Invalid
from atlas.pointinpolygon import ray_casting

Point = tuple[float, float]
Edge = tuple[Point, Point]


class BandedPolygon:
    def __init__(self, polygon: list[Point], bands: int = 32) -> None:
        if polygon is None or len(polygon) < 3:
            raise Invalid("a polygon needs at least three vertices")
        if bands <= 0:
            raise Invalid("band count must be positive")
        self._polygon = list(polygon)
        ys = [p[1] for p in polygon]
        self._min_y = min(ys)
        self._max_y = max(ys)
        self._bands = bands
        self._lists: list[list[Edge]] = [[] for _ in range(bands)]
        self.edges_tested = 0
        n = len(polygon)
        for i in range(n):
            a, b = polygon[i], polygon[(i + 1) % n]
            lo, hi = self._band_of(min(a[1], b[1])), self._band_of(max(a[1], b[1]))
            for band in range(lo, hi + 1):
                self._lists[band].append((a, b))

    def _band_of(self, y: float) -> int:
        span = self._max_y - self._min_y
        if span == 0:
            return 0
        index = int((y - self._min_y) / span * self._bands)
        return max(0, min(self._bands - 1, index))

    def contains(self, point: Point) -> bool:
        x, y = point
        if y < self._min_y or y > self._max_y:
            self.edges_tested = 0
            return False
        edges = self._lists[self._band_of(y)]
        self.edges_tested = len(edges)
        inside = False
        for (x1, y1), (x2, y2) in edges:
            if (y1 > y) != (y2 > y):
                x_cross = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
                if x < x_cross:
                    inside = not inside
        return inside

    def edge_count(self) -> int:
        return len(self._polygon)

    def plain(self, point: Point) -> bool:
        return ray_casting(point, self._polygon)
