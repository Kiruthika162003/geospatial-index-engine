"""Geofence: which named regions contain a point, with a box prefilter doing most of the work.

A geofence lookup asks which of many named regions, delivery zones,
districts, restricted areas, contain a given point, and it is the
workhorse behind location-triggered alerts and per-region billing.
The exact answer is a point-in-polygon test against every region,
which is the polygon's edge count per region per point, and it is
mostly wasted, since a point in one corner of the map is nowhere near
most regions. The prefilter that removes the waste is the bounding
box: each region stores its box at load time, and a lookup tests the
cheap box first, running the exact ray cast only for regions whose
box contains the point. A box containment is four comparisons where
the ray cast is a pass over every edge, and since regions are small
relative to the map, the box rejects most of them outright. The
survey measures three things. The verdict is identical to the exact
test against every region, since a point outside a region's box is
certainly outside the region, so the prefilter never drops a true
hit. The fraction of regions reaching the exact test falls as the
regions get smaller relative to the map, from most of them when a few
large regions tile the map to a few percent when many small ones dot
it. And overlapping regions are all reported, since the lookup is a
membership test per region rather than a first-match, so a point in
two nested zones names both, which a first-match lookup would get
wrong. The finding worth stating is that the box prefilter preserves
every verdict while cutting the exact tests to a fraction set by how
small the regions are, and that overlap is reported rather than
resolved. This module stores named regions with their boxes and looks
up all containing regions, and a survey confirms the verdict against
the exact scan and measures the prefilter's pass rate.
"""

from __future__ import annotations

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.pointinpolygon import ray_casting

Point = tuple[float, float]


class Geofence:
    def __init__(self) -> None:
        self._regions: list[tuple[str, list[Point], BBox]] = []
        self.exact_tests = 0

    def add(self, name: str, polygon: list[Point]) -> None:
        if not name:
            raise Invalid("a region needs a name")
        if polygon is None or len(polygon) < 3:
            raise Invalid("a region needs at least three vertices")
        self._regions.append((name, list(polygon), BBox.from_points(polygon)))

    def __len__(self) -> int:
        return len(self._regions)

    def containing(self, point: Point) -> list[str]:
        self.exact_tests = 0
        names = []
        for name, polygon, box in self._regions:
            if not box.contains_point(*point):
                continue
            self.exact_tests += 1
            if ray_casting(point, polygon):
                names.append(name)
        return names

    def containing_exact(self, point: Point) -> list[str]:
        # the brute reference: ray cast against every region, no prefilter
        return [name for name, polygon, _ in self._regions if ray_casting(point, polygon)]
