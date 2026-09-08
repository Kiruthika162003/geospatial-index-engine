"""Line of sight: a target is visible when its sight line crosses no obstacle segment.

Whether one point can see another across a field of walls is the
question behind coverage planning, sensor placement, and the
visibility graphs that path planners route through. Its answer is a
crossing test: draw the sight line from viewer to target, and the
target is visible exactly when that line crosses none of the obstacle
segments, since any crossing puts a wall between them. Testing every
obstacle per target is the brute answer and it is what the survey uses
as ground truth; the module adds the one filter that costs nothing and
prunes most obstacles, a bounding-box rejection, because an obstacle
whose box does not overlap the sight line's box cannot cross it, and
only the survivors get the exact orientation-sign crossing test. Two
measurements characterize visibility as a quantity. The visible
fraction of scattered targets falls as the obstacle density rises,
smoothly, from everything visible with no walls toward almost nothing
behind a dense field, which is the curve a planner reads to decide
how many sensors a cluttered site needs. And the box filter prunes a
fraction of the crossing tests that rises as the obstacles get shorter
relative to the field, the same locality that the segment sweep
exploits, while never changing the verdict, because a box overlap is a
necessary condition for a crossing. Endpoints matter: a target sitting
exactly on an obstacle counts as blocked, since the sight line touches
the wall, which is the safe convention for coverage. The finding worth
stating is that visibility is a per-target crossing count of zero, its
fraction over a field decays with obstacle density, and the box filter
skips most of the crossing tests without altering a single verdict, so
line of sight is cheap to compute exactly. This module answers
visibility with and without the box filter, and a survey measures the
decay curve and the pruning against the brute count.
"""

from __future__ import annotations

from atlas.errors import Invalid
from atlas.segmentintersect import segments_intersect

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _boxes_overlap(a: Point, b: Point, c: Point, d: Point) -> bool:
    return not (
        max(a[0], b[0]) < min(c[0], d[0])
        or max(c[0], d[0]) < min(a[0], b[0])
        or max(a[1], b[1]) < min(c[1], d[1])
        or max(c[1], d[1]) < min(a[1], b[1])
    )


class Visibility:
    def __init__(self, obstacles: list[Segment]) -> None:
        if obstacles is None:
            raise Invalid("obstacles must not be None")
        self.obstacles = list(obstacles)
        self.tests = 0  # exact crossing tests run by the most recent query

    def visible(self, viewer: Point, target: Point, use_box_filter: bool = True) -> bool:
        self.tests = 0
        for c, d in self.obstacles:
            if use_box_filter and not _boxes_overlap(viewer, target, c, d):
                continue
            self.tests += 1
            if segments_intersect(viewer, target, c, d):
                return False
        return True

    def visible_fraction(self, viewer: Point, targets: list[Point]) -> float:
        if not targets:
            raise Invalid("need at least one target")
        seen = sum(1 for t in targets if self.visible(viewer, t))
        return seen / len(targets)
