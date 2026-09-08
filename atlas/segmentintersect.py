"""Segment intersection: decide crossing by orientation signs, before computing any point.

Whether two line segments cross is a question that seems to call for
solving where their infinite lines meet and checking the point lies on
both, but that path divides by a quantity that is zero for parallel
segments and loses precision for nearly-parallel ones. The orientation
test decides the boolean without any of that. The orientation of an
ordered triple of points is the sign of a cross product: positive if
the three turn counterclockwise, negative if clockwise, zero if
collinear. Two segments, one from A to B and one from C to D, cross in
the ordinary case exactly when A and B lie on opposite sides of the
line CD and C and D lie on opposite sides of the line AB, which is four
orientation signs and no division. The special case is collinearity,
when an orientation comes out zero: then the segments lie on the same
line and cross only if they overlap, which a simple bounding-box
containment check settles. So the crossing decision is exact integer or
float sign arithmetic, robust where the line-intersection formula is
fragile, and the actual intersection point is computed only afterward,
only when it is wanted, and only when the segments are not parallel. The
finding worth stating is that separating the does-it-cross question,
answered by orientation signs, from the where-does-it-cross question,
answered by the line formula, keeps the common query cheap and robust
and confines the fragile division to the case that truly needs it. This
module tests intersection by orientation and computes the point when it
exists, and a survey checks the boolean against a brute parametric
solver including the collinear-overlap and touching-endpoint cases.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]


def _orient(a: Point, b: Point, c: Point) -> int:
    val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    # assuming p is collinear with a,b: is it within the bounding box?
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    # collinear touch/overlap cases
    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    return bool(o4 == 0 and _on_segment(c, d, b))


def intersection_point(a: Point, b: Point, c: Point, d: Point) -> Point | None:
    if not segments_intersect(a, b, c, d):
        return None
    denom = (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0])
    if denom == 0:
        raise Invalid("segments are collinear; a single crossing point is undefined")
    t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / denom
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
