"""Cohen-Sutherland: clip a line to a rectangle, rejecting most cases with one bit test.

Clipping a line segment to a rectangle is a constant operation in map
and screen rendering, and Cohen-Sutherland makes the common cases
nearly free with a clever encoding. It assigns each endpoint a four-bit
outcode, one bit for each way the point can be outside the rectangle,
left of it, right of it, below, above; a point inside has outcode zero.
Two bitwise tests then dispatch most segments without any arithmetic.
If the two endpoints' outcodes are both zero, or more precisely if
their bitwise OR is zero, both endpoints are inside and the whole
segment is accepted trivially. If their bitwise AND is nonzero, the two
endpoints share an outside bit, meaning both lie beyond the same edge,
so the segment cannot cross the rectangle and is rejected trivially,
again with no computation. Only when neither test fires, when the
segment straddles the boundary, does the algorithm do real work:
compute where the segment crosses one violated edge, replace the
outside endpoint with that crossing, recompute its outcode, and repeat,
each iteration clipping against one edge until the segment is wholly
accepted or rejected. The point of the outcodes is that the expensive
intersection arithmetic runs only for the genuinely straddling
segments, while segments entirely outside past one edge, the large
majority when a small rectangle sits in a big scene, are thrown out by
a single AND. The finding worth stating is that the trivial-reject test
removes most fully-outside segments before any division, so the
average cost is dominated by the cheap bit tests, not the arithmetic.
This module clips a segment to a rectangle by outcodes, and a survey
measures how large a fraction of random segments are settled by the
trivial accept or reject alone.
"""

from __future__ import annotations

from atlas.bbox import BBox
from atlas.errors import Invalid

Point = tuple[float, float]

_INSIDE = 0
_LEFT = 1
_RIGHT = 2
_BOTTOM = 4
_TOP = 8


def _outcode(x: float, y: float, box: BBox) -> int:
    code = _INSIDE
    if x < box.min_x:
        code |= _LEFT
    elif x > box.max_x:
        code |= _RIGHT
    if y < box.min_y:
        code |= _BOTTOM
    elif y > box.max_y:
        code |= _TOP
    return code


def clip(a: Point, b: Point, box: BBox) -> tuple[Point, Point] | None:
    if box is None:
        raise Invalid("clip rectangle must not be None")
    x0, y0 = a
    x1, y1 = b
    code0 = _outcode(x0, y0, box)
    code1 = _outcode(x1, y1, box)
    while True:
        if code0 == 0 and code1 == 0:
            return ((x0, y0), (x1, y1))  # trivial accept
        if code0 & code1:
            return None  # trivial reject
        outside = code0 or code1
        if outside & _TOP:
            x = x0 + (x1 - x0) * (box.max_y - y0) / (y1 - y0)
            y = box.max_y
        elif outside & _BOTTOM:
            x = x0 + (x1 - x0) * (box.min_y - y0) / (y1 - y0)
            y = box.min_y
        elif outside & _RIGHT:
            y = y0 + (y1 - y0) * (box.max_x - x0) / (x1 - x0)
            x = box.max_x
        else:  # _LEFT
            y = y0 + (y1 - y0) * (box.min_x - x0) / (x1 - x0)
            x = box.min_x
        if outside == code0:
            x0, y0 = x, y
            code0 = _outcode(x0, y0, box)
        else:
            x1, y1 = x, y
            code1 = _outcode(x1, y1, box)


def is_trivial(a: Point, b: Point, box: BBox) -> bool:
    # true when the segment is settled by accept or reject with no arithmetic
    code0 = _outcode(a[0], a[1], box)
    code1 = _outcode(b[0], b[1], box)
    return (code0 == 0 and code1 == 0) or bool(code0 & code1)
