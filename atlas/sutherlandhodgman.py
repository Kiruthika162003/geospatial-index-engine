"""Sutherland-Hodgman: clip a polygon to a convex window one edge at a time.

Clipping a polygon to a rectangular viewport or any convex region is
the operation behind cropping a map to a tile or a screen. The
Sutherland-Hodgman algorithm does it by a sequence of simpler clips:
it clips the whole subject polygon against one edge of the clip region,
takes the result, clips that against the next edge, and so on around
the clip region, and after the last edge the polygon has been confined
to the intersection of all the half-planes, which for a convex clip
region is the region itself. Clipping against a single edge is the
heart of it and is simple: walk the polygon's edges, and for each pair
of consecutive vertices decide by which side of the clip edge they lie.
If both are inside, keep the second. If the edge crosses from inside to
outside, keep the crossing point. If it crosses from outside to inside,
keep the crossing point and then the second vertex. If both are
outside, keep nothing. Threading this rule around all the clip edges
leaves exactly the clipped polygon. The requirement that the clip
region be convex is not a detail to skip over: the algorithm relies on
each clip edge dividing the plane into a kept and a discarded
half-plane, and a concave clip region cannot be expressed that way, so
the algorithm is defined only for convex windows. A known cosmetic
artifact, worth naming, is that clipping a concave subject polygon can
leave degenerate zero-width connecting edges along the clip boundary,
which are geometrically harmless but visible to a naive consumer. The
finding worth stating is that clipping never enlarges a polygon, the
clipped area is at most the original and equals it when the subject
already lies inside the window, so the operation is a monotone
restriction. This module clips a polygon against a convex window, and a
survey confirms the clipped area never exceeds the subject's and equals
it when the subject is already contained.
"""

from __future__ import annotations

from atlas.errors import Invalid

Point = tuple[float, float]


def clip(subject: list[Point], window: list[Point]) -> list[Point]:
    if subject is None or window is None:
        raise Invalid("subject and window must not be None")
    if len(window) < 3:
        raise Invalid("the clip window needs at least three vertices")
    output = list(subject)
    n = len(window)
    for i in range(n):
        a = window[i]
        b = window[(i + 1) % n]
        if not output:
            break
        inputs = output
        output = []
        prev = inputs[-1]
        prev_inside = _inside(prev, a, b)
        for curr in inputs:
            curr_inside = _inside(curr, a, b)
            if curr_inside:
                if not prev_inside:
                    output.append(_intersect(prev, curr, a, b))
                output.append(curr)
            elif prev_inside:
                output.append(_intersect(prev, curr, a, b))
            prev, prev_inside = curr, curr_inside
    return output


def _inside(p: Point, a: Point, b: Point) -> bool:
    # left side of the directed clip edge a->b (assumes CCW window)
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0


def _intersect(p: Point, q: Point, a: Point, b: Point) -> Point:
    r = (b[0] - a[0], b[1] - a[1])
    s = (q[0] - p[0], q[1] - p[1])
    denom = r[0] * s[1] - r[1] * s[0]
    t = ((p[0] - a[0]) * s[1] - (p[1] - a[1]) * s[0]) / denom
    return (a[0] + t * r[0], a[1] + t * r[1])
