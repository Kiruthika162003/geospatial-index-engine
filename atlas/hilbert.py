"""Hilbert curve: a space-filling order whose consecutive points are always adjacent.

The Morton code is cheap but has seams: sorting by it makes most
nearby points nearby, yet crossing a quadrant boundary can throw two
neighbors far apart. The Hilbert curve is the space-filling curve that
removes the seams entirely. It visits every cell of a two-to-the-order
by two-to-the-order grid exactly once, and its defining property is
that consecutive cells in the visit order are always grid-adjacent,
one step apart, never diagonal and never a jump. There is no boundary
it crosses with a leap, because the curve is built recursively by
rotating and reflecting the sub-curves in each quadrant so their ends
meet their neighbors' beginnings. That perfect step-by-step continuity
is why the Hilbert order gives the tightest locality of the common
space-filling curves, and why spatial indexes that sort or pack data
by a linear key often choose it over Z-order despite its more involved
arithmetic. The mapping is a reversible bijection between a distance
along the curve and a grid cell, computed not by interleaving bits but
by walking the order levels from the top, at each level using two bits
of the distance to pick a quadrant and rotating the coordinate frame
so the next level is interpreted correctly. Encode and decode invert
each other exactly, cell for distance and distance for cell. The
finding worth stating, and the one the survey pins, is that the
Hilbert curve's worst consecutive jump is exactly one, against the
Morton curve's jump of sixteen on the same grid, so Hilbert buys
seamless locality at the cost of harder encoding. This module encodes
a cell to its Hilbert distance and back, and a survey measures the
worst consecutive jump to confirm it is one.
"""

from __future__ import annotations

from atlas.errors import Invalid


def encode(x: int, y: int, order: int) -> int:
    if order <= 0:
        raise Invalid("order must be positive")
    side = 1 << order
    if not (0 <= x < side and 0 <= y < side):
        raise Invalid("coordinates must lie within the curve's grid")
    rx = ry = 0
    distance = 0
    s = side >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        distance += s * s * ((3 * rx) ^ ry)
        x, y = _rotate(s, x, y, rx, ry)
        s >>= 1
    return distance


def decode(distance: int, order: int) -> tuple[int, int]:
    if order <= 0:
        raise Invalid("order must be positive")
    side = 1 << order
    if not 0 <= distance < side * side:
        raise Invalid("distance must lie within the curve's length")
    x = y = 0
    t = distance
    s = 1
    while s < side:
        rx = 1 & (t >> 1)
        ry = 1 & (t ^ rx)
        x, y = _rotate(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t >>= 2
        s <<= 1
    return x, y


def _rotate(s: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    # rotate/reflect the quadrant so sub-curves connect end to end
    if ry == 0:
        if rx == 1:
            x = s - 1 - x
            y = s - 1 - y
        x, y = y, x
    return x, y
