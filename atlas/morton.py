"""Morton code: interleave the bits of x and y so 2D points sort along a Z curve.

A spatial index often wants to lay two-dimensional points out in one
dimension, so that a plain sorted array or a B-tree can hold them, and
the trick is to do it without throwing away too much of the nearness
that made them spatial. The Morton code, or Z-order code, does this by
interleaving the bits of the two coordinates: take the bits of x and
y and weave them together, x's bit then y's bit then x's next bit and
so on, into a single integer. Sorting points by their Morton code
traces a recursive Z-shaped path through the plane, visiting a
quadrant fully before moving to the next, which keeps most nearby
points nearby in the ordering. The word most is the honest part. The
Z-order has seams: crossing certain quadrant boundaries makes the code
jump far even though the points barely moved, because a high bit flips
while low bits reset, so two points a single step apart in space can
be far apart in Morton order. This is the price of the interleave's
simplicity, and it is why the Hilbert curve, which never makes such a
jump, is preferred when locality must be tight. Encoding and decoding
are exact inverses: de-interleaving the code recovers x and y with no
loss, so the code is a reversible relabeling, not an approximation.
The finding worth stating, and the one the survey measures, is that
sorting by Morton code makes consecutive points close on average but
leaves a tail of long jumps at the quadrant seams, so Z-order buys
cheap, reversible, mostly-local ordering, not perfectly-local
ordering. This module encodes and decodes Morton codes for
non-negative integer coordinates, and a survey measures the average
and worst consecutive jump along the Z curve.
"""

from __future__ import annotations

from atlas.errors import Invalid

_MAX_BITS = 32


def _part1by1(n: int) -> int:
    # spread the low bits of n so each occupies an even position
    n &= 0xFFFFFFFF
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2)) & 0x3333333333333333
    return (n | (n << 1)) & 0x5555555555555555


def _compact1by1(n: int) -> int:
    n &= 0x5555555555555555
    n = (n | (n >> 1)) & 0x3333333333333333
    n = (n | (n >> 2)) & 0x0F0F0F0F0F0F0F0F
    n = (n | (n >> 4)) & 0x00FF00FF00FF00FF
    n = (n | (n >> 8)) & 0x0000FFFF0000FFFF
    return (n | (n >> 16)) & 0x00000000FFFFFFFF


def encode(x: int, y: int) -> int:
    if x < 0 or y < 0:
        raise Invalid("Morton coordinates must be non-negative")
    if x >= (1 << _MAX_BITS) or y >= (1 << _MAX_BITS):
        raise Invalid("coordinates must fit in 32 bits")
    return _part1by1(x) | (_part1by1(y) << 1)


def decode(code: int) -> tuple[int, int]:
    if code < 0:
        raise Invalid("a Morton code must not be negative")
    return _compact1by1(code), _compact1by1(code >> 1)
