"""The error family: every refusal in this package speaks a full sentence.

Spatial code fails in ways flat code does not: a coordinate can
be a perfectly good number and still fall off the edge of the
world, three points can each be valid and still refuse to make a
triangle, and a query can be well formed and still ask an empty
tree for a neighbor it does not have. The hierarchy gives each
failure its own name so callers can catch what they mean: Invalid
for requests that were never going to work, Missing for a thing
that should exist and does not, Outside for a coordinate beyond
the domain it must lie in, and Degenerate for geometry that has
collapsed, a zero-area polygon or a set of collinear points asked
to bound an area, where the honest answer is that the shape is not
one the operation can act on.
"""

from __future__ import annotations


class AtlasError(Exception):
    pass


class Invalid(AtlasError):
    pass


class Missing(AtlasError):
    pass


class Outside(AtlasError):
    pass


class Degenerate(AtlasError):
    pass
