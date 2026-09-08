"""Shoelace formula: polygon area and centroid from a signed sum of cross products.

The area of a simple polygon, however concave, follows from a sum that
crosses each edge's endpoints, the shoelace formula, so named because
the multiplications lace across like threading a shoe. Walk the
vertices in order and accumulate, for each edge, the cross product of
its two endpoints, x of one times y of the next minus x of the next
times y of the one; half that sum is the signed area. The word signed
carries real information: the sign records orientation. A polygon whose
vertices run counterclockwise gives a positive signed area, one running
clockwise gives a negative one, and the absolute value is the area
either way. So a single cheap sum answers two questions at once, how
big the polygon is and which way it is wound, and orientation matters
because many algorithms, containment tests and triangulations among
them, assume or must detect a consistent winding. The same signed
areas weight the centroid, the polygon's area center of mass: it is not
the plain average of the vertices, which is pulled toward wherever the
vertices happen to bunch, but a sum of edge midpoints weighted by the
signed cross products, divided by six times the signed area. The finding
worth stating is that the shoelace sum is exact for any simple polygon
in one linear pass, and its sign is the orientation, so area and winding
come from the same arithmetic rather than needing separate work. This
module computes signed area, area, and centroid, and a survey checks the
area against a triangle-fan decomposition and confirms reversing the
vertex order flips the sign but not the magnitude.
"""

from __future__ import annotations

from atlas.errors import Degenerate, Invalid

Point = tuple[float, float]


def signed_area(polygon: list[Point]) -> float:
    if polygon is None or len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    total = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2


def area(polygon: list[Point]) -> float:
    return abs(signed_area(polygon))


def is_counterclockwise(polygon: list[Point]) -> bool:
    return signed_area(polygon) > 0


def centroid(polygon: list[Point]) -> Point:
    a = signed_area(polygon)
    if a == 0:
        raise Degenerate("a zero-area polygon has no centroid")
    cx = cy = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    factor = 1 / (6 * a)
    return (cx * factor, cy * factor)
