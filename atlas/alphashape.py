"""Alpha shape: a concave hull that tightens as alpha falls and relaxes to the convex hull.

The convex hull of a scatter of points wraps them in the tightest
convex polygon, and for a crescent, a ring, or a coastline that is
the wrong answer, since it fills the bays with empty space. The alpha
shape is the concave hull that follows the points into their bays,
and it comes almost free from the Delaunay triangulation. Every
Delaunay triangle has a circumradius; keep only the triangles whose
circumradius is below a threshold, one over alpha, and the outer
boundary of the kept triangles is the alpha shape. The intuition is a
disk of radius one over alpha rolled around the points: it can enter
any gap wider than its diameter, and the triangles it can pass through
are the ones removed. So alpha is a resolution knob with two limits
that the survey measures rather than describes. As alpha goes to
zero the radius bound goes to infinity, every triangle is kept, and
the alpha shape is exactly the convex hull, its area equal to the
hull's. As alpha rises the bound tightens, the large slivers across
bays are the first to go, the shape hugs the concavities, and its
area falls below the hull's; push alpha far enough and the shape
fragments, losing triangles inside the true body as well, until at
the limit nothing survives. The useful alpha is the one that carves
the bays without breaking the body, and on a crescent of points the
survey finds a range where the alpha shape recovers the crescent's
concave inner edge while a convex hull fills the whole bay. The
finding worth stating is that the alpha-shape area is a monotone
non-increasing function of alpha that starts at the convex hull area
and reaches zero, with the concave features carved out between, so
one knob moves continuously from the convex hull to nothing. This
module builds the alpha shape by circumradius filtering of Delaunay
triangles, and a survey measures its area against alpha on a crescent.
"""

from __future__ import annotations

import math

from atlas.delaunay import triangulate
from atlas.errors import Invalid

Point = tuple[float, float]
Triangle = tuple[Point, Point, Point]


def circumradius(tri: Triangle) -> float:
    (ax, ay), (bx, by), (cx, cy) = tri
    a = math.hypot(bx - cx, by - cy)
    b = math.hypot(ax - cx, ay - cy)
    c = math.hypot(ax - bx, ay - by)
    area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    if area2 == 0:
        return math.inf
    return a * b * c / (2 * area2)


def alpha_triangles(points: list[Point], alpha: float) -> list[Triangle]:
    if points is None or len(set(points)) < 3:
        raise Invalid("need at least three distinct points")
    if alpha < 0:
        raise Invalid("alpha must not be negative")
    bound = math.inf if alpha == 0 else 1.0 / alpha
    return [t for t in triangulate(points) if circumradius(t) <= bound]


def alpha_area(points: list[Point], alpha: float) -> float:
    total = 0.0
    for (ax, ay), (bx, by), (cx, cy) in alpha_triangles(points, alpha):
        total += abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2
    return total


def boundary_edges(triangles: list[Triangle]) -> set[tuple[Point, Point]]:
    # edges belonging to exactly one kept triangle form the shape's outline
    count: dict[tuple[Point, Point], int] = {}
    for t in triangles:
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            key = (a, b) if a <= b else (b, a)
            count[key] = count.get(key, 0) + 1
    return {e for e, n in count.items() if n == 1}
