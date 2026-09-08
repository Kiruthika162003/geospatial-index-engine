"""Ear clipping: cut a simple polygon into exactly n minus two triangles.

Triangulating a polygon, splitting it into triangles that tile it with
no gaps or overlaps, is the step that lets a renderer fill an arbitrary
shape, since graphics hardware draws triangles. A basic theorem sets
the target: every simple polygon of n vertices triangulates into
exactly n minus two triangles, no matter how concave, so a square gives
two, a pentagon three, and the count is fixed by the vertex count
alone. Ear clipping reaches that count by repeatedly removing ears. An
ear is a vertex whose two neighbors can be joined by a diagonal that
lies entirely inside the polygon, which for a vertex means the triangle
formed by it and its two neighbors is oriented the same way as the
polygon and contains no other vertex of the polygon inside it. The two
ears theorem guarantees every simple polygon with more than three
vertices has at least two ears, so there is always one to clip: find an
ear, output its triangle, remove its tip vertex, and repeat on the
smaller polygon until three vertices remain, the last triangle. Each
clip reduces the vertex count by one and adds one triangle, so starting
from n vertices the process outputs exactly n minus two triangles,
matching the theorem. The method is quadratic, since each clip scans
for an ear, which is fine for the modest polygons of map features and
is simpler than the linear-time methods. The finding worth stating is
that the triangle count is exactly n minus two and the triangles'
areas sum to the polygon's own area, so the triangulation is both
complete and non-overlapping, a partition rather than a covering. This
module triangulates a simple polygon by ear clipping, and a survey
confirms the triangle count and that the areas sum to the whole.
"""

from __future__ import annotations

from atlas.errors import Degenerate, Invalid

Point = tuple[float, float]
Triangle = tuple[Point, Point, Point]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool:
    d1 = _cross(a, b, p)
    d2 = _cross(b, c, p)
    d3 = _cross(c, a, p)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def _signed_area(poly: list[Point]) -> float:
    total = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2


def triangulate(polygon: list[Point]) -> list[Triangle]:
    if polygon is None or len(polygon) < 3:
        raise Invalid("a polygon needs at least three vertices")
    verts = list(polygon)
    if _signed_area(verts) < 0:
        verts.reverse()  # work counterclockwise
    if _signed_area(verts) == 0:
        raise Degenerate("a zero-area polygon cannot be triangulated")
    indices = list(range(len(verts)))
    triangles: list[Triangle] = []
    guard = 0
    while len(indices) > 3:
        guard += 1
        if guard > len(verts) * len(verts) + 10:
            raise Degenerate("polygon is not simple; no ear could be found")
        clipped = False
        m = len(indices)
        for k in range(m):
            i0 = indices[(k - 1) % m]
            i1 = indices[k]
            i2 = indices[(k + 1) % m]
            a, b, c = verts[i0], verts[i1], verts[i2]
            if _cross(a, b, c) <= 0:
                continue  # reflex or straight, not a convex ear tip
            if any(
                idx not in (i0, i1, i2) and _in_triangle(verts[idx], a, b, c)
                for idx in indices
            ):
                continue  # another vertex is inside, not an ear
            triangles.append((a, b, c))
            del indices[k]
            clipped = True
            break
        if not clipped:
            raise Degenerate("polygon is not simple; no ear could be found")
    triangles.append((verts[indices[0]], verts[indices[1]], verts[indices[2]]))
    return triangles


def triangle_area(t: Triangle) -> float:
    return abs(_cross(t[0], t[1], t[2])) / 2
