from __future__ import annotations

import math
import random

import pytest

from atlas.earclipping import triangle_area, triangulate
from atlas.errors import Degenerate, Invalid
from atlas.shoelace import area


class TestKnownShapes:
    def test_a_square_gives_two_triangles(self):
        tris = triangulate([(0, 0), (2, 0), (2, 2), (0, 2)])
        assert len(tris) == 2
        assert sum(triangle_area(t) for t in tris) == pytest.approx(4.0)

    def test_a_concave_l_gives_four(self):
        shape = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        tris = triangulate(shape)
        assert len(tris) == 4
        assert sum(triangle_area(t) for t in tris) == pytest.approx(area(shape))


class TestTheInvariants:
    def test_convex_polygons_give_n_minus_two_and_full_area(self):
        rng = random.Random(47)
        for _ in range(5000):
            k = rng.randint(3, 12)
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
            cx, cy, r = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(1, 5)
            poly = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles]
            tris = triangulate(poly)
            assert len(tris) == k - 2
            assert sum(triangle_area(t) for t in tris) == pytest.approx(area(poly))

    def test_concave_star_polygons_also_partition_exactly(self):
        rng = random.Random(48)
        for _ in range(3000):
            k = rng.randint(4, 14)
            poly = []
            for i in range(k):
                a = 2 * math.pi * i / k
                r = rng.uniform(1, 3) if i % 2 else rng.uniform(3, 5)
                poly.append((r * math.cos(a), r * math.sin(a)))
            tris = triangulate(poly)
            assert len(tris) == k - 2
            assert sum(triangle_area(t) for t in tris) == pytest.approx(area(poly))


class TestRefusals:
    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            triangulate([(0, 0), (1, 1)])

    def test_a_degenerate_polygon_is_refused(self):
        with pytest.raises(Degenerate):
            triangulate([(0, 0), (1, 0), (2, 0), (3, 0)])  # collinear, zero area
