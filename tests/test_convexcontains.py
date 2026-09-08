from __future__ import annotations

import math
import random

import pytest

from atlas.convexcontains import ConvexPolygon
from atlas.errors import Invalid


def _regular(k):
    step = 2 * math.pi / k
    return ConvexPolygon([(10 * math.cos(i * step), 10 * math.sin(i * step)) for i in range(k)])


class TestVerdict:
    def test_it_matches_ray_casting_on_random_convex_polygons(self):
        rng = random.Random(133)
        for _ in range(200):
            k = rng.randint(3, 64)
            step = 2 * math.pi / k
            angles = [i * step + rng.uniform(-0.3 * step, 0.3 * step) for i in range(k)]
            poly = ConvexPolygon([(10 * math.cos(a), 10 * math.sin(a)) for a in angles])
            for _ in range(50):
                q = (rng.uniform(-12, 12), rng.uniform(-12, 12))
                assert poly.contains(q) == poly.contains_by_ray(q)

    def test_boundary_and_corner_points_count_inside(self):
        sq = ConvexPolygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        assert sq.contains((4, 2))
        assert sq.contains((0, 0))
        assert sq.contains((2, 2))
        assert not sq.contains((5, 2))


class TestLogarithmicCost:
    def test_orientation_tests_grow_like_log_n_not_n(self):
        rng = random.Random(134)
        means = []
        for k in (8, 16, 32, 64, 128, 256):
            poly = _regular(k)
            total = 0
            for _ in range(200):
                poly.contains((rng.uniform(-9, 9), rng.uniform(-9, 9)))
                total += poly.tests
            means.append(total / 200)
        # measured 5.5, 6.9, 7.9, 9.0, 10.0, 11.0: about log2(n) + 3, one more per doubling
        for k, mean in zip((8, 16, 32, 64, 128, 256), means, strict=True):
            assert mean == pytest.approx(math.log2(k) + 3, abs=0.6)
            assert mean < k
        assert means == sorted(means)


class TestRefusals:
    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            ConvexPolygon([(0, 0), (1, 1)])

    def test_a_clockwise_or_concave_polygon_is_refused(self):
        with pytest.raises(Invalid):
            ConvexPolygon([(0, 0), (0, 4), (4, 4), (4, 0)])  # clockwise
        with pytest.raises(Invalid):
            ConvexPolygon([(0, 0), (4, 0), (1, 1), (0, 4)])  # a dent
