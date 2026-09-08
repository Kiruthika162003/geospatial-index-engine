from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.ringvalidate import first_crossing, has_repeated_vertex, is_simple


class TestCalibratingShapes:
    def test_a_square_and_a_triangle_are_simple(self):
        assert is_simple([(0, 0), (4, 0), (4, 4), (0, 4)])
        assert is_simple([(0, 0), (1, 0), (0, 1)])

    def test_a_bowtie_crosses_on_its_diagonals(self):
        bowtie = [(0, 0), (4, 4), (4, 0), (0, 4)]
        assert not is_simple(bowtie)
        assert first_crossing(bowtie) == (0, 2)

    def test_a_pinched_ring_touches_itself(self):
        pinched = [(0, 0), (4, 0), (4, 2), (2, 1), (4, 4), (0, 4), (2, 1)]
        assert not is_simple(pinched)
        assert has_repeated_vertex(pinched)
        assert first_crossing(pinched) == (2, 5)

    def test_a_concave_l_is_simple(self):
        assert is_simple([(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])


class TestRandomRings:
    def test_convex_polygons_are_always_simple(self):
        rng = random.Random(115)
        for _ in range(500):
            k = rng.randint(4, 12)
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
            assert is_simple([(10 * math.cos(a), 10 * math.sin(a)) for a in angles])

    def test_star_shaped_combs_are_simple_when_the_origin_is_inside(self):
        # angles spread round the full circle keep the origin inside, so radial
        # pushes preserve the angular order and the comb stays simple; the earlier
        # half-circle construction failed 29 of 500 exactly where the origin fell outside
        rng = random.Random(115)
        for _ in range(500):
            k = rng.randint(4, 12)
            step = 2 * math.pi / k
            angles = [i * step + rng.uniform(-0.45 * step, 0.45 * step) for i in range(k)]
            comb = []
            for i, a in enumerate(angles):
                r = 4.0 if i % 2 else 10.0  # every other vertex pushed inward radially
                comb.append((r * math.cos(a), r * math.sin(a)))
            assert is_simple(comb)

    def test_shuffling_a_polygons_vertices_almost_always_breaks_it(self):
        rng = random.Random(116)
        simple = 0
        for _ in range(500):
            k = rng.randint(4, 12)
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
            poly = [(10 * math.cos(a), 10 * math.sin(a)) for a in angles]
            rng.shuffle(poly)
            simple += is_simple(poly)
        assert simple < 100  # measured about 27 of 500


class TestRefusals:
    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            is_simple([(0, 0), (1, 1)])

    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            has_repeated_vertex(None)
