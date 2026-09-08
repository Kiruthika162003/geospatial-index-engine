from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.pointinpolygon import (
    contains_by_winding,
    ray_casting,
    winding_number,
)


class TestBasicShapes:
    def test_a_square(self):
        sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
        assert ray_casting((2, 2), sq)
        assert not ray_casting((5, 2), sq)
        assert contains_by_winding((2, 2), sq)
        assert not contains_by_winding((5, 2), sq)

    def test_a_concave_l_shape(self):
        shape = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        assert not ray_casting((3, 3), shape)  # inside the notch is outside
        assert ray_casting((1, 3), shape)  # up the arm is inside


class TestAgreementOnSimplePolygons:
    def test_ray_and_winding_agree_everywhere_on_simple_polygons(self):
        rng = random.Random(21)
        shape = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
        square = [(0, 0), (6, 0), (6, 6), (0, 6)]
        for _ in range(50000):
            p = (rng.uniform(-1, 7), rng.uniform(-1, 7))
            assert ray_casting(p, shape) == contains_by_winding(p, shape)
            assert ray_casting(p, square) == contains_by_winding(p, square)


class TestDivergenceOnSelfIntersection:
    def test_a_pentagram_center_is_outside_by_parity_but_inside_by_winding(self):
        outer = [
            (math.cos(math.radians(90 + 72 * i)), math.sin(math.radians(90 + 72 * i)))
            for i in range(5)
        ]
        star = [outer[(2 * i) % 5] for i in range(5)]  # connect every second vertex
        center = (0.0, 0.0)
        assert ray_casting(center, star) is False  # even crossings
        assert winding_number(center, star) == 2  # boundary wraps twice
        assert contains_by_winding(center, star) is True


class TestRefusals:
    def test_too_few_vertices_is_refused(self):
        with pytest.raises(Invalid):
            ray_casting((0, 0), [(0, 0), (1, 1)])

    def test_none_polygon_is_refused(self):
        with pytest.raises(Invalid):
            winding_number((0, 0), None)
