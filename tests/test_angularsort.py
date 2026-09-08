from __future__ import annotations

import math
import random

import pytest

from atlas.angularsort import is_tie, sort_around, sort_by_atan2
from atlas.errors import Invalid

ORIGIN = (0, 0)


class TestOrder:
    def test_the_compass_points_come_out_counterclockwise_from_east(self):
        pts = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        assert sort_around(ORIGIN, pts) == [
            (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
        ]

    def test_it_agrees_with_the_arctangent_order_on_random_floats(self):
        rng = random.Random(105)
        for _ in range(3000):
            pivot = (rng.uniform(-5, 5), rng.uniform(-5, 5))
            n = rng.randint(1, 30)
            pts = [(rng.uniform(-50, 50), rng.uniform(-50, 50)) for _ in range(n)]
            assert sort_around(pivot, pts) == sort_by_atan2(pivot, pts)

    def test_the_pivot_itself_is_excluded(self):
        assert sort_around(ORIGIN, [(0, 0), (1, 1)]) == [(1, 1)]


class TestTies:
    def test_collinear_points_are_exact_ties_and_sort_nearer_first(self):
        assert sort_around(ORIGIN, [(6, 3), (2, 1), (4, 2)]) == [(2, 1), (4, 2), (6, 3)]
        assert is_tie(ORIGIN, (2, 1), (6, 3))

    def test_integer_scalar_multiples_tie_exactly_and_atan2_happens_to_agree(self):
        # the refuted guess said atan2 would sometimes split these; it never did
        rng = random.Random(106)
        for _ in range(20000):
            a = (rng.randint(1, 10**6), rng.randint(1, 10**6))
            k = rng.randint(2, 5)
            b = (a[0] * k, a[1] * k)
            assert is_tie(ORIGIN, a, b)
            assert math.atan2(a[1], a[0]) == math.atan2(b[1], b[0])

    def test_opposite_directions_are_not_ties(self):
        assert not is_tie(ORIGIN, (1, 1), (-1, -1))


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            sort_around(None, [(1, 1)])
