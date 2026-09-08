from __future__ import annotations

import random

import pytest

from atlas.closestpair import _dist2, closest_pair
from atlas.errors import Invalid


class TestClosestPair:
    def test_a_concrete_case(self):
        d, a, b = closest_pair([(0, 0), (5, 5), (1, 1), (9, 9), (1.2, 1.1)])
        assert d == pytest.approx((0.2**2 + 0.1**2) ** 0.5)
        assert {a, b} == {(1, 1), (1.2, 1.1)}

    def test_two_points(self):
        d, _, _ = closest_pair([(0, 0), (3, 4)])
        assert d == pytest.approx(5.0)

    def test_it_matches_brute_force(self):
        rng = random.Random(63)
        for _ in range(20000):
            n = rng.randint(2, 50)
            pts = [(rng.randint(-100, 100), rng.randint(-100, 100)) for _ in range(n)]
            d, _, _ = closest_pair(pts)
            brute = min(
                _dist2(p, q) for i, p in enumerate(pts) for q in pts[i + 1 :]
            )
            assert d * d == pytest.approx(brute)

    def test_it_scales_to_many_points(self):
        rng = random.Random(1)
        big = [(rng.uniform(0, 10000), rng.uniform(0, 10000)) for _ in range(50000)]
        d, a, b = closest_pair(big)
        assert d >= 0
        assert _dist2(a, b) == pytest.approx(d * d)


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            closest_pair(None)

    def test_a_single_point_is_refused(self):
        with pytest.raises(Invalid):
            closest_pair([(0, 0)])
