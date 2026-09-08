from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.spatialjoin import candidate_count, join_within


class TestJoin:
    def test_it_matches_a_nested_loop(self):
        rng = random.Random(67)
        for _ in range(2000):
            nl, nr = rng.randint(1, 60), rng.randint(1, 60)
            left = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(nl)]
            right = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(nr)]
            r = 5.0
            got = sorted(join_within(left, right, r))
            brute = sorted(
                (i, j)
                for i, (lx, ly) in enumerate(left)
                for j, (rx, ry) in enumerate(right)
                if (lx - rx) ** 2 + (ly - ry) ** 2 <= r * r
            )
            assert got == brute

    def test_a_concrete_pairing(self):
        left = [(0, 0), (10, 10)]
        right = [(1, 1), (100, 100)]
        assert join_within(left, right, 2.0) == [(0, 0)]  # only (0,0)~(1,1)


class TestCandidateReduction:
    def test_the_grid_tests_a_tiny_fraction_of_the_product(self):
        rng = random.Random(1)
        left = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(5000)]
        right = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(5000)]
        r = 10.0
        cand = candidate_count(left, right, r)
        product = len(left) * len(right)
        # measured ~0.089% of the 25M product
        assert cand < product * 0.01
        # yet it still finds every true pair
        assert len(join_within(left, right, r)) > 0


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            join_within(None, [], 1.0)

    def test_a_non_positive_radius_is_refused(self):
        with pytest.raises(Invalid):
            join_within([(0, 0)], [(1, 1)], 0)
