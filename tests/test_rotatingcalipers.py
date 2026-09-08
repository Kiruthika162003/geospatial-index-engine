from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.rotatingcalipers import _dist2, diameter


class TestDiameter:
    def test_a_square_diameter_is_its_diagonal(self):
        d, a, b = diameter([(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)])
        assert d == pytest.approx(math.sqrt(200))
        assert _dist2(a, b) == pytest.approx(200)

    def test_two_points(self):
        d, _a, _b = diameter([(0, 0), (3, 4)])
        assert d == pytest.approx(5.0)

    def test_it_matches_the_brute_farthest_pair(self):
        rng = random.Random(61)
        for _ in range(20000):
            n = rng.randint(2, 40)
            pts = [(rng.randint(-50, 50), rng.randint(-50, 50)) for _ in range(n)]
            if len(set(pts)) < 2:
                continue
            d, _, _ = diameter(pts)
            brute = max(
                _dist2(p, q) for i, p in enumerate(pts) for q in pts[i + 1 :]
            )
            assert d * d == pytest.approx(brute)

    def test_the_farthest_pair_are_hull_extremes_on_a_disk(self):
        rng = random.Random(2)
        disk = [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(3000)]
        d, a, b = diameter(disk)
        # the diameter of a unit disk sample approaches 2 and is between rim points
        assert 1.8 < d <= 2 * math.sqrt(2)
        assert math.hypot(*a) > 0.9
        assert math.hypot(*b) > 0.9


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            diameter(None)

    def test_fewer_than_two_distinct_points_is_refused(self):
        with pytest.raises(Invalid):
            diameter([(1, 1), (1, 1)])
