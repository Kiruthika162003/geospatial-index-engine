from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.kcenter import (
    covering_radius,
    greedy,
    greedy_ratio,
    optimal,
    start_spread,
    tight_pairs,
)


class TestAgainstBruteForce:
    def test_greedy_averages_thirty_percent_above_optimum_and_never_reaches_two(self):
        rng = random.Random(205)
        ratios = []
        for _ in range(120):
            n, k = rng.randint(12, 14), rng.randint(2, 4)
            pts = [(rng.random(), rng.random()) for _ in range(n)]
            ratios.append(greedy_ratio(pts, k))
        assert sum(ratios) / 120 == pytest.approx(1.3036, abs=1e-3)
        assert max(ratios) == pytest.approx(1.8586, abs=1e-3)
        assert max(ratios) < 2.0
        assert sum(1 for r in ratios if r < 1 + 1e-9) == 8

    def test_the_start_swings_the_radius_by_forty_percent(self):
        rng = random.Random(205)
        for _ in range(120):
            n = rng.randint(12, 14)
            rng.randint(2, 4)
            for _ in range(n):
                rng.random()
                rng.random()
        pts = [(rng.random(), rng.random()) for _ in range(14)]
        low, high = start_spread(pts, 3)
        best = optimal(pts, 3)[1]
        assert (low, high) == pytest.approx((0.3562, 0.497), abs=1e-3)
        assert (low / best, high / best) == pytest.approx((1.2033, 1.6788), abs=1e-3)


class TestTheAdversary:
    def test_tight_pairs_are_solved_exactly(self):
        assert greedy_ratio(tight_pairs(4, 0.5), 4) == pytest.approx(1.0)
        assert greedy_ratio(tight_pairs(3, 1.0, 4.0), 3) == pytest.approx(1.0)
        assert covering_radius(tight_pairs(4, 0.5), greedy(tight_pairs(4, 0.5), 4)) == 0.5


class TestTheRadiusCurve:
    def test_the_radius_falls_irregularly_with_k(self):
        rng = random.Random(207)
        pts = [(rng.random(), rng.random()) for _ in range(400)]
        radii = [covering_radius(pts, greedy(pts, k)) for k in (1, 2, 4, 8, 16, 32, 64)]
        assert radii == sorted(radii, reverse=True)
        assert radii[0] > 0.5
        assert radii[-1] < 0.15
        assert radii[0] / radii[-1] > 5


class TestRefusals:
    def test_bad_k_start_and_size_are_refused(self):
        with pytest.raises(Invalid):
            greedy([(0, 0)], 2)
        with pytest.raises(Invalid):
            greedy([(0, 0), (1, 1)], 1, start=5)
        with pytest.raises(Invalid):
            optimal([(i, 0) for i in range(30)], 10)
        with pytest.raises(Invalid):
            covering_radius([(0, 0)], [])
