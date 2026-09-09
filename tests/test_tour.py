from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.tour import crossings, length, nearest_neighbour, optimal, start_spread, two_opt


class TestSmallLayouts:
    def test_nearest_neighbour_is_a_tenth_long_and_two_opt_finds_most_optima(self):
        rng = random.Random(206)
        nn_ratios, opt_ratios, hits = [], [], 0
        for _ in range(60):
            n = rng.choice((8, 9))
            stops = [(rng.random(), rng.random()) for _ in range(n)]
            best = optimal(stops)[1]
            nn = nearest_neighbour(stops)
            improved, _ = two_opt(nn)
            nn_ratios.append(length(nn) / best)
            opt_ratios.append(length(improved) / best)
            hits += abs(length(improved) - best) < 1e-9
        assert sum(nn_ratios) / 60 == pytest.approx(1.0847, abs=1e-3)
        assert max(nn_ratios) == pytest.approx(1.3063, abs=1e-3)
        assert sum(opt_ratios) / 60 == pytest.approx(1.0062, abs=1e-3)
        assert max(opt_ratios) == pytest.approx(1.0928, abs=1e-3)
        assert hits == 47


class TestFiftyStops:
    def test_two_opt_removes_every_crossing_and_recovers_a_fifth(self):
        rng = random.Random(208)
        stops = [(rng.random(), rng.random()) for _ in range(50)]
        nn = nearest_neighbour(stops)
        improved, reversals = two_opt(nn)
        assert crossings(nn) > 0
        assert crossings(improved) == 0
        assert reversals > 0
        assert length(improved) < length(nn)
        assert 0.05 < 1 - length(improved) / length(nn) < 0.35

    def test_a_random_start_does_about_as_well(self):
        rng = random.Random(209)
        ratios = []
        for _ in range(10):
            stops = [(rng.random(), rng.random()) for _ in range(50)]
            from_nn = two_opt(nearest_neighbour(stops))[0]
            order = list(stops)
            rng.shuffle(order)
            from_random = two_opt(order)[0]
            ratios.append(length(from_random) / length(from_nn))
        assert 0.9 < min(ratios) < 1.0 < max(ratios) < 1.2 or all(r < 1.2 for r in ratios)
        assert sum(ratios) / 10 < 1.1

    def test_the_start_swings_nearest_neighbour_by_a_third(self):
        rng = random.Random(210)
        stops = [(rng.random(), rng.random()) for _ in range(50)]
        low, high = start_spread(stops)
        assert 0.15 < (high - low) / low < 0.6


class TestSmallCases:
    def test_a_square_and_a_crossed_square(self):
        assert length([(0, 0), (1, 0), (1, 1), (0, 1)]) == 4.0
        crossed = [(0, 0), (1, 1), (1, 0), (0, 1)]
        assert crossings(crossed) == 1
        fixed, reversals = two_opt(crossed)
        assert reversals == 1
        assert crossings(fixed) == 0
        assert length(fixed) == 4.0

    def test_refusals(self):
        with pytest.raises(Invalid):
            optimal([(i, i) for i in range(11)])
        with pytest.raises(Invalid):
            nearest_neighbour([])
        with pytest.raises(Invalid):
            nearest_neighbour([(0, 0)], start=3)
