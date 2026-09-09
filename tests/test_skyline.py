from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.skyline import (
    anti_correlated,
    brute_force,
    correlated,
    dominates,
    harmonic,
    independent,
    rounded,
    skyline,
)


class TestExactness:
    def test_the_sweep_matches_brute_force_with_and_without_ties(self):
        rng = random.Random(211)
        for _ in range(200):
            pts = independent(rng.randint(1, 60), rng)
            assert skyline(pts) == brute_force(pts)
            r = rounded(pts, 0.1)
            assert skyline(r) == brute_force(r)

    def test_duplicates_stay_on_the_skyline_together(self):
        pts = [(0.2, 0.2), (0.2, 0.2), (0.5, 0.1), (0.1, 0.5), (0.3, 0.3)]
        assert skyline(pts) == [(0.1, 0.5), (0.2, 0.2), (0.2, 0.2), (0.5, 0.1)]

    def test_dominance_is_strict_somewhere(self):
        assert dominates((0, 0), (1, 1))
        assert not dominates((0, 1), (1, 0))
        assert not dominates((1, 1), (1, 1))


class TestSizes:
    @pytest.mark.parametrize(("n", "mean"), [(10, 2.84), (100, 5.23), (1000, 7.72)])
    def test_independent_attributes_sit_on_the_harmonic_number(self, n, mean):
        rng = random.Random(211 + n)
        sizes = [len(skyline(independent(n, rng))) for _ in range(200)]
        observed = sum(sizes) / 200
        assert abs(observed - harmonic(n)) < 0.4
        assert observed == pytest.approx(mean, abs=0.4)
        assert harmonic(n) == pytest.approx(math.log(n) + 0.5772, abs=0.06)

    def test_anti_correlation_lengthens_the_front_by_its_sharpness(self):
        rng = random.Random(213)
        readings = []
        for noise in (0.0, 0.01, 0.05, 0.2):
            sizes = [len(skyline(anti_correlated(1000, noise, rng))) for _ in range(10)]
            readings.append(sum(sizes) / 10)
        assert readings[0] == 1000.0
        assert readings == sorted(readings, reverse=True)
        assert 100 < readings[1] < 160
        assert 25 < readings[2] < 50
        assert 8 < readings[3] < 20

    def test_correlation_collapses_the_front(self):
        rng = random.Random(214)
        sizes = [len(skyline(correlated(1000, 0.05, rng))) for _ in range(20)]
        assert 2 < sum(sizes) / 20 < 7

    def test_rounding_to_a_coarse_grid_leaves_a_handful(self):
        rng = random.Random(211)
        r = rounded(independent(1000, rng), 0.1)
        assert len(set(r)) <= 121
        assert len(skyline(r)) == len(brute_force(r)) <= 5


class TestRefusals:
    def test_empty_and_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            skyline([])
        with pytest.raises(Invalid):
            brute_force([])
        with pytest.raises(Invalid):
            harmonic(0)
        with pytest.raises(Invalid):
            rounded([(1, 1)], 0)
