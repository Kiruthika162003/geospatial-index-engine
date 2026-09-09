from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.sdellipse import (
    axis_ratio,
    bearing_gap,
    coverage,
    ellipse,
    gaussian_coverage,
    isotropic,
    stretched,
)


class TestIsotropicScatters:
    def test_the_ratio_wanders_by_an_amount_the_count_sets(self):
        rng = random.Random(251)
        for n, mean, worst in ((100, 1.1331, 1.3114), (1000, 1.0432, 1.123)):
            ratios = [axis_ratio(isotropic(n, rng)) for _ in range(200)]
            assert sum(ratios) / 200 == pytest.approx(mean, abs=1e-3)
            assert max(ratios) == pytest.approx(worst, abs=1e-3)
            assert min(ratios) > 1.0


class TestStretchedScatters:
    @pytest.mark.parametrize("bearing", [30.0, 170.0, 95.0])
    def test_the_bearing_and_ratio_return_on_average(self, bearing):
        rng = random.Random(int(bearing))
        gaps, ratios = [], []
        for _ in range(60):
            _, a, b, found = ellipse(stretched(500, 3.0, bearing, rng))
            gaps.append(bearing_gap(found, bearing))
            ratios.append(a / b)
        assert sum(gaps) / 60 < 1.0
        assert max(gaps) < 4.0
        assert sum(ratios) / 60 == pytest.approx(3.0, abs=0.06)
        assert 2.5 < min(ratios) < max(ratios) < 3.6

    def test_the_half_turn_convention(self):
        rng = random.Random(251)
        assert 165 < ellipse(stretched(500, 3.0, 170.0, rng))[3] < 176
        flat = [(0, 0), (1, 0), (2, 0), (0, 0.1), (1, -0.1), (2, 0.05)]
        assert ellipse(flat)[3] == pytest.approx(90.72, abs=0.01)
        assert bearing_gap(179.0, 1.0) == 2.0
        assert bearing_gap(30.0, 120.0) == 90.0


class TestCoverage:
    def test_one_sigma_holds_39_percent_and_two_sigma_86(self):
        rng = random.Random(251)
        for _ in range(200):
            isotropic(100, rng)
        for _ in range(200):
            isotropic(1000, rng)
        for _ in range(300):
            stretched(500, 3.0, 30.0, rng)
        points = isotropic(20000, rng)
        assert coverage(points, 1.0) == pytest.approx(0.3945, abs=2e-3)
        assert coverage(points, 2.0) == pytest.approx(0.8656, abs=2e-3)
        assert coverage(points, 3.0) == pytest.approx(0.9897, abs=2e-3)
        assert gaussian_coverage(1.0) == pytest.approx(0.3935, abs=1e-4)
        assert gaussian_coverage(2.0) == pytest.approx(0.8647, abs=1e-4)


class TestRefusals:
    def test_too_few_coincident_and_collinear_points_are_refused(self):
        with pytest.raises(Invalid):
            ellipse([(0, 0), (1, 1)])
        with pytest.raises(Invalid):
            ellipse([(1, 1)] * 5)
        with pytest.raises(Invalid):
            axis_ratio([(0, 0), (1, 1), (2, 2)])
