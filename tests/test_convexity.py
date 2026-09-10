from __future__ import annotations

import math

import pytest

from atlas.convexity import area, convexity, hull, mean_convexity, star, star_guess, star_law
from atlas.errors import Invalid


class TestCircles:
    @pytest.mark.parametrize(
        ("noise", "reading"),
        [(0.0, 1.0), (0.5, 0.9888), (1.0, 0.9745), (2.0, 0.9441), (5.0, 0.8533)],
    )
    def test_noise_costs_convexity(self, noise, reading):
        assert mean_convexity(360, 100.0, noise, 900) == pytest.approx(reading, abs=1e-4)
        if noise:
            assert 0.02 < (1 - reading) / noise < 0.03


class TestStars:
    @pytest.mark.parametrize(
        ("n", "readings", "factor"),
        [
            (5, (0.309, 0.618, 0.9271), 1.236),
            (8, (0.2706, 0.5412, 0.8118), 1.0824),
            (12, (0.2588, 0.5176, 0.7765), 1.0353),
        ],
    )
    def test_the_ratio_over_the_cosine(self, n, readings, factor):
        for ratio, reading in zip((0.25, 0.5, 0.75), readings, strict=True):
            ring = star(n, 100.0, 100.0 * ratio)
            assert convexity(ring) == pytest.approx(reading, abs=1e-4)
            assert star_law(n, ratio) == pytest.approx(reading, abs=1e-4)
            assert reading / ratio == pytest.approx(factor, abs=1e-3)
            assert len(hull(ring)) == n
        assert 1 / math.cos(math.pi / n) == pytest.approx(factor, abs=1e-3)
        assert star_guess(5, 0.25) == pytest.approx(0.2236, abs=1e-4)
        assert convexity(star(n, 100.0, 100.0)) == pytest.approx(1.0)
        assert len(hull(star(n, 100.0, 100.0))) == 2 * n


class TestPieces:
    def test_hulls_areas_and_refusals(self):
        square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert convexity(square) == 1.0
        assert hull(square) == square
        assert area(square) == 1.0
        with pytest.raises(Invalid):
            area(square[:2])
        with pytest.raises(Invalid):
            hull([(0.0, 0.0), (1.0, 1.0)])
        with pytest.raises(Invalid):
            star(2, 1.0, 0.5)
        with pytest.raises(Invalid):
            star_law(5, 0.0)
