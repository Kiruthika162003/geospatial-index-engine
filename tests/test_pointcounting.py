from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.pointcounting import (
    aligned_grid_bias,
    fitted_exponent,
    grid_errors,
    grid_estimate,
    random_errors,
    random_estimate,
    random_law,
    rms_error,
    true_area,
)

SIDE, CENTRE, RADIUS = 100.0, (50.0, 50.0), 30.0


class TestDesigns:
    @pytest.mark.parametrize(
        ("n_side", "grid", "rand", "bias"),
        [
            (5, 0.12601, 0.30689, 0.27324),
            (20, 0.01584, 0.08379, -0.0097),
            (80, 0.00207, 0.01984, -0.00307),
        ],
    )
    def test_grid_beats_random_and_the_fixed_grid_is_biased(self, n_side, grid, rand, bias):
        n = n_side * n_side
        assert grid_errors(
            n_side, SIDE, CENTRE, RADIUS, random.Random(840), 200
        ) == pytest.approx(grid, abs=1e-5)
        assert random_errors(n, SIDE, CENTRE, RADIUS, random.Random(841), 200) == pytest.approx(
            rand, abs=1e-5
        )
        share = true_area(RADIUS) / SIDE**2
        assert random_law(n, share) == pytest.approx(rand, rel=0.06)
        assert aligned_grid_bias(n_side, SIDE, CENTRE, RADIUS) == pytest.approx(bias, abs=1e-5)
        assert rand / grid > 2

    def test_the_exponents(self):
        counts = [25, 100, 400, 1600, 6400]
        grid = [0.12601, 0.042, 0.01584, 0.00617, 0.00207]
        rand = [0.30689, 0.16056, 0.08379, 0.03977, 0.01984]
        assert fitted_exponent(counts, grid) == pytest.approx(-0.731, abs=1e-3)
        assert fitted_exponent(counts, rand) == pytest.approx(-0.496, abs=1e-3)


class TestPieces:
    def test_estimates_and_refusals(self):
        assert grid_estimate(1, SIDE, CENTRE, RADIUS) == SIDE**2
        assert random_estimate(50, SIDE, CENTRE, 0.0, random.Random(1)) == 0.0
        assert true_area(1.0) == pytest.approx(math.pi)
        assert rms_error([2.0, 2.0], 2.0) == 0.0
        with pytest.raises(Invalid):
            grid_estimate(0, SIDE, CENTRE, RADIUS)
        with pytest.raises(Invalid):
            random_estimate(0, SIDE, CENTRE, RADIUS, random.Random(1))
        with pytest.raises(Invalid):
            rms_error([], 1.0)
        with pytest.raises(Invalid):
            fitted_exponent([1], [1.0])
