from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.verticalerror import (
    error_by_shift,
    mean,
    noisy,
    plane,
    rms_difference,
    rms_slope,
    rolling,
    sample,
    shifted,
    vertical_law,
)

SHIFTS = [0.25, 0.5, 1.0, 2.0]


class TestPlanes:
    @pytest.mark.parametrize("slope", [0.1, 0.5])
    def test_along_across_and_diagonal(self, slope):
        grid = plane(81, slope)
        assert rms_slope(grid, 5) == pytest.approx(slope, abs=1e-9)
        along = [e for _, e in error_by_shift(grid, SHIFTS, 0.0, 5)]
        assert along == pytest.approx([s * slope for s in SHIFTS], abs=1e-9)
        diagonal = [e for _, e in error_by_shift(grid, SHIFTS, 45.0, 5)]
        assert diagonal == pytest.approx([s * slope / math.sqrt(2) for s in SHIFTS], abs=1e-9)
        assert diagonal == pytest.approx([vertical_law(s, slope) for s in SHIFTS], abs=1e-9)
        across = [e for _, e in error_by_shift(grid, SHIFTS, 90.0, 5)]
        assert across == pytest.approx([0.0] * 4, abs=1e-9)


class TestRolling:
    def test_the_averaged_law_holds_on_isotropic_ground(self):
        grid = rolling(81, 20.0, 5.0)
        g = rms_slope(grid, 5)
        assert g == pytest.approx(1.0926, abs=1e-4)
        along = [round(e, 4) for _, e in error_by_shift(grid, SHIFTS, 0.0, 5)]
        assert along == [0.1929, 0.3857, 0.7714, 1.5254]
        for angle in (0.0, 45.0, 90.0):
            for s, e in error_by_shift(grid, SHIFTS[:3], angle, 5):
                assert abs(e / vertical_law(s, g) - 1) < 0.03

    def test_noise_adds_its_own_differences(self):
        grid = noisy(rolling(81, 20.0, 5.0), 0.3, random.Random(870))
        assert rms_slope(grid, 5) == pytest.approx(1.1351, abs=1e-4)
        along = [round(e, 4) for _, e in error_by_shift(grid, SHIFTS, 0.0, 5)]
        assert along == [0.2212, 0.4424, 0.8849, 1.5874]
        assert along[0] > vertical_law(0.25, 1.1351)


class TestPieces:
    def test_sampling_and_refusals(self):
        grid = [[0.0, 1.0], [2.0, 3.0]]
        assert sample(grid, 0.5, 0.5) == 1.5
        assert shifted(grid, 0.0, 0.0) == grid
        assert mean([1.0, 3.0]) == 2.0
        with pytest.raises(Invalid):
            shifted([], 1.0, 1.0)
        with pytest.raises(Invalid):
            rms_difference(grid, grid, 1)
        with pytest.raises(Invalid):
            mean([])
