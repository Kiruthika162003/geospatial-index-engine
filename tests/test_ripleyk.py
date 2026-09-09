from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.ripleyk import (
    clustered,
    jittered_grid,
    k_function,
    l_excess,
    random_envelope,
    random_scatter,
)


def _patterns():
    rng = random.Random(196)
    return random_scatter(150, rng), clustered(15, 10, 0.02, rng), jittered_grid(12, 0.1, rng)


class TestThreePatterns:
    def test_random_clustered_and_grid_read_as_they_should(self):
        scatter, clus, grid = _patterns()
        assert (len(scatter), len(clus), len(grid)) == (150, 150, 144)
        for r, rand_v, clus_v, grid_v in (
            (0.02, 0.0013, 0.0488, -0.02),
            (0.05, 0.001, 0.0817, -0.05),
            (0.1, 0.0061, 0.0822, -0.0034),
            (0.2, 0.0005, 0.0608, 0.0105),
            (0.3, 0.0017, 0.0692, -0.0031),
        ):
            assert l_excess(scatter, r, 1, 1) == pytest.approx(rand_v, abs=1e-3)
            assert l_excess(clus, r, 1, 1) == pytest.approx(clus_v, abs=1e-3)
            assert l_excess(grid, r, 1, 1) == pytest.approx(grid_v, abs=1e-3)

    def test_the_grid_reads_exactly_minus_r_below_its_spacing(self):
        _, _, grid = _patterns()
        assert k_function(grid, 0.05, 1, 1) == 0.0
        assert l_excess(grid, 0.05, 1, 1) == -0.05

    def test_the_clustered_pattern_clears_the_random_envelope_tenfold(self):
        scatter, clus, _ = _patterns()
        low, high = random_envelope(150, 0.1, 20, random.Random(197))
        assert (low, high) == pytest.approx((-0.0059, 0.0087), abs=1e-3)
        assert low <= l_excess(scatter, 0.1, 1, 1) <= high
        assert l_excess(clus, 0.1, 1, 1) > 9 * high


class TestTheEdge:
    def test_without_correction_a_random_scatter_turns_regular_at_large_radii(self):
        scatter, _, _ = _patterns()
        for r, raw, corrected in (
            (0.1, 1.054, 1.125),
            (0.2, 0.862, 1.005),
            (0.3, 0.786, 1.011),
            (0.4, 0.705, 1.01),
        ):
            disc = math.pi * r * r
            assert k_function(scatter, r, 1, 1, False) / disc == pytest.approx(raw, abs=2e-3)
            assert k_function(scatter, r, 1, 1) / disc == pytest.approx(corrected, abs=2e-3)
        assert l_excess(scatter, 0.4, 1, 1, False) == pytest.approx(-0.0641, abs=1e-3)
        assert l_excess(scatter, 0.4, 1, 1) == pytest.approx(0.0019, abs=1e-3)


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            k_function([(0, 0)], 0.1, 1, 1)
        with pytest.raises(Invalid):
            k_function([(0, 0), (1, 1)], 0, 1, 1)
        with pytest.raises(Invalid):
            k_function([(0, 0), (1, 1)], 0.1, 0, 1)
