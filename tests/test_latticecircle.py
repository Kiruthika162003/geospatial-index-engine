from __future__ import annotations

import itertools
import math

import pytest

from atlas.errors import Invalid
from atlas.latticecircle import (
    disc,
    disc_count,
    error_exponent,
    gauss_error,
    ring,
    ring_density,
    ring_straddle,
)


class TestTheRing:
    @pytest.mark.parametrize(
        ("radius", "cells", "density"), [(5, 28, 5.6), (50, 284, 5.68), (1000, 5656, 5.656)]
    )
    def test_the_density_settles_on_four_root_two(self, radius, cells, density):
        assert len(ring(radius)) == cells
        assert ring_density(radius) == pytest.approx(density, abs=1e-3)
        assert abs(ring_density(1000) - 4 * math.sqrt(2)) < 1e-3

    @pytest.mark.parametrize("radius", [5, 10, 50, 100, 500])
    def test_the_ring_straddles_its_circle_by_half_a_cell(self, radius):
        outside, farthest = ring_straddle(radius)
        assert 0.45 < outside < 0.6
        assert 0.38 < farthest < 0.5
        assert all(math.hypot(*c) <= radius + 0.5 for c in ring(radius))


class TestTheDisc:
    @pytest.mark.parametrize(
        ("radius", "count", "error"),
        [
            (10, 317, 2.8),
            (20, 1257, 0.4),
            (50, 7845, -9.0),
            (100, 31417, 1.1),
            (1000, 3141549, -43.7),
        ],
    )
    def test_gauss_counts_and_errors(self, radius, count, error):
        assert disc_count(radius) == count
        assert gauss_error(radius) == pytest.approx(error, abs=0.1)
        if radius >= 20:
            assert abs(gauss_error(radius)) / (math.pi * radius * radius) < 1.2e-3
        if radius >= 500:
            assert abs(gauss_error(radius)) / (math.pi * radius * radius) < 1e-4

    def test_the_set_matches_the_count_and_the_fitted_exponent_exceeds_the_bound(self):
        assert all(len(disc(r)) == disc_count(r) for r in (7, 13, 30))
        radii = [10 * k for k in range(1, 201)]
        assert error_exponent(radii) == pytest.approx(0.731, abs=0.01)
        signs = [gauss_error(r) > 0 for r in radii]
        assert sum(signs) / len(signs) == pytest.approx(0.035, abs=1e-3)
        assert sum(1 for a, b in itertools.pairwise(signs) if a != b) == 11


class TestRefusals:
    def test_negative_radii_and_single_fits_are_refused(self):
        with pytest.raises(Invalid):
            ring(-1)
        with pytest.raises(Invalid):
            disc(-1)
        with pytest.raises(Invalid):
            error_exponent([10])
