from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.heatmap import Heatmap


class TestMassConservation:
    def test_two_hundred_points_deposit_two_hundred_units(self):
        rng = random.Random(117)
        hm = Heatmap(-5, -5, 15, 15, cell=0.1, bandwidth=1.0)
        for _ in range(200):
            hm.add((rng.uniform(0, 10), rng.uniform(0, 10)))
        # the tail past four bandwidths is the only loss: measured 199.98
        assert hm.total_mass() == pytest.approx(200.0, abs=0.1)

    def test_a_single_point_carries_one_unit(self):
        hm = Heatmap(-10, -10, 10, 10, cell=0.05, bandwidth=1.0)
        hm.add((0.025, 0.025))
        assert hm.total_mass() == pytest.approx(1.0, abs=1e-3)


class TestPeakLaw:
    @pytest.mark.parametrize("bandwidth", [0.5, 1.0, 2.0])
    def test_the_peak_is_one_over_two_pi_bandwidth_squared(self, bandwidth):
        hm = Heatmap(-10, -10, 10, 10, cell=0.05, bandwidth=bandwidth)
        hm.add((0.025, 0.025))  # exactly on a cell center
        assert hm.peak() == pytest.approx(1 / (2 * math.pi * bandwidth**2), rel=1e-3)

    def test_quartering_the_bandwidth_raises_the_peak_sixteenfold(self):
        narrow = Heatmap(-10, -10, 10, 10, cell=0.05, bandwidth=0.5)
        wide = Heatmap(-10, -10, 10, 10, cell=0.05, bandwidth=2.0)
        narrow.add((0.025, 0.025))
        wide.add((0.025, 0.025))
        assert narrow.peak() / wide.peak() == pytest.approx(16.0, rel=1e-3)


class TestSmoothing:
    def test_a_small_bandwidth_keeps_a_valley_between_two_points(self):
        hm = Heatmap(-5, -5, 10, 5, cell=0.05, bandwidth=0.3)
        hm.add((0.025, 0.025))
        hm.add((3.025, 0.025))
        assert hm.value_at((1.525, 0.025)) < 0.01 * hm.value_at((0.025, 0.025))

    def test_a_large_bandwidth_melts_them_into_one_hill(self):
        hm = Heatmap(-5, -5, 10, 5, cell=0.05, bandwidth=2.0)
        hm.add((0.025, 0.025))
        hm.add((3.025, 0.025))
        # the midpoint is now the top of the merged hill: measured 1.14x a point's density
        assert hm.value_at((1.525, 0.025)) > hm.value_at((0.025, 0.025))


class TestRefusals:
    def test_non_positive_cell_or_bandwidth_is_refused(self):
        with pytest.raises(Invalid):
            Heatmap(0, 0, 1, 1, cell=0, bandwidth=1)
        with pytest.raises(Invalid):
            Heatmap(0, 0, 1, 1, cell=0.1, bandwidth=0)

    def test_an_empty_extent_is_refused(self):
        with pytest.raises(Invalid):
            Heatmap(0, 0, 0, 1, cell=0.1, bandwidth=1)

    def test_a_point_outside_the_grid_cannot_be_read(self):
        hm = Heatmap(0, 0, 1, 1, cell=0.1, bandwidth=1)
        with pytest.raises(Invalid):
            hm.value_at((5, 5))
