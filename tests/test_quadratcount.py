from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.quadratcount import (
    chi_squared,
    clustered,
    counts,
    jittered_grid,
    log_spacing,
    random_scatter,
    ratio,
    ratio_curve,
)

SIDES = (2, 4, 5, 8, 10, 16, 20, 25, 40)


def _patterns():
    rng = random.Random(204)
    return random_scatter(400, rng), clustered(20, 20, 0.02, rng), jittered_grid(20, 0.1, rng)


class TestTheCurves:
    def test_a_single_random_scatter_wanders_round_one(self):
        scatter, _, _ = _patterns()
        readings = dict(ratio_curve(scatter, SIDES))
        assert readings[2] == pytest.approx(0.467, abs=1e-3)
        assert readings[4] == pytest.approx(1.003, abs=1e-3)
        assert readings[8] == pytest.approx(1.382, abs=1e-3)
        assert readings[40] == pytest.approx(1.046, abs=1e-3)
        assert chi_squared(counts(scatter, 10)) == (pytest.approx(132.0), 99)

    def test_many_random_scatters_average_to_one(self):
        for side, mean, low, high in ((10, 1.0348, 0.682, 1.263), (20, 1.0159, 0.857, 1.158)):
            scatters = [random_scatter(400, random.Random(300 + i)) for i in range(30)]
            values = [ratio(s, side) for s in scatters]
            assert sum(values) / 30 == pytest.approx(mean, abs=1e-3)
            assert (min(values), max(values)) == pytest.approx((low, high), abs=1e-3)

    def test_the_clustered_ratio_peaks_at_a_few_cluster_diameters_then_falls(self):
        _, clus, _ = _patterns()
        readings = dict(ratio_curve(clus, SIDES))
        assert readings[40] == pytest.approx(3.237, abs=1e-3)
        assert readings[4] == pytest.approx(16.683, abs=1e-3)
        assert readings[2] == pytest.approx(7.607, abs=1e-3)
        assert max(readings, key=readings.get) == 4

    def test_the_grid_reads_zero_exactly_when_the_cells_tile_it(self):
        _, _, grid = _patterns()
        readings = dict(ratio_curve(grid, SIDES))
        for side in (2, 4, 5, 10, 20):
            assert readings[side] == 0.0
        assert readings[8] == pytest.approx(0.249, abs=1e-3)
        assert readings[16] == pytest.approx(0.334, abs=1e-3)
        assert readings[25] == pytest.approx(0.361, abs=1e-3)
        assert readings[40] == pytest.approx(0.75, abs=1e-2)  # Bernoulli with p = 1/4: 1 - p
        assert log_spacing(20, 20) == pytest.approx(0.0, abs=1e-12)
        assert log_spacing(40, 20) == pytest.approx(-1.0, abs=1e-12)


class TestAccounting:
    def test_counts_sum_to_the_point_count_at_every_cell_size(self):
        for pts in _patterns():
            for side in SIDES:
                assert sum(counts(pts, side)) == len(pts)

    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            counts([(2, 0)], 2)
        with pytest.raises(Invalid):
            ratio([], 2)
        with pytest.raises(Invalid):
            counts([(0.5, 0.5)], 0)
        with pytest.raises(Invalid):
            counts([(0.5, 0.5)], 2, size=0)
