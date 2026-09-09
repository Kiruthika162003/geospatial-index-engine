from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.gearyc import gearys_c, one_minus_i_gap, outlier_shifts, shuffle_band, torus_c_and_i
from atlas.morani import blocks, checkerboard, gradient, morans_i


class TestAgainstOneMinusI:
    @pytest.mark.parametrize(
        ("size", "checker", "block", "ramp"),
        [(4, 1.875, 0.3125, 0.1875), (10, 1.98, 0.11, 0.03), (20, 1.995, 0.0525, 0.0075)],
    )
    def test_the_bounded_grid_readings(self, size, checker, block, ramp):
        assert gearys_c(checkerboard(size)) == pytest.approx(checker, abs=1e-4)
        assert gearys_c(blocks(size)) == pytest.approx(block, abs=1e-4)
        assert gearys_c(gradient(size)) == pytest.approx(ramp, abs=1e-4)
        assert one_minus_i_gap(checkerboard(size)) < 0
        assert abs(one_minus_i_gap(blocks(size))) < abs(one_minus_i_gap(gradient(size)))

    @pytest.mark.parametrize(("size", "gap"), [(4, 0.0375), (10, 0.00273), (20, 0.00036)])
    def test_even_a_torus_leaves_a_gap_near_one_over_n(self, size, gap):
        c, i = torus_c_and_i(gradient(size))
        assert (1 - i) - c == pytest.approx(gap, abs=2e-4)
        assert (1 - i) - c > 0

    def test_a_shuffle_reads_one_within_a_spread_of_0_08(self):
        mean, sd = shuffle_band(gradient(10), 500, random.Random(249))
        assert mean == pytest.approx(0.9999, abs=1e-3)
        assert sd == pytest.approx(0.0759, abs=2e-3)


class TestOutliers:
    def test_an_outlier_moves_c_only_a_fifth_more_than_i(self):
        dc, di = outlier_shifts(gradient(10), 5, 5, 100.0)
        assert (dc, di) == pytest.approx((0.8797, -0.7297), abs=1e-3)
        assert abs(dc / di) == pytest.approx(1.21, abs=0.01)
        dc, di = outlier_shifts(blocks(10), 5, 5, 50.0)
        assert abs(dc / di) == pytest.approx(1.11, abs=0.01)
        assert morans_i(gradient(10)) == pytest.approx(0.8889, abs=1e-4)


class TestRefusals:
    def test_constant_single_and_empty_grids_are_refused(self):
        with pytest.raises(Invalid):
            gearys_c([[1.0, 1.0]])
        with pytest.raises(Invalid):
            gearys_c([[1.0]])
        with pytest.raises(Invalid):
            gearys_c([])
        with pytest.raises(Invalid):
            shuffle_band(gradient(4), 1, random.Random(0))
