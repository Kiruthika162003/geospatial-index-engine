from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.nodata import (
    POLICIES,
    hill,
    hole_fraction,
    mean_under,
    punch_block,
    punch_high,
    punch_random,
    ramp,
    resolve,
    slope_under,
    std_under,
    zero_bias_law,
)

RAMP = ramp(60, 1.0)
TRUE_MEAN = 129.5


class TestRandomHoles:
    @pytest.mark.parametrize(
        ("share", "ignore", "zero", "zero_slope", "mean_slope", "near_slope"),
        [
            (0.05, -0.014, -6.237, 12.9271, 2.1558, 0.999),
            (0.1, -0.061, -12.682, 23.9282, 3.3613, 0.9978),
            (0.3, 0.029, -39.046, 54.6331, 6.5754, 0.9929),
        ],
    )
    def test_zero_drags_and_the_mean_fill_steps(
        self, share, ignore, zero, zero_slope, mean_slope, near_slope
    ):
        holes = punch_random(RAMP, share, random.Random(710))
        assert mean_under(holes, "ignore") - TRUE_MEAN == pytest.approx(ignore, abs=1e-3)
        assert mean_under(holes, "zero") - TRUE_MEAN == pytest.approx(zero, abs=1e-3)
        assert zero_bias_law(TRUE_MEAN, hole_fraction(holes)) == pytest.approx(zero, abs=0.1)
        assert mean_under(holes, "fill_mean") == pytest.approx(
            mean_under(holes, "ignore"), abs=1e-9
        )
        assert slope_under(holes, "ignore") == pytest.approx(1.0, abs=1e-9)
        assert slope_under(holes, "zero") == pytest.approx(zero_slope, abs=1e-4)
        assert slope_under(holes, "fill_mean") == pytest.approx(mean_slope, abs=1e-4)
        assert slope_under(holes, "fill_neighbours") == pytest.approx(near_slope, abs=1e-4)
        assert abs(mean_under(holes, "fill_neighbours") - TRUE_MEAN) < 0.005
        assert (
            std_under(holes, "zero")
            > std_under(holes, "ignore")
            > std_under(holes, "fill_mean")
        )


class TestBiasedHoles:
    @pytest.mark.parametrize(
        ("share", "ignore", "zero", "near"),
        [(0.05, -1.614, -2.521, -0.102), (0.2, -6.141, -8.724, -1.453)],
    )
    def test_missing_peaks_bias_every_policy(self, share, ignore, zero, near):
        field = hill(60, 50.0)
        true_mean = sum(sum(r) for r in field) / 3600
        holes = punch_high(field, share)
        assert mean_under(holes, "ignore") - true_mean == pytest.approx(ignore, abs=1e-3)
        assert mean_under(holes, "zero") - true_mean == pytest.approx(zero, abs=1e-3)
        assert mean_under(holes, "fill_neighbours") - true_mean == pytest.approx(near, abs=1e-3)

    def test_a_block_hole_on_the_ramp(self):
        block = punch_block(RAMP, 20, 20, 10)
        assert mean_under(block, "ignore") - TRUE_MEAN == pytest.approx(0.143, abs=1e-3)
        assert mean_under(block, "zero") - TRUE_MEAN == pytest.approx(-3.458, abs=1e-3)
        assert mean_under(block, "fill_neighbours") - TRUE_MEAN == pytest.approx(0.0, abs=1e-9)
        assert slope_under(block, "zero") == pytest.approx(1.6723, abs=1e-4)
        assert slope_under(block, "fill_neighbours") == pytest.approx(1.0, abs=1e-9)
        assert resolve(block, "fill_neighbours")[25][25] == pytest.approx(127.375, abs=1e-3)
        assert RAMP[25][25] == 125.0


class TestRefusals:
    def test_bad_policies_and_empty_grids(self):
        assert len(POLICIES) == 4
        with pytest.raises(Invalid):
            resolve(RAMP, "guess")
        with pytest.raises(Invalid):
            resolve([[None, None], [None, None]], "fill_neighbours")
        with pytest.raises(Invalid):
            mean_under([[None]], "ignore")
        with pytest.raises(Invalid):
            punch_random(RAMP, 1.5, random.Random(1))
        with pytest.raises(Invalid):
            hole_fraction([])
