from __future__ import annotations

import pytest

from atlas.errors import Invalid
from atlas.regrid import (
    downsample_mean,
    downsample_nearest,
    edge_band_width,
    mean,
    rms,
    round_trip,
    round_trip_errors,
    smooth_field,
    spread,
    step_field,
    stripe_field,
    upsample_bilinear,
    upsample_nearest,
)

SIZE = 64


class TestSmooth:
    @pytest.mark.parametrize(
        ("factor", "expected"),
        [
            (2, (0.04904, 0.03468, 0.03529, 0.00772)),
            (4, (0.12896, 0.0772, 0.10502, 0.02665)),
            (8, (0.28059, 0.15551, 0.23807, 0.0865)),
        ],
    )
    def test_mean_down_and_bilinear_up_is_six_times_better(self, factor, expected):
        field = smooth_field(SIZE)
        assert spread(field) == pytest.approx(0.5, abs=1e-9)
        errors = round_trip_errors(field, factor)
        assert tuple(errors.values()) == pytest.approx(expected, abs=1e-5)
        assert errors["nearest-nearest"] / errors["mean-bilinear"] > 3

    def test_eight_waves_by_four_are_lost(self):
        field = smooth_field(SIZE, 8.0)
        errors = round_trip_errors(field, 4)
        assert errors["nearest-nearest"] == pytest.approx(0.5, abs=1e-9)
        assert errors["mean-bilinear"] == pytest.approx(0.4847, abs=1e-4)


class TestSharp:
    @pytest.mark.parametrize(("factor", "blur"), [(2, 0.04419), (4, 0.06988), (8, 0.10126)])
    def test_the_step_is_exact_under_nearest_and_blurred_a_factor_wide(self, factor, blur):
        field = step_field(SIZE)
        errors = round_trip_errors(field, factor)
        assert errors["nearest-nearest"] == 0.0
        assert errors["mean-nearest"] == 0.0
        assert errors["mean-bilinear"] == pytest.approx(blur, abs=1e-5)
        blurred = round_trip(field, factor, "mean", "bilinear")
        assert edge_band_width(field, blurred) == factor
        assert edge_band_width(field, round_trip(field, factor, "nearest", "nearest")) == 0
        assert mean(blurred) == pytest.approx(0.5, abs=1e-12)

    @pytest.mark.parametrize(("period", "blur"), [(4, 0.2461), (8, 0.1712), (16, 0.1169)])
    def test_stripes_below_the_factor_are_exact_and_bilinear_blurs(self, period, blur):
        field = stripe_field(SIZE, period)
        errors = round_trip_errors(field, 2)
        assert errors["nearest-nearest"] == 0.0
        assert errors["nearest-bilinear"] == pytest.approx(blur, abs=1e-4)

    def test_at_the_period_nearest_aliases_worse_than_mean_flattens(self):
        field = stripe_field(SIZE, 4)
        assert spread(downsample_nearest(field, 4)) == 0.0
        assert spread(downsample_mean(field, 4)) == 0.0
        assert mean(downsample_mean(field, 4)) == 0.5
        errors = round_trip_errors(field, 4)
        assert errors["nearest-nearest"] == pytest.approx(0.7071, abs=1e-4)
        assert errors["mean-nearest"] == pytest.approx(0.5, abs=1e-9)


class TestPiecesAndRefusals:
    def test_shapes_and_values(self):
        grid = [[1.0, 2.0], [3.0, 4.0]]
        assert upsample_nearest(grid, 2)[1] == [1.0, 1.0, 2.0, 2.0]
        fine = upsample_bilinear(grid, 2)
        assert len(fine) == 4 and len(fine[0]) == 4
        assert fine[0][0] == 1.0 and fine[3][3] == 4.0
        assert fine[1][1] == pytest.approx(1.75)
        assert downsample_mean(fine, 4) == [[pytest.approx(2.5)]]
        assert rms(grid, grid) == 0.0

    def test_refusals(self):
        with pytest.raises(Invalid):
            downsample_nearest([], 2)
        with pytest.raises(Invalid):
            downsample_mean([[1.0, 2.0, 3.0]], 2)
        with pytest.raises(Invalid):
            upsample_nearest([[1.0]], 0)
        with pytest.raises(Invalid):
            upsample_bilinear([[1.0]], 0)
        with pytest.raises(Invalid):
            rms([[1.0]], [[1.0, 2.0]])
        with pytest.raises(Invalid):
            stripe_field(8, 1)
