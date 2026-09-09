from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.variogram import bump_field, empirical, exponential, fit, sample_field, variance


def _noise(rng):
    return [(rng.uniform(0, 100), rng.uniform(0, 100), rng.gauss(0, 1)) for _ in range(400)]


class TestBumpFields:
    @pytest.mark.parametrize(
        ("width", "range_", "sill", "var", "first"),
        [
            (5.0, 37.6, 0.2878, 0.258, 0.061),
            (10.0, 27.6, 0.5986, 0.6278, 0.024),
            (20.0, 80.0, 2.1047, 1.5467, 0.005),
        ],
    )
    def test_the_exponential_range_is_not_a_width_but_the_sill_tracks_the_variance(
        self, width, range_, sill, var, first
    ):
        rng = random.Random(255)
        for earlier in (5.0, 10.0, 20.0):
            if earlier == width:
                break
            sample_field(bump_field(earlier, 40, rng), 400, rng)
        samples = sample_field(bump_field(width, 40, rng), 400, rng)
        curve = empirical(samples, 4.0, 80.0)
        nugget, fitted_sill, fitted_range = fit(curve, 80.0)
        assert fitted_range == pytest.approx(range_, abs=0.5)
        assert fitted_sill == pytest.approx(sill, abs=1e-3)
        assert variance(samples) == pytest.approx(var, abs=1e-3)
        assert curve[0][1] / fitted_sill == pytest.approx(first, abs=2e-3)
        assert nugget == pytest.approx(curve[0][1])
        assert min(c for _, _, c in curve) > 300


class TestNoiseAndTrend:
    def test_white_noise_is_pure_nugget(self):
        rng = random.Random(255)
        for width in (5.0, 10.0, 20.0):
            sample_field(bump_field(width, 40, rng), 400, rng)
        noise = _noise(rng)
        curve = empirical(noise, 4.0, 80.0)
        nugget, sill, fitted_range = fit(curve, 80.0)
        assert curve[0][1] == pytest.approx(0.9425, abs=1e-3)
        assert nugget / sill == pytest.approx(1.0)
        assert fitted_range == pytest.approx(0.4, abs=1e-6)

    def test_a_plane_rises_as_the_square_of_the_lag_and_hits_the_cap(self):
        rng = random.Random(255)
        for width in (5.0, 10.0, 20.0):
            sample_field(bump_field(width, 40, rng), 400, rng)
        plane = [(x, y, 0.1 * x + 0.05 * y) for x, y, _ in _noise(rng)]
        curve = empirical(plane, 4.0, 80.0)
        ratios = [curve[k][1] / curve[0][1] for k in (1, 3, 7)]
        assert ratios == pytest.approx([4.9, 23.8, 106.7], abs=0.1)
        assert fit(curve, 80.0)[2] == 80.0

    def test_the_model_starts_at_the_nugget_and_reaches_the_sill(self):
        model = exponential(0.1, 1.0, 10.0)
        assert model(0) == 0.0
        assert model(1e-9) == pytest.approx(0.1, abs=1e-6)
        assert model(10.0) == pytest.approx(0.1 + 0.9 * 0.9502, abs=1e-3)
        assert model(100.0) == pytest.approx(1.0, abs=1e-9)


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            empirical([(0, 0, 1)], 1, 10)
        with pytest.raises(Invalid):
            empirical([(0, 0, 1), (1, 1, 2)], 0, 10)
        with pytest.raises(Invalid):
            fit([(1.0, 0.5, 10), (2.0, 0.6, 10)], 10.0)
        with pytest.raises(Invalid):
            fit([(1.0, 0.5, 10), (2.0, 0.6, 10), (3.0, 0.7, 10)], 0)
