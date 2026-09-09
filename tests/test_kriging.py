from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.idw import best_power
from atlas.kriging import coverage, krige, rms_error, weights_at
from atlas.variogram import bump_field, empirical, exponential, fit, sample_field


def _setup():
    rng = random.Random(256)
    field = bump_field(10.0, 40, rng)
    samples = sample_field(field, 300, rng)
    _nugget, sill, range_ = fit(empirical(samples, 4.0, 80.0), 80.0)
    probes = [(rng.uniform(5, 95), rng.uniform(5, 95)) for _ in range(200)]
    return field, samples, exponential(0.0, sill, range_), sill, range_, probes


class TestExactnessAndError:
    def test_exact_at_a_sample_and_1_43_times_better_than_idw(self):
        field, samples, model, sill, range_, probes = _setup()
        assert (round(sill, 4), round(range_, 2)) == (1.4081, 31.6)
        x, y, v = samples[0]
        estimate, variance = krige(samples, model, x, y)
        assert estimate == pytest.approx(v, abs=1e-9)
        assert variance == 0.0
        kriging = rms_error(samples, model, field, probes)
        assert kriging == pytest.approx(0.1331, abs=1e-3)
        _, idw = best_power(samples, field, probes, (1, 2, 3, 4, 6))
        assert idw / kriging == pytest.approx(1.43, abs=0.02)


class TestTheErrorBar:
    def test_the_variance_is_4_7_times_too_wide_for_a_bump_field(self):
        field, samples, model, _, _, probes = _setup()
        sds = [math.sqrt(krige(samples, model, px, py)[1]) for px, py in probes]
        assert sum(sds) / 200 == pytest.approx(0.6244, abs=1e-3)
        assert coverage(samples, model, field, probes, 1.0) == 1.0
        assert coverage(samples, model, field, probes, 0.5) == pytest.approx(0.965, abs=1e-3)
        assert coverage(samples, model, field, probes, 0.1) == pytest.approx(0.765, abs=1e-3)
        assert krige(samples, model, 50.0, 50.0)[1] == pytest.approx(0.4132, abs=1e-3)


class TestScreeningAndNugget:
    def test_a_close_pair_shares_the_weight_of_one_lone_sample(self):
        _, _, model, _, _, _ = _setup()
        layout = [(-10.0, 0.0, 1.0), (-10.0, 0.5, 1.0), (10.0, 0.0, 1.0)]
        w = weights_at(layout, model, 0.0, 0.0)
        assert w == pytest.approx([0.2579, 0.2489, 0.4932], abs=1e-3)
        assert sum(w) == pytest.approx(1.0, abs=1e-12)

    def test_a_nugget_keeps_the_sample_exact_but_steps_beside_it(self):
        _, samples, _, sill, range_, _ = _setup()
        nugget_model = exponential(0.3 * sill, sill, range_)
        x, y, v = samples[0]
        estimate, variance = krige(samples, nugget_model, x, y)
        assert estimate == pytest.approx(v, abs=1e-9)
        assert variance == 0.0
        beside, beside_variance = krige(samples, nugget_model, x + 1e-6, y)
        assert beside == pytest.approx(-0.1401, abs=1e-3)
        assert abs(beside - v) == pytest.approx(0.0105, abs=1e-3)
        assert beside_variance == pytest.approx(0.668, abs=1e-2)


class TestRefusals:
    def test_too_few_samples_neighbours_and_duplicates_are_refused(self):
        model = exponential(0.0, 1.0, 10.0)
        with pytest.raises(Invalid):
            krige([(0, 0, 1)], model, 1, 1)
        with pytest.raises(Invalid):
            krige([(0, 0, 1), (1, 1, 2)], model, 1, 1, 1)
        with pytest.raises(Invalid):
            krige([(0, 0, 1), (0, 0, 2), (1, 1, 3)], model, 0.5, 0.5, 3)
        with pytest.raises(Invalid):
            rms_error([(0, 0, 1), (1, 1, 2)], model, _flat, [])


def _flat(_x: float, _y: float) -> float:
    return 0.0
