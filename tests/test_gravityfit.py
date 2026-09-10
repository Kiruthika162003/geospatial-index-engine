from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.gravityfit import (
    expected_flows,
    log_fit,
    observed_flows,
    poisson,
    poisson_fit,
    total_flow,
    towns,
    zero_share,
)


@pytest.fixture(scope="module")
def places():
    return towns(30, random.Random(740))


class TestClean:
    def test_both_fits_recover_a_clean_exponent(self, places):
        origins, sizes = places
        expected = expected_flows(origins, sizes, 1.5, 2.0)
        assert log_fit(origins, sizes, expected) == pytest.approx((1.5, 2.0), abs=1e-6)
        assert poisson_fit(origins, sizes, expected) == pytest.approx(1.5, abs=1e-4)


class TestNoisyCounts:
    @pytest.mark.parametrize(
        ("beta", "scale", "total", "zeros", "log_mean", "poisson_mean"),
        [
            (1.0, 2.0, 123, 0.897, 0.439, 1.06),
            (1.0, 20.0, 1268, 0.513, 0.71, 1.003),
            (1.0, 200.0, 12612, 0.068, 0.972, 0.998),
            (2.0, 200.0, 851, 0.774, 1.195, 1.995),
        ],
    )
    def test_the_log_fit_loses_its_centre_with_the_zeros(
        self, places, beta, scale, total, zeros, log_mean, poisson_mean
    ):
        origins, sizes = places
        expected = expected_flows(origins, sizes, beta, scale)
        logs, poissons, empties, totals = [], [], [], []
        for k in range(10):
            observed = observed_flows(expected, random.Random(741 + k))
            totals.append(total_flow(observed))
            empties.append(zero_share(observed))
            logs.append(log_fit(origins, sizes, observed)[0])
            poissons.append(poisson_fit(origins, sizes, observed))
        assert round(sum(totals) / 10) == total
        assert sum(empties) / 10 == pytest.approx(zeros, abs=1e-3)
        assert sum(logs) / 10 == pytest.approx(log_mean, abs=1e-3)
        assert sum(poissons) / 10 == pytest.approx(poisson_mean, abs=1e-3)
        assert abs(poisson_mean - beta) < abs(log_mean - beta)


class TestPieces:
    def test_poisson_draws_and_refusals(self):
        rng = random.Random(1)
        draws = [poisson(3.0, rng) for _ in range(4000)]
        assert sum(draws) / 4000 == pytest.approx(3.0, abs=0.1)
        assert poisson(0.0, rng) == 0
        assert 500 < poisson(1000.0, rng) < 1500
        with pytest.raises(Invalid):
            expected_flows([(0, 0), (1, 1)], [1.0], 1.0, 1.0)
        with pytest.raises(Invalid):
            log_fit([(0, 0), (1, 1)], [1.0, 1.0], [[0, 0], [0, 0]])
