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
    return towns(25, random.Random(740))


def _fits(places, beta, scale, draws=10):
    origins, sizes = places
    expected = expected_flows(origins, sizes, beta, scale)
    logs, pois, zeros = [], [], []
    for k in range(draws):
        observed = observed_flows(expected, random.Random(741 + k))
        zeros.append(zero_share(observed))
        logs.append(log_fit(origins, sizes, observed)[0])
        pois.append(poisson_fit(origins, sizes, observed))
    return sum(logs) / draws, sum(pois) / draws, sum(zeros) / draws, total_flow(expected) / 600


class TestNoiseFree:
    def test_both_fits_recover_the_truth(self, places):
        origins, sizes = places
        expected = expected_flows(origins, sizes, 2.0, 50.0)
        beta, scale = log_fit(origins, sizes, expected)
        assert (beta, scale) == pytest.approx((2.0, 50.0), abs=1e-6)
        assert poisson_fit(origins, sizes, expected) == pytest.approx(2.0, abs=1e-6)


class TestPoissonCounts:
    @pytest.mark.parametrize(
        ("beta", "scale", "log_read", "pois_read", "zeros", "flow"),
        [
            (1.0, 5.0, 0.492, 1.01, 0.816, 0.32),
            (1.0, 50.0, 0.813, 1.001, 0.316, 3.16),
            (1.0, 500.0, 1.019, 0.997, 0.007, 31.64),
            (2.0, 50.0, 0.925, 2.018, 0.925, 0.21),
            (2.0, 500.0, 1.318, 1.996, 0.684, 2.14),
        ],
    )
    def test_the_log_fit_reads_low_and_the_poisson_fit_holds(
        self, places, beta, scale, log_read, pois_read, zeros, flow
    ):
        logs, pois, zero, mean_flow = _fits(places, beta, scale)
        assert logs == pytest.approx(log_read, abs=1e-3)
        assert pois == pytest.approx(pois_read, abs=1e-3)
        assert zero == pytest.approx(zeros, abs=1e-3)
        assert mean_flow == pytest.approx(flow, abs=0.01)
        assert logs < beta or zero < 0.05
        assert abs(pois - beta) < 0.02 * beta or mean_flow < 0.25


class TestPieces:
    def test_poisson_draws_and_refusals(self, places):
        rng = random.Random(1)
        draws = [poisson(3.0, rng) for _ in range(20000)]
        assert sum(draws) / 20000 == pytest.approx(3.0, abs=0.05)
        assert poisson(0.0, rng) == 0
        assert poisson(1000.0, rng) > 800
        origins, sizes = places
        with pytest.raises(Invalid):
            expected_flows(origins, sizes[:3], 1.0, 1.0)
        with pytest.raises(Invalid):
            log_fit(origins, sizes, [[0.0] * 25 for _ in range(25)])
