from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.idw import best_power, estimate, nearest_value, rms_error, slope_at_sample


def plane(x: float, y: float) -> float:
    return 2.0 + 0.5 * x - 0.3 * y


def bowl(x: float, y: float) -> float:
    return (x - 5) ** 2 / 10 + (y - 5) ** 2 / 10


def _layout():
    rng = random.Random(193)
    samples_xy = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(60)]
    probes = [(rng.uniform(1, 9), rng.uniform(1, 9)) for _ in range(400)]
    return samples_xy, probes


class TestPower:
    def test_the_error_bottoms_at_a_middling_power_on_both_fields(self):
        samples_xy, probes = _layout()
        for field, expected, best in (
            (plane, {0.5: 1.035, 2: 0.2983, 3: 0.1724, 4: 0.1716, 10: 0.2309}, 4),
            (bowl, {0.5: 0.6696, 2: 0.3318, 3: 0.2358, 4: 0.2158, 10: 0.262}, 4),
        ):
            samples = [(x, y, field(x, y)) for x, y in samples_xy]
            for p, err in expected.items():
                assert rms_error(samples, field, probes, p) == pytest.approx(err, abs=1e-3)
            assert best_power(samples, field, probes, (0.5, 1, 2, 3, 4, 6, 10))[0] == best

    def test_a_high_power_closes_on_the_nearest_sample_mosaic(self):
        samples_xy, probes = _layout()
        samples = [(x, y, plane(x, y)) for x, y in samples_xy]
        mosaic = math.sqrt(
            sum((nearest_value(samples, *p) - plane(*p)) ** 2 for p in probes) / len(probes)
        )
        assert mosaic == pytest.approx(0.2779, abs=1e-3)
        assert abs(rms_error(samples, plane, probes, 10) - mosaic) < 0.05
        assert abs(rms_error(samples, plane, probes, 40) - mosaic) < 0.01


class TestTheBullseye:
    def test_the_estimate_is_exact_and_flat_at_every_sample(self):
        samples_xy, _ = _layout()
        samples = [(x, y, plane(x, y)) for x, y in samples_xy]
        assert all(estimate(samples, x, y) == v for x, y, v in samples)
        close = max(math.hypot(*slope_at_sample(samples, i, 1e-3)) for i in range(60))
        far = max(math.hypot(*slope_at_sample(samples, i, 0.5)) for i in range(60))
        assert close < 1e-3
        assert far == pytest.approx(0.623, abs=1e-2)

    def test_between_two_samples_the_estimate_hugs_the_near_one_except_at_power_one(self):
        two = [(0.0, 0.0, 0.0), (10.0, 0.0, 10.0)]
        assert estimate(two, 5, 0) == 5.0
        assert estimate(two, 1, 0) == pytest.approx(0.12195, abs=1e-4)
        assert estimate(two, 1, 0, 4.0) == pytest.approx(0.00152, abs=1e-4)
        assert estimate(two, 1, 0, 1.0) == pytest.approx(1.0)
        assert estimate(two, 9, 0) == pytest.approx(9.878, abs=1e-3)


class TestTheEdge:
    def test_beyond_the_samples_the_estimate_levels_toward_their_mean(self):
        samples_xy, _ = _layout()
        samples = [(x, y, plane(x, y)) for x, y in samples_xy]
        mean = sum(s[2] for s in samples) / 60
        assert mean == pytest.approx(3.012, abs=1e-3)
        assert estimate(samples, 20, 5) == pytest.approx(3.515, abs=1e-2)
        assert estimate(samples, 50, 5) == pytest.approx(3.184, abs=1e-2)
        assert estimate(samples, -10, 5) == pytest.approx(2.492, abs=1e-2)
        assert plane(20, 5) == 10.5


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            estimate([], 0, 0)
        with pytest.raises(Invalid):
            estimate([(0, 0, 1)], 1, 1, power=0)
        with pytest.raises(Invalid):
            nearest_value([], 0, 0)
        with pytest.raises(Invalid):
            rms_error([(0, 0, 1)], plane, [])
        with pytest.raises(Invalid):
            best_power([(0, 0, 1)], plane, [(1, 1)], [])
