from __future__ import annotations

import math
import random

import pytest

from atlas.bearingstats import (
    angular_gap,
    arithmetic_mean,
    circular_mean,
    circular_std_deg,
    circular_variance,
    resultant_length,
    rotate,
)
from atlas.errors import Degenerate, Invalid


def _cluster() -> list[float]:
    rng = random.Random(179)
    return [(5 + rng.gauss(0, 8)) % 360 for _ in range(200)]


class TestTheLoudCase:
    def test_350_and_10_average_to_north_not_south(self):
        assert circular_mean([350, 10]) == 0.0
        assert arithmetic_mean([350, 10]) == 180.0
        assert resultant_length([350, 10]) == pytest.approx(0.98481, abs=1e-5)
        assert circular_variance([350, 10]) == pytest.approx(0.01519, abs=1e-5)
        assert circular_std_deg([350, 10]) == pytest.approx(10.026, abs=1e-3)


class TestTheQuietCase:
    def test_wrapped_members_drag_the_arithmetic_mean_a_hundred_degrees(self):
        cluster = _cluster()
        assert sum(1 for b in cluster if b > 180) == 58
        assert circular_mean(cluster) == pytest.approx(4.908, abs=1e-3)
        assert arithmetic_mean(cluster) == pytest.approx(109.31, abs=1e-2)
        assert angular_gap(circular_mean(cluster), 5) < 0.1
        assert angular_gap(arithmetic_mean(cluster), 5) == pytest.approx(104.31, abs=1e-2)

    def test_rotation_invariance_holds_for_the_vector_mean_only(self):
        cluster = _cluster()
        base = circular_mean(cluster)
        worst_circ = worst_arith = 0.0
        for by in range(0, 360, 15):
            rot = rotate(cluster, by)
            worst_circ = max(worst_circ, angular_gap((circular_mean(rot) - by) % 360, base))
            arith = (arithmetic_mean(rot) - by) % 360
            worst_arith = max(worst_arith, angular_gap(arith, arithmetic_mean(cluster) % 360))
        assert worst_circ < 1e-9
        assert worst_arith == pytest.approx(147.6, abs=0.1)


class TestSpread:
    @pytest.mark.parametrize(("sigma", "tolerance"), [(5, 1e-4), (60, 2e-3)])
    def test_circular_std_matches_the_ordinary_one_on_gaussian_bearings(self, sigma, tolerance):
        rng = random.Random(182 + sigma)
        bearings = [180 + rng.gauss(0, sigma) for _ in range(5000)]
        mean = sum(bearings) / len(bearings)
        ordinary = math.sqrt(sum((b - mean) ** 2 for b in bearings) / len(bearings))
        assert circular_std_deg(bearings) / ordinary == pytest.approx(1.0, abs=tolerance)

    def test_a_uniform_spread_has_no_resultant(self):
        assert resultant_length([i * 36.0 for i in range(10)]) < 1e-12
        assert circular_std_deg([0, 180]) > 400


class TestGaps:
    def test_short_way_round(self):
        assert angular_gap(350, 10) == 20.0
        assert angular_gap(0, 180) == 180.0
        assert angular_gap(90, 270) == 180.0


class TestRefusals:
    def test_cancelling_and_empty_inputs_are_refused(self):
        with pytest.raises(Degenerate):
            circular_mean([0, 180])
        with pytest.raises(Invalid):
            circular_mean([])
        with pytest.raises(Invalid):
            arithmetic_mean([])
