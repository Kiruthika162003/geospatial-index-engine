from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Degenerate, Invalid
from atlas.trilateration import dilution, locate, normal_matrix_determinant, residual

TRUTH = (3.0, 4.0)


def _ring(angles: tuple[int, ...]) -> list[tuple[float, float]]:
    return [
        (3 + 10 * math.cos(math.radians(a)), 4 + 10 * math.sin(math.radians(a))) for a in angles
    ]


SPREAD = _ring((0, 120, 240))
BUNCHED = _ring((80, 90, 100))
COLLINEAR = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]


def _exact(beacons):
    return [math.hypot(x - TRUTH[0], y - TRUTH[1]) for x, y in beacons]


class TestExactness:
    def test_exact_distances_recover_the_truth(self):
        for beacons in (SPREAD, BUNCHED):
            p = locate(beacons, _exact(beacons))
            assert math.hypot(p[0] - 3, p[1] - 4) < 1e-12
            assert residual(beacons, _exact(beacons), p) < 1e-12

    def test_random_layouts_recover_the_truth(self):
        rng = random.Random(180)
        for _ in range(300):
            beacons = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(3, 6)]
            t = (rng.uniform(-100, 100), rng.uniform(-100, 100))
            ds = [math.hypot(x - t[0], y - t[1]) for x, y in beacons]
            q = locate(beacons, ds)
            assert math.hypot(q[0] - t[0], q[1] - t[1]) < 1e-8


class TestDilution:
    def test_bunched_beacons_amplify_noise_eightyfold(self):
        assert dilution(SPREAD, TRUTH, 0.1, 2000, random.Random(181)) == pytest.approx(
            1.144, abs=0.01
        )
        assert dilution(BUNCHED, TRUTH, 0.1, 2000, random.Random(181)) == pytest.approx(
            80.0, abs=0.5
        )
        four = [*SPREAD, (3.0, 14.0)]
        assert dilution(four, TRUTH, 0.1, 2000, random.Random(181)) == pytest.approx(
            1.073, abs=0.01
        )

    def test_the_determinant_tells_the_geometry_before_any_noise(self):
        assert normal_matrix_determinant(SPREAD) == pytest.approx(1_080_000.0)
        assert normal_matrix_determinant(BUNCHED) == pytest.approx(4.454, abs=1e-3)
        assert normal_matrix_determinant(COLLINEAR) == 0


class TestBias:
    def test_a_common_bias_cancels_in_the_position_and_shows_in_the_residual(self):
        biased = [d + 0.5 for d in _exact(SPREAD)]
        p = locate(SPREAD, biased)
        assert p == pytest.approx(TRUTH, abs=1e-12)
        assert residual(SPREAD, biased, p) == pytest.approx(0.5, abs=1e-12)


class TestRefusals:
    def test_collinear_beacons_are_refused(self):
        with pytest.raises(Degenerate):
            locate(COLLINEAR, [5, 0, 5])

    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            locate([(0, 0), (1, 0)], [1, 1])
        with pytest.raises(Invalid):
            locate(SPREAD, [1, 2])
        with pytest.raises(Invalid):
            locate(SPREAD, [1, -2, 3])
