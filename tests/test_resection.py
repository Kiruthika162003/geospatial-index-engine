from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Degenerate, Invalid
from atlas.resection import (
    bearing_to,
    circumcircle,
    dilution,
    locate,
    on_circle,
    residual_deg,
    subtended_angles,
)

TRUTH = (3.0, 4.0)
TRIANGLE = [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0)]


def _ring(angles, radius=10.0):
    return [on_circle(TRUTH, radius, a) for a in angles]


class TestExactness:
    def test_noise_free_bearings_recover_the_observer(self):
        for landmarks in (_ring((0, 120, 240)), _ring((80, 90, 100)), _ring((0, 90))):
            bearings = [bearing_to(TRUTH, lm) for lm in landmarks]
            p = locate(landmarks, bearings)
            assert math.hypot(p[0] - 3, p[1] - 4) < 1e-12
            assert residual_deg(landmarks, bearings, p) < 1e-9

    def test_random_layouts_recover_the_observer(self):
        rng = random.Random(184)
        for _ in range(300):
            count = rng.randint(2, 6)
            landmarks = [(rng.uniform(-100, 100), rng.uniform(-100, 100)) for _ in range(count)]
            t = (rng.uniform(-100, 100), rng.uniform(-100, 100))
            q = locate(landmarks, [bearing_to(t, lm) for lm in landmarks])
            assert math.hypot(q[0] - t[0], q[1] - t[1]) < 1e-8

    def test_bearings_are_compass_bearings(self):
        assert bearing_to((0, 0), (0, 1)) == 0.0
        assert bearing_to((0, 0), (1, 0)) == 90.0
        assert bearing_to((0, 0), (-1, -1)) == 225.0


class TestDilution:
    def test_range_multiplies_the_error_and_bunching_adds_to_it(self):
        near = dilution(_ring((0, 120, 240)), TRUTH, 0.5, 2000, random.Random(183))
        far = dilution(_ring((0, 120, 240), 100.0), TRUTH, 0.5, 2000, random.Random(183))
        bunched = dilution(_ring((80, 90, 100)), TRUTH, 0.5, 2000, random.Random(183))
        assert near == pytest.approx(0.2025, abs=1e-3)
        assert far == pytest.approx(2.025, abs=1e-2)
        assert far / near == pytest.approx(10.0, abs=1e-6)
        assert bunched == pytest.approx(0.717, abs=1e-2)


class TestTheDangerCircle:
    def test_with_a_compass_the_circle_is_an_ordinary_place(self):
        center, radius = circumcircle(*TRIANGLE)
        assert center == pytest.approx((5.0, 2.4375))
        assert radius == pytest.approx(5.5625)
        rim, half = on_circle(center, radius, 200), on_circle(center, radius / 2, 200)
        on = dilution(TRIANGLE, rim, 0.5, 2000, random.Random(185))
        mid = dilution(TRIANGLE, half, 0.5, 2000, random.Random(185))
        at_center = dilution(TRIANGLE, center, 0.5, 2000, random.Random(185))
        assert on == pytest.approx(0.1616, abs=2e-3)
        assert mid == pytest.approx(0.1626, abs=2e-3)
        assert at_center == pytest.approx(0.1155, abs=2e-3)
        assert on / at_center < 1.5  # the guessed collapse did not happen

    def test_without_a_compass_every_point_on_the_arc_looks_the_same(self):
        center, radius = circumcircle(*TRIANGLE)
        base = subtended_angles(on_circle(center, radius, 150), TRIANGLE)
        assert base == pytest.approx([115.989, 302.005], abs=1e-3)
        for angle in range(150, 211, 10):
            seen = subtended_angles(on_circle(center, radius, angle), TRIANGLE)
            assert seen == pytest.approx(base, abs=1e-9)
        inside = subtended_angles(on_circle(center, radius * 0.5, 180), TRIANGLE)[0]
        outside = subtended_angles(on_circle(center, radius * 1.5, 180), TRIANGLE)[0]
        assert [inside, outside] == pytest.approx([172.13, 80.50], abs=1e-2)


class TestRefusals:
    def test_parallel_bearings_and_bad_inputs_are_refused(self):
        with pytest.raises(Degenerate):
            locate([(0, 0), (0, 10)], [0.0, 0.0])
        with pytest.raises(Degenerate):
            circumcircle((0, 0), (1, 1), (2, 2))
        with pytest.raises(Invalid):
            locate([(0, 0)], [1])
        with pytest.raises(Invalid):
            locate(TRIANGLE, [1, 2])
