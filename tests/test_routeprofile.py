from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.routeprofile import (
    ascent_descent,
    ground_length,
    hill,
    jittered,
    profile,
    sample,
    steepest_grade,
)

ROUTE = [(5.0, 40.0), (75.0, 40.0)]
HILL = hill(81, 100.0, 35.0)


class TestTheSmoothHill:
    @pytest.mark.parametrize(
        ("step", "ascent"), [(0.25, 100.0), (1, 100.0), (2, 97.143), (8, 88.889)]
    )
    def test_fine_steps_read_the_peak_and_coarse_steps_clip_it(self, step, ascent):
        prof = profile(HILL, ROUTE, step)
        up, down = ascent_descent(prof)
        assert up == pytest.approx(ascent, abs=1e-3)
        assert down == pytest.approx(ascent, abs=1e-3)
        assert steepest_grade(prof) == pytest.approx(100 / 35, abs=1e-3)

    def test_the_ground_length_and_the_bilinear_sample(self):
        prof = profile(HILL, ROUTE, 1)
        assert ground_length(prof) / 70.0 == pytest.approx(3.0271, abs=1e-3)
        assert sample(HILL, 40, 40) == HILL[40][40]
        assert sample(HILL, 40.5, 40) == pytest.approx((HILL[40][40] + HILL[40][41]) / 2)

    def test_a_meter_of_noise_does_not_move_a_steep_hills_ascent(self):
        noisy = jittered(HILL, 1.0, random.Random(259))
        for step in (0.25, 1, 4):
            up, _ = ascent_descent(profile(noisy, ROUTE, step))
            assert up == pytest.approx(100.0, abs=1.0)


class TestNoisyGround:
    def test_flat_noisy_ground_reads_42_meters_of_false_ascent_at_fine_steps(self):
        rng = random.Random(260)
        flat = jittered([[0.0] * 81 for _ in range(81)], 1.0, rng)
        readings = {}
        for step in (0.25, 1, 2, 4, 8):
            readings[step] = ascent_descent(profile(flat, ROUTE, step))[0]
        assert readings[0.25] == pytest.approx(42.4, abs=0.1)
        assert readings[1] == pytest.approx(readings[0.25])
        assert readings[2] == pytest.approx(21.0, abs=0.1)
        assert readings[8] == pytest.approx(4.2, abs=0.1)
        fine = profile(flat, ROUTE, 0.25)
        assert ascent_descent(fine, 5.0)[0] == 0.0
        assert ascent_descent(fine, 2.0)[0] == pytest.approx(13.3, abs=0.1)

    def test_a_gentle_slope_inflates_1_69_times_and_a_3_meter_threshold_restores_it(self):
        rng = random.Random(260)
        jittered([[0.0] * 81 for _ in range(81)], 1.0, rng)
        gentle = jittered([[0.5 * c for c in range(81)] for _ in range(81)], 1.0, rng)
        fine = profile(gentle, ROUTE, 0.25)
        assert ascent_descent(fine)[0] == pytest.approx(59.2, abs=0.1)
        assert ascent_descent(fine, 3.0)[0] == pytest.approx(35.2, abs=0.1)
        assert ascent_descent(fine, 5.0)[0] == pytest.approx(32.4, abs=0.1)
        assert ascent_descent(profile(gentle, ROUTE, 4))[0] == pytest.approx(35.4, abs=0.1)


class TestRefusals:
    def test_bad_grids_routes_steps_and_thresholds_are_refused(self):
        with pytest.raises(Invalid):
            profile([], ROUTE, 1)
        with pytest.raises(Invalid):
            profile(HILL, [(0, 0)], 1)
        with pytest.raises(Invalid):
            profile(HILL, ROUTE, 0)
        with pytest.raises(Invalid):
            ascent_descent([(0, 0), (1, 1)], -1)
