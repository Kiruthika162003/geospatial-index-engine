from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.gpsjitter import (
    break_even_step,
    inflation,
    inflation_law,
    jittered,
    length,
    mean,
    moving_step_law,
    smoothed,
    speed_readings,
    still_step_law,
    still_track,
    straight_track,
    thinned,
)


class TestStill:
    @pytest.mark.parametrize(
        ("sigma", "step", "law"),
        [(1.0, 1.7619, 1.7725), (3.0, 5.2858, 5.3174), (5.0, 8.8097, 8.8623)],
    )
    def test_a_still_receiver_walks_1_77_sigma_a_fix(self, sigma, step, law):
        still = still_track(2001)
        steps = [
            length(jittered(still, sigma, random.Random(731 + k))) / 2000 for k in range(5)
        ]
        assert sum(steps) / 5 == pytest.approx(step, abs=1e-4)
        assert still_step_law(sigma) == pytest.approx(law, abs=1e-4)
        assert 0.99 < (sum(steps) / 5) / law < 1.0
        assert mean(speed_readings(still, sigma, 10.0, random.Random(732))) == pytest.approx(
            law / 10, abs=0.02
        )


class TestMoving:
    @pytest.mark.parametrize(
        ("step", "read", "law", "smooth", "thin"),
        [
            (1.0, 5.3942, 5.3947, 1.4075, 1.3948),
            (5.0, 1.4043, 1.4033, 1.0133, 1.0135),
            (10.0, 1.0964, 1.095, 1.0026, 1.0035),
            (20.0, 1.0229, 1.0225, 0.9999, 1.001),
        ],
    )
    def test_inflation_and_its_cures(self, step, read, law, smooth, thin):
        track = straight_track(2001, step)
        assert inflation(track, 3.0, random.Random(733)) == pytest.approx(read, abs=1e-4)
        assert moving_step_law(step, 3.0, 50000, 1) / step == pytest.approx(law, abs=1e-4)
        noisy = jittered(track, 3.0, random.Random(734))
        assert length(smoothed(noisy, 5)) / length(track) == pytest.approx(smooth, abs=1e-4)
        assert length(thinned(noisy, 5)) / length(track) == pytest.approx(thin, abs=1e-4)
        if step >= 10:
            assert inflation_law(step, 3.0) == pytest.approx(read, abs=0.01)

    def test_the_break_even_step(self):
        assert break_even_step(3.0) == pytest.approx(3 * math.sqrt(10))
        assert inflation_law(break_even_step(3.0), 3.0) == pytest.approx(1.1)
        assert inflation_law(5.0, 3.0) == pytest.approx(1.36)


class TestRefusals:
    def test_bad_sigmas_tracks_windows_and_steps(self):
        with pytest.raises(Invalid):
            jittered([(0.0, 0.0)], -1.0, random.Random(1))
        with pytest.raises(Invalid):
            still_track(1)
        with pytest.raises(Invalid):
            inflation(still_track(5), 1.0, random.Random(1))
        with pytest.raises(Invalid):
            speed_readings(still_track(5), 1.0, 0.0, random.Random(1))
        with pytest.raises(Invalid):
            smoothed(still_track(5), 4)
        with pytest.raises(Invalid):
            thinned(still_track(5), 0)
        with pytest.raises(Invalid):
            inflation_law(0.0, 1.0)
        with pytest.raises(Invalid):
            mean([])
