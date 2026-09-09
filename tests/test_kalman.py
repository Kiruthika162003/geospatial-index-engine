from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.kalman import rms_error, smooth

CLEAN_TURN = [(float(t), 0.0) for t in range(100)] + [(99.0, float(t)) for t in range(1, 101)]


def _jittered(truth, seed=163, sigma=1.0):
    rng = random.Random(seed)
    return [(x + rng.gauss(0, sigma), y + rng.gauss(0, sigma)) for x, y in truth]


class TestStraightLineGain:
    def test_less_process_noise_smooths_harder(self):
        truth = [(float(t), 0.0) for t in range(200)]
        noisy = _jittered(truth)
        raw = rms_error(noisy, truth)
        gains = [
            raw / rms_error(smooth(noisy, process=q, measure=1.0), truth)
            for q in (1.0, 0.1, 0.01)
        ]
        assert gains == pytest.approx([1.18, 1.47, 1.90], abs=0.05)
        assert gains == sorted(gains)

    def test_a_clean_straight_converges_to_the_truth(self):
        truth = [(float(t), 0.0) for t in range(100)]
        s = smooth(truth, process=0.01, measure=1.0)
        assert math.hypot(s[-1][0] - 99, s[-1][1]) < 1e-6


class TestCornerLag:
    def test_on_a_clean_turn_the_lag_grows_as_smoothing_strengthens(self):
        peaks, settles = [], []
        for q in (1.0, 0.1, 0.01):
            s = smooth(CLEAN_TURN, process=q, measure=1.0)
            errs = [
                math.hypot(s[t][0] - CLEAN_TURN[t][0], s[t][1] - CLEAN_TURN[t][1])
                for t in range(100, 140)
            ]
            peaks.append(max(errs))
            settles.append(next(k for k in range(len(errs)) if all(e < 0.1 for e in errs[k:])))
        assert peaks == pytest.approx([0.25, 0.73, 1.59], abs=0.03)
        assert settles == [2, 6, 12]

    def test_with_jitter_the_corner_winner_depends_on_the_draw(self):
        # two guesses refuted in turn: neither setting reliably wins a jittered corner,
        # because the lag added and the jitter removed are the same size here
        def corner(seed, q):
            noisy = _jittered(CLEAN_TURN, seed=seed)
            s = smooth(noisy, process=q, measure=1.0)
            return max(
                math.hypot(s[t][0] - CLEAN_TURN[t][0], s[t][1] - CLEAN_TURN[t][1])
                for t in range(100, 112)
            )

        assert corner(160, 0.01) == pytest.approx(1.42, abs=0.05)
        assert corner(160, 1.0) == pytest.approx(2.15, abs=0.05)
        assert corner(160, 0.01) < corner(160, 1.0)  # heavy smoothing wins this draw
        assert corner(163, 0.01) == pytest.approx(1.99, abs=0.05)
        assert corner(163, 1.0) == pytest.approx(1.42, abs=0.05)
        assert corner(163, 0.01) > corner(163, 1.0)  # and loses this one

    def test_heavy_smoothing_wins_the_jittered_corner_about_half_the_time(self):
        wins = 0
        for seed in range(160, 180):
            noisy = _jittered(CLEAN_TURN, seed=seed)
            errors = {}
            for q in (1.0, 0.01):
                s = smooth(noisy, process=q, measure=1.0)
                errors[q] = max(
                    math.hypot(s[t][0] - CLEAN_TURN[t][0], s[t][1] - CLEAN_TURN[t][1])
                    for t in range(100, 112)
                )
            wins += errors[0.01] < errors[1.0]
        assert wins == 8  # of 20: a coin flip, not a rule


class TestRefusals:
    def test_an_empty_trace_is_refused(self):
        with pytest.raises(Invalid):
            smooth([], 1.0, 1.0)

    def test_non_positive_noise_or_dt_is_refused(self):
        with pytest.raises(Invalid):
            smooth([(0, 0)], process=0, measure=1.0)
        with pytest.raises(Invalid):
            smooth([(0, 0)], process=1.0, measure=1.0, dt=0)

    def test_mismatched_traces_are_refused(self):
        with pytest.raises(Invalid):
            rms_error([(0, 0)], [(0, 0), (1, 1)])
