from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.speedprofile import flag_implausible, speeds, spread

CLEAN = [(10.0 * t, 0.0, float(t)) for t in range(200)]


class TestCleanTrace:
    def test_a_constant_speed_trace_gives_a_flat_profile(self):
        s = speeds(CLEAN)
        assert min(s) == max(s) == 10.0
        assert spread(s) == 0.0
        assert flag_implausible(CLEAN, ceiling=50.0) == []


class TestNoiseScaling:
    def test_speed_noise_grows_as_the_interval_shrinks(self):
        rng = random.Random(167)
        spreads = []
        for dt in (4.0, 2.0, 1.0, 0.5):
            trace = [
                (10.0 * t * dt + rng.gauss(0, 2), rng.gauss(0, 2), t * dt) for t in range(400)
            ]
            spreads.append(spread(speeds(trace)))
        assert spreads == pytest.approx([0.074, 0.140, 0.262, 0.424], abs=0.01)
        assert spreads == sorted(spreads)
        # doubling per halving until the noise nears the step length, where it compresses
        assert spreads[1] / spreads[0] == pytest.approx(1.89, abs=0.1)
        assert spreads[2] / spreads[1] == pytest.approx(1.87, abs=0.1)
        assert spreads[3] / spreads[2] == pytest.approx(1.62, abs=0.1)


class TestGlitches:
    def test_one_teleported_fix_flags_exactly_two_segments(self):
        glitched = list(CLEAN)
        x, y, t = glitched[100]
        glitched[100] = (x + 500.0, y, t)
        assert flag_implausible(glitched, ceiling=50.0) == [99, 100]


class TestRefusals:
    def test_fewer_than_two_fixes_is_refused(self):
        with pytest.raises(Invalid):
            speeds([(0, 0, 0)])

    def test_non_increasing_times_are_refused(self):
        with pytest.raises(Invalid):
            speeds([(0, 0, 1.0), (1, 0, 1.0)])

    def test_a_non_positive_ceiling_is_refused(self):
        with pytest.raises(Invalid):
            flag_implausible(CLEAN, ceiling=0)
