from __future__ import annotations

import math
import random

import pytest

from atlas.deadreckoning import (
    advance,
    bias_error_fraction,
    distance_run,
    end_error,
    reckon,
    split_error,
    track,
    with_heading_bias,
    with_heading_noise,
)
from atlas.errors import Invalid

START = (0.0, 0.0)


def _legs(count: int) -> list[tuple[float, float, float]]:
    return [(30.0, 10.0, 1.0)] * count


class TestExactness:
    def test_a_hundred_legs_northeast_land_where_they_should(self):
        end = reckon(START, [(45.0, 10.0, 1.0)] * 100)
        assert end == pytest.approx((1000 / math.sqrt(2), 1000 / math.sqrt(2)))
        assert len(track(START, [(45.0, 10.0, 1.0)] * 100)) == 101
        assert distance_run([(45.0, 10.0, 1.0)] * 100) == 1000.0

    def test_a_square_closes(self):
        square = [(0, 1, 10), (90, 1, 10), (180, 1, 10), (270, 1, 10)]
        assert reckon(START, square) == pytest.approx((0.0, 0.0), abs=1e-12)


class TestHeadingBias:
    @pytest.mark.parametrize(
        ("bias", "fraction"), [(1.0, 0.01745), (2.0, 0.0349), (5.0, 0.08724)]
    )
    def test_the_error_is_the_chord_of_the_bias_times_the_distance(self, bias, fraction):
        for count in (25, 100):
            legs = _legs(count)
            error = end_error(START, legs, with_heading_bias(legs, bias))
            assert error / distance_run(legs) == pytest.approx(fraction, abs=1e-5)
            assert error / distance_run(legs) == pytest.approx(bias_error_fraction(bias))


class TestHeadingNoise:
    def test_the_error_grows_as_the_square_root_of_the_leg_count(self):
        rng = random.Random(186)
        rms = {}
        along = {}
        across = {}
        for count in (25, 100, 400):
            legs = _legs(count)
            sq = al = ac = 0.0
            for _ in range(500):
                noisy = with_heading_noise(legs, 1.0, rng)
                sq += end_error(START, legs, noisy) ** 2
                a, c = split_error(START, legs, noisy)
                al += a * a
                ac += c * c
            rms[count] = math.sqrt(sq / 500)
            along[count] = math.sqrt(al / 500)
            across[count] = math.sqrt(ac / 500)
        assert rms[25] == pytest.approx(0.951, abs=1e-2)
        assert rms[100] == pytest.approx(1.844, abs=1e-2)
        assert rms[400] == pytest.approx(3.635, abs=1e-2)
        assert rms[100] / rms[25] == pytest.approx(1.94, abs=0.03)
        assert rms[400] / rms[100] == pytest.approx(1.97, abs=0.03)
        # the error lives across track
        assert across[25] == pytest.approx(0.951, abs=1e-2)
        assert along[25] == pytest.approx(0.04, abs=1e-2)
        assert along[400] == pytest.approx(0.612, abs=1e-2)
        # a hundred noisy legs end a tenth as far off as a one-degree bias
        bias = end_error(START, _legs(100), with_heading_bias(_legs(100), 1.0))
        assert rms[100] / bias == pytest.approx(0.106, abs=3e-3)


class TestSpeedBias:
    def test_a_speed_bias_errs_entirely_along_track(self):
        legs = _legs(100)
        fast = [(h, s * 1.02, d) for h, s, d in legs]
        along, across = split_error(START, legs, fast)
        assert along == pytest.approx(20.0, abs=1e-9)
        assert abs(across) < 1e-9
        assert end_error(START, legs, fast) / distance_run(legs) == pytest.approx(0.02)


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            advance(START, 0, -1, 1)
        with pytest.raises(Invalid):
            split_error(START, [(0, 0, 1)], [(0, 0, 1)])
