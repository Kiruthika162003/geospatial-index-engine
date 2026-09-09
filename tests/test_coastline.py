from __future__ import annotations

import math
import random

import pytest

from atlas.coastline import circle, divider_length, divider_walk, koch, richardson_slope
from atlas.errors import Invalid


class TestTheCircle:
    @pytest.mark.parametrize(("span", "steps"), [(1.0, 6), (0.5, 12), (0.1, 62), (0.02, 314)])
    def test_the_walk_takes_the_chord_count(self, span, steps):
        ring = circle(3600)
        taken, leftover = divider_walk(ring, span)
        assert taken == steps
        assert taken == math.floor(2 * math.pi / (2 * math.asin(span / 2)) + 1e-9)
        assert leftover < span

    def test_the_slope_reads_dimension_one(self):
        assert richardson_slope(circle(3600), [0.5, 0.2, 0.1, 0.05, 0.02]) == pytest.approx(
            -0.0129, abs=1e-3
        )


class TestTheKochCurve:
    def test_power_of_three_spans_take_powers_of_four_and_fit_the_dimension(self):
        curve = koch(6)
        for p in (1, 2, 3, 4, 5):
            steps, leftover = divider_walk(curve, 3.0**-p)
            assert steps == 4**p
            assert leftover < 1e-12
        spans = [3.0**-p for p in (1, 2, 3, 4, 5)]
        expected = 1 - math.log(4) / math.log(3)
        assert richardson_slope(curve, spans) == pytest.approx(expected, abs=1e-4)

    def test_the_staircase_between_the_scales(self):
        curve = koch(6)
        assert divider_length(curve, 0.5) == pytest.approx(1.0)
        assert divider_length(curve, 0.25) == pytest.approx(1.0)
        assert divider_length(curve, 0.15) == pytest.approx(1.2)
        assert divider_length(curve, 0.02) == pytest.approx(2.56)
        assert divider_length(curve, 0.01) == pytest.approx(2.56)
        between = [0.5, 0.25, 0.15, 0.08, 0.05, 0.02, 0.01]
        assert richardson_slope(curve, between) == pytest.approx(-0.2804, abs=1e-3)


class TestTheWalk:
    def test_a_random_walk_reads_steeper_than_minus_one(self):
        rng = random.Random(248)
        walk = [(0.0, 0.0)]
        for _ in range(5000):
            a = rng.uniform(0, 2 * math.pi)
            x, y = walk[-1]
            walk.append((x + 0.01 * math.cos(a), y + 0.01 * math.sin(a)))
        slope = richardson_slope(walk, [0.5, 0.3, 0.2, 0.1, 0.05])
        assert slope == pytest.approx(-1.272, abs=1e-2)


class TestRefusals:
    def test_bad_spans_lines_and_fits_are_refused(self):
        with pytest.raises(Invalid):
            divider_walk(koch(2), 0)
        with pytest.raises(Invalid):
            divider_walk([(0, 0)], 1)
        with pytest.raises(Invalid):
            richardson_slope(koch(2), [0.1])
