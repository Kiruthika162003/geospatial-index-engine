from __future__ import annotations

import pytest

from atlas.chaikin import (
    first_pass_bound,
    length,
    limit_departure,
    max_departure,
    point_count,
    smooth,
    smooth_once,
    zigzag,
)
from atlas.errors import Invalid
from atlas.jarvismarch import convex_hull

ZIGZAG = zigzag(10)
SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _inside_hull(hull, pt) -> bool:
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        if (bx - ax) * (pt[1] - ay) - (by - ay) * (pt[0] - ax) < -1e-12:
            return False
    return True


class TestOnePass:
    def test_the_corner_is_cut_at_the_quarter_points(self):
        assert smooth_once([(0, 0), (1, 0), (1, 1)]) == [
            (0, 0),
            (0.25, 0.0),
            (0.75, 0.0),
            (1.0, 0.25),
            (1.0, 0.75),
            (1, 1),
        ]

    def test_the_point_count_doubles_exactly(self):
        for passes in range(1, 6):
            assert len(smooth(ZIGZAG, passes)) == 11 * 2**passes
            assert point_count(11, passes) == 11 * 2**passes


class TestShrinkage:
    def test_a_right_angle_zigzag_loses_13_percent_then_a_quarter_of_that_each_pass(self):
        assert length(ZIGZAG) == 10.0
        lengths = [length(smooth(ZIGZAG, p)) for p in range(0, 9)]
        steps = [1 - lengths[i + 1] / lengths[i] for i in range(8)]
        assert steps[0] == pytest.approx(0.1318, abs=1e-4)
        assert steps[1] == pytest.approx(0.0327, abs=1e-4)
        assert steps[2] == pytest.approx(0.0084, abs=1e-4)
        ratios = [steps[i] / steps[i + 1] for i in range(7)]
        assert ratios[:3] == pytest.approx([4.037, 3.884, 3.978], abs=1e-3)
        for ratio in ratios[3:]:
            assert ratio == pytest.approx(4.0, abs=0.01)
        assert 1 - lengths[8] / 10.0 == pytest.approx(0.16955, abs=1e-4)

    def test_a_closed_square_loses_15_percent_then_settles_near_19(self):
        one = length(smooth(SQUARE, 1, closed=True), closed=True)
        six = length(smooth(SQUARE, 6, closed=True), closed=True)
        assert one == pytest.approx(3.41421, 1e-5)
        assert six == pytest.approx(3.24661, 1e-5)
        assert len(smooth(SQUARE, 3, closed=True)) == 32


class TestDeparture:
    def test_the_first_pass_departs_by_nothing_and_later_passes_halve_toward_an_eighth(self):
        departures = [max_departure(ZIGZAG, smooth(ZIGZAG, p)) for p in range(1, 9)]
        assert departures[0] == 0.0
        assert departures[1] == pytest.approx(0.0625)
        assert departures[2] == pytest.approx(0.09375)
        assert departures[7] == pytest.approx(0.12402, abs=1e-4)
        assert departures[7] < limit_departure(ZIGZAG) == 0.125
        assert first_pass_bound(ZIGZAG) == 0.25  # holds, loose by two

    def test_the_line_stays_inside_the_hull_with_its_ends_fixed(self):
        hull = convex_hull(list(ZIGZAG))
        smoothed = smooth(ZIGZAG, 6)
        assert all(_inside_hull(hull, pt) for pt in smoothed)
        assert smoothed[0] == ZIGZAG[0]
        assert smoothed[-1] == ZIGZAG[-1]


class TestRefusals:
    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            smooth_once([(0, 0)])
        with pytest.raises(Invalid):
            smooth(ZIGZAG, -1)
        with pytest.raises(Invalid):
            point_count(1, 1)
