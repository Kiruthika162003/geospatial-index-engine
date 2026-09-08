from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Degenerate, Invalid
from atlas.minrect import axis_aligned_area, minimum_rectangle

RECT_10_BY_4 = [(0, 0), (10, 0), (10, 4), (0, 4)]


def _rotate(points, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return [(x * c - y * s, x * s + y * c) for x, y in points]


class TestGain:
    def test_an_axis_aligned_shape_gains_nothing(self):
        area, angle, _, _ = minimum_rectangle(RECT_10_BY_4)
        assert area == pytest.approx(40.0)
        assert axis_aligned_area(RECT_10_BY_4) == pytest.approx(40.0)
        assert angle == pytest.approx(0.0)

    def test_a_forty_five_degree_tilt_gains_the_elongation_factor(self):
        tilted = _rotate(RECT_10_BY_4, 45)
        area, _, w, h = minimum_rectangle(tilted)
        box = axis_aligned_area(tilted)
        assert area == pytest.approx(40.0)
        assert box == pytest.approx(98.0)
        # (w + h)^2 / (2 w h) for 10 by 4 is 196 / 80 = 2.45, not the guessed 2
        assert box / area == pytest.approx(2.45)
        assert sorted([round(w, 6), round(h, 6)]) == [4.0, 10.0]

    def test_a_square_at_forty_five_gains_exactly_two(self):
        square = _rotate([(0, 0), (6, 0), (6, 6), (0, 6)], 45)
        area, _, _, _ = minimum_rectangle(square)
        assert axis_aligned_area(square) / area == pytest.approx(2.0)

    def test_a_thin_diagonal_strip_gains_enormously(self):
        rng = random.Random(97)
        strip = [(i, i + rng.uniform(-0.05, 0.05)) for i in range(0, 100, 5)]
        strip += [(i, i + 0.5) for i in range(0, 100, 5)]
        area, _, _, _ = minimum_rectangle(strip)
        assert axis_aligned_area(strip) / area > 50  # measured about 174


class TestNeverLooser:
    def test_the_minimum_rectangle_never_exceeds_the_box(self):
        rng = random.Random(97)
        ratios = []
        for _ in range(3000):
            n = rng.randint(3, 30)
            pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
            area, _, _, _ = minimum_rectangle(pts)
            box = axis_aligned_area(pts)
            assert area <= box + 1e-9
            ratios.append(area / box)
        assert sum(ratios) / len(ratios) < 0.95  # measured mean 0.905


class TestRefusals:
    def test_fewer_than_three_distinct_points_is_refused(self):
        with pytest.raises(Invalid):
            minimum_rectangle([(0, 0), (1, 1), (1, 1)])

    def test_collinear_points_are_degenerate(self):
        with pytest.raises(Degenerate):
            minimum_rectangle([(0, 0), (1, 1), (2, 2), (3, 3)])

    def test_an_empty_box_is_refused(self):
        with pytest.raises(Invalid):
            axis_aligned_area([])
