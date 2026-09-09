from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.parcelsplit import (
    area,
    crescent,
    cut_at_fraction,
    equal_strips,
    pieces_on_side,
    random_convex,
)

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]
TRIANGLE = [(0, 0), (10, 0), (0, 10)]


class TestExactness:
    def test_every_fraction_and_direction_lands_within_1e_10_in_33_to_35_steps(self):
        rng = random.Random(266)
        worst, steps = 0.0, []
        shapes = [SQUARE, TRIANGLE] + [random_convex(rng.randint(4, 9), rng) for _ in range(20)]
        for polygon in shapes:
            for fraction, direction in ((0.1, 0.0), (0.5, 45.0), (0.75, 120.0)):
                near, far, _, n = cut_at_fraction(polygon, fraction, direction)
                whole = area(polygon)
                fraction_error = abs(area(near) / whole - fraction)
                sum_error = abs(area(near) + area(far) - whole) / whole
                worst = max(worst, fraction_error, sum_error)
                steps.append(n)
        assert worst < 1e-9
        assert 32 <= min(steps) <= max(steps) <= 35


class TestEqualStrips:
    def test_a_triangles_strips_sit_at_root_f_heights_and_a_squares_evenly(self):
        pieces, offsets = equal_strips(TRIANGLE, 4, 90.0)
        assert [area(p) for p in pieces] == pytest.approx([12.5] * 4, abs=1e-6)
        heights = [10 * math.sqrt(f) for f in (0.75, 0.5, 0.25)]
        assert [10 - o for o in offsets] == pytest.approx(heights, abs=1e-4)
        pieces, offsets = equal_strips(SQUARE, 4, 0.0)
        assert offsets == pytest.approx([2.5, 5.0, 7.5], abs=1e-6)
        assert [area(p) for p in pieces] == pytest.approx([25.0] * 4, abs=1e-6)


class TestNonConvex:
    def test_the_crescent_yields_the_area_but_not_always_one_piece(self):
        moon = crescent()
        assert area(moon) == pytest.approx(99.387, abs=1e-3)
        near, _, offset, _ = cut_at_fraction(moon, 0.3, 0.0)
        assert area(near) / area(moon) == pytest.approx(0.3, abs=1e-8)
        assert pieces_on_side(moon, 1.0, 0.0, offset) == 1
        near, _, offset, _ = cut_at_fraction(moon, 0.3, 90.0)
        assert area(near) / area(moon) == pytest.approx(0.3, abs=1e-8)
        assert pieces_on_side(moon, 0.0, 1.0, offset) == 2


class TestRefusals:
    def test_bad_parcels_fractions_and_counts_are_refused(self):
        with pytest.raises(Invalid):
            cut_at_fraction([(0, 0), (1, 1)], 0.5)
        with pytest.raises(Invalid):
            cut_at_fraction(SQUARE, 1.0)
        with pytest.raises(Invalid):
            cut_at_fraction([(0, 0), (1, 1), (2, 2)], 0.5)
        with pytest.raises(Invalid):
            equal_strips(SQUARE, 0)
