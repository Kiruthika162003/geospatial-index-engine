from __future__ import annotations

import math
import random

import pytest

from atlas.buffer import offset, perimeter, polygon_area, steiner_area
from atlas.errors import Invalid

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


class TestSteiner:
    def test_the_formula_on_a_square(self):
        assert steiner_area(SQUARE, 2.0) == pytest.approx(100 + 40 * 2 + math.pi * 4)
        assert perimeter(SQUARE) == pytest.approx(40.0)
        assert polygon_area(SQUARE) == pytest.approx(100.0)

    def test_buffering_by_zero_changes_nothing(self):
        assert polygon_area(offset(SQUARE, 0.0, 4)) == pytest.approx(100.0)
        assert steiner_area(SQUARE, 0.0) == pytest.approx(100.0)


class TestTheFanApproachesFromBelow:
    def test_shortfall_shrinks_with_arc_resolution(self):
        exact = steiner_area(SQUARE, 2.0)
        shortfalls = [
            (exact - polygon_area(offset(SQUARE, 2.0, steps))) / exact
            for steps in (1, 2, 4, 8, 32, 128)
        ]
        assert shortfalls[0] == pytest.approx(0.02371, abs=1e-4)
        assert shortfalls[-1] < 1e-5
        assert shortfalls == sorted(shortfalls, reverse=True)
        assert all(s >= -1e-12 for s in shortfalls)

    def test_the_fan_never_exceeds_steiner_on_random_convex_polygons(self):
        rng = random.Random(135)
        for _ in range(300):
            k = rng.randint(3, 12)
            step = 2 * math.pi / k
            angles = [i * step + rng.uniform(-0.3 * step, 0.3 * step) for i in range(k)]
            r = rng.uniform(5, 15)
            poly = [(r * math.cos(a), r * math.sin(a)) for a in angles]
            d = rng.uniform(0.5, 5)
            exact = steiner_area(poly, d)
            assert polygon_area(offset(poly, d, 8)) <= exact + 1e-9
            assert polygon_area(offset(poly, d, 64)) == pytest.approx(exact, rel=1e-3)


class TestRefusals:
    def test_a_concave_or_clockwise_polygon_is_refused(self):
        with pytest.raises(Invalid):
            offset([(0, 0), (0, 10), (10, 10), (10, 0)], 1.0)  # clockwise
        with pytest.raises(Invalid):
            steiner_area([(0, 0), (10, 0), (2, 2), (0, 10)], 1.0)  # dented

    def test_a_negative_distance_is_refused(self):
        with pytest.raises(Invalid):
            offset(SQUARE, -1.0)

    def test_zero_arc_steps_is_refused(self):
        with pytest.raises(Invalid):
            offset(SQUARE, 1.0, arc_steps=0)
