from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.rasterize import (
    center_fill,
    filled_area,
    inside_fill,
    perimeter,
    touch_fill,
    true_area,
)

SQUARE = [[(1.25, 1.25), (5.25, 1.25), (5.25, 5.25), (1.25, 5.25)]]
TRIANGLE = [[(0.3, 0.2), (9.1, 1.4), (4.7, 8.6)]]


class TestTheSquare:
    def test_the_center_rule_is_exact_and_the_other_two_bracket_it_by_the_perimeter(self):
        assert true_area(SQUARE) == 16.0
        assert filled_area(center_fill(SQUARE, 8, 8, 1.0), 1.0) == 16.0
        touch = filled_area(touch_fill(SQUARE, 8, 8, 1.0), 1.0)
        inside = filled_area(inside_fill(SQUARE, 8, 8, 1.0), 1.0)
        assert (touch, inside) == (25.0, 9.0)
        assert (touch - 16.0) + (16.0 - inside) == perimeter(SQUARE[0]) * 1.0


class TestTheTriangle:
    def test_center_errors_shrink_and_change_sign_while_biases_halve_with_the_cell(self):
        truth = true_area(TRIANGLE)
        assert truth == pytest.approx(34.32)
        readings = {}
        for cells in (10, 20, 40, 80, 160):
            size = 10.0 / cells
            center = filled_area(center_fill(TRIANGLE, cells, cells, size), size)
            touch = filled_area(touch_fill(TRIANGLE, cells, cells, size), size)
            inside = filled_area(inside_fill(TRIANGLE, cells, cells, size), size)
            readings[cells] = (center - truth, touch - truth, inside - truth, size)
        assert readings[10][0] == pytest.approx(-1.32, abs=1e-2)
        assert readings[80][0] == pytest.approx(0.039, abs=1e-3)
        assert readings[160][0] == pytest.approx(-0.004, abs=1e-3)
        assert readings[40][1] == pytest.approx(4.368, abs=1e-3)
        assert readings[40][2] == pytest.approx(-4.32, abs=1e-3)
        for cells in (20, 40, 80, 160):
            _, touch_bias, inside_bias, size = readings[cells]
            band = perimeter(TRIANGLE[0]) * size
            assert 0.55 < touch_bias / band < 0.75
            assert 0.55 < -inside_bias / band < 0.75
            assert 1.2 < (touch_bias - inside_bias) / band < 1.3


class TestRandomPolygons:
    def test_the_center_rule_errs_positive_as_often_as_negative(self):
        rng = random.Random(220)
        positive = negative = 0
        magnitudes = []
        for _ in range(100):
            k = rng.randint(3, 8)
            angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(k))
            ring = [
                (5 + rng.uniform(2, 4.5) * math.cos(a), 5 + rng.uniform(2, 4.5) * math.sin(a))
                for a in angles
            ]
            error = filled_area(center_fill([ring], 40, 40, 0.25), 0.25) - true_area([ring])
            positive += error > 0
            negative += error < 0
            magnitudes.append(abs(error) / true_area([ring]))
        assert (positive, negative) == (50, 50)
        assert sum(magnitudes) / 100 == pytest.approx(0.0144, abs=1e-3)


class TestHoles:
    def test_the_parity_rule_subtracts_the_inner_ring(self):
        outer = [(1, 1), (9, 1), (9, 9), (1, 9)]
        inner = [(3, 3), (7, 3), (7, 7), (3, 7)]
        assert filled_area(center_fill([outer, inner], 10, 10, 1.0), 1.0) == 48.0
        assert true_area([outer, inner]) == 48.0


class TestRefusals:
    def test_bad_rings_and_grids_are_refused(self):
        with pytest.raises(Invalid):
            center_fill([[(0, 0), (1, 1)]], 4, 4, 1.0)
        with pytest.raises(Invalid):
            center_fill(SQUARE, 0, 4, 1.0)
        with pytest.raises(Invalid):
            center_fill([], 4, 4, 1.0)
