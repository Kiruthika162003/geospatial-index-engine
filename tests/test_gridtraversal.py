from __future__ import annotations

import random

import pytest

from atlas.bresenham import line
from atlas.errors import Invalid
from atlas.gridtraversal import grid_lines_crossed, is_four_connected, traverse


class TestKnownWalks:
    def test_a_diagonal_visits_every_corner_cell(self):
        # Bresenham gives 4 cells on this diagonal; the traversal must not cut the corners
        assert traverse(0.5, 0.5, 3.5, 3.5) == [
            (0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3),
        ]

    def test_a_segment_inside_one_cell(self):
        assert traverse(2.2, 2.2, 2.8, 2.7) == [(2, 2)]

    def test_an_axis_aligned_walk(self):
        assert traverse(0.5, 0.5, 3.5, 0.5) == [(0, 0), (1, 0), (2, 0), (3, 0)]


class TestInvariants:
    def test_count_is_one_plus_grid_lines_crossed_and_four_connected(self):
        rng = random.Random(85)
        for _ in range(10000):
            x0, y0 = rng.uniform(-30, 30), rng.uniform(-30, 30)
            x1, y1 = rng.uniform(-30, 30), rng.uniform(-30, 30)
            cells = traverse(x0, y0, x1, y1)
            assert len(cells) == grid_lines_crossed(x0, y0, x1, y1) + 1
            assert is_four_connected(cells)

    def test_it_is_a_superset_of_bresenham_with_more_cells(self):
        rng = random.Random(86)
        ratios = []
        for _ in range(5000):
            x0, y0, x1, y1 = (rng.randint(-30, 30) for _ in range(4))
            walked = set(traverse(x0 + 0.5, y0 + 0.5, x1 + 0.5, y1 + 0.5))
            thin = set(line(x0, y0, x1, y1))
            assert thin <= walked
            ratios.append(len(walked) / len(thin))
        # measured mean 1.43, max 1.98: a pure diagonal nearly doubles the cells
        assert 1.3 < sum(ratios) / len(ratios) < 1.6
        assert max(ratios) < 2.0


class TestRefusals:
    def test_non_finite_endpoints_are_refused(self):
        with pytest.raises(Invalid):
            traverse(0, 0, float("inf"), 1)
