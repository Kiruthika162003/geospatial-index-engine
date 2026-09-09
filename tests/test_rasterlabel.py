from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.rasterlabel import (
    checkerboard,
    diagonal,
    flood_label,
    label,
    patch_count,
    patch_sizes,
    random_grid,
    spans,
)


class TestKnownShapes:
    def test_a_diagonal_is_one_patch_under_eight_and_ten_under_four(self):
        assert patch_count(diagonal(10)) == 10
        assert patch_count(diagonal(10), eight=True) == 1

    def test_a_checkerboard_is_one_patch_under_eight_and_fifty_under_four(self):
        assert patch_count(checkerboard(10)) == 50
        assert patch_count(checkerboard(10), eight=True) == 1
        assert patch_sizes(checkerboard(10), eight=True) == [50]


class TestAgainstFloodFill:
    def test_the_two_pass_labelling_agrees_with_flood_fill_on_every_cell(self):
        rng = random.Random(221)
        for _ in range(100):
            grid = random_grid(rng.randint(1, 15), rng.randint(1, 15), rng.random(), rng)
            for eight in (False, True):
                assert label(grid, eight) == flood_label(grid, eight)


class TestRandomGrids:
    def test_the_count_ratio_and_the_percolation_gap(self):
        rng = random.Random(221)
        for _ in range(100):
            random_grid(rng.randint(1, 15), rng.randint(1, 15), rng.random(), rng)
        readings = {}
        for density in (0.2, 0.33, 0.4, 0.5, 0.6, 0.7):
            fours = eights = spans_four = spans_eight = 0
            for _ in range(20):
                grid = random_grid(40, 40, density, rng)
                fours += patch_count(grid)
                eights += patch_count(grid, True)
                spans_four += spans(grid)
                spans_eight += spans(grid, True)
            readings[density] = (fours / eights, spans_four, spans_eight)
        assert readings[0.2][0] == pytest.approx(1.63, abs=0.02)
        assert readings[0.5][0] == pytest.approx(11.01, abs=0.05)
        assert readings[0.6][0] == pytest.approx(18.05, abs=0.05)
        assert readings[0.4][1:] == (0, 10)
        assert readings[0.5][1:] == (0, 19)
        assert readings[0.6][1:] == (13, 20)
        assert readings[0.7][1:] == (20, 20)

    def test_the_largest_patch_at_half_density(self):
        grid = random_grid(40, 40, 0.5, random.Random(222))
        assert patch_sizes(grid)[0] == 175
        assert patch_sizes(grid, True)[0] == 782


class TestRefusals:
    def test_an_empty_grid_is_refused(self):
        with pytest.raises(Invalid):
            label([])
        with pytest.raises(Invalid):
            flood_label([[]])
