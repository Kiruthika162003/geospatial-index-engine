from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.regionquadtree import RegionQuadtree, disc, random_grid, rectangles


class TestTheDisc:
    def test_leaves_follow_the_boundary_and_double_per_doubling_of_side(self):
        leaves = {}
        for side in (32, 64, 128, 256):
            tree = RegionQuadtree(disc(side))
            leaves[side] = tree.leaves()
            assert tree.leaves() == 3 * tree.internal() + 1
            assert tree.decode() == disc(side)
        assert leaves == {32: 244, 64: 532, 128: 1084, 256: 2020}
        ratios = [leaves[2 * s] / leaves[s] for s in (32, 64, 128)]
        assert ratios == pytest.approx([2.18, 2.038, 1.863], abs=1e-3)
        assert all(1.5 < r < 2.5 for r in ratios)


class TestExtremes:
    def test_noise_cannot_be_compressed_and_a_blank_is_one_node(self):
        grid = random_grid(64, random.Random(232))
        tree = RegionQuadtree(grid)
        assert tree.leaves() == 3733
        assert tree.decode() == grid
        blank = [[False] * 64 for _ in range(64)]
        assert RegionQuadtree(blank).leaves() == 1
        assert RegionQuadtree(blank).depth() == 0

    def test_a_checkerboard_is_the_worst_case(self):
        checker = [[(r + c) % 2 == 0 for c in range(64)] for r in range(64)]
        tree = RegionQuadtree(checker)
        assert tree.leaves() == 4096
        assert tree.internal() == 1365
        assert tree.depth() == 6


class TestAlignment:
    def test_a_one_cell_shift_costs_a_factor_of_59(self):
        aligned = RegionQuadtree(rectangles(128, 0))
        shifted = RegionQuadtree(rectangles(128, 1))
        assert (aligned.leaves(), aligned.depth()) == (25, 3)
        assert (shifted.leaves(), shifted.depth()) == (1471, 7)
        assert shifted.leaves() / aligned.leaves() > 55
        assert RegionQuadtree(rectangles(128, 3)).leaves() == 1459
        assert shifted.decode() == rectangles(128, 1)


class TestRefusals:
    def test_non_square_and_empty_grids_are_refused(self):
        with pytest.raises(Invalid):
            RegionQuadtree([[True, False, True]])
        with pytest.raises(Invalid):
            RegionQuadtree([[True] * 3 for _ in range(3)])
        with pytest.raises(Invalid):
            RegionQuadtree([])
