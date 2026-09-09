from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.lidarheights import (
    TreeIndex,
    canopy_height,
    chm,
    dsm,
    dtm,
    empty_fraction,
    expected_shortfall,
    matched_tops,
    plant_trees,
    scan,
    terrain_bias,
    top_shortfall,
    tree_tops,
)

SIZE = 200.0


@pytest.fixture(scope="module")
def trees():
    return plant_trees(40, SIZE, random.Random(310))


@pytest.fixture(scope="module")
def dense_scan(trees):
    return scan(trees, SIZE, 4.0, 0.3, random.Random(313))


class TestBareGround:
    @pytest.mark.parametrize(
        ("cell", "bias"), [(1.0, -0.0536), (2.0, -0.175), (5.0, -0.4901), (10.0, -0.995)]
    )
    def test_the_terrain_sits_under_half_a_cells_drop_low(self, cell, bias):
        returns = scan([], SIZE, 4.0, 1.0, random.Random(312), slope=0.2)
        low = terrain_bias(dtm(returns, SIZE, cell), cell, 0.2, [])[1]
        high = terrain_bias(dsm(returns, SIZE, cell), cell, 0.2, [])[1]
        assert low == pytest.approx(bias, abs=1e-4)
        assert high == pytest.approx(-bias, abs=5e-4)
        assert abs(bias) < 0.1 * cell

    def test_noise_pulls_the_terrain_down_by_the_minimum_of_sixteen(self, trees):
        returns = scan(trees, SIZE, 4.0, 1.0, random.Random(314), noise=0.3)
        clear = terrain_bias(dtm(returns, SIZE, 2.0), 2.0, 0.0, trees)[1]
        assert clear == pytest.approx(-0.5237, abs=1e-4)
        assert pytest.approx(-0.5298, abs=1e-4) == -0.3 * 1.766


class TestUnderTheCanopy:
    @pytest.mark.parametrize(
        ("penetration", "bias", "shortfall"),
        [(0.5, 0.0, 1.653), (0.1, 0.903, 1.922), (0.0, 3.497, 10.022)],
    )
    def test_the_terrain_rises_as_the_ground_stops_seeing_pulses(
        self, trees, penetration, bias, shortfall
    ):
        returns = scan(trees, SIZE, 4.0, penetration, random.Random(311))
        terrain = dtm(returns, SIZE, 2.0)
        under, clear = terrain_bias(terrain, 2.0, 0.0, trees)
        assert under == pytest.approx(bias, abs=1e-3)
        assert clear == 0.0
        heights = chm(dsm(returns, SIZE, 2.0), terrain)
        assert top_shortfall(heights, 2.0, trees) == pytest.approx(shortfall, abs=1e-3)

    def test_a_one_meter_cell_with_four_pulses_reads_high(self, dense_scan, trees):
        fine = terrain_bias(dtm(dense_scan, SIZE, 1.0), 1.0, 0.0, trees)[0]
        coarse = terrain_bias(dtm(dense_scan, SIZE, 2.0), 2.0, 0.0, trees)[0]
        assert fine == pytest.approx(1.86, abs=0.01)
        assert coarse == pytest.approx(0.009, abs=1e-3)


class TestTops:
    @pytest.mark.parametrize(
        ("density", "empty", "shortfall", "guess", "hit"),
        [(0.25, 0.3704, 10.152, 4.752, 33), (1.0, 0.0202, 3.634, 2.376, 40)],
    )
    def test_shortfall_and_hits_by_density(self, trees, density, empty, shortfall, guess, hit):
        returns = scan(trees, SIZE, density, 0.3, random.Random(313))
        terrain = dtm(returns, SIZE, 2.0)
        assert empty_fraction(terrain) == pytest.approx(empty, abs=1e-4)
        heights = chm(dsm(returns, SIZE, 2.0), terrain)
        assert top_shortfall(heights, 2.0, trees) == pytest.approx(shortfall, abs=1e-3)
        assert expected_shortfall(trees, density) == pytest.approx(guess, abs=1e-3)
        assert matched_tops(tree_tops(heights, 5.0, 2), 2.0, trees) == (hit, 0)

    def test_the_cell_size_trades_doubles_for_misses(self, dense_scan, trees):
        expected = {
            (1.0, 1): (91, 40, 0),
            (1.0, 2): (40, 40, 0),
            (4.0, 2): (35, 35, 0),
            (8.0, 1): (32, 28, 4),
            (8.0, 3): (14, 14, 0),
        }
        for (cell, radius), (found, hit, stray) in expected.items():
            heights = chm(dsm(dense_scan, SIZE, cell), dtm(dense_scan, SIZE, cell))
            tops = tree_tops(heights, 5.0, radius)
            assert (len(tops), *matched_tops(tops, cell, trees)) == (found, hit, stray)
        heights = chm(dsm(dense_scan, SIZE, 2.0), dtm(dense_scan, SIZE, 2.0))
        assert top_shortfall(heights, 2.0, trees) == pytest.approx(1.36, abs=1e-3)


class TestIndexAndRefusals:
    def test_the_index_agrees_with_the_brute_force_height(self, trees):
        index = TreeIndex(trees)
        rng = random.Random(315)
        for _ in range(500):
            x, y = rng.uniform(0, SIZE), rng.uniform(0, SIZE)
            assert index.height(x, y) == canopy_height(trees, x, y)

    def test_bad_inputs(self, trees):
        with pytest.raises(Invalid):
            scan(trees, SIZE, 0, 0.5, random.Random(1))
        with pytest.raises(Invalid):
            scan(trees, SIZE, 1, 1.5, random.Random(1))
        with pytest.raises(Invalid):
            dtm([(0, 0, 0)], SIZE, 0)
        with pytest.raises(Invalid):
            tree_tops([[1.0]], 0.0, 0)
        with pytest.raises(Invalid):
            TreeIndex(trees, 0)
