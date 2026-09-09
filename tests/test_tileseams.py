from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.tileseams import (
    box_filter,
    cost_ratio,
    halo_needed,
    noise_grid,
    repeated,
    seam_cells,
    seam_fraction_law,
    tiled,
    worst_gap,
)


@pytest.fixture(scope="module")
def grid():
    return noise_grid(96, random.Random(560))


class TestSinglePass:
    @pytest.mark.parametrize(
        ("radius", "tile", "fractions", "gap"),
        [
            (1, 16, (0.1975, 0.0), 0.2758),
            (1, 32, (0.0816, 0.0), 0.2143),
            (3, 16, (0.5273, 0.3733, 0.1975, 0.0), 0.1594),
            (3, 32, (0.2344, 0.1597, 0.0816, 0.0), 0.1377),
        ],
    )
    def test_the_spoiled_band_is_the_reach_less_the_halo(
        self, grid, radius, tile, fractions, gap
    ):
        whole = box_filter(grid, radius)
        for halo, fraction in enumerate(fractions):
            pieced = tiled(grid, tile, halo, radius)
            assert seam_cells(whole, pieced) / 96**2 == pytest.approx(fraction, abs=1e-4)
            assert seam_fraction_law(96, tile, radius, halo) == pytest.approx(
                fraction, abs=1e-4
            )
            if halo == 0:
                assert worst_gap(whole, pieced) == pytest.approx(gap, abs=1e-4)
        assert worst_gap(whole, tiled(grid, tile, radius, radius)) == 0.0


class TestRepeatedPasses:
    @pytest.mark.parametrize(
        ("radius", "passes", "tile", "fractions"),
        [
            (1, 2, 16, (0.3733, 0.1975, 0.0)),
            (2, 3, 16, (0.8594, 0.7704, 0.6597, 0.5273, 0.3733, 0.1975, 0.0)),
            (2, 3, 32, (0.4375, 0.3733, 0.3056, 0.2344, 0.1597, 0.0816, 0.0)),
        ],
    )
    def test_each_pass_reaches_another_radius(self, grid, radius, passes, tile, fractions):
        whole = repeated(grid, radius, passes)
        assert halo_needed(radius, passes) == radius * passes
        for halo, fraction in enumerate(fractions):
            pieced = tiled(grid, tile, halo, radius, passes)
            assert seam_cells(whole, pieced) / 96**2 == pytest.approx(fraction, abs=1e-4)
            assert seam_fraction_law(96, tile, radius, halo, passes) == pytest.approx(
                fraction, abs=1e-4
            )
            assert seam_fraction_law(96, tile, radius, halo) <= fraction + 1e-9

    def test_the_halos_price(self):
        assert cost_ratio(96, 16, 3) == pytest.approx(1.8906, abs=1e-4)
        assert cost_ratio(96, 32, 3) == pytest.approx(1.4102, abs=1e-4)
        assert cost_ratio(1000, 256, 3) == pytest.approx(1.0474, abs=1e-4)


class TestRefusals:
    def test_bad_grids_radii_tiles_and_passes(self):
        with pytest.raises(Invalid):
            box_filter([], 1)
        with pytest.raises(Invalid):
            box_filter([[1.0]], -1)
        with pytest.raises(Invalid):
            repeated([[1.0]], 1, 0)
        with pytest.raises(Invalid):
            tiled([[1.0]], 0, 0, 1)
        with pytest.raises(Invalid):
            tiled([[1.0]], 1, -1, 1)
