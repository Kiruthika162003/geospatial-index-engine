from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.nearestneighbourindex import (
    HEXAGONAL_CEILING,
    expected_random_distance,
    families,
    hexagonal_lattice,
    index,
    lattice_area,
    nearest_distances,
    square_grid,
)


class TestLattices:
    def test_a_square_grid_reads_two_with_its_own_area(self):
        grid = square_grid(20)
        assert index(grid, lattice_area(grid, 1.0, False)) == pytest.approx(2.0)
        assert index(grid, 19.0 * 19.0) == pytest.approx(2.1053, abs=1e-4)

    def test_a_hexagonal_lattice_reads_the_ceiling(self):
        hexes = hexagonal_lattice(20)
        assert min(nearest_distances(hexes)) == pytest.approx(1.0)
        assert max(nearest_distances(hexes)) == pytest.approx(1.0)
        assert index(hexes, lattice_area(hexes, 1.0, True)) == pytest.approx(HEXAGONAL_CEILING)
        assert pytest.approx(2.1491, abs=1e-4) == HEXAGONAL_CEILING


class TestRandomScatters:
    @pytest.mark.parametrize(
        ("n", "bias", "spread"), [(100, 0.0314, 0.1713), (400, 0.0316, 0.1312)]
    )
    def test_the_edge_bias_did_not_fall_as_one_over_root_n(self, n, bias, spread):
        values = []
        for seed in range(10):
            rng = random.Random(198 + seed + n)
            values.append(index([(rng.random(), rng.random()) for _ in range(n)], 1.0))
        assert sum(values) / 10 - 1 == pytest.approx(bias, abs=1e-3)
        assert max(values) - min(values) == pytest.approx(spread, abs=1e-3)

    def test_the_index_is_scale_free(self):
        rng = random.Random(199)
        for _ in range(3):
            rng.random()
        pts = [(rng.random(), rng.random()) for _ in range(100)]
        doubled = [(2 * x, 2 * y) for x, y in pts]
        assert index(doubled, 4.0) == pytest.approx(index(pts, 1.0))


class TestFamilies:
    def test_tighter_families_read_lower(self):
        rng = random.Random(199)
        spreads = (0.001, 0.005, 0.02)
        readings = [index(families(50, 10, spread, rng), 1.0) for spread in spreads]
        assert readings == pytest.approx([0.0318, 0.1489, 0.5717], abs=1e-3)
        assert readings == sorted(readings)


class TestExpectation:
    def test_the_random_expectation_is_half_over_root_density(self):
        assert expected_random_distance(100, 1.0) == 0.05
        assert expected_random_distance(100, 4.0) == pytest.approx(0.1)
        assert expected_random_distance(1, math.pi) == pytest.approx(0.5 * math.sqrt(math.pi))

    def test_bad_inputs_are_refused(self):
        with pytest.raises(Invalid):
            index([(0, 0)], 1.0)
        with pytest.raises(Invalid):
            expected_random_distance(0, 1)
        with pytest.raises(Invalid):
            expected_random_distance(5, 0)
