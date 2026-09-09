from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.thinning import (
    distance_thin,
    grid_thin,
    hexagonal_bound,
    min_pair_distance,
    nearest_neighbour_index,
    occupied_cells,
)


def _uniform():
    rng = random.Random(241)
    return [(rng.random(), rng.random()) for _ in range(5000)]


class TestUniformPoints:
    @pytest.mark.parametrize(
        ("spacing", "grid_kept", "distance_kept", "fraction"),
        [(0.02, 2147, 1185, 0.41), (0.05, 400, 255, 0.552), (0.1, 100, 68, 0.589)],
    )
    def test_kept_counts_and_the_fraction_of_the_hexagonal_bound(
        self, spacing, grid_kept, distance_kept, fraction
    ):
        points = _uniform()
        grid = grid_thin(points, spacing)
        assert len(grid) == grid_kept == occupied_cells(points, spacing)
        kept = distance_thin(points, spacing)
        assert len(kept) == distance_kept
        assert len(kept) / hexagonal_bound(spacing) == pytest.approx(fraction, abs=2e-3)
        assert min_pair_distance(kept) >= spacing

    def test_the_grid_set_is_already_regular_and_the_distance_set_more_so(self):
        points = _uniform()
        assert nearest_neighbour_index(points[:1000], 1.0) == pytest.approx(1.0, abs=0.05)
        readings = {}
        for spacing in (0.05, 0.1):
            readings[spacing] = (
                nearest_neighbour_index(grid_thin(points, spacing), 1.0),
                nearest_neighbour_index(grid_thin(points, spacing, nearest_center=True), 1.0),
                nearest_neighbour_index(distance_thin(points, spacing), 1.0),
            )
        assert readings[0.05] == pytest.approx((1.296, 1.71, 1.746), abs=0.01)
        assert readings[0.1] == pytest.approx((1.318, 1.857, 1.782), abs=0.01)
        for first, center, distance in readings.values():
            assert 1.2 < first < center
            assert distance > 1.5

    def test_the_walk_order_moves_the_count_by_a_few_percent(self):
        points = _uniform()
        kept = []
        for seed in range(5):
            order = list(points)
            random.Random(seed).shuffle(order)
            kept.append(len(distance_thin(order, 0.05)))
        assert kept == [249, 258, 245, 257, 255]
        assert (max(kept) - min(kept)) / min(kept) < 0.06


class TestClusters:
    def test_the_grid_keeps_more_per_cluster_than_the_distance_method(self):
        rng = random.Random(241)
        for _ in range(5000):
            rng.random()
            rng.random()
        clusters = []
        for _ in range(30):
            cx, cy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
            for _ in range(100):
                clusters.append((cx + rng.gauss(0, 0.01), cy + rng.gauss(0, 0.01)))
        for spacing, grid_kept, distance_kept in ((0.05, 85, 29), (0.02, 247, 122)):
            assert len(grid_thin(clusters, spacing)) == grid_kept
            assert len(distance_thin(clusters, spacing)) == distance_kept


class TestRefusals:
    def test_bad_sizes_and_counts_are_refused(self):
        with pytest.raises(Invalid):
            grid_thin([(0, 0)], 0)
        with pytest.raises(Invalid):
            distance_thin([(0, 0)], 0)
        with pytest.raises(Invalid):
            hexagonal_bound(0)
        with pytest.raises(Invalid):
            nearest_neighbour_index([(0, 0)], 1.0)
        with pytest.raises(Invalid):
            min_pair_distance([(0, 0)])
        with pytest.raises(Invalid):
            occupied_cells([(0, 0)], 0)
