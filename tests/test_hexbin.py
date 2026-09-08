from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid
from atlas.hexbin import (
    coefficient_of_variation,
    hex_area,
    hex_bins,
    neighbor_distances,
    square_bins,
    square_side_for_hex,
)
from atlas.hexgrid import from_point, to_point

SIZE = 3.0


def _points(seed=141, count=20000):
    rng = random.Random(seed)
    return [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(count)]


class TestConservation:
    def test_hex_and_square_bins_both_sum_to_the_point_count(self):
        pts = _points()
        assert sum(hex_bins(pts, SIZE).values()) == len(pts)
        assert sum(square_bins(pts, square_side_for_hex(SIZE)).values()) == len(pts)

    def test_the_comparison_square_has_the_hexagons_area(self):
        side = square_side_for_hex(SIZE)
        assert side * side == pytest.approx(hex_area(SIZE))


class TestGeometry:
    def test_hex_neighbors_sit_at_one_distance_where_squares_have_two(self):
        distances = neighbor_distances(from_point(50, 50, SIZE), SIZE)
        assert len(distances) == 6
        assert max(distances) - min(distances) < 1e-9
        assert distances[0] == pytest.approx(SIZE * math.sqrt(3))
        side = square_side_for_hex(SIZE)
        assert side * math.sqrt(2) > side  # the diagonal neighbor is farther than the edge one

    def test_the_hexagon_has_a_shorter_boundary_at_equal_area(self):
        side = square_side_for_hex(SIZE)
        assert 1 - (6 * SIZE) / (4 * side) == pytest.approx(0.069, abs=0.002)


class TestNoiseIsNearlyEqual:
    def test_the_hex_noise_advantage_is_marginal_on_random_points(self):
        # the refuted guess expected a clear reduction; measured 0.1446 vs 0.1466
        pts = _points()
        side = square_side_for_hex(SIZE)
        hex_interior = {
            k: v for k, v in hex_bins(pts, SIZE).items()
            if 10 < to_point(k, SIZE)[0] < 90 and 10 < to_point(k, SIZE)[1] < 90
        }
        sq_interior = {
            k: v for k, v in square_bins(pts, side).items()
            if 10 < (k[0] + 0.5) * side < 90 and 10 < (k[1] + 0.5) * side < 90
        }
        cv_hex = coefficient_of_variation(hex_interior)
        cv_sq = coefficient_of_variation(sq_interior)
        assert cv_hex == pytest.approx(0.1446, abs=0.005)
        assert cv_sq == pytest.approx(0.1466, abs=0.005)
        assert 0.95 < cv_hex / cv_sq < 1.0


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            hex_bins(None, 1.0)

    def test_a_non_positive_size_is_refused(self):
        with pytest.raises(Invalid):
            hex_bins([(0, 0)], 0)
        with pytest.raises(Invalid):
            square_bins([(0, 0)], 0)

    def test_an_empty_count_map_has_zero_variation(self):
        assert coefficient_of_variation({}) == 0.0
