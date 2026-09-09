from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.spatialweights import (
    distance_band,
    grid_points,
    inverse_distance,
    k_nearest,
    morans_i,
    neighbour_counts,
    row_standardize,
    smooth_field,
)

POINTS = grid_points(20)
FIELD = smooth_field(POINTS)


class TestTheBand:
    @pytest.mark.parametrize(
        ("radius", "binary", "standardized", "low", "high"),
        [
            (1.0, 0.9748, 0.9885, 2, 4),
            (2.0, 0.9497, 0.9732, 5, 12),
            (4.0, 0.8614, 0.909, 16, 48),
            (8.0, 0.5925, 0.6634, 57, 196),
        ],
    )
    def test_i_falls_with_the_radius_and_standardization_lifts_it_by_the_edge_effect(
        self, radius, binary, standardized, low, high
    ):
        weights = distance_band(POINTS, radius)
        counts = neighbour_counts(weights)
        assert (min(counts), max(counts)) == (low, high)
        assert morans_i(FIELD, weights) == pytest.approx(binary, abs=1e-3)
        lifted = morans_i(FIELD, row_standardize(weights))
        assert lifted == pytest.approx(standardized, abs=1e-3)


class TestOtherSchemes:
    def test_k_nearest_reads_the_same_binary_and_standardized(self):
        for k, reading in ((4, 0.987078), (8, 0.981273)):
            weights = k_nearest(POINTS, k)
            assert set(neighbour_counts(weights)) == {k}
            binary = morans_i(FIELD, weights)
            assert binary == pytest.approx(reading, abs=1e-5)
            assert morans_i(FIELD, row_standardize(weights)) == pytest.approx(binary, abs=1e-9)

    def test_inverse_distance_and_the_shuffled_floor(self):
        weights = inverse_distance(POINTS, 4.0)
        assert morans_i(FIELD, weights) == pytest.approx(0.8909, abs=1e-3)
        assert morans_i(FIELD, row_standardize(weights)) == pytest.approx(0.9313, abs=1e-3)
        shuffled = list(FIELD)
        random.Random(267).shuffle(shuffled)
        schemes = (distance_band(POINTS, 1.0), distance_band(POINTS, 4.0), k_nearest(POINTS, 8))
        for scheme in schemes:
            assert abs(morans_i(shuffled, row_standardize(scheme))) < 0.02


class TestRefusals:
    def test_bad_radii_counts_and_shapes_are_refused(self):
        with pytest.raises(Invalid):
            distance_band(POINTS, 0)
        with pytest.raises(Invalid):
            k_nearest(POINTS, 0)
        with pytest.raises(Invalid):
            morans_i([1.0, 1.0], distance_band([(0, 0), (1, 0)], 2.0))
        with pytest.raises(Invalid):
            morans_i(FIELD[:5], distance_band(POINTS, 1.0))
        with pytest.raises(Invalid):
            morans_i([1.0, 2.0], [{}, {}])
