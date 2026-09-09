from __future__ import annotations

import math
import random

import pytest

from atlas.errors import Invalid, Outside
from atlas.geodesicgrid import GeodesicGrid, expected_counts, spread


class TestCounts:
    @pytest.mark.parametrize("frequency", [1, 2, 4, 8])
    def test_vertices_triangles_and_twelve_pentagons(self, frequency):
        grid = GeodesicGrid(frequency)
        vertices, triangles = expected_counts(frequency)
        assert len(grid.vertices) == vertices
        assert len(grid.triangles) == triangles
        counts = grid.neighbour_counts()
        assert counts[5] == 12
        assert counts.get(6, 0) == vertices - 12


class TestEvenness:
    @pytest.mark.parametrize(
        ("frequency", "area_spread", "edge_spread"),
        [
            (1, 1.0, 1.0),
            (2, 1.2031, 1.135),
            (4, 1.5166, 1.2856),
            (8, 1.7354, 1.379),
            (16, 1.8595, 1.4286),
        ],
    )
    def test_the_spread_keeps_climbing_and_the_areas_sum_to_the_sphere(
        self, frequency, area_spread, edge_spread
    ):
        grid = GeodesicGrid(frequency)
        areas = grid.areas_km2()
        assert spread(areas) == pytest.approx(area_spread, abs=1e-3)
        assert spread(grid.edge_lengths()) == pytest.approx(edge_spread, abs=1e-3)
        assert sum(areas) / (4 * math.pi * 6371.0088**2) == pytest.approx(1.0, abs=1e-8)


class TestLocation:
    def test_a_triangles_centroid_locates_back_to_it(self):
        grid = GeodesicGrid(6)
        rng = random.Random(257)
        for _ in range(100):
            lat, lon = math.degrees(math.asin(rng.uniform(-1, 1))), rng.uniform(-180, 180)
            t = grid.locate(lat, lon)
            assert grid.locate(*grid.centroid(t)) == t

    def test_refusals(self):
        with pytest.raises(Invalid):
            GeodesicGrid(0)
        with pytest.raises(Outside):
            GeodesicGrid(2).locate(91, 0)
