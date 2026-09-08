from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.geohash import bounds, encode
from atlas.geohashneighbors import adjacent, neighbors, nine_cells


class TestAdjacency:
    def test_a_known_north_neighbor(self):
        assert adjacent("u4pruyd", "n") == "u4pruyf"

    def test_opposite_steps_cancel(self):
        rng = random.Random(57)
        for _ in range(20000):
            gh = encode(rng.uniform(-70, 70), rng.uniform(-170, 170), 7)
            assert adjacent(adjacent(gh, "n"), "s") == gh
            assert adjacent(adjacent(gh, "e"), "w") == gh

    def test_neighbors_returns_all_eight(self):
        n = neighbors("gcpuv")
        assert set(n) == {"n", "s", "e", "w", "ne", "nw", "se", "sw"}
        assert len(set(n.values())) == 8

    def test_nine_cells_includes_the_center(self):
        cells = nine_cells("gcpuv")
        assert cells[0] == "gcpuv"
        assert len(cells) == 9


class TestSeamCoverage:
    def test_the_nine_cells_cover_the_local_neighborhood_a_single_cell_misses(self):
        rng = random.Random(57)
        cell = bounds(encode(40, 10, 6))
        dlat, dlon = cell[2] - cell[0], cell[3] - cell[1]
        missed_single = missed_nine = total = 0
        for _ in range(5000):
            lat, lon = rng.uniform(-60, 60), rng.uniform(-160, 160)
            q = encode(lat, lon, 6)
            cells = set(nine_cells(q))
            for _ in range(5):
                nb = encode(
                    lat + rng.uniform(-dlat * 0.9, dlat * 0.9),
                    lon + rng.uniform(-dlon * 0.9, dlon * 0.9),
                    6,
                )
                total += 1
                if nb != q:
                    missed_single += 1
                if nb not in cells:
                    missed_nine += 1
        assert missed_nine == 0  # the nine cells have no seam gaps
        assert missed_single > total * 0.5  # a single cell leaks most of them


class TestRefusals:
    def test_an_empty_geohash_is_refused(self):
        with pytest.raises(Invalid):
            adjacent("", "n")

    def test_a_bad_direction_is_refused(self):
        with pytest.raises(Invalid):
            adjacent("gcpuv", "up")
