from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.gridindex import GridIndex


class TestRangeQuery:
    def test_range_matches_brute(self):
        rng = random.Random(39)
        for _ in range(2000):
            g = GridIndex(cell_size=5.0)
            pts = [
                (rng.uniform(0, 100), rng.uniform(0, 100))
                for _ in range(rng.randint(1, 80))
            ]
            for p in pts:
                g.insert(p)
            box = BBox(12, 20, 63, 58)
            assert sorted(g.range_query(box)) == sorted(
                p for p in pts if box.contains_point(*p)
            )

    def test_length_tracks_insertions(self):
        g = GridIndex(1.0)
        for p in [(0.5, 0.5), (1.5, 1.5)]:
            g.insert(p)
        assert len(g) == 2


class TestBucketOccupancy:
    def test_clustering_overloads_a_single_bucket(self):
        rng = random.Random(39)
        uniform = GridIndex(5.0)
        clustered = GridIndex(5.0)
        for _ in range(4000):
            uniform.insert((rng.uniform(0, 100), rng.uniform(0, 100)))
            clustered.insert((rng.uniform(49, 51), rng.uniform(49, 51)))
        # uniform spreads across many cells with small buckets
        assert uniform.occupied_cells() > 300
        assert uniform.max_bucket() < 40
        # clustered collapses into a handful of cells, one holding a quarter of all
        assert clustered.occupied_cells() <= 6
        assert clustered.max_bucket() > 4000 * 0.2


class TestRefusals:
    def test_a_non_positive_cell_size_is_refused(self):
        with pytest.raises(Invalid):
            GridIndex(0)
