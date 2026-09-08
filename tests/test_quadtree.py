from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.quadtree import QuadTree


class TestRangeQuery:
    def test_range_matches_brute(self):
        rng = random.Random(13)
        for _ in range(2000):
            qt = QuadTree(BBox(0, 0, 100, 100), capacity=4)
            pts = [
                (rng.uniform(0, 100), rng.uniform(0, 100))
                for _ in range(rng.randint(1, 80))
            ]
            for p in pts:
                qt.insert(p)
            box = BBox(10, 10, 55, 45)
            got = sorted(qt.range_query(box))
            brute = sorted(p for p in pts if box.contains_point(*p))
            assert got == brute

    def test_length_tracks_insertions(self):
        qt = QuadTree(BBox(0, 0, 10, 10))
        for p in [(1, 1), (2, 2), (3, 3)]:
            qt.insert(p)
        assert len(qt) == 3


class TestDepthIsSpaceDriven:
    def test_clustering_deepens_the_tree_far_more_than_count(self):
        rng = random.Random(13)

        def uniform_depth():
            qt = QuadTree(BBox(0, 0, 100, 100), 4)
            for _ in range(500):
                qt.insert((rng.uniform(0, 100), rng.uniform(0, 100)))
            return qt.depth()

        def clustered_depth():
            qt = QuadTree(BBox(0, 0, 100, 100), 4)
            for _ in range(500):
                qt.insert((rng.uniform(49.9, 50.1), rng.uniform(49.9, 50.1)))
            return qt.depth()

        u = sum(uniform_depth() for _ in range(20)) / 20
        c = sum(clustered_depth() for _ in range(20)) / 20
        # same 500 points: clustered is far deeper (measured ~6 vs ~15)
        assert u < 9
        assert c > 12
        assert c > u * 1.8


class TestRefusals:
    def test_a_non_positive_capacity_is_refused(self):
        with pytest.raises(Invalid):
            QuadTree(BBox(0, 0, 1, 1), capacity=0)

    def test_a_point_outside_the_region_is_refused(self):
        qt = QuadTree(BBox(0, 0, 10, 10))
        with pytest.raises(Invalid):
            qt.insert((20, 20))
