from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid, Missing
from atlas.kdtree import KDTree, _dist2


class TestQueriesMatchBrute:
    def test_nearest_knn_and_range_all_match_brute(self):
        rng = random.Random(11)
        nn_visit_fraction = []
        for _ in range(2000):
            n = rng.randint(1, 60)
            pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
            tree = KDTree(pts)
            q = (rng.uniform(0, 100), rng.uniform(0, 100))

            got = tree.nearest(q)
            brute = min(pts, key=lambda p: _dist2(p, q))
            assert _dist2(got, q) == _dist2(brute, q)
            nn_visit_fraction.append(tree.visits / n)

            k = min(5, n)
            gk = tree.k_nearest(q, k)
            bk = sorted(pts, key=lambda p: _dist2(p, q))[:k]
            assert sorted(_dist2(p, q) for p in gk) == sorted(_dist2(p, q) for p in bk)

            box = BBox(20, 20, 70, 60)
            got_range = sorted(tree.range_query(box))
            brute_range = sorted(p for p in pts if box.contains_point(*p))
            assert got_range == brute_range

        assert sum(nn_visit_fraction) / len(nn_visit_fraction) < 0.6


class TestPruning:
    def test_nearest_visits_a_tiny_fraction_on_a_large_tree(self):
        rng = random.Random(1)
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(5000)]
        tree = KDTree(pts)
        visits = []
        for _ in range(500):
            q = (rng.uniform(0, 1000), rng.uniform(0, 1000))
            tree.nearest(q)
            visits.append(tree.visits)
        mean_visits = sum(visits) / len(visits)
        # measured ~19 nodes out of 5000; a brute scan would touch all 5000
        assert mean_visits < 100


class TestRefusals:
    def test_none_points_is_refused(self):
        with pytest.raises(Invalid):
            KDTree(None)

    def test_nearest_on_an_empty_tree_is_refused(self):
        with pytest.raises(Missing):
            KDTree([]).nearest((0, 0))

    def test_a_non_positive_k_is_refused(self):
        with pytest.raises(Invalid):
            KDTree([(1, 1)]).k_nearest((0, 0), 0)
