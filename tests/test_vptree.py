from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid, Missing
from atlas.vptree import VPTree, brute_nearest, great_circle


def _scatter(n: int, rng) -> list[tuple[float, float]]:
    return [(rng.random(), rng.random()) for _ in range(n)]


class TestExactness:
    def test_every_answer_matches_brute_force(self):
        rng = random.Random(212)
        for n in (500, 2000):
            pts = _scatter(n, rng)
            tree = VPTree(pts)
            for _ in range(100):
                q = (rng.random(), rng.random())
                _, d = tree.nearest(q)
                assert d == pytest.approx(brute_nearest(pts, q)[1], abs=1e-12)

    def test_a_single_point_tree(self):
        assert VPTree([(0, 0)]).nearest((1, 1)) == ((0, 0), pytest.approx(2**0.5))


class TestPruning:
    def test_visits_rise_by_about_a_node_and_a_half_per_doubling(self):
        rng = random.Random(212)
        means = {}
        for n in (500, 1000, 2000, 4000, 8000):
            pts = _scatter(n, rng)
            tree = VPTree(pts)
            queries = [(rng.random(), rng.random()) for _ in range(200)]
            total = 0
            for q in queries:
                tree.nearest(q)
                total += tree.visits
                brute_nearest(pts, q)
            means[n] = total / 200
        assert means[500] == pytest.approx(16.9, abs=0.1)
        assert means[8000] == pytest.approx(23.1, abs=0.1)
        assert means[8000] / 8000 < 0.003
        steps = [means[2 * n] - means[n] for n in (500, 1000, 2000, 4000)]
        assert all(0.5 < s < 3.0 for s in steps)

    def test_the_sphere_prunes_like_the_plane(self):
        rng = random.Random(215)
        pts = [(rng.uniform(-80, 80), rng.uniform(-180, 180)) for _ in range(2000)]
        tree = VPTree(pts, metric=great_circle)
        total = 0
        for _ in range(50):
            q = (rng.uniform(-80, 80), rng.uniform(-180, 180))
            _, d = tree.nearest(q)
            total += tree.visits
            assert d == pytest.approx(brute_nearest(pts, q, great_circle)[1], abs=1e-9)
        assert 12 < total / 50 < 30

    def test_a_central_vantage_is_barely_worse_than_a_random_one(self):
        rng = random.Random(216)
        pts = _scatter(4000, rng)
        queries = [(rng.random(), rng.random()) for _ in range(200)]

        def mean_visits(tree: VPTree) -> float:
            total = 0
            for q in queries:
                tree.nearest(q)
                total += tree.visits
            return total / 200

        central = mean_visits(VPTree(pts, central_vantage=True))
        randoms = [mean_visits(VPTree(pts, seed=s)) for s in range(5)]
        assert central > min(randoms)
        assert central / (sum(randoms) / 5) < 1.15


class TestRefusals:
    def test_an_empty_tree_is_refused(self):
        with pytest.raises(Invalid):
            VPTree([])
        with pytest.raises(Missing):
            brute_nearest([], (0, 0))
