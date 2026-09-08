from __future__ import annotations

import random

import pytest

from atlas.balltree import BallTree, _dist
from atlas.errors import Invalid, Missing


class TestNearest:
    def test_it_matches_brute_and_prunes(self):
        rng = random.Random(49)
        fractions = []
        for _ in range(2000):
            n = rng.randint(1, 60)
            pts = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(n)]
            tree = BallTree(pts)
            q = (rng.uniform(0, 100), rng.uniform(0, 100))
            got = tree.nearest(q)
            brute = min(pts, key=lambda p: _dist(p, q))
            assert _dist(got, q) == pytest.approx(_dist(brute, q))
            fractions.append(tree.visits / n)
        assert sum(fractions) / len(fractions) < 0.6

    def test_a_large_tree_visits_few_leaves(self):
        rng = random.Random(1)
        pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(5000)]
        tree = BallTree(pts)
        visits = []
        for _ in range(500):
            q = (rng.uniform(0, 1000), rng.uniform(0, 1000))
            tree.nearest(q)
            visits.append(tree.visits)
        assert sum(visits) / len(visits) < 100  # measured ~22 of 5000


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            BallTree(None)

    def test_nearest_on_empty_is_refused(self):
        with pytest.raises(Missing):
            BallTree([]).nearest((0, 0))
