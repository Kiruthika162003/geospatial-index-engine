from __future__ import annotations

import random

import pytest

from atlas.errors import Invalid
from atlas.intervaltree import IntervalTree


class TestQueries:
    def test_a_concrete_stab(self):
        t = IntervalTree([(1, 5), (3, 8), (10, 12), (6, 7)])
        assert sorted(t.stab(4)) == [(1, 5), (3, 8)]
        assert t.stab(9) == []

    def test_stab_and_overlap_match_brute(self):
        rng = random.Random(65)
        for _ in range(5000):
            n = rng.randint(1, 80)
            ivs = []
            for _ in range(n):
                a = rng.randint(0, 100)
                ivs.append((a, a + rng.randint(0, 20)))
            t = IntervalTree(ivs)
            pt = rng.randint(0, 120)
            assert sorted(t.stab(pt)) == sorted(
                iv for iv in ivs if iv[0] <= pt <= iv[1]
            )
            ql = rng.randint(0, 100)
            qh = ql + rng.randint(0, 15)
            assert sorted(t.overlap(ql, qh)) == sorted(
                iv for iv in ivs if iv[0] <= qh and ql <= iv[1]
            )


class TestPruning:
    def test_a_narrow_stab_visits_few_nodes(self):
        rng = random.Random(1)
        ivs = [(a, a + 5) for a in (rng.randint(0, 100000) for _ in range(20000))]
        t = IntervalTree(ivs)
        t.stab(50000)
        assert t.visits < 100  # measured ~17 of 20000


class TestRefusals:
    def test_none_is_refused(self):
        with pytest.raises(Invalid):
            IntervalTree(None)

    def test_an_inverted_interval_is_refused(self):
        with pytest.raises(Invalid):
            IntervalTree([(5, 1)])

    def test_an_inverted_query_is_refused(self):
        with pytest.raises(Invalid):
            IntervalTree([(1, 2)]).overlap(5, 1)
