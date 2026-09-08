from __future__ import annotations

import random

import pytest

from atlas.bbox import BBox
from atlas.errors import Invalid
from atlas.rtree import RTree


class TestQueriesMatchBrute:
    def test_range_queries_match_a_brute_scan(self):
        rng = random.Random(17)
        for _ in range(1500):
            tree = RTree(max_entries=6)
            items = []
            n = rng.randint(1, 60)
            for i in range(n):
                x, y = rng.uniform(0, 100), rng.uniform(0, 100)
                w, h = rng.uniform(0, 5), rng.uniform(0, 5)
                b = BBox(x, y, x + w, y + h)
                tree.insert(b, i)
                items.append((b, i))
            assert len(tree) == n
            q = BBox(20, 20, 60, 55)
            assert sorted(tree.query(q)) == sorted(
                i for b, i in items if b.intersects(q)
            )


class TestBalanceAndOverlap:
    def test_the_tree_stays_shallow_as_it_grows(self):
        rng = random.Random(1)
        tree = RTree(max_entries=8)
        for i in range(3000):
            x, y = rng.uniform(0, 1000), rng.uniform(0, 1000)
            tree.insert(BBox(x, y, x + 1, y + 1), i)
        assert len(tree) == 3000
        assert len(tree.query(BBox(0, 0, 1000, 1000))) == 3000
        # 3000 items under branching factor 8 should be only a few levels deep
        assert tree._height() <= 5

    def test_sibling_overlap_is_real(self):
        # the R-tree's signature cost: unlike a space partition, siblings overlap
        rng = random.Random(2)
        tree = RTree(max_entries=8)
        for i in range(2000):
            x, y = rng.uniform(0, 1000), rng.uniform(0, 1000)
            tree.insert(BBox(x, y, x + 1, y + 1), i)
        assert tree.sibling_overlap() > 0


class TestRefusals:
    def test_too_small_a_fanout_is_refused(self):
        with pytest.raises(Invalid):
            RTree(max_entries=3)

    def test_a_none_box_is_refused(self):
        with pytest.raises(Invalid):
            RTree().insert(None, 1)
